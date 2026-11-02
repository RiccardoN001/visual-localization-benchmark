"""Visual localization against a rendered reference database.

For each query: global retrieval of the top-k references, 2D-2D matching against each candidate,
geometric rerank, then absolute PnP using the depth map of the best reference.

    python -m loc.localize --dataset keble --retriever salad --matcher lightglue
"""
import argparse
import logging
import os
import statistics
import time
import traceback
from pathlib import Path

os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
os.environ.setdefault("XFORMERS_DISABLED", "1")

import numpy as np  # noqa: E402
import torch  # noqa: E402

# Compute nodes without internet access: resolve torch.hub repositories from the local cache.
if os.environ.get("LOC_OFFLINE_HUB"):
    from hpc import offline_hub  # noqa: E402,F401

from loc import paths, vram  # noqa: E402
from loc.capture_io import save_capture_trajectory  # noqa: E402
from loc.database import build_or_load_index, search  # noqa: E402
from loc.datasets import SCENES, Dataset  # noqa: E402
from loc.evaluate import THRESHOLDS, recall, threshold_key  # noqa: E402
from loc.geometry import K_from, pose_error, query_pose_from_pnp  # noqa: E402
from loc.matchers import MATCHER_NAMES, get_matcher  # noqa: E402
from loc.pose import PnPConfig, match_candidates, solve_pnp  # noqa: E402
from loc.retrievers import RETRIEVER_NAMES, get_retriever  # noqa: E402
from loc.utils import load_json, parse_options, save_json, setup_logging  # noqa: E402

log = logging.getLogger("loc")

STAGES = ("retrieval", "match", "pnp", "estimate", "total")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", required=True, choices=list(SCENES))
    p.add_argument("--retriever", required=True, choices=RETRIEVER_NAMES)
    p.add_argument("--matcher", required=True, choices=MATCHER_NAMES)
    p.add_argument("--k", type=int, default=3, help="number of retrieved candidates")
    p.add_argument("--retriever-opt", action="append", metavar="KEY=VALUE",
                   help="override a retriever constructor argument (repeatable)")
    p.add_argument("--matcher-opt", action="append", metavar="KEY=VALUE",
                   help="override a matcher constructor argument (repeatable)")
    p.add_argument("--pnp-reprojection-error", type=float, default=PnPConfig.reprojection_error)
    p.add_argument("--pnp-iterations", type=int, default=PnPConfig.iterations)
    p.add_argument("--pnp-confidence", type=float, default=PnPConfig.confidence)
    p.add_argument("--rerank-threshold", type=float, default=PnPConfig.rerank_threshold)
    p.add_argument("--query", default=None, help="process a single query (qid or timestamp)")
    p.add_argument("--query-stride", type=int, default=1, help="keep one capture stop every N")
    p.add_argument("--limit", type=int, default=0, help="process only the first N queries")
    p.add_argument("--resume", action="store_true",
                   help="reuse queries that already have a result file instead of recomputing them")
    p.add_argument("--no-save-matches", dest="save_matches", action="store_false",
                   help="do not store the matched keypoints of every query (they are large, and "
                        "writing them is part of total_sec)")
    p.add_argument("--name", default=None, help="run name (default: <dataset>_<retriever>_<matcher>_k<k>)")
    p.add_argument("--data-root", type=Path, default=paths.DATA_ROOT)
    p.add_argument("--output-root", type=Path, default=paths.OUTPUT_ROOT)
    return p.parse_args()


def select_queries(dataset, query, limit):
    queries = dataset.queries()
    if query:
        queries = [q for q in queries if query in (q["qid"], q["timestamp"])]
        if not queries:
            raise SystemExit(f"query '{query}' not found")
    if limit:
        queries = queries[:limit]
    if not queries:
        raise SystemExit("no queries to process")
    return queries


def result_path(dirs, qid):
    return dirs["results"] / f"{qid.replace('/', '__')}_result.json"


def localize_query(q, retriever, index, metadata, matcher, dataset, dirs, args, pnp_config):
    t_start = time.perf_counter()
    query_path = Path(q["image_path"])
    stem = q["qid"].replace("/", "__")

    # Retrieval. Peak counters are reset before each stage, otherwise they would report the
    # all-time maximum (usually model loading) instead of the peak of this stage.
    retriever.reset_timers()
    vram.reset_peak()
    t = time.perf_counter()
    embedding = retriever.embed(query_path)
    vram.sync()
    t_embed = time.perf_counter() - t
    vram_retrieval = (vram.peak_mb(), vram.peak_reserved_mb())

    t = time.perf_counter()
    candidates = search(index, metadata, embedding, args.k)   # FAISS runs on the CPU
    t_faiss = time.perf_counter() - t
    t_retrieval = t_embed + t_faiss
    save_json({"query": str(query_path), "candidates": candidates},
              dirs["retrieval"] / f"{stem}_retrieval.json")

    # Matching against every candidate, plus the geometric rerank.
    matcher.reset_timers()
    vram.reset_peak()
    t = time.perf_counter()
    ranked = match_candidates(matcher, query_path, candidates, pnp_config)
    t_match = time.perf_counter() - t
    vram_match = (vram.peak_mb(), vram.peak_reserved_mb())
    if args.save_matches:
        save_json({"query": str(query_path), "matches": [
            {**{k: r[k] for k in ("candidate", "inlier_count", "raw_matches")},
             "pts_q": r["pts_q"].tolist(), "pts_r": r["pts_r"].tolist()} for r in ranked]},
            dirs["matches"] / f"{stem}_matches.json")

    # PnP with the best candidate.
    t_pnp = 0.0
    pnp_detail = {}
    if not ranked:
        result = {"success": False, "error": "no retrieval candidates"}
    else:
        best = ranked[0]
        reference = best["candidate"]["metadata"]
        t = time.perf_counter()
        pnp, pnp_detail = solve_pnp(best["pts_q"], best["pts_r"], reference,
                                    K_from(*q["K_query"]), dataset.read_depth, pnp_config)
        t_pnp = time.perf_counter() - t

        if pnp is None:
            result = {"success": False, "error": "PnP failed or not enough valid depth"}
        else:
            rvec, tvec, num_inliers = pnp
            R_c2w, t_c2w = query_pose_from_pnp(rvec, tvec, reference["R_c2w"], reference["t_c2w"])
            result = {
                "success": True,
                "query_R_c2w": R_c2w.tolist(), "query_t_c2w": t_c2w.tolist(),
                "relative_rvec": rvec.tolist(), "relative_tvec": tvec.tolist(),
                "pnp_inliers": num_inliers,
                "reference_pose": reference,
                "chosen_rank": best["candidate"]["rank"],
                "chosen_inliers": best["inlier_count"],
                "raw_matches": best["raw_matches"],
                # Distance to the chosen reference tells pipeline failures apart from gaps in the
                # database coverage.
                "dist_to_reference_m": float(np.linalg.norm(np.array(reference["t_c2w"]) - t_c2w)),
            }
            if q["gt_t_c2w"] is not None:
                dt, dr = pose_error(R_c2w, t_c2w, q["gt_R_c2w"], q["gt_t_c2w"])
                result["gt_position_error_m"], result["gt_rotation_error_deg"] = dt, dr

    result.update({
        "qid": q["qid"], "query_image": str(query_path),
        "query_timestamp": q["timestamp"], "query_sensor_id": q["sensor_id"],
        # Same schema for every retriever/matcher combination. estimate_sec excludes JSON I/O,
        # total_sec is the wall-clock time of the query.
        "timing": {
            "retrieval_sec": t_retrieval,
            "match_sec": t_match,
            "pnp_sec": t_pnp,
            "estimate_sec": t_retrieval + t_match + t_pnp,
            "total_sec": time.perf_counter() - t_start,
            "detail": {
                "retrieval": {"embed_sec": t_embed, "faiss_sec": t_faiss,
                              **retriever.timing_detail()},
                "match": matcher.timing_detail(),
                "pnp": pnp_detail,
            },
        },
        "vram_peak_mb": {"retrieval": vram_retrieval[0], "match": vram_match[0]},
        "vram_peak_reserved_mb": {"retrieval": vram_retrieval[1], "match": vram_match[1]},
    })
    save_json(result, result_path(dirs, q["qid"]))
    return result


def median(xs):
    return statistics.median(xs) if xs else None


def mean(xs):
    return statistics.mean(xs) if xs else None


def format_mb(value):
    return "n/a" if value is None else f"{value:.0f} MB"


def summarize(results, queries, resumed):
    """Aggregate timing, accuracy and memory over the run."""
    timings = {s: [r["timing"][f"{s}_sec"] for r in results] for s in STAGES}
    # Memory is aggregated only over queries computed in this invocation: resumed results did no
    # GPU work here, and mixing them would make the medians depend on how the run was split.
    fresh = [r for r, was_resumed in zip(results, resumed) if not was_resumed]
    peaks = {kind: {s: [r[kind][s] for r in fresh if r.get(kind, {}).get(s) is not None]
                    for s in ("retrieval", "match")}
             for kind in ("vram_peak_mb", "vram_peak_reserved_mb")}

    with_gt = [r for r, q in zip(results, queries) if q["gt_t_c2w"] is not None]
    errors = [(r["gt_position_error_m"], r["gt_rotation_error_deg"])
              for r in with_gt if r.get("gt_position_error_m") is not None]
    return {
        "num_queries": len(results),
        "num_success": sum(1 for r in results if r.get("success")),
        "num_resumed": sum(resumed),
        "timing_mean_sec": {s: mean(v) for s, v in timings.items()},
        "timing_median_sec": {s: median(v) for s, v in timings.items()},
        "gt_num_with_gt": len(with_gt),
        "gt_num_estimated": len(errors),
        "gt_median_position_error_m": median([e[0] for e in errors]),
        "gt_median_rotation_error_deg": median([e[1] for e in errors]),
        "gt_recall": recall(errors, len(with_gt), THRESHOLDS),
        "dist_to_reference_median_m": median([r["dist_to_reference_m"] for r in results
                                              if r.get("success")]),
        "peaks": peaks,
    }


def main():
    args = parse_args()
    setup_logging()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    name = args.name or paths.experiment_name(args.dataset, args.retriever, args.matcher, args.k)
    dirs = paths.run_dirs(args.output_root, name)
    log.info("dataset=%s retriever=%s matcher=%s k=%d device=%s", args.dataset, args.retriever,
             args.matcher, args.k, device)
    log.info("output: %s", dirs["root"])
    log.info("HuggingFace: %s", "using HF_TOKEN" if os.environ.get("HF_TOKEN")
             else "HF_TOKEN not set, using cached credentials")

    dataset = Dataset(args.dataset, args.data_root, args.query_stride)
    pnp_config = PnPConfig(args.pnp_reprojection_error, args.pnp_iterations, args.pnp_confidence,
                           args.rerank_threshold)

    # Model weights are measured as the allocated-memory difference around each construction.
    # Both models stay resident for the whole run, so this is the only point where the cost of a
    # single component can be read. The index build sits in between and is measured separately,
    # since on a cold cache it runs the retriever on every reference image.
    t_setup = time.perf_counter()
    vram_baseline = vram.allocated_mb()
    retriever = get_retriever(args.retriever, device, **parse_options(args.retriever_opt))
    vram_after_retriever = vram.allocated_mb()

    vram.reset_peak()
    index, metadata = build_or_load_index(retriever, dataset.references(), dataset.db_root,
                                          args.output_root)
    vram_db_build_peak = vram.peak_mb()

    vram_before_matcher = vram.allocated_mb()
    matcher = get_matcher(args.matcher, device, **parse_options(args.matcher_opt))
    vram_after_matcher = vram.allocated_mb()
    setup_time = time.perf_counter() - t_setup
    log.info("setup done in %.1fs", setup_time)

    queries = select_queries(dataset, args.query, args.limit)
    log.info("%d queries (%d with ground truth)", len(queries),
             sum(q["gt_t_c2w"] is not None for q in queries))

    results, resumed, processed = [], [], []
    t_batch = time.perf_counter()
    for i, q in enumerate(queries, 1):
        try:
            r, was_resumed = None, False
            path = result_path(dirs, q["qid"])
            if args.resume and path.exists():
                try:
                    r, was_resumed = load_json(path), True
                except ValueError:
                    log.warning("unreadable result for %s, recomputing", q["qid"])
            if r is None:
                r = localize_query(q, retriever, index, metadata, matcher, dataset, dirs, args,
                                   pnp_config)
        except Exception as exc:  # noqa: BLE001  one failing query must not stop the run
            log.error("[%d/%d] %s: %s: %s", i, len(queries), q["qid"], type(exc).__name__, exc)
            traceback.print_exc()
            continue
        results.append(r)
        resumed.append(was_resumed)
        processed.append(q)
        err = (f" err={r['gt_position_error_m']:.2f}m/{r['gt_rotation_error_deg']:.1f}deg"
               if r.get("gt_position_error_m") is not None else "")
        status = "resumed" if was_resumed else ("ok" if r.get("success") else "failed")
        log.info("[%d/%d] %s: %s %.2fs%s", i, len(queries), q["qid"], status,
                 r["timing"]["total_sec"], err)
    batch_time = time.perf_counter() - t_batch
    if any(resumed):
        log.info("%d/%d queries resumed from existing results, %d recomputed", sum(resumed),
                 len(queries), len(queries) - sum(resumed))

    rows = [(r["query_timestamp"], r["query_sensor_id"], r["query_R_c2w"], r["query_t_c2w"])
            for r in results if r.get("success")]
    save_capture_trajectory(rows, dirs["root"] / "estimated_trajectories.txt")
    log.info("%d estimated poses -> %s", len(rows), dirs["root"] / "estimated_trajectories.txt")

    stats = summarize(results, processed, resumed)
    peaks = stats.pop("peaks")
    stage_peaks = [vram_db_build_peak, *peaks["vram_peak_mb"]["retrieval"],
                   *peaks["vram_peak_mb"]["match"]]
    stage_peaks = [p for p in stage_peaks if p is not None]
    summary = {
        "dataset": args.dataset, "retriever": args.retriever, "matcher": args.matcher, "k": args.k,
        "name": name,
        "retriever_config": retriever.config_summary(),
        "matcher_config": matcher.config_summary(),
        "pnp_config": vars(pnp_config),
        "setup_time_sec": setup_time,
        "batch_time_sec": batch_time,
        **stats,
        # GPU memory in MB (None on CPU). See "Timing and memory measurement" in the README.
        "vram_mb": {
            "baseline": vram_baseline,
            "retriever_weights": vram.delta_mb(vram_baseline, vram_after_retriever),
            "matcher_weights": vram.delta_mb(vram_before_matcher, vram_after_matcher),
            "db_build_peak": vram_db_build_peak,
            "retrieval_peak_median": median(peaks["vram_peak_mb"]["retrieval"]),
            "match_peak_median": median(peaks["vram_peak_mb"]["match"]),
            "retrieval_peak_reserved_median": median(peaks["vram_peak_reserved_mb"]["retrieval"]),
            "match_peak_reserved_median": median(peaks["vram_peak_reserved_mb"]["match"]),
            "max_stage_peak": max(stage_peaks) if stage_peaks else None,
            # The caching allocator does not return memory to the driver during the run, so the
            # reserved memory at the end is its high-water mark.
            "run_reserved_peak": vram.reserved_mb(),
            "nvml_process_at_end": vram.nvml_process_mb(),
            "nvml_device_at_end": vram.nvml_device_mb(),
        },
    }
    save_json(summary, dirs["root"] / "batch_timing.json")

    log.info("%s + %s (k=%d): %d/%d queries localized in %.0fs", args.retriever, args.matcher,
             args.k, summary["num_success"], len(results), batch_time)
    for stage in STAGES:
        if summary["timing_median_sec"][stage] is not None:
            log.info("  %-9s mean %.3fs  median %.3fs", stage, summary["timing_mean_sec"][stage],
                     summary["timing_median_sec"][stage])
    vram_mb = summary["vram_mb"]
    if vram_mb["retriever_weights"] is not None:
        log.info("  VRAM %s: weights %s, median peak %s (reserved %s)", args.retriever,
                 format_mb(vram_mb["retriever_weights"]), format_mb(vram_mb["retrieval_peak_median"]),
                 format_mb(vram_mb["retrieval_peak_reserved_median"]))
        log.info("  VRAM %s: weights %s, median peak %s (reserved %s)", args.matcher,
                 format_mb(vram_mb["matcher_weights"]), format_mb(vram_mb["match_peak_median"]),
                 format_mb(vram_mb["match_peak_reserved_median"]))
        log.info("  VRAM run: %s reserved by torch, %s used by the process (NVML)",
                 format_mb(vram_mb["run_reserved_peak"]), format_mb(vram_mb["nvml_process_at_end"]))
    if summary["gt_num_with_gt"]:
        log.info("  ground truth: %d queries, %d localized, %d failed (counted as misses)",
                 summary["gt_num_with_gt"], summary["gt_num_estimated"],
                 summary["gt_num_with_gt"] - summary["gt_num_estimated"])
        if summary["gt_num_estimated"]:
            log.info("  median error %.2f m / %.2f deg, median distance to the reference %.2f m",
                     summary["gt_median_position_error_m"], summary["gt_median_rotation_error_deg"],
                     summary["dist_to_reference_median_m"])
        log.info("  recall over %d queries with ground truth: %s", summary["gt_num_with_gt"],
                 "  ".join(f"{threshold_key(m, d)}="
                           f"{100 * summary['gt_recall'][threshold_key(m, d)]:.1f}%"
                           for m, d in THRESHOLDS))


if __name__ == "__main__":
    main()
