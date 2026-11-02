"""Evaluate estimated poses against the ground truth of a dataset.

    python -m loc.evaluate --dataset keble --results output/localization/keble_salad_lightglue_k3
    python -m loc.evaluate --dataset keble --results <run> --thresholds 0.25:2,0.5:5,5:10
"""
import argparse
import json
from pathlib import Path

import numpy as np

from .capture_io import load_trajectories
from .datasets import SCENES, Dataset
from .geometry import pose_error
from .paths import DATA_ROOT

# Recall thresholds as (meters, degrees). A query counts as localized when both errors are below.
THRESHOLDS = [(0.10, 1.0), (0.25, 2.0), (0.50, 3.0), (1.00, 5.0)]


def parse_thresholds(spec):
    """'0.1:1,1:5' -> [(0.1, 1.0), (1.0, 5.0)]."""
    out = []
    for part in filter(None, (p.strip() for p in spec.split(","))):
        try:
            m, d = part.split(":")
            out.append((float(m), float(d)))
        except ValueError:
            raise SystemExit(f"invalid threshold '{part}': expected <meters>:<degrees>, e.g. 0.1:1") from None
    return out or THRESHOLDS


def threshold_key(m, d):
    return f"{m}m_{d}deg"


def recall(errors, denominator, thresholds):
    """errors: list of (position_m, rotation_deg) of the localized queries."""
    if not denominator:
        return {threshold_key(m, d): None for m, d in thresholds}
    return {threshold_key(m, d): sum(1 for dt, dr in errors if dt <= m and dr <= d) / denominator
            for m, d in thresholds}


def error_stats(values):
    if not values:
        return None
    return {"p25": float(np.percentile(values, 25)), "p75": float(np.percentile(values, 75)),
            "max": float(np.max(values))}


def evaluate(estimated_path, queries, thresholds=THRESHOLDS, denominator="with-gt"):
    """Recall and median errors of a Capture trajectory against the query ground truth.

    With denominator='with-gt' recall is computed over the queries that have ground truth, and a
    failed query counts as a miss. With 'all', queries without ground truth count as misses too.
    """
    gt = {(q["timestamp"], q["sensor_id"]): q for q in queries if q["gt_t_c2w"] is not None}
    estimated = load_trajectories(estimated_path)

    errors = [pose_error(R, t, gt[key]["gt_R_c2w"], gt[key]["gt_t_c2w"])
              for key, (R, t) in estimated.items() if key in gt]
    denom = len(queries) if denominator == "all" else len(gt)
    return {
        "num_queries": len(queries),
        "num_with_gt": len(gt),
        "num_estimated": len(errors),
        "num_estimated_without_gt": sum(1 for key in estimated if key not in gt),
        "denominator": denominator,
        "thresholds": [list(t) for t in thresholds],
        "median_position_error_m": float(np.median([e[0] for e in errors])) if errors else None,
        "median_rotation_error_deg": float(np.median([e[1] for e in errors])) if errors else None,
        "position_error_stats_m": error_stats([e[0] for e in errors]),
        "rotation_error_stats_deg": error_stats([e[1] for e in errors]),
        "recall": recall(errors, denom, thresholds),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", required=True, choices=list(SCENES))
    p.add_argument("--results", type=Path, required=True,
                   help="run directory containing estimated_trajectories.txt, or the file itself")
    p.add_argument("--data-root", type=Path, default=DATA_ROOT)
    p.add_argument("--thresholds", default="",
                   help="comma-separated <meters>:<degrees> pairs (default: "
                        + ",".join(f"{m}:{d:g}" for m, d in THRESHOLDS) + ")")
    p.add_argument("--denominator", choices=("with-gt", "all"), default="with-gt")
    p.add_argument("--out", type=Path, default=None,
                   help="output JSON (default: evaluation.json next to the trajectory)")
    args = p.parse_args()

    est_path = args.results / "estimated_trajectories.txt" if args.results.is_dir() else args.results
    if not est_path.exists():
        raise SystemExit(f"not found: {est_path}")

    queries = Dataset(args.dataset, args.data_root).queries()
    report = {"dataset": args.dataset, "estimated_trajectories": str(est_path),
              **evaluate(est_path, queries, parse_thresholds(args.thresholds), args.denominator)}

    denom = report["num_queries"] if args.denominator == "all" else report["num_with_gt"]
    print(f"{args.dataset}: {report['num_estimated']} estimated poses, "
          f"{report['num_with_gt']} queries with ground truth (denominator: {args.denominator})")
    if report["num_estimated_without_gt"]:
        print(f"{report['num_estimated_without_gt']} estimated poses without ground truth, ignored")
    if args.denominator == "all" and report["num_queries"] != report["num_with_gt"]:
        print(f"{report['num_queries'] - report['num_with_gt']} queries without ground truth "
              "counted as misses")
    if report["median_position_error_m"] is not None:
        pos, rot = report["position_error_stats_m"], report["rotation_error_stats_deg"]
        print(f"median error: {report['median_position_error_m']:.3f} m / "
              f"{report['median_rotation_error_deg']:.3f} deg")
        print(f"position error: p25={pos['p25']:.3f} p75={pos['p75']:.3f} max={pos['max']:.1f} m")
        print(f"rotation error: p25={rot['p25']:.3f} p75={rot['p75']:.3f} max={rot['max']:.1f} deg")
    for key, value in report["recall"].items():
        count = "" if value is None else f"  ({round(value * denom)}/{denom})"
        print(f"  recall {key:>14}: {'n/a' if value is None else f'{100 * value:5.1f}%'}{count}")

    out = args.out or est_path.with_name("evaluation.json")
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"report written to {out}")


if __name__ == "__main__":
    main()
