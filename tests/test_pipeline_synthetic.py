# End-to-end test of retrieval, rerank, PnP, outputs and evaluation on a synthetic scene, with an
# oracle retriever and an oracle matcher. It needs neither a GPU, nor model weights, nor datasets.
from argparse import Namespace

import cv2
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

pytest.importorskip("faiss")

from loc import paths  # noqa: E402
from loc.capture_io import quantize_depth, save_capture_trajectory, save_pinhole_sensors, stem_for  # noqa: E402
from loc.database import build_or_load_index  # noqa: E402
from loc.datasets import Dataset  # noqa: E402
from loc.evaluate import evaluate  # noqa: E402
from loc.geometry import K_from  # noqa: E402
from loc.localize import localize_query  # noqa: E402
from loc.matchers import Matcher  # noqa: E402
from loc.pose import PnPConfig, solve_pnp  # noqa: E402
from loc.retrievers import Retriever  # noqa: E402

W, H, F = 320, 240, 200.0
K = (F, F, W / 2, H / 2)
PLANE_DEPTH = 10.0   # every reference camera looks along world +Z at the plane z = 10


class OracleRetriever(Retriever):
    name = "oracle"

    def __init__(self, embeddings):
        super().__init__("cpu")
        self.embeddings = embeddings

    def embed(self, img_path):
        return self.embeddings[str(img_path)][None, :]


class OracleMatcher(Matcher):
    name = "oracle"

    # Exact correspondences: reference pixels lifted with the plane depth, projected in the query.
    def __init__(self, poses):
        super().__init__("cpu")
        self.poses = poses

    def match(self, query_path, ref_path):
        R_q, t_q = self.poses[str(query_path)]
        R_r, t_r = self.poses[str(ref_path)]
        u, v = np.meshgrid(np.arange(10, W - 10, 15.0), np.arange(10, H - 10, 15.0))
        pts_r = np.stack([u.ravel(), v.ravel()], axis=1)
        X_r = np.stack([(pts_r[:, 0] - K[2]) / F, (pts_r[:, 1] - K[3]) / F,
                        np.ones(len(pts_r))], axis=1) * PLANE_DEPTH
        X_q = (R_q.T @ ((R_r @ X_r.T).T + t_r - t_q).T).T
        pts_q = X_q[:, :2] / X_q[:, 2:] * F + np.array(K[2:])
        inside = (pts_q[:, 0] >= 0) & (pts_q[:, 0] < W) & (pts_q[:, 1] >= 0) & (pts_q[:, 1] < H)
        return pts_q[inside].astype(np.float32), pts_r[inside].astype(np.float32)


def make_scene(root):
    """Three reference renders and two queries near the second reference."""
    capture, db = root / "KEBLE" / "capture", root / "KEBLE" / "db"
    (capture / "raw_data").mkdir(parents=True)
    (db / "images").mkdir(parents=True)
    (db / "depth_png").mkdir()
    image = np.full((H, W, 3), 128, np.uint8)

    refs = [(str(i), "render", np.eye(3), np.array([2.0 * i, 0.0, 0.0])) for i in range(3)]
    for ts, sensor, _, _ in refs:
        cv2.imwrite(str(db / "images" / f"{stem_for(ts, sensor)}.jpg"), image)
        cv2.imwrite(str(db / "depth_png" / f"{stem_for(ts, sensor)}.png"),
                    quantize_depth(np.full((H, W), PLANE_DEPTH)))
    save_capture_trajectory(refs, db / "db_trajectories.txt")
    save_pinhole_sensors([("render", "synthetic", W, H, *K)], db / "db_sensors.txt")

    queries = [("100", "phone", Rotation.from_euler("y", 3, degrees=True).as_matrix(),
                np.array([2.3, 0.1, 0.2])),
               ("101", "phone", Rotation.from_euler("x", -2, degrees=True).as_matrix(),
                np.array([1.8, -0.1, -0.3]))]
    with open(capture / "images.txt", "w", encoding="utf-8") as f:
        for ts, sensor, _, _ in queries:
            cv2.imwrite(str(capture / "raw_data" / f"{ts}.jpg"), image)
            f.write(f"{ts}, {sensor}, {ts}.jpg\n")
    save_capture_trajectory(queries, capture / "trajectories.txt")
    save_pinhole_sensors([("phone", "synthetic", W, H, *K)], capture / "sensors.txt")
    return refs, queries


def test_missing_depth_map_is_reported(tmp_path, caplog):
    pts = np.zeros((4, 2), np.float32)
    reference = {"depth_path": str(tmp_path / "missing.png"), "K_ref": K}
    pnp, _ = solve_pnp(pts, pts, reference, K_from(*K), None, PnPConfig())
    assert pnp is None
    assert "missing depth map" in caplog.text


def test_pipeline_recovers_the_query_poses(tmp_path):
    make_scene(tmp_path / "data")
    dataset = Dataset("keble", tmp_path / "data")
    references, queries = dataset.references(), dataset.queries()

    poses = {e["image_path"]: (np.array(e["R_c2w"]), np.array(e["t_c2w"])) for e in references}
    poses.update({q["image_path"]: (np.array(q["gt_R_c2w"]), np.array(q["gt_t_c2w"])) for q in queries})
    # The closest reference (index 1) gets the most similar embedding, the others rank lower.
    embeddings = {e["image_path"]: np.eye(3, dtype=np.float32)[i] for i, e in enumerate(references)}
    for q in queries:
        embeddings[q["image_path"]] = np.array([0.3, 0.9, 0.3], np.float32)

    retriever, matcher = OracleRetriever(embeddings), OracleMatcher(poses)
    output_root = tmp_path / "output"
    index, metadata = build_or_load_index(retriever, references, dataset.db_root, output_root)
    dirs = paths.run_dirs(output_root, "test")
    args = Namespace(k=3, save_matches=True)

    results = [localize_query(q, retriever, index, metadata, matcher, dataset, dirs, args, PnPConfig())
               for q in queries]
    for r in results:
        assert r["success"]
        # Every candidate sees the same plane, so any of them gives an exact pose after the rerank.
        assert r["chosen_inliers"] is not None
        assert r["gt_position_error_m"] < 1e-3 and r["gt_rotation_error_deg"] < 1e-2
        assert set(r["timing"]) >= {"retrieval_sec", "match_sec", "pnp_sec", "estimate_sec", "total_sec"}
        assert (dirs["results"] / f"{r['qid']}_result.json").exists()

    rows = [(r["query_timestamp"], r["query_sensor_id"], r["query_R_c2w"], r["query_t_c2w"])
            for r in results]
    save_capture_trajectory(rows, dirs["root"] / "estimated_trajectories.txt")
    report = evaluate(dirs["root"] / "estimated_trajectories.txt", queries)
    assert report["num_estimated"] == 2
    assert all(v == 1.0 for v in report["recall"].values())

    # A second build reuses the cached index.
    index2, _ = build_or_load_index(retriever, references, dataset.db_root, output_root)
    assert index2.ntotal == index.ntotal == 3


def test_intrinsics_are_read_per_camera(tmp_path):
    make_scene(tmp_path / "data")
    q = Dataset("keble", tmp_path / "data").queries()[0]
    assert np.allclose(K_from(*q["K_query"]), K_from(*K))
