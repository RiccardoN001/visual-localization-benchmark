import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from ..paths import THIRD_PARTY_ROOT
from . import Matcher

REPO_DIR = THIRD_PARTY_ROOT / "SuperGluePretrainedNetwork"


def import_matching(repo_dir):
    """Import models.matching.Matching from the cloned repository without keeping it on sys.path.

    The clone exposes a top-level package named `models`, and so does the SALAD torch.hub code.
    The clone is therefore only importable during this call, and its modules are removed again.
    """
    root = Path(repo_dir)
    if not (root / "models" / "matching.py").exists():
        raise SystemExit(f"SuperGlue repository not found in {root}: "
                         f"run `python -m hpc.prefetch_models --only superglue`")

    saved = {k: sys.modules.pop(k) for k in list(sys.modules)
             if k == "models" or k.startswith("models.")}
    sys.path.insert(0, str(root))
    try:
        import models.matching
        return models.matching.Matching
    finally:
        sys.path.remove(str(root))
        for k in [k for k in sys.modules if k == "models" or k.startswith("models.")]:
            del sys.modules[k]
        sys.modules.update(saved)


class SuperGlueMatcher(Matcher):
    name = "superglue"

    def __init__(self, device, weights="outdoor", max_kpts=2048, kp_threshold=0.005,
                 match_threshold=0.2, sinkhorn_iterations=20, repo_dir=REPO_DIR, **cfg):
        super().__init__(device, weights=weights, max_kpts=max_kpts, kp_threshold=kp_threshold,
                         match_threshold=match_threshold, sinkhorn_iterations=sinkhorn_iterations,
                         **cfg)
        Matching = import_matching(repo_dir)
        self.model = Matching({
            "superpoint": {"nms_radius": 4, "keypoint_threshold": kp_threshold,
                           "max_keypoints": max_kpts},
            "superglue": {"weights": weights, "sinkhorn_iterations": sinkhorn_iterations,
                          "match_threshold": match_threshold},
        }).eval().to(device)
        self._query_cache = None
        self.reset_timers()

    def reset_timers(self):
        self._times = {"query_load_sec": 0.0, "query_extract_sec": 0.0, "query_cache_hits": 0,
                       "ref_load_sec": 0.0, "superglue_match_sec": 0.0}

    def timing_detail(self):
        return dict(self._times)

    def _load_gray(self, path):
        # The Magic Leap SuperPoint expects a single-channel image in [0, 1] at full resolution.
        arr = np.asarray(Image.open(path).convert("L"), dtype=np.float32) / 255.0
        return torch.from_numpy(arr)[None, None].to(self.device)

    @torch.inference_mode()
    def match(self, query_path, ref_path):
        # Matching.forward skips SuperPoint for keys that are already present ('keypoints0', ...),
        # so the query prediction is cached and only the reference is extracted again.
        query_path = str(query_path)
        if self._query_cache is not None and self._query_cache[0] == query_path:
            _, pred_q, image_q = self._query_cache
            self._times["query_cache_hits"] += 1
        else:
            t = time.perf_counter()
            image_q = self._load_gray(query_path)
            self._times["query_load_sec"] += time.perf_counter() - t
            t = time.perf_counter()
            pred_q = {k + "0": v for k, v in self.model.superpoint({"image": image_q}).items()}
            self._times["query_extract_sec"] += time.perf_counter() - t
            self._query_cache = (query_path, pred_q, image_q)

        t = time.perf_counter()
        image_r = self._load_gray(ref_path)
        self._times["ref_load_sec"] += time.perf_counter() - t

        t = time.perf_counter()
        pred = self.model({**pred_q, "image0": image_q, "image1": image_r})
        self._times["superglue_match_sec"] += time.perf_counter() - t

        # Inputs are not copied to the output, so the query keypoints are read from pred_q.
        kps_q = pred_q["keypoints0"][0].cpu().numpy()
        kps_r = pred["keypoints1"][0].cpu().numpy()
        matches = pred["matches0"][0].cpu().numpy()   # -1 = unmatched
        valid = matches > -1
        if not valid.any():
            return np.zeros((0, 2), np.float32), np.zeros((0, 2), np.float32)
        return kps_q[valid].astype(np.float32), kps_r[matches[valid]].astype(np.float32)
