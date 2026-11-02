import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

from ..paths import THIRD_PARTY_ROOT
from . import Matcher

REPO_DIR = THIRD_PARTY_ROOT / "omniglue"


class OmniGlueMatcher(Matcher):
    name = "omniglue"

    # OmniGlue runs on TensorFlow and needs its own environment (see "Installation" in the README).
    def __init__(self, device, min_conf=0.02, repo_dir=REPO_DIR, **cfg):
        super().__init__(device, min_conf=min_conf, **cfg)
        root = Path(repo_dir)
        weights = {"og_export": root / "models" / "og_export",
                   "sp_export": root / "models" / "sp_v6",
                   "dino_export": root / "models" / "dinov2_vitb14_pretrain.pth"}
        missing = [str(p) for p in weights.values() if not p.exists()]
        if missing:
            raise SystemExit("OmniGlue weights not found:\n  " + "\n  ".join(missing))

        # dino_extract.py imports `third_party.dinov2`, which lives at the repository root.
        for p in (str(root / "src"), str(root)):
            if p not in sys.path:
                sys.path.insert(0, p)
        import omniglue
        from omniglue import omniglue_extract

        from ._omniglue_patch import patch_omniglue

        self.og = omniglue.OmniGlue(**{k: str(v) for k, v in weights.items()})
        patch_omniglue(self.og, device)
        self._dino_dim = omniglue_extract.DINO_FEATURE_DIM
        self._match_threshold = omniglue_extract.MATCH_THRESHOLD
        self.min_conf = min_conf
        self._query_cache = None
        self.reset_timers()

    def reset_timers(self):
        self._times = {"query_load_sec": 0.0, "query_extract_sec": 0.0, "query_cache_hits": 0,
                       "ref_load_sec": 0.0, "ref_extract_sec": 0.0, "omniglue_match_sec": 0.0}

    def timing_detail(self):
        return dict(self._times)

    def _extract(self, path):
        """SuperPoint features and DINOv2 keypoint descriptors, as in OmniGlue.FindMatches."""
        import tensorflow as tf
        from omniglue import dino_extract

        t = time.perf_counter()
        image = np.array(Image.open(path).convert("RGB"))
        load_sec = time.perf_counter() - t
        h, w = image.shape[:2]
        sp = self.og.sp_extract(image)
        dino_desc = dino_extract.get_dino_descriptors(
            self.og.dino_extract(image), tf.convert_to_tensor(sp[0], dtype=tf.float32),
            tf.convert_to_tensor(h, dtype=tf.int32), tf.convert_to_tensor(w, dtype=tf.int32),
            self._dino_dim)
        return (sp, dino_desc, h, w), load_sec

    def match(self, query_path, ref_path):
        from omniglue import utils as og_utils

        # Upstream FindMatches re-extracts both images at every call. The same steps are run here
        # so that the query features can be cached across the retrieved candidates.
        query_path = str(query_path)
        if self._query_cache is not None and self._query_cache[0] == query_path:
            feats_q = self._query_cache[1]
            self._times["query_cache_hits"] += 1
        else:
            t = time.perf_counter()
            feats_q, load_sec = self._extract(query_path)
            self._times["query_load_sec"] += load_sec
            self._times["query_extract_sec"] += time.perf_counter() - t - load_sec
            self._query_cache = (query_path, feats_q)

        t = time.perf_counter()
        feats_r, load_sec = self._extract(ref_path)
        self._times["ref_load_sec"] += load_sec
        self._times["ref_extract_sec"] += time.perf_counter() - t - load_sec

        t = time.perf_counter()
        (sp_q, dd_q, h_q, w_q), (sp_r, dd_r, h_r, w_r) = feats_q, feats_r
        inputs = self.og._construct_inputs(w_q, h_q, w_r, h_r, sp_q, sp_r, dd_q, dd_r)
        soft = self.og.matcher.signatures["serving_default"](**inputs)["soft_assignment"][:, :-1, :-1]
        match_matrix = og_utils.soft_assignment_to_match_matrix(
            soft, self._match_threshold).numpy().squeeze()
        idx = np.argwhere(match_matrix)
        # Drop matches on keypoints with zero confidence, as upstream does.
        idx = idx[(sp_q[2][idx[:, 0]] > 0.0) & (sp_r[2][idx[:, 1]] > 0.0)] if len(idx) else idx
        conf = np.asarray(soft)[0, idx[:, 0], idx[:, 1]] if len(idx) else np.zeros(0)
        self._times["omniglue_match_sec"] += time.perf_counter() - t

        if self.min_conf is not None:
            idx = idx[conf > self.min_conf]
        if len(idx) == 0:
            return np.zeros((0, 2), np.float32), np.zeros((0, 2), np.float32)
        # SuperPoint keypoints are already in full-resolution pixels.
        return sp_q[0][idx[:, 0]].astype(np.float32), sp_r[0][idx[:, 1]].astype(np.float32)
