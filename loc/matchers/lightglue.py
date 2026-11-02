import time

import numpy as np
import torch

from . import Matcher


class LightGlueMatcher(Matcher):
    name = "lightglue"

    def __init__(self, device, max_kpts=2048, resize=None, **cfg):
        super().__init__(device, max_kpts=max_kpts, resize=resize, **cfg)
        from lightglue import LightGlue, SuperPoint

        self.extractor = SuperPoint(max_num_keypoints=max_kpts).eval().to(device)
        self.matcher = LightGlue(features="superpoint").eval().to(device)
        # resize=None keeps the full resolution, so keypoints are already in original pixels.
        self.resize = resize
        self._query_cache = None
        self.reset_timers()

    def reset_timers(self):
        self._times = {"query_load_sec": 0.0, "query_extract_sec": 0.0, "query_cache_hits": 0,
                       "ref_load_sec": 0.0, "ref_extract_sec": 0.0, "lightglue_match_sec": 0.0}

    def timing_detail(self):
        return dict(self._times)

    @torch.inference_mode()
    def match(self, query_path, ref_path):
        from lightglue.utils import load_image, rbd

        # match() is called once per retrieved candidate with the same query, so the query
        # features are cached and only the reference side is extracted again.
        query_path = str(query_path)
        if self._query_cache is not None and self._query_cache[0] == query_path:
            feats_q = self._query_cache[1]
            self._times["query_cache_hits"] += 1
        else:
            t = time.perf_counter()
            image_q = load_image(query_path, resize=self.resize).to(self.device)
            self._times["query_load_sec"] += time.perf_counter() - t
            t = time.perf_counter()
            feats_q = self.extractor.extract(image_q)
            self._times["query_extract_sec"] += time.perf_counter() - t
            self._query_cache = (query_path, feats_q)

        t = time.perf_counter()
        image_r = load_image(str(ref_path), resize=self.resize).to(self.device)
        self._times["ref_load_sec"] += time.perf_counter() - t
        t = time.perf_counter()
        feats_r = self.extractor.extract(image_r)
        self._times["ref_extract_sec"] += time.perf_counter() - t

        t = time.perf_counter()
        out = self.matcher({"image0": feats_q, "image1": feats_r})
        self._times["lightglue_match_sec"] += time.perf_counter() - t

        feats_q, feats_r, out = (rbd(x) for x in (feats_q, feats_r, out))
        matches = out["matches"]
        if matches.shape[0] == 0:
            return np.zeros((0, 2), np.float32), np.zeros((0, 2), np.float32)
        pts_q = feats_q["keypoints"][matches[:, 0]].cpu().numpy().astype(np.float32)
        pts_r = feats_r["keypoints"][matches[:, 1]].cpu().numpy().astype(np.float32)
        return pts_q, pts_r
