import time

import numpy as np
import torch

from . import Matcher
from ._keypoint_features import SaladClusterProbabilities
from ._mutual_nn import mutual_nn_matches


class SaladMatcher(Matcher):
    name = "salad_matcher"

    # SuperPoint keypoints described by the SALAD cluster probabilities of their image region and
    # matched by mutual nearest neighbour (Bhattacharyya coefficient). The matcher loads its own
    # SALAD model, so it works with any retriever.
    def __init__(self, device, max_kpts=2048, min_score=None, ratio=None, **cfg):
        super().__init__(device, max_kpts=max_kpts, min_score=min_score, ratio=ratio, **cfg)
        from lightglue import SuperPoint

        from ..retrievers.salad import SaladRetriever

        self.extractor = SuperPoint(max_num_keypoints=max_kpts).eval().to(device)
        self.clusters = SaladClusterProbabilities(SaladRetriever(device))
        self.min_score, self.ratio = min_score, ratio
        self._query_cache = None
        self.reset_timers()

    def reset_timers(self):
        self._times = {"query_extract_sec": 0.0, "query_cache_hits": 0,
                       "ref_extract_sec": 0.0, "mutual_nn_sec": 0.0}

    def timing_detail(self):
        return dict(self._times)

    @torch.inference_mode()
    def _describe(self, img_path):
        from lightglue.utils import load_image, rbd

        image = load_image(str(img_path)).to(self.device)
        kps = rbd(self.extractor.extract(image))["keypoints"].cpu().numpy().astype(np.float32)
        if len(kps) == 0:
            return kps, torch.zeros(0, 1, device=self.device)
        return kps, self.clusters.descriptors(img_path, kps)

    @torch.inference_mode()
    def match(self, query_path, ref_path):
        query_path = str(query_path)
        if self._query_cache is not None and self._query_cache[0] == query_path:
            kps_q, desc_q = self._query_cache[1]
            self._times["query_cache_hits"] += 1
        else:
            t = time.perf_counter()
            kps_q, desc_q = self._describe(query_path)
            self._times["query_extract_sec"] += time.perf_counter() - t
            self._query_cache = (query_path, (kps_q, desc_q))

        t = time.perf_counter()
        kps_r, desc_r = self._describe(ref_path)
        self._times["ref_extract_sec"] += time.perf_counter() - t

        t = time.perf_counter()
        idx_q, idx_r, _ = mutual_nn_matches(desc_q, desc_r, "bhattacharyya",
                                            min_score=self.min_score, ratio=self.ratio)
        self._times["mutual_nn_sec"] += time.perf_counter() - t
        if len(idx_q) == 0:
            return np.zeros((0, 2), np.float32), np.zeros((0, 2), np.float32)
        return kps_q[idx_q.cpu().numpy()], kps_r[idx_r.cpu().numpy()]
