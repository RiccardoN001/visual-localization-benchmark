import time

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image

from . import Matcher
from ._keypoint_features import bilinear_lookup
from ._mutual_nn import mutual_nn_matches


class DINOv2Matcher(Matcher):
    name = "dinov2_matcher"

    # SuperPoint keypoints described by DINOv2 patch features and matched by mutual nearest
    # neighbour (cosine similarity). Input sizes must be multiples of the 14-pixel ViT patch.
    def __init__(self, device, max_kpts=2048, min_score=None, ratio=None, height=336, width=448,
                 hub_repo="facebookresearch/dinov2", hub_model="dinov2_vitl14", **cfg):
        super().__init__(device, max_kpts=max_kpts, min_score=min_score, ratio=ratio,
                         height=height, width=width, hub_model=hub_model, **cfg)
        from lightglue import SuperPoint

        self.extractor = SuperPoint(max_num_keypoints=max_kpts).eval().to(device)
        self.model = torch.hub.load(hub_repo, hub_model, trust_repo=True).eval().to(device)
        # A plain resize keeps a fixed mapping between patches and normalized image coordinates.
        self.transform = T.Compose([
            T.Resize((height, width)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.grid_size = (height // 14, width // 14)
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
        """Full-resolution SuperPoint keypoints (N, 2) and their DINOv2 descriptors [N, C]."""
        from lightglue.utils import load_image, rbd

        image = load_image(str(img_path)).to(self.device)
        kps = rbd(self.extractor.extract(image))["keypoints"].cpu().numpy().astype(np.float32)
        if len(kps) == 0:
            return kps, torch.zeros(0, 1, device=self.device)

        pil = Image.open(img_path).convert("RGB")
        x = self.transform(pil).unsqueeze(0).to(self.device)
        tokens = self.model.forward_features(x)["x_norm_patchtokens"]   # [1, gh*gw, C]
        gh, gw = self.grid_size
        grid = tokens[0].reshape(gh, gw, tokens.shape[-1]).permute(2, 0, 1).contiguous()
        return kps, bilinear_lookup(grid, kps, *pil.size)

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
        idx_q, idx_r, _ = mutual_nn_matches(desc_q, desc_r, "cosine",
                                            min_score=self.min_score, ratio=self.ratio)
        self._times["mutual_nn_sec"] += time.perf_counter() - t
        if len(idx_q) == 0:
            return np.zeros((0, 2), np.float32), np.zeros((0, 2), np.float32)
        return kps_q[idx_q.cpu().numpy()], kps_r[idx_r.cpu().numpy()]
