import time

import numpy as np
import torch
from PIL import Image

from . import Matcher


def _to_numpy(x):
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    return np.asarray(x)


class RoMaMatcher(Matcher):
    name = "roma"

    def __init__(self, device, setting="precise", certainty=0.5, max_kp=2000, n_sample=5000, **cfg):
        super().__init__(device, setting=setting, certainty=certainty, max_kp=max_kp,
                         n_sample=n_sample, **cfg)
        from romav2 import RoMaV2

        self.model = RoMaV2()
        # A run labelled with a preset must really use it, so a failure here stops the run.
        self.model.apply_setting(setting)
        self.certainty, self.max_kp, self.n_sample = certainty, max_kp, n_sample
        self._times = {"dinov3": 0.0, "coarse": 0.0, "vgg_features": 0.0, "conv_refine": 0.0}
        self._register_timing_hooks()

    def _register_timing_hooks(self):
        """Accumulate the time of the four internal stages of RoMa v2 with forward hooks.

        f = DINOv3 backbone, matcher = coarse matching, refiner_features = VGG features,
        refiners = convolutional refiners. Each module runs several times per match() call.
        """
        def hooks(key):
            starts = []

            def pre(module, inputs):
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                starts.append(time.perf_counter())

            def post(module, inputs, output):
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                if starts:
                    self._times[key] += time.perf_counter() - starts.pop()

            return pre, post

        modules = [("dinov3", getattr(self.model, "f", None)),
                   ("coarse", getattr(self.model, "matcher", None)),
                   ("vgg_features", getattr(self.model, "refiner_features", None))]
        modules += [("conv_refine", m) for m in (getattr(self.model, "refiners", {}) or {}).values()]
        for key, module in modules:
            if module is not None:
                pre, post = hooks(key)
                module.register_forward_pre_hook(pre)
                module.register_forward_hook(post)

    def reset_timers(self):
        for k in self._times:
            self._times[k] = 0.0

    def timing_detail(self):
        return {f"{k}_sec": v for k, v in self._times.items()}

    def _dense_match(self, query_path, ref_path, Wq, Hq, Wr, Hr):
        """Sampled correspondences in pixels with their per-match certainty."""
        preds = self.model.match(str(query_path), str(ref_path))
        matches, overlaps, precision_AB, precision_BA = self.model.sample(preds, self.n_sample)
        kpts_q, kpts_r = self.model.to_pixel_coordinates(matches, Hq, Wq, Hr, Wr)
        kpts_q, kpts_r = _to_numpy(kpts_q), _to_numpy(kpts_r)
        # The certainty is the returned array whose length equals the number of matches.
        for candidate in (overlaps, precision_AB, precision_BA):
            cert = _to_numpy(candidate).reshape(-1)
            if cert.shape[0] == len(kpts_q):
                return kpts_q, kpts_r, cert
        return kpts_q, kpts_r, np.ones(len(kpts_q), dtype=np.float32)

    def match(self, query_path, ref_path):
        Wq, Hq = Image.open(query_path).size
        Wr, Hr = Image.open(ref_path).size
        kpts_q, kpts_r, cert = self._dense_match(query_path, ref_path, Wq, Hq, Wr, Hr)

        keep = cert >= self.certainty
        pts_q, pts_r, cert = kpts_q[keep], kpts_r[keep], cert[keep]
        # Keep only the most certain matches: RANSAC cost grows with the number of points.
        if self.max_kp and len(pts_q) > self.max_kp:
            top = np.argsort(cert)[::-1][:self.max_kp]
            pts_q, pts_r = pts_q[top], pts_r[top]
        return pts_q.astype(np.float32), pts_r.astype(np.float32)
