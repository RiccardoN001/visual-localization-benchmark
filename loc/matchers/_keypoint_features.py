import inspect

import torch
from PIL import Image


def bilinear_lookup(grid, keypoints_xy, W, H):
    """Bilinear interpolation of a [D, gh, gw] patch grid at full-resolution keypoints -> [N, D].

    Patch (r, c) covers [c/gw, (c+1)/gw) x [r/gh, (r+1)/gh) of the normalized image, with its
    center at ((c+0.5)/gw, (r+0.5)/gh). This holds because images are resized to the network
    input without cropping or padding.
    """
    D, gh, gw = grid.shape
    kp = torch.from_numpy(keypoints_xy).to(grid.device).float()
    gx = (kp[:, 0] / W * gw - 0.5).clamp(0, gw - 1)
    gy = (kp[:, 1] / H * gh - 0.5).clamp(0, gh - 1)

    c0, r0 = gx.floor().long(), gy.floor().long()
    c1, r1 = (c0 + 1).clamp(max=gw - 1), (r0 + 1).clamp(max=gh - 1)
    tx = (gx - c0.float()).unsqueeze(1)
    ty = (gy - r0.float()).unsqueeze(1)

    return ((1 - ty) * (1 - tx) * grid[:, r0, c0].T + (1 - ty) * tx * grid[:, r0, c1].T
            + ty * (1 - tx) * grid[:, r1, c0].T + ty * tx * grid[:, r1, c1].T)


class SaladClusterProbabilities:
    def __init__(self, salad_retriever):
        self.salad = salad_retriever
        self.aggregator = salad_retriever.model.aggregator
        # get_matching_probs is a module-level function of the SALAD aggregator source file.
        self._get_matching_probs = inspect.getmodule(self.aggregator).get_matching_probs
        self._captured = None
        # The hook stores the backbone feature map that SALAD passes to its aggregator.
        self.aggregator.register_forward_pre_hook(self._capture)

    def _capture(self, module, args):
        self._captured = args[0]

    @torch.inference_mode()
    def grid(self, img_path):
        """SALAD cluster assignment probabilities of every patch -> [m, gh, gw].

        The dustbin is removed and the remaining probabilities renormalized, otherwise the dustbin
        dominates and all patches look alike.
        """
        self._captured = None
        self.salad.embed(img_path)
        if self._captured is None:
            raise RuntimeError("the SALAD aggregator input was not captured")
        features, _ = self._captured                                   # [1, C, gh, gw]
        scores = self.aggregator.score(features).flatten(2)            # [1, m, n]
        probs = torch.exp(self._get_matching_probs(scores, self.aggregator.dust_bin, 3))
        clusters = probs[0, :-1, :]
        clusters = clusters / clusters.sum(dim=0, keepdim=True).clamp_min(1e-9)
        return clusters.reshape(clusters.shape[0], features.shape[-2], features.shape[-1])

    def descriptors(self, img_path, keypoints_xy):
        W, H = Image.open(img_path).size
        return bilinear_lookup(self.grid(img_path), keypoints_xy, W, H)
