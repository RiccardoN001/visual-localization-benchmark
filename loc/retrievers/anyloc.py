import logging
import sys
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image

from ..paths import THIRD_PARTY_ROOT
from . import Retriever

REPO_DIR = THIRD_PARTY_ROOT / "AnyLoc"

log = logging.getLogger(__name__)


class AnyLocRetriever(Retriever):
    name = "anyloc"

    # Official AnyLoc configuration: "value" facet of DINOv2 ViT-L/14 layer 18, hard VLAD with 32
    # clusters, intra-normalization. The feature extractor and VLAD come from the official repo.
    def __init__(self, device, model="dinov2_vitl14", layer=18, facet="value", num_clusters=32,
                 seed=0, height=336, width=448, repo_dir=REPO_DIR, **cfg):
        super().__init__(device, model=model, layer=layer, facet=facet, num_clusters=num_clusters,
                         seed=seed, height=height, width=width, **cfg)
        # The official repository is not a package: its demo/utilities.py module is imported directly.
        sys.path.append(str(Path(repo_dir) / "demo"))
        from utilities import VLAD, DinoV2ExtractFeatures

        self._VLAD = VLAD
        self.extractor = DinoV2ExtractFeatures(model, layer, facet, device=device)
        self.num_clusters, self.seed = num_clusters, seed
        self.vlad = None
        self.transform = T.Compose([
            T.Resize((height, width)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    @torch.inference_mode()
    def _patches(self, img_path):
        x = self.transform(Image.open(img_path).convert("RGB")).unsqueeze(0).to(self.device)
        return self.extractor(x)[0]

    def prepare(self, db_image_paths, cache_dir):
        """Fit the VLAD vocabulary on the reference images, or load it from cache_dir.

        The official VLAD class caches its cluster centers in cache_dir/c_centers.pt and
        fit(None) reloads them without extracting any feature.
        """
        self.vlad = self._VLAD(self.num_clusters, desc_dim=None, intra_norm=True, norm_descs=True,
                               dist_mode="cosine", vlad_mode="hard", cache_dir=str(cache_dir))
        if self.vlad.can_use_cache_vlad():
            log.info("Loading VLAD vocabulary from %s", cache_dir)
            self.vlad.fit(None)
        else:
            log.info("Fitting VLAD vocabulary (%d clusters) on %d references", self.num_clusters,
                     len(db_image_paths))
            feats = []
            for i, p in enumerate(db_image_paths, 1):
                feats.append(self._patches(p))
                if i % 200 == 0:
                    log.info("  VLAD vocabulary: %d/%d", i, len(db_image_paths))
            torch.manual_seed(self.seed)
            np.random.seed(self.seed)
            self.vlad.fit(torch.cat(feats, dim=0))

        centers = self.vlad.c_centers
        if centers is None or int(centers.shape[0]) != self.num_clusters:
            raise SystemExit(f"VLAD vocabulary in {cache_dir} does not have {self.num_clusters} "
                             "clusters: delete it and run again")
        # VLAD.generate works on CPU tensors.
        self.vlad.c_centers = centers.cpu()
        if getattr(self.vlad, "kmeans", None) is not None:
            self.vlad.kmeans.centroids = self.vlad.kmeans.centroids.cpu()
        log.info("VLAD vocabulary ready: %d clusters x %d dims", centers.shape[0], centers.shape[1])

    @torch.inference_mode()
    def embed(self, img_path):
        if self.vlad is None:
            raise RuntimeError("prepare() must be called before embed()")
        desc = self.vlad.generate(self._patches(img_path).cpu())   # already L2-normalized
        return desc.detach().cpu().numpy().astype("float32")[None, :]
