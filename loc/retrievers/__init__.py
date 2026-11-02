import importlib
import logging
import time

from ..utils import check_options

log = logging.getLogger(__name__)

# name -> (module, class, preset options). Modules are imported lazily by get_retriever.
RETRIEVER_SPECS = {
    "salad":   (".salad", "SaladRetriever", {}),
    "megaloc": (".megaloc", "MegaLocRetriever", {}),
    "anyloc":  (".anyloc", "AnyLocRetriever", {}),
    "netvlad": (".netvlad", "NetVLADRetriever", {}),
    "dinov2":  (".dinov2", "DINOv2Retriever", {}),
    "dinov3":  (".dinov3", "DINOv3Retriever", {}),
}
RETRIEVER_NAMES = tuple(RETRIEVER_SPECS)


class Retriever:
    name = "?"

    def __init__(self, device, **cfg):
        self.device = device
        self.cfg = cfg

    def prepare(self, db_image_paths, cache_dir):
        """Database-dependent setup, called before any embed() (AnyLoc fits its VLAD vocabulary).

        It runs both when the index is built and when it is loaded from the cache, so query and
        reference descriptors always live in the same space.
        """

    def embed(self, img_path):
        """Return a (1, D) float32 L2-normalized global descriptor."""
        raise NotImplementedError

    def config_summary(self):
        return dict(self.cfg)

    def timing_detail(self):
        return {}

    def reset_timers(self):
        pass


def get_retriever(name, device, **options):
    if name not in RETRIEVER_SPECS:
        raise SystemExit(f"unknown retriever '{name}': choose from {list(RETRIEVER_SPECS)}")
    module_name, cls_name, preset = RETRIEVER_SPECS[name]
    cls = getattr(importlib.import_module(module_name, package=__name__), cls_name)
    check_options(cls, options)
    t = time.perf_counter()
    retriever = cls(device, **{**preset, **options})
    retriever.name = name
    log.info("%s loaded in %.1fs: %s", name, time.perf_counter() - t, retriever.config_summary())
    return retriever
