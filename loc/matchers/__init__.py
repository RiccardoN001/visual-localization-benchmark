import importlib
import logging
import time

from ..utils import check_options

log = logging.getLogger(__name__)

# name -> (module, class, preset options). Two entries may share a class and differ only in the
# preset, as roma_precise and roma_turbo do. Modules are imported lazily by get_matcher, so a
# missing optional backend only matters when that matcher is requested.
MATCHER_SPECS = {
    "lightglue":      (".lightglue", "LightGlueMatcher", {}),
    "superglue":      (".superglue", "SuperGlueMatcher", {"weights": "outdoor"}),
    "roma_precise":   (".roma", "RoMaMatcher", {"setting": "precise"}),
    "roma_turbo":     (".roma", "RoMaMatcher", {"setting": "turbo"}),
    "omniglue":       (".omniglue", "OmniGlueMatcher", {}),
    "dinov2_matcher": (".dinov2_matcher", "DINOv2Matcher", {}),
    "salad_matcher":  (".salad_matcher", "SaladMatcher", {}),
}
MATCHER_NAMES = tuple(MATCHER_SPECS)


class Matcher:
    name = "?"

    def __init__(self, device, **cfg):
        self.device = device
        self.cfg = cfg

    def match(self, query_path, ref_path):
        """Return (pts_q, pts_r): two (N, 2) float32 arrays in full-resolution pixel coordinates.

        Full resolution matters because pts_r is used to look up the reference depth map.
        """
        raise NotImplementedError

    def config_summary(self):
        """Configuration saved with the results of a run."""
        return dict(self.cfg)

    def timing_detail(self):
        """Optional internal timers accumulated since the last reset_timers()."""
        return {}

    def reset_timers(self):
        pass


def get_matcher(name, device, **options):
    if name not in MATCHER_SPECS:
        raise SystemExit(f"unknown matcher '{name}': choose from {list(MATCHER_SPECS)}")
    module_name, cls_name, preset = MATCHER_SPECS[name]
    cls = getattr(importlib.import_module(module_name, package=__name__), cls_name)
    check_options(cls, options)
    t = time.perf_counter()
    matcher = cls(device, **{**preset, **options})
    matcher.name = name
    log.info("%s loaded in %.1fs: %s", name, time.perf_counter() - t, matcher.config_summary())
    return matcher
