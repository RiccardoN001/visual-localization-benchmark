import logging
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .geometry import K_from, backproject

log = logging.getLogger(__name__)


@dataclass
class PnPConfig:
    reprojection_error: float = 8.0   # pixels, OpenCV default
    iterations: int = 100             # OpenCV default
    confidence: float = 0.99          # OpenCV default
    rerank_threshold: float = 3.0     # pixels, RANSAC threshold of the fundamental-matrix rerank


def match_candidates(matcher, query_path, candidates, config):
    """Match the query against every candidate and sort them by geometric inliers.

    With a single candidate there is nothing to rerank, so the fundamental-matrix RANSAC is skipped
    and inlier_count is None.
    """
    ranked = []
    for c in candidates:
        pts_q, pts_r = matcher.match(query_path, c["metadata"]["image_path"])
        inliers = None
        if len(candidates) > 1:
            inliers = 0
            if len(pts_q) > 8:
                _, mask = cv2.findFundamentalMat(pts_q, pts_r, cv2.FM_RANSAC,
                                                 config.rerank_threshold, 0.99)
                if mask is not None:
                    inliers = int(np.sum(mask))
        ranked.append({"candidate": c, "inlier_count": inliers, "raw_matches": int(len(pts_q)),
                       "pts_q": pts_q, "pts_r": pts_r})
    ranked.sort(key=lambda x: x["inlier_count"] or 0, reverse=True)
    return ranked


def solve_pnp(pts_q, pts_r, reference, K_query, read_depth, config):
    """Absolute PnP from 2D-2D matches lifted to 3D with the reference depth map.

    The depth is back-projected with the reference intrinsics, while PnP uses the query
    intrinsics: the two differ when query and database cameras are not the same.
    Returns (rvec, tvec, num_inliers) or None, plus the time spent in each step.
    """
    detail = {"depth_load_sec": 0.0, "backproject_sec": 0.0, "solve_sec": 0.0}
    if len(pts_r) < 4:
        return None, detail
    if not Path(reference["depth_path"]).exists():
        log.error("missing depth map: %s", reference["depth_path"])
        return None, detail

    t = time.perf_counter()
    depth = read_depth(reference["depth_path"])
    detail["depth_load_sec"] = time.perf_counter() - t
    if depth is None:
        return None, detail

    t = time.perf_counter()
    pts3d, valid = backproject(pts_r, depth, K_from(*reference["K_ref"]))
    detail["backproject_sec"] = time.perf_counter() - t
    if len(pts3d) < 4:
        return None, detail

    t = time.perf_counter()
    ok, rvec, tvec, inliers = cv2.solvePnPRansac(
        pts3d, pts_q[valid], K_query, distCoeffs=np.zeros((5, 1)),
        iterationsCount=config.iterations, reprojectionError=config.reprojection_error,
        confidence=config.confidence, flags=cv2.SOLVEPNP_EPNP)
    detail["solve_sec"] = time.perf_counter() - t
    if not ok:
        return None, detail
    return (rvec.flatten(), tvec.flatten(), int(len(inliers) if inliers is not None else 0)), detail
