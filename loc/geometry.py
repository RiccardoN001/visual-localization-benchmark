import cv2
import numpy as np
from scipy.spatial.transform import Rotation

MAX_DEPTH_M = 1000.0


def K_from(fx, fy, cx, cy):
    return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)


def backproject(pts2d, depth_map, K):
    """2D points + depth map -> 3D points in the camera frame, and the indices of the valid points.

    Depth is the Z coordinate along the optical axis; 0, non-finite or very large values are invalid.
    """
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    h, w = depth_map.shape[:2]

    pts2d = np.asarray(pts2d, dtype=np.float64)
    u = np.round(pts2d[:, 0]).astype(np.int64)
    v = np.round(pts2d[:, 1]).astype(np.int64)

    in_bounds = (u >= 0) & (u < w) & (v >= 0) & (v < h)
    # Clipping only keeps the lookup in range; out-of-bounds points are discarded by `valid`.
    z = depth_map[np.clip(v, 0, h - 1), np.clip(u, 0, w - 1)].astype(np.float64)
    valid = in_bounds & (z > 0) & (z <= MAX_DEPTH_M) & np.isfinite(z)

    x = (pts2d[:, 0] - cx) * z / fx
    y = (pts2d[:, 1] - cy) * z / fy
    pts3d = np.stack([x, y, z], axis=1)[valid].astype(np.float32)
    return pts3d, np.nonzero(valid)[0]


def query_pose_from_pnp(rvec, tvec, R_ref_c2w, t_ref_c2w):
    """Absolute cam2world pose of the query.

    PnP receives 3D points in the reference camera frame, so (rvec, tvec) is T_query_from_ref and
    T_world_from_query = T_world_from_ref @ inv(T_query_from_ref).
    """
    R_q_from_r, _ = cv2.Rodrigues(np.asarray(rvec, dtype=np.float64).reshape(3, 1))
    t_q_from_r = np.asarray(tvec, dtype=np.float64).reshape(3)

    R_r_from_q = R_q_from_r.T
    t_r_from_q = -R_r_from_q @ t_q_from_r

    R_ref_c2w = np.asarray(R_ref_c2w, dtype=np.float64)
    t_ref_c2w = np.asarray(t_ref_c2w, dtype=np.float64)
    return R_ref_c2w @ R_r_from_q, R_ref_c2w @ t_r_from_q + t_ref_c2w


def pose_error(R_est, t_est, R_gt, t_gt):
    """(position error in meters, rotation error in degrees) between two cam2world poses."""
    dt = float(np.linalg.norm(np.asarray(t_est, float) - np.asarray(t_gt, float)))
    dR = np.asarray(R_est, float).T @ np.asarray(R_gt, float)
    dr = float(np.degrees(np.linalg.norm(Rotation.from_matrix(dR).as_rotvec())))
    return dt, dr
