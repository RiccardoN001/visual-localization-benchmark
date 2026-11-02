# Readers and writers for the LaMAR Capture text format (CAPTURE.md in microsoft/lamar-benchmark).
# Poses are sensor-to-world (cam2world), quaternions are stored as (qw, qx, qy, qz), and a row is
# identified by the pair (timestamp, sensor_id): NavVis repeats the same sensor_id at every stop.
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

# Rotation from the body frame (x forward, y left, z up) to the optical camera frame
# (x right, y down, z forward), as columns: optical axes expressed in body coordinates.
R_BODY_TO_OPTICAL = np.array([[0.0, 0.0, 1.0],
                              [-1.0, 0.0, 0.0],
                              [0.0, -1.0, 0.0]])

# Depth maps are stored as uint16 PNG with depth_m = value / DEPTH_SCALE (4 mm steps, max ~262 m).
DEPTH_SCALE = 250.0


def read_rows(path):
    """Rows of a Capture file as lists of fields, skipping comments and empty lines."""
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            yield [tok.strip() for tok in line.split(",")]


def qt_to_Rt(qw, qx, qy, qz, tx, ty, tz):
    R = Rotation.from_quat([qx, qy, qz, qw]).as_matrix()
    return R, np.array([tx, ty, tz], dtype=np.float64)


def Rt_to_qt(R, t):
    """(R, t) -> [qw, qx, qy, qz, tx, ty, tz] as Python floats (keeps the text output readable)."""
    q = Rotation.from_matrix(np.asarray(R, float)).as_quat()
    t = np.asarray(t, float)
    return [float(q[3]), float(q[0]), float(q[1]), float(q[2]),
            float(t[0]), float(t[1]), float(t[2])]


def invert_Rt(R, t):
    Ri = np.asarray(R, float).T
    return Ri, -(Ri @ np.asarray(t, float))


def stem_for(timestamp, sensor_id):
    """Unique file stem for one pose, shared by the renderer, the data loader and the evaluation."""
    return f"{timestamp}__{sensor_id.replace('/', '__')}"


def load_cameras(path):
    """{sensor_id: dict(width, height, fx, fy, cx, cy)} for the PINHOLE cameras of a sensors.txt."""
    cams = {}
    for row in read_rows(path):
        if len(row) < 10 or row[2] != "camera":
            continue
        if row[3] != "PINHOLE":
            raise ValueError(f"unsupported camera model for {row[0]}: {row[3]}")
        fx, fy, cx, cy = map(float, row[6:10])
        cams[row[0]] = dict(width=int(row[4]), height=int(row[5]), fx=fx, fy=fy, cx=cx, cy=cy)
    return cams


def load_trajectories(path):
    """{(timestamp, sensor_id): (R_c2w, t_c2w)}."""
    return {(row[0], row[1]): qt_to_Rt(*map(float, row[2:9])) for row in read_rows(path)}


def load_alignment_global(path, label="pose_graph_optimized"):
    """Session-to-global transform of a LaMAR session: p_global = R @ p_session + t."""
    for row in read_rows(path):
        if row[0] == label:
            return qt_to_Rt(*map(float, row[2:9]))
    raise KeyError(f"label '{label}' not found in {path}")


def read_tum(path):
    """TUM trajectory (timestamp tx ty tz qx qy qz qw) -> list of (timestamp_str, R, t)."""
    poses = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.split()
            if len(parts) < 8 or parts[0].startswith("#"):
                continue
            tx, ty, tz, qx, qy, qz, qw = map(float, parts[1:8])
            R = Rotation.from_quat([qx, qy, qz, qw]).as_matrix()
            poses.append((parts[0], R, np.array([tx, ty, tz])))
    return poses


def save_capture_trajectory(rows, path):
    """rows: iterable of (timestamp, sensor_id, R_c2w, t_c2w)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("# timestamp, device_id, qw, qx, qy, qz, tx, ty, tz\n")
        for ts, sensor_id, R, t in rows:
            f.write(f"{ts}, {sensor_id}, " + ", ".join(repr(v) for v in Rt_to_qt(R, t)) + "\n")
    return path


def save_pinhole_sensors(cameras, path):
    """cameras: iterable of (sensor_id, name, width, height, fx, fy, cx, cy)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("# sensor_id, name, sensor_type, [sensor_params]+\n")
        for sensor_id, name, w, h, *params in cameras:
            # float() first: repr() of a numpy scalar is not a plain number with numpy >= 2.
            f.write(f"{sensor_id}, {name}, camera, PINHOLE, {int(w)}, {int(h)}, "
                    + ", ".join(repr(float(v)) for v in params) + "\n")
    return path


def quantize_depth(depth_m, scale=DEPTH_SCALE):
    """Metric depth -> uint16 image for PNG storage. Out-of-range and non-finite values become 0."""
    q = np.asarray(depth_m, dtype=np.float64) * scale
    q[(q > np.iinfo(np.uint16).max) | (q < 0) | ~np.isfinite(q)] = 0
    return q.round().astype(np.uint16)
