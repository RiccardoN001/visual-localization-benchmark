"""Convert the San Andrea scene (available on request) to the layout used by the pipeline.

Input folder: GT-poses.txt (query ground truth), queryImages_v2/ (query JPEGs), poses.json,
meshTexture/ and meshTextureDepth/ (database renders and EXR depth maps, made with Blender).
Output: <out>/capture and <out>/db.

    python scripts/prepare/sanandrea_prepare.py --scene <raw_sanandrea> --out data/SANANDREA
"""
import argparse
import json
import os
import re
import shutil
import sys
from pathlib import Path

os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")

import cv2  # noqa: E402
import numpy as np  # noqa: E402
from scipy.spatial.transform import Rotation  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from loc.capture_io import (  # noqa: E402
    DEPTH_SCALE,
    load_cameras,
    quantize_depth,
    save_capture_trajectory,
    save_pinhole_sensors,
    stem_for,
)

# Blender cameras look along -Z with +Y up; OpenCV cameras look along +Z with +Y down.
BLENDER_TO_CV = np.diag([1.0, -1.0, -1.0])
# Axes of the ground-truth camera frame expressed in the OpenCV optical frame, applied on the right.
GT_AXES = Rotation.from_euler("z", -90, degrees=True) * Rotation.from_euler("x", 180, degrees=True)

W, H = 1944, 2592   # portrait images
# Pinhole intrinsics (f, cx, cy) of the three query cameras. The database was rendered with cam0,
# whose principal point is the empirically corrected one used to produce the renders.
CAMERAS = {
    "cam0": (1016.883409, 1011.483097, (2592 - 1) - 1272.187046),
    "cam1": (1019.195937, 965.804886, (2592 - 1) - 1273.649697),
    "cam2": (1016.339241, 982.422838, (2592 - 1) - 1254.845941),
}
DB_SENSOR = "render"

# Query ids look like cam0_pic100_Wednesday-07-41-43-495153.
ID_RE = re.compile(r"^(cam\d)_pic(\d+)_(\w+)-(\d{2})-(\d{2})-(\d{2})-(\d{6})$")


def parse_query_id(qid):
    """Query id -> (microseconds of the day, camera). All images were taken on the same day."""
    m = ID_RE.match(qid)
    if not m:
        raise SystemExit(f"unrecognized query id: {qid!r}")
    cam, _, _, hh, mm, ss, us = m.groups()
    return (int(hh) * 3600 + int(mm) * 60 + int(ss)) * 1_000_000 + int(us), cam


def load_gt(path):
    """GT-poses.txt (id, x, y, z, omega, phi, kappa in degrees) -> [(ts, cam, qid, R_c2w, t)]."""
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = [tok.strip() for tok in line.split(",")]
        if not line.strip() or line.startswith("#"):
            continue
        ts, cam = parse_query_id(parts[0])
        omega, phi, kappa = map(float, parts[4:7])
        R = Rotation.from_euler("ZYX", [kappa, phi, omega], degrees=True) * GT_AXES
        rows.append((ts, cam, parts[0], R.as_matrix(), np.array(list(map(float, parts[1:4])))))
    return sorted(rows, key=lambda r: (r[0], r[1]))


def load_db_poses(path):
    """poses.json (Blender) -> [(id, R_c2w, t)] in the OpenCV convention."""
    return sorted(((p["id"], Rotation.from_euler("xyz", p["rotation_euler"]).as_matrix() @ BLENDER_TO_CV,
                    np.array(p["position"], dtype=float))
                   for p in json.loads(path.read_text(encoding="utf-8"))), key=lambda p: p[0])


def write_capture(out, gt_rows, image_dir, copy_images):
    (out / "raw_data" / "images").mkdir(parents=True, exist_ok=True)
    save_pinhole_sensors([(cam, f"sanandrea {cam}", W, H, f, f, cx, cy)
                          for cam, (f, cx, cy) in CAMERAS.items()], out / "sensors.txt")
    id_map = {}
    with open(out / "images.txt", "w", encoding="utf-8") as f:
        f.write("# timestamp, sensor_id, image_path\n")
        for ts, cam, qid, _, _ in gt_rows:
            src = image_dir / f"{qid}.jpeg"
            if not src.exists():
                raise SystemExit(f"missing query image {src}")
            stem = stem_for(ts, cam)
            if copy_images:
                shutil.copyfile(src, out / "raw_data" / "images" / f"{stem}.jpeg")
            f.write(f"{ts}, {cam}, images/{stem}.jpeg\n")
            id_map[stem] = qid
    save_capture_trajectory([(ts, cam, R, t) for ts, cam, _, R, t in gt_rows],
                            out / "trajectories.txt")
    # Mapping back to the original query ids.
    (out / "id_map.json").write_text(json.dumps(id_map, indent=2), encoding="utf-8")


def write_db(out, poses, render_dir, depth_dir, jpeg_quality, convert_images):
    (out / "images").mkdir(parents=True, exist_ok=True)
    (out / "depth_png").mkdir(exist_ok=True)
    f, cx, cy = CAMERAS["cam0"]
    save_pinhole_sensors([(DB_SENSOR, "sanandrea render (cam0 intrinsics)", W, H, f, f, cx, cy)],
                         out / "db_sensors.txt")
    save_capture_trajectory([(pid, DB_SENSOR, R, t) for pid, R, t in poses],
                            out / "db_trajectories.txt")
    (out / "depth_scale.json").write_text(json.dumps(
        {"depth_scale": DEPTH_SCALE, "dtype": "uint16", "unit": "meters", "invalid": 0},
        indent=2), encoding="utf-8")
    if not convert_images:
        return
    for pid, _, _ in poses:
        stem = stem_for(pid, DB_SENSOR)
        img = cv2.imread(str(render_dir / f"{pid}.png"), cv2.IMREAD_COLOR)   # drops alpha
        depth = cv2.imread(str(depth_dir / f"{pid}.exr"), cv2.IMREAD_UNCHANGED)
        if img is None or depth is None:
            raise SystemExit(f"cannot read the render or depth of {pid}")
        cv2.imwrite(str(out / "images" / f"{stem}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
        depth = depth if depth.ndim == 2 else depth[:, :, 0]
        cv2.imwrite(str(out / "depth_png" / f"{stem}.png"), quantize_depth(depth))


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--scene", type=Path, required=True, help="raw San Andrea folder")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--jpeg-quality", type=int, default=95)
    p.add_argument("--skip-images", action="store_true", help="write only the metadata files")
    args = p.parse_args()

    gt_rows = load_gt(args.scene / "GT-poses.txt")
    poses = load_db_poses(args.scene / "poses.json")
    print(f"{len(gt_rows)} queries with ground truth, {len(poses)} reference poses")
    write_capture(args.out / "capture", gt_rows, args.scene / "queryImages_v2", not args.skip_images)
    write_db(args.out / "db", poses, args.scene / "meshTexture", args.scene / "meshTextureDepth",
             args.jpeg_quality, not args.skip_images)

    cams = load_cameras(args.out / "capture" / "sensors.txt")
    if set(cams) != set(CAMERAS):
        raise SystemExit(f"sensors.txt could not be read back: {sorted(cams)}")
    print(f"read back: {len(cams)} query sensors")
    print(f"{len(gt_rows)} queries and {len(poses)} references written to {args.out}")


if __name__ == "__main__":
    main()
