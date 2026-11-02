"""Convert an Oxford Spires sequence to the Capture layout used by the pipeline.

The fisheye images are undistorted once to pinhole images (balance=0, no black borders), and the
TUM ground truth (body frame: x forward, y left, z up) is converted to optical camera poses.

    python scripts/prepare/oxford_prepare.py --images <sequence>/cam0 --calib <sequence>/../cam0.yaml \\
        --gt <sequence>/gt-tum.txt --out data/KEBLE/capture
"""
import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from loc.capture_io import R_BODY_TO_OPTICAL, read_tum, save_capture_trajectory, save_pinhole_sensors  # noqa: E402

SENSOR_ID = "cam0"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def load_fisheye_calibration(path):
    """K, D and image size from a Kalibr/ROS camera_info YAML with the equidistant model."""
    y = yaml.safe_load(path.read_text(encoding="utf-8"))
    if y.get("distortion_model") != "equidistant":
        raise SystemExit(f"{path}: expected the 'equidistant' (OpenCV fisheye) distortion model")
    K = np.array(y["camera_matrix"]["data"], dtype=np.float64).reshape(3, 3)
    D = np.array(y["distortion_coefficients"]["data"], dtype=np.float64).reshape(-1, 1)
    return K, D, int(y["image_width"]), int(y["image_height"])


def undistort_images(src_dir, dst_dir, K, D, size, jpeg_quality, overwrite):
    """Undistort every image with the same remap tables. -> (new K, image stems)."""
    K_new = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(K, D, size, np.eye(3), balance=0.0)
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), K_new, size, cv2.CV_16SC2)
    files = sorted(f for f in src_dir.iterdir() if f.suffix.lower() in IMAGE_EXTENSIONS)
    if not files:
        raise SystemExit(f"no images in {src_dir}")
    dst_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    done = present = 0
    for i, src in enumerate(files, 1):
        dst = dst_dir / f"{src.stem}.jpg"
        if dst.exists() and not overwrite:
            present += 1
            continue
        img = cv2.imread(str(src), cv2.IMREAD_COLOR)
        if img is None:
            print(f"unreadable image skipped: {src.name}")
            continue
        cv2.imwrite(str(dst), cv2.remap(img, map1, map2, cv2.INTER_LINEAR),
                    [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
        done += 1
        if done % 500 == 0:
            eta = (time.time() - t0) / i * (len(files) - i)
            print(f"  [{i}/{len(files)}] ETA {eta / 60:.0f} min", flush=True)
    print(f"  undistorted {done}, already present {present} ({time.time() - t0:.0f}s)")
    return K_new, [f.stem for f in files]


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--images", type=Path, required=True, help="folder of the fisheye cam0 images")
    p.add_argument("--calib", type=Path, required=True, help="cam0.yaml")
    p.add_argument("--gt", type=Path, required=True, help="gt-tum.txt")
    p.add_argument("--out", type=Path, required=True, help="output capture/ folder")
    p.add_argument("--jpeg-quality", type=int, default=95)
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    K, D, W, H = load_fisheye_calibration(args.calib)
    print(f"{args.calib.name}: {W}x{H} fx={K[0, 0]:.2f} fy={K[1, 1]:.2f} cx={K[0, 2]:.2f} "
          f"cy={K[1, 2]:.2f} equidistant D={D.ravel().round(5).tolist()}")
    print(f"undistort {args.images} -> {args.out / 'raw_data' / 'images'}")
    K_new, stems = undistort_images(args.images, args.out / "raw_data" / "images", K, D, (W, H),
                                    args.jpeg_quality, args.overwrite)
    print(f"K after undistortion: fx={K_new[0, 0]:.2f} fy={K_new[1, 1]:.2f} cx={K_new[0, 2]:.2f} "
          f"cy={K_new[1, 2]:.2f}")
    save_pinhole_sensors([(SENSOR_ID, "oxford cam0 undistorted", W, H, K_new[0, 0], K_new[1, 1],
                           K_new[0, 2], K_new[1, 2])], args.out / "sensors.txt")
    with open(args.out / "images.txt", "w", encoding="utf-8") as f:
        f.write("# timestamp, sensor_id, image_path\n")
        for stem in sorted(stems, key=float):
            f.write(f"{stem}, {SENSOR_ID}, images/{stem}.jpg\n")

    # Timestamps stay strings: they are the key to the image file names.
    gt_rows = read_tum(args.gt)
    if not gt_rows:
        raise SystemExit(f"{args.gt} is empty")
    have = set(stems)
    gt = [(ts, SENSOR_ID, R_body @ R_BODY_TO_OPTICAL, t) for ts, R_body, t in gt_rows if ts in have]
    save_capture_trajectory(gt, args.out / "trajectories.txt")

    # Images recorded outside the ground-truth time window have no pose and are not evaluated.
    # An image without a pose inside the window instead points to a timestamp matching problem.
    times = [float(ts) for ts, _, _ in gt_rows]
    matched = {row[0] for row in gt}
    orphans = [float(s) for s in stems if s not in matched]
    inside = sum(min(times) <= s <= max(times) for s in orphans)
    print(f"ground truth: {len(gt)}/{len(stems)} queries matched ({len(gt_rows)} rows in {args.gt.name}), "
          "body frame converted to optical camera poses")
    if inside:
        print(f"warning: {inside} queries inside the ground-truth time window have no pose: "
              "check the timestamp matching")
    if len(orphans) > inside:
        print(f"{len(orphans) - inside} queries outside the ground-truth time window (not evaluated)")
    print(f"{len(stems)} images, {len(gt)} with ground truth -> {args.out}")


if __name__ == "__main__":
    main()
