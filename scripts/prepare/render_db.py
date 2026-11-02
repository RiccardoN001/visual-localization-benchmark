"""Render the reference database (RGB + depth) from a colored mesh at the given camera poses.

Ray casting runs on the CPU with Open3D (Embree). All poses use the intrinsics of one camera of
<root>/sensors.txt. Mesh and poses must be in the same frame as the query ground truth.

    python scripts/prepare/render_db.py --root data/HGE/capture \\
        --mesh-path <navvis_session>/proc/meshes/mesh_simplified.ply \\
        --poses data/HGE/db_poses.txt --out data/HGE/db
"""
import os

os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")

import argparse  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import open3d as o3d  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from loc.capture_io import (  # noqa: E402
    DEPTH_SCALE,
    load_cameras,
    qt_to_Rt,
    quantize_depth,
    read_rows,
    save_pinhole_sensors,
    stem_for,
)

INVALID_ID = o3d.t.geometry.RaycastingScene.INVALID_ID
# Resolution of the low-cost coverage check run before every full render.
PROBE_W, PROBE_H = 90, 120


def load_scene(mesh_path):
    """Mesh -> (raycasting scene, triangles (F, 3), vertex colors (V, 3) in [0, 1])."""
    t0 = time.time()
    print(f"loading mesh {mesh_path.name} ({mesh_path.stat().st_size / 1e9:.1f} GB)...", flush=True)
    mesh = o3d.t.io.read_triangle_mesh(str(mesh_path))
    mesh.vertex.positions = mesh.vertex.positions.to(o3d.core.Dtype.Float32)
    mesh.triangle.indices = mesh.triangle.indices.to(o3d.core.Dtype.UInt32)
    if "colors" not in mesh.vertex:
        raise SystemExit(f"{mesh_path} has no vertex colors")

    colors = mesh.vertex["colors"].numpy()
    colors = colors.astype(np.float32) / 255.0 if colors.dtype == np.uint8 else colors.astype(np.float32)
    # A few vertices of photogrammetry meshes have no valid color.
    invalid = ~np.isfinite(colors)
    if invalid.any():
        print(f"  {int(invalid.any(axis=1).sum())} vertices with a non-finite color set to black",
              flush=True)
        colors[invalid] = 0.0
    # Open3D divides PLY colors by 255 even when they are already floats in [0, 1].
    if colors.size and colors.mean() <= 2.0 / 255.0:
        colors = colors * 255.0
        print("  colors already normalized: Open3D division by 255 undone", flush=True)

    triangles = mesh.triangle.indices.numpy()
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(mesh)
    print(f"  {len(colors):,} vertices, {len(triangles):,} triangles, BVH ready in "
          f"{time.time() - t0:.1f}s", flush=True)
    return scene, triangles, colors


def build_rays(R_c2w, t_c2w, fx, fy, cx, cy, W, H):
    """Pinhole rays in the world frame, (H*W, 6).

    Directions have z = 1 in the camera frame, so the hit distance is the depth along the optical
    axis. Pixel centers are at +0.5 (COLMAP convention).
    """
    xs = (np.arange(W, dtype=np.float32) + 0.5 - cx) / fx
    ys = (np.arange(H, dtype=np.float32) + 0.5 - cy) / fy
    gx, gy = np.meshgrid(xs, ys)
    dirs = np.stack([gx.ravel(), gy.ravel(), np.ones(gx.size, np.float32)], axis=1)
    rays = np.empty((dirs.shape[0], 6), np.float32)
    rays[:, :3] = t_c2w.astype(np.float32)
    rays[:, 3:] = dirs @ R_c2w.T.astype(np.float32)
    return rays


def render(scene, triangles, colors, R_c2w, t_c2w, cam, W=None, H=None):
    """-> (rgb uint8 HxWx3, depth float32 HxW with 0 where nothing is hit, hit ratio)."""
    W, H = W or cam["width"], H or cam["height"]
    sx, sy = W / cam["width"], H / cam["height"]
    rays = build_rays(R_c2w, t_c2w, cam["fx"] * sx, cam["fy"] * sy, cam["cx"] * sx, cam["cy"] * sy,
                      W, H)
    ans = scene.cast_rays(o3d.core.Tensor(rays))
    hit = ans["geometry_ids"].numpy() != INVALID_ID

    depth = np.where(hit, ans["t_hit"].numpy(), 0.0).astype(np.float32)
    rgb = np.zeros((H * W, 3), np.float32)
    if hit.any():
        tri = triangles[ans["primitive_ids"].numpy()[hit]]
        uv = ans["primitive_uvs"].numpy()[hit]
        u, v = uv[:, 0:1], uv[:, 1:2]
        # Barycentric weights in Open3D: (1 - u - v, u, v).
        rgb[hit] = (1.0 - u - v) * colors[tri[:, 0]] + u * colors[tri[:, 1]] + v * colors[tri[:, 2]]
    rgb = (np.clip(rgb, 0, 1) * 255).round().astype(np.uint8).reshape(H, W, 3)
    return rgb, depth.reshape(H, W), float(hit.mean())


def select_camera(cams, sensor):
    if sensor in cams:
        return sensor
    if len(cams) == 1:
        only = next(iter(cams))
        print(f"sensor '{sensor}' not found, using the only one: '{only}'")
        return only
    raise SystemExit(f"sensor '{sensor}' not found: available {sorted(cams)}")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", type=Path, required=True,
                   help="Capture folder whose sensors.txt provides the intrinsics")
    p.add_argument("--mesh-path", type=Path, required=True)
    p.add_argument("--poses", type=Path, required=True, help="camera poses in Capture format")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--sensor", default="cam0_center",
                   help="camera providing the intrinsics (ignored if sensors.txt has only one)")
    p.add_argument("--depth-format", choices=["png", "exr"], default="png")
    p.add_argument("--min-hit-ratio", type=float, default=0.5,
                   help="skip poses whose view of the mesh covers less than this fraction")
    p.add_argument("--jpeg-quality", type=int, default=95)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    cams = load_cameras(args.root / "sensors.txt")
    sensor = select_camera(cams, args.sensor)
    cam = cams[sensor]
    print(f"intrinsics of {sensor}: {cam['width']}x{cam['height']} fx={cam['fx']:g} fy={cam['fy']:g} "
          f"cx={cam['cx']:.2f} cy={cam['cy']:.2f}")
    poses = list(read_rows(args.poses))
    if args.limit:
        poses = poses[:args.limit]
    print(f"{len(poses)} poses from {args.poses}")

    depth_dir = args.out / ("depth_png" if args.depth_format == "png" else "depth")
    depth_ext = ".png" if args.depth_format == "png" else ".exr"
    (args.out / "images").mkdir(parents=True, exist_ok=True)
    depth_dir.mkdir(parents=True, exist_ok=True)
    if args.depth_format == "png":
        (args.out / "depth_scale.json").write_text(json.dumps(
            {"depth_scale": DEPTH_SCALE, "dtype": "uint16", "unit": "meters", "invalid": 0},
            indent=2), encoding="utf-8")

    scene, triangles, colors = load_scene(args.mesh_path)
    written, rendered, skipped, present = [], 0, 0, 0
    t_start = time.time()
    for i, row in enumerate(poses, 1):
        ts, dev = row[0], row[1]
        R_c2w, t_c2w = qt_to_Rt(*map(float, row[2:9]))
        out_img = args.out / "images" / f"{stem_for(ts, dev)}.jpg"
        out_depth = depth_dir / f"{stem_for(ts, dev)}{depth_ext}"
        if not args.overwrite and out_img.exists() and out_depth.exists():
            written.append(row)
            present += 1
            continue

        t_pose = time.time()
        _, _, coverage = render(scene, triangles, colors, R_c2w, t_c2w, cam, PROBE_W, PROBE_H)
        if coverage < args.min_hit_ratio:
            skipped += 1
            continue
        rgb, depth, hit_ratio = render(scene, triangles, colors, R_c2w, t_c2w, cam)
        cv2.imwrite(str(out_img), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
                    [cv2.IMWRITE_JPEG_QUALITY, args.jpeg_quality])
        cv2.imwrite(str(out_depth), quantize_depth(depth) if args.depth_format == "png" else depth)
        written.append(row)
        rendered += 1
        valid = depth > 0
        median_depth = float(np.median(depth[valid])) if valid.any() else 0.0
        eta = (time.time() - t_start) / i * (len(poses) - i)
        print(f"[{i}/{len(poses)}] {dev}  hit={hit_ratio:.2f}  med={median_depth:5.2f}m  "
              f"{time.time() - t_pose:4.1f}s  ETA {eta / 60:.0f} min", flush=True)

    # Poses are copied verbatim from the input, only for the renders that exist on disk.
    with open(args.out / "db_trajectories.txt", "w", encoding="utf-8") as f:
        f.write("# timestamp, device_id, qw, qx, qy, qz, tx, ty, tz\n")
        for row in written:
            f.write(", ".join(row[:9]) + "\n")
    devices = dict.fromkeys(row[1] for row in written)
    save_pinhole_sensors([(dev, f"synthetic render ({sensor} intrinsics)", cam["width"],
                           cam["height"], cam["fx"], cam["fy"], cam["cx"], cam["cy"])
                          for dev in devices], args.out / "db_sensors.txt")
    print(f"{rendered} rendered, {skipped} poses skipped for low coverage, {present} already "
          f"present, {(time.time() - t_start) / 60:.1f} min")
    print(f"{len(written)} renders in {args.out}")


if __name__ == "__main__":
    main()
