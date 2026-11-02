"""Turn positions sampled on the mesh into database camera poses (Capture format).

Each unique sampled position gets --yaws equally spaced headings at a fixed --pitch. Positions
outside the height range covered by the query trajectory (plus --z-margin) are discarded.

    python scripts/prepare/make_db_poses.py --points <sampled_positions.json> \\
        --reference-trajectories data/HGE/capture/trajectories.txt --out data/HGE/db_poses.txt
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from loc.capture_io import R_BODY_TO_OPTICAL, save_capture_trajectory  # noqa: E402


def reference_z_range(path):
    """(zmin, zmax) of a Capture (comma-separated) or TUM (space-separated) trajectory."""
    z = []
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if line and not line.startswith("#"):
            z.append(float(line.split(",")[8]) if "," in line else float(line.split()[3]))
    if not z:
        raise SystemExit(f"no poses in {path}")
    return min(z), max(z)


def camera_rotation(yaw_deg, pitch_deg):
    """cam2world rotation (optical frame) from a heading around world Z and an upward pitch."""
    R_body = Rotation.from_euler("z", yaw_deg, degrees=True).as_matrix()
    R_pitch = Rotation.from_euler("x", pitch_deg, degrees=True).as_matrix()
    return R_body @ R_BODY_TO_OPTICAL @ R_pitch


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--points", type=Path, required=True,
                   help='JSON list of entries with a "position": [x, y, z] field')
    p.add_argument("--reference-trajectories", type=Path, required=True,
                   help="query trajectory (Capture or TUM) used for the height filter")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--yaws", type=int, default=4)
    p.add_argument("--pitch", type=float, default=0.0, help="upward pitch in degrees")
    p.add_argument("--z-margin", type=float, default=0.25)
    p.add_argument("--no-z-filter", action="store_true")
    p.add_argument("--fov-deg", type=float, default=2 * np.degrees(np.arctan(640 / 960)),
                   help="horizontal field of view of the queries, used only for the overlap printed "
                        "at the end (default: NavVis)")
    args = p.parse_args()

    pts = np.array([e["position"] for e in json.loads(args.points.read_text(encoding="utf-8"))])
    # Unique positions in a stable order, so that device ids do not change between runs.
    _, first = np.unique(np.round(pts, 4), axis=0, return_index=True)
    positions = pts[np.sort(first)]
    print(f"{len(pts)} entries -> {len(positions)} unique positions")

    if not args.no_z_filter:
        zmin, zmax = reference_z_range(args.reference_trajectories)
        lo, hi = zmin - args.z_margin, zmax + args.z_margin
        keep = (positions[:, 2] >= lo) & (positions[:, 2] <= hi)
        print(f"height filter z in [{lo:.2f}, {hi:.2f}] (query range {zmin:.2f}..{zmax:.2f} "
              f"+/- {args.z_margin}): kept {keep.sum()}, dropped {(~keep).sum()}")
        for q in positions[~keep]:
            print(f"  dropped (x, y, z) = {np.round(q, 2).tolist()}")
        positions = positions[keep]
    if len(positions) == 0:
        raise SystemExit("no positions left: check --z-margin or use --no-z-filter")

    yaws = np.arange(args.yaws) * (360.0 / args.yaws)
    rows = []
    for i, pos in enumerate(positions):
        for yaw in yaws:
            # Synthetic increasing timestamp; the device id is the key of the database entry.
            rows.append((len(rows), f"synth/pt{i:04d}_yaw{int(round(yaw)):03d}",
                         camera_rotation(yaw, args.pitch), pos))
    save_capture_trajectory(rows, args.out)
    print(f"wrote {len(rows)} poses ({len(positions)} positions x {args.yaws} yaws) to {args.out}")
    print(f"  pitch {args.pitch:g} deg upward, yaw step {360.0 / args.yaws:g} deg")
    # A query is at most half a yaw step away from the closest rendered heading.
    overlap = args.fov_deg - 180.0 / args.yaws
    print(f"  worst query misalignment {180.0 / args.yaws:.1f} deg -> overlap {overlap:.1f} of "
          f"{args.fov_deg:.1f} deg of field of view ({100 * overlap / args.fov_deg:.0f}%)")


if __name__ == "__main__":
    main()
