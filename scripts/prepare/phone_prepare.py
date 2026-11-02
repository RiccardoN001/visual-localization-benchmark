"""Prepare the LaMAR phone queries (sessions/query_val_phone) of a scene for the pipeline.

The phone ground truth (proc/alignment_trajectories.txt) is in the global LaMAR frame, while the
NavVis mesh and the rendered database are in the frame of the NavVis session. The ground truth is
moved to the NavVis frame with the inverse of the session alignment, and only the official queries
of queries.txt are kept.

    python scripts/prepare/phone_prepare.py --session <HGE>/sessions/query_val_phone \\
        --alignment <HGE>/<navvis_session>/proc/alignment_global.txt --out data/HGEPHONE/capture
"""
import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from loc.capture_io import (  # noqa: E402
    invert_Rt,
    load_alignment_global,
    load_cameras,
    qt_to_Rt,
    read_rows,
    save_capture_trajectory,
    save_pinhole_sensors,
)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--session", type=Path, required=True, help="sessions/query_val_phone folder")
    p.add_argument("--alignment", type=Path, required=True,
                   help="proc/alignment_global.txt of the NavVis session used for the database")
    p.add_argument("--out", type=Path, required=True, help="output capture/ folder")
    p.add_argument("--skip-images", action="store_true", help="write only the metadata files")
    args = p.parse_args()

    queries = [(r[0], r[1]) for r in read_rows(args.session / "queries.txt")]
    cams = load_cameras(args.session / "sensors.txt")
    images = {r[1]: r[2] for r in read_rows(args.session / "images.txt")}
    gt_global = {r[1]: qt_to_Rt(*map(float, r[2:9]))
                 for r in read_rows(args.session / "proc" / "alignment_trajectories.txt")}
    print(f"{len(queries)} official queries, {len(cams)} sensors, {len(gt_global)} global "
          "ground-truth poses")
    missing = [s for _, s in queries if s not in cams or s not in images or s not in gt_global]
    if missing:
        raise SystemExit(f"{len(missing)} queries without camera, image or ground truth: {missing[:3]}")

    # p_global = R @ p_navvis + t, so the inverse maps global poses to the NavVis frame.
    R_g2n, t_g2n = invert_Rt(*load_alignment_global(args.alignment))

    (args.out / "raw_data").mkdir(parents=True, exist_ok=True)
    rows = []
    with open(args.out / "images.txt", "w", encoding="utf-8") as f:
        f.write("# timestamp, sensor_id, image_path\n")
        for ts, sensor_id in queries:
            name = Path(images[sensor_id]).name
            if not args.skip_images:
                shutil.copyfile(args.session / "raw_data" / images[sensor_id],
                                args.out / "raw_data" / name)
            f.write(f"{ts}, {sensor_id}, {name}\n")
            R, t = gt_global[sensor_id]
            rows.append((ts, sensor_id, R_g2n @ R, R_g2n @ t + t_g2n))

    # Phone intrinsics change from frame to frame (autofocus): one sensor per query.
    save_pinhole_sensors([(s, "phone query", cams[s]["width"], cams[s]["height"], cams[s]["fx"],
                           cams[s]["fy"], cams[s]["cx"], cams[s]["cy"]) for _, s in queries],
                         args.out / "sensors.txt")
    save_capture_trajectory(rows, args.out / "trajectories.txt")
    print(f"{len(rows)} queries written to {args.out}")


if __name__ == "__main__":
    main()
