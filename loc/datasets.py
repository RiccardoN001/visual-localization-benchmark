# After data preparation every scene is stored as <data_root>/<FOLDER>/{capture,db}:
#   capture/  sensors.txt, images.txt, trajectories.txt and raw_data/  (queries + ground truth)
#   db/       db_sensors.txt, db_trajectories.txt, images/, depth_png/, depth_scale.json
#             (the reference database rendered from the mesh)
import json
import os
from dataclasses import dataclass
from pathlib import Path

# Must be set before cv2 is imported, otherwise EXR depth maps cannot be read.
os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from .capture_io import DEPTH_SCALE, load_cameras, qt_to_Rt, read_rows, stem_for  # noqa: E402
from .paths import DATA_ROOT  # noqa: E402


@dataclass(frozen=True)
class Scene:
    folder: str
    # Phone scenes localize against the database of their base scene, which is not duplicated.
    db_folder: str = None


SCENES = {
    # LaMAR, NavVis scanner images as queries
    "hge": Scene("HGE"),
    "cab": Scene("CAB"),
    "lin": Scene("LIN"),
    # LaMAR, phone queries against the NavVis-rendered database of the same building
    "hgephone": Scene("HGEPHONE", db_folder="HGE"),
    "linphone": Scene("LINPHONE", db_folder="LIN"),
    # Oxford Spires
    "bodleian": Scene("BODLEIAN"),
    "christchurch": Scene("CHRISTCHURCH"),
    "keble": Scene("KEBLE"),
    "observatory": Scene("OBSERVATORY"),
    # San Andrea (available on request)
    "sanandrea": Scene("SANANDREA"),
}


class Dataset:
    def __init__(self, name, data_root=DATA_ROOT, query_stride=1):
        if name not in SCENES:
            raise SystemExit(f"unknown dataset '{name}': choose from {list(SCENES)}")
        scene = SCENES[name]
        self.name = name
        self.query_root = Path(data_root) / scene.folder / "capture"
        self.db_root = Path(data_root) / (scene.db_folder or scene.folder) / "db"
        self.query_stride = max(1, query_stride)
        self.depth_scale = self._load_depth_scale()

    def _load_depth_scale(self):
        cfg = self.db_root / "depth_scale.json"
        if cfg.exists():
            return float(json.loads(cfg.read_text(encoding="utf-8"))["depth_scale"])
        return DEPTH_SCALE

    def references(self):
        """Rendered database images with their cam2world pose, intrinsics and depth map."""
        traj = self.db_root / "db_trajectories.txt"
        if not traj.exists():
            raise SystemExit(f"database not found: {traj} (see 'Data preparation' in the README)")
        cams = load_cameras(self.db_root / "db_sensors.txt")
        # PNG depth is the default; EXR is accepted for databases rendered with --depth-format exr.
        depth_dir, ext = self.db_root / "depth_png", ".png"
        if not depth_dir.is_dir():
            depth_dir, ext = self.db_root / "depth", ".exr"

        entries = []
        for row in read_rows(traj):
            ts, sensor_id = row[0], row[1]
            c = cams[sensor_id]
            R, t = qt_to_Rt(*map(float, row[2:9]))
            stem = stem_for(ts, sensor_id)
            entries.append({
                "pose_id": stem, "timestamp": ts, "sensor_id": sensor_id,
                "image_path": str(self.db_root / "images" / f"{stem}.jpg"),
                "depth_path": str(depth_dir / f"{stem}{ext}"),
                "K_ref": [c["fx"], c["fy"], c["cx"], c["cy"]],
                "width": c["width"], "height": c["height"],
                "R_c2w": R.tolist(), "t_c2w": t.tolist(),
            })
        return entries

    def queries(self):
        """Query images sorted by time, with intrinsics and ground truth (None when missing)."""
        cams = load_cameras(self.query_root / "sensors.txt")
        gt = {(r[0], r[1]): qt_to_Rt(*map(float, r[2:9]))
              for r in read_rows(self.query_root / "trajectories.txt")}
        rows = [(r[0], r[1], r[2]) for r in read_rows(self.query_root / "images.txt")]

        # The stride subsamples capture stops and keeps every camera of a kept stop. Timestamps
        # are sorted as floats (Oxford uses decimal seconds) but kept as the original strings.
        stops = sorted({ts for ts, _, _ in rows}, key=float)
        keep = set(stops[::self.query_stride])

        entries = []
        for ts, sensor_id, rel in rows:
            if ts not in keep or sensor_id not in cams:
                continue
            c = cams[sensor_id]
            entry = {
                "qid": stem_for(ts, sensor_id), "timestamp": ts, "sensor_id": sensor_id,
                "image_path": str(self.query_root / "raw_data" / rel),
                "K_query": [c["fx"], c["fy"], c["cx"], c["cy"]],
                "width": c["width"], "height": c["height"],
                "gt_R_c2w": None, "gt_t_c2w": None,
            }
            if (ts, sensor_id) in gt:
                R, t = gt[(ts, sensor_id)]
                entry["gt_R_c2w"], entry["gt_t_c2w"] = R.tolist(), t.tolist()
            entries.append(entry)
        entries.sort(key=lambda e: (float(e["timestamp"]), e["sensor_id"]))
        return entries

    def read_depth(self, path):
        """Depth map in meters (float32), 0 = invalid."""
        path = Path(path)
        if path.suffix.lower() == ".png":
            d = cv2.imread(str(path), cv2.IMREAD_ANYDEPTH)
            return None if d is None else (d.astype(np.float32) / self.depth_scale)
        d = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if d is None:
            return None
        return (d if d.ndim == 2 else d[:, :, 0]).astype(np.float32)
