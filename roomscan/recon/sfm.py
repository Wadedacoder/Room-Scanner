"""COLMAP structure-from-motion for video frames, run in a child process.

pycolmap and torch each bundle an OpenMP runtime and abort if both load in one process (E5), so the pipeline never
imports pycolmap itself: `run_sfm` launches this module as a script and reads back JSON.

The lens is usually unknown for video (iPhone .MOV files carry no focal length), so COLMAP starts from a guess and
refines the focal length during bundle adjustment. The refined focal is reported; it also tells us which lens was
used (0.5x vs 1x), which the file itself doesn't record.
"""

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class SfmResult:
    c2w: dict[str, np.ndarray]  # frame name -> 4x4 camera-to-world (OpenCV), arbitrary scale
    points: np.ndarray  # (P,3) sparse 3D points
    obs: dict[str, tuple[np.ndarray, np.ndarray]]  # frame -> (pixel uv (n,2), point index (n,))
    K: np.ndarray  # refined intrinsics at the frame resolution
    pieces: list[int]  # registered frames per reconstructed piece, largest first
    n_frames: int


def run_sfm(frames_dir: Path, names: list[str], focal_guess: float, work: Path) -> SfmResult:
    work.mkdir(parents=True, exist_ok=True)
    out = work / "sfm.json"
    cmd = [sys.executable, "-m", "roomscan.recon.sfm", str(frames_dir), str(work), str(focal_guess), json.dumps(names)]
    subprocess.run(cmd, check=True, capture_output=True, text=True)
    d = json.loads(out.read_text())
    return SfmResult({k: np.array(v) for k, v in d["c2w"].items()}, np.array(d["points"]).reshape(-1, 3),
                     {k: (np.array(v[0]).reshape(-1, 2), np.array(v[1], int)) for k, v in d["obs"].items()},
                     np.array(d["K"]), d["pieces"], len(names))


def _child(frames_dir: str, work: str, focal_guess: str, names_json: str) -> None:
    import shutil

    import pycolmap

    names = json.loads(names_json)
    work_p = Path(work)
    db = work_p / "database.db"
    if db.exists():
        db.unlink()
    shutil.rmtree(work_p / "sparse", ignore_errors=True)
    from PIL import Image

    w, h = Image.open(Path(frames_dir) / names[0]).size
    reader = pycolmap.ImageReaderOptions()
    reader.camera_model = "SIMPLE_RADIAL"
    reader.camera_params = f"{float(focal_guess)},{w / 2},{h / 2},0"
    pycolmap.extract_features(db, frames_dir, image_names=names, camera_mode=pycolmap.CameraMode.SINGLE,
                              reader_options=reader, device=pycolmap.Device.cpu)
    pair = pycolmap.SequentialPairingOptions()
    pair.overlap = 10
    if hasattr(pair, "loop_detection"):
        pair.loop_detection = False  # needs a vocabulary tree file; drift is handled downstream
    pycolmap.match_sequential(db, pairing_options=pair, device=pycolmap.Device.cpu)
    opts = pycolmap.IncrementalPipelineOptions()
    for k, v in (("ba_refine_focal_length", True), ("ba_refine_principal_point", False),
                 ("ba_refine_extra_params", True)):
        if hasattr(opts, k):
            setattr(opts, k, v)
    maps = pycolmap.incremental_mapping(db, frames_dir, work_p / "sparse", options=opts)
    if not maps:
        raise SystemExit("COLMAP registered no frames")
    ranked = sorted(maps.values(), key=lambda r: r.num_reg_images(), reverse=True)
    rec = ranked[0]
    cam = next(iter(rec.cameras.values()))
    f, cx, cy = cam.params[0], cam.params[1], cam.params[2]
    pid_index, pts = {}, []
    for pid, p in rec.points3D.items():
        pid_index[pid] = len(pts)
        pts.append(p.xyz.tolist())
    c2w, obs = {}, {}
    for img in rec.images.values():
        if not img.has_pose:
            continue
        cfw = img.cam_from_world() if callable(img.cam_from_world) else img.cam_from_world
        T = np.eye(4)
        T[:3, :4] = cfw.matrix()
        c2w[img.name] = np.linalg.inv(T).tolist()
        uv, idx = [], []
        for p2 in img.points2D:
            if p2.has_point3D() and p2.point3D_id in pid_index:
                uv.append(p2.xy.tolist())
                idx.append(pid_index[p2.point3D_id])
        obs[img.name] = [uv, idx]
    out = {"c2w": c2w, "points": pts, "obs": obs, "K": [[f, 0, cx], [0, f, cy], [0, 0, 1]],
           "pieces": [r.num_reg_images() for r in ranked]}
    (work_p / "sfm.json").write_text(json.dumps(out))


if __name__ == "__main__":
    _child(*sys.argv[1:5])
