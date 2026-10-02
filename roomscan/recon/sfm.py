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
class Piece:
    c2w: dict[str, np.ndarray]  # frame name -> 4x4 camera-to-world (OpenCV), this piece's own arbitrary scale
    points: np.ndarray  # (P,3)
    obs: dict[str, tuple[np.ndarray, np.ndarray]]  # frame -> (pixel uv (n,2), point index (n,))


@dataclass
class SfmResult:
    c2w: dict[str, np.ndarray]  # largest piece (kept for callers that use one piece)
    points: np.ndarray
    obs: dict[str, tuple[np.ndarray, np.ndarray]]
    K: np.ndarray  # refined intrinsics at the frame resolution (largest piece)
    pieces: list[int]  # registered frames per reconstructed piece, largest first
    n_frames: int
    all_pieces: list[Piece] = None  # every piece with >= 8 frames, largest first (fix loop: join them)


def run_sfm(frames_dir: Path, names: list[str], focal_guess: float, work: Path, matcher: str = "sequential",
            overlap: int = 10, min_inliers: int = 30, learned_matches: Path | None = None) -> SfmResult:
    """learned_matches: .npz from roomscan.recon.learned_matches (DISK + LightGlue) used instead of SIFT matching."""
    work.mkdir(parents=True, exist_ok=True)
    from roomscan.recon import checkpoint as ck

    key = ck.key_for("sfm", meta={"names": names, "focal_guess": round(float(focal_guess), 3), "matcher": matcher,
                                  "overlap": overlap, **({"min_inliers": min_inliers} if min_inliers != 30 else {}),
                                  **({"learned": learned_matches.name} if learned_matches else {})}, files=[frames_dir / n for n in names])
    done = work / f"sfm_{key}.json"
    if ck.enabled() and done.exists():  # checkpoint: same frames + settings already solved
        return _parse(json.loads(done.read_text()), len(names))
    out = work / "sfm.json"
    cmd = [sys.executable, "-m", "roomscan.recon.sfm", str(frames_dir), str(work), str(focal_guess), json.dumps(names),
           matcher, str(overlap), str(min_inliers), str(learned_matches or "")]
    r = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if r.returncode != 0:  # surface the child's own error (sweep, 2026-10-03: a bare CalledProcessError hid the cause)
        tail = (r.stderr or r.stdout or "").strip().splitlines()[-6:]
        raise RuntimeError(f"COLMAP step failed (exit {r.returncode}): " + " | ".join(tail))
    out.replace(done)
    return _parse(json.loads(done.read_text()), len(names))


def _piece(d) -> Piece:
    return Piece({k: np.array(v) for k, v in d["c2w"].items()}, np.array(d["points"]).reshape(-1, 3),
                 {k: (np.array(v[0]).reshape(-1, 2), np.array(v[1], int)) for k, v in d["obs"].items()})


def _parse(d: dict, n: int) -> SfmResult:
    main = _piece(d)
    return SfmResult(main.c2w, main.points, main.obs, np.array(d["K"]), d["pieces"], n,
                     [_piece(p) for p in d.get("all", [])] or [main])


def _child(frames_dir: str, work: str, focal_guess: str, names_json: str, matcher: str = "sequential",
           overlap: str = "10", min_inliers: str = "30", learned: str = "") -> None:
    import shutil

    import pycolmap

    # Determinism (E10): with default threading and seeds, two runs of the same video tracked 92 vs 65 frames and gave
    # -3.4% vs -33.5% area. Fixed seed + single-threaded mapping makes reruns identical.
    pycolmap.set_random_seed(0)
    names = json.loads(names_json)
    work_p = Path(work)
    db = work_p / "database.db"
    # A run killed mid-write leaves SQLite's -wal/-shm side files; with only database.db removed, COLMAP then fails to
    # open a fresh database ("No registered database factory succeeded"). Remove all three.
    for f in (db, work_p / "database.db-wal", work_p / "database.db-shm"):
        f.unlink(missing_ok=True)
    shutil.rmtree(work_p / "sparse", ignore_errors=True)
    from PIL import Image

    w, h = Image.open(Path(frames_dir) / names[0]).size
    reader = pycolmap.ImageReaderOptions()
    reader.camera_model = "SIMPLE_RADIAL"
    reader.camera_params = f"{float(focal_guess)},{w / 2},{h / 2},0"
    ext = pycolmap.FeatureExtractionOptions()
    ext.num_threads = 1  # parallel extraction writes features in a varying order -> different reconstructions
    pycolmap.extract_features(db, frames_dir, image_names=names, camera_mode=pycolmap.CameraMode.SINGLE,
                              reader_options=reader, extraction_options=ext, device=pycolmap.Device.cpu)
    mopt = pycolmap.FeatureMatchingOptions()
    mopt.num_threads = 1
    if learned:
        from roomscan.recon.learned_matches import import_into_colmap

        import_into_colmap(db, Path(learned))
    elif matcher == "exhaustive":
        pycolmap.match_exhaustive(db, matching_options=mopt, device=pycolmap.Device.cpu)
    else:
        pair = pycolmap.SequentialPairingOptions()
        pair.overlap = int(overlap)
        if hasattr(pair, "loop_detection"):
            pair.loop_detection = False  # needs a vocabulary tree file
        pycolmap.match_sequential(db, matching_options=mopt, pairing_options=pair, device=pycolmap.Device.cpu)
    # The mapper is very sensitive to its starting pair: one random run tracked 190 frames, another 99 (E10). Matching
    # is done once; mapping is tried with a few FIXED seeds and the reconstruction that tracks the most frames wins.
    # Same input -> same seeds -> same result.
    best = None
    for seed in (0, 1, 2, 3):
        pycolmap.set_random_seed(seed)
        opts = pycolmap.IncrementalPipelineOptions()
        opts.random_seed = seed
        opts.num_threads = 1
        if int(min_inliers) < 30:
            # E22: register blurrier frames through turns (COLMAP defaults: 30 inliers, ratio 0.25, 15 matches)
            opts.mapper.abs_pose_min_num_inliers = int(min_inliers)
            opts.mapper.abs_pose_min_inlier_ratio = 0.15
            opts.min_num_matches = 10
        for k, v in (("ba_refine_focal_length", True), ("ba_refine_principal_point", False),
                     ("ba_refine_extra_params", True)):
            if hasattr(opts, k):
                setattr(opts, k, v)
        out_dir = work_p / "sparse" / f"seed{seed}"
        out_dir.mkdir(parents=True, exist_ok=True)
        maps = pycolmap.incremental_mapping(db, frames_dir, out_dir, options=opts)
        if maps and (best is None or max(r.num_reg_images() for r in maps.values()) >
                     max(r.num_reg_images() for r in best.values())):
            best = maps
    maps = best
    if not maps:
        raise SystemExit("COLMAP registered no frames")
    ranked = sorted(maps.values(), key=lambda r: r.num_reg_images(), reverse=True)
    def export(rec):
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
        return {"c2w": c2w, "points": pts, "obs": obs}

    rec = ranked[0]
    cam = next(iter(rec.cameras.values()))
    f, cx, cy = cam.params[0], cam.params[1], cam.params[2]
    out = export(rec)
    out |= {"K": [[f, 0, cx], [0, f, cy], [0, 0, 1]], "pieces": [r.num_reg_images() for r in ranked],
            "all": [export(r) for r in ranked if r.num_reg_images() >= 8]}
    (work_p / "sfm.json").write_text(json.dumps(out))


if __name__ == "__main__":
    _child(*sys.argv[1:9])
