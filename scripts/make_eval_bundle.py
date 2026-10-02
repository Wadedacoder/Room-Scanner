"""Pack proxy photo/video sets + LiDAR ground truth into a small, self-contained bundle for model evaluation
(runs off-machine, e.g. Kaggle). Each set is a folder:

    <set>/images/IMG_0001.jpg ...     upright RGB, what the model sees
    <set>/gt.npz                      per image: lidar depth (m) + confidence, rotated like the image,
                                      K at image resolution, T_world_cam of the *upright* camera (OpenCV c2w, gravity +Y up)

Sets: photos_<mode>_<room>_n<N> for every proxy photo set, plus video60: 60 frames sampled evenly over the walk.

Usage: python scripts/make_eval_bundle.py data/raw/lidar/c00a170fe1 -o data/derived/eval_bundle
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_photo_tier import frame_angles, read_frames, save_jpeg, upright  # noqa: E402

from roomscan.io.stray import StrayCapture  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def gt_for(cap: StrayCapture, idx: int, k90: int) -> tuple[np.ndarray, np.ndarray]:
    return np.rot90(cap.depth(idx), k90).copy(), np.rot90(cap.confidence(idx), k90).copy()


# Rotating the image by k*90 deg is rotating the camera about its optical axis: p_orig_cam = Q[k] @ p_new_cam.
ROLL = {
    0: np.eye(3),
    1: np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1.0]]),  # np.rot90(img, 1): u' = v, v' = W-1-u
    -1: np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1.0]]),  # np.rot90(img, -1): u' = H-1-v, v' = u
}


def upright_pose(T: np.ndarray, k90: int) -> np.ndarray:
    T = T.copy()
    T[:3, :3] = T[:3, :3] @ ROLL[k90]
    return T


def write_set(dst: Path, cap: StrayCapture, items: list[tuple[str, int, np.ndarray, np.ndarray, int]]) -> None:
    (dst / "images").mkdir(parents=True, exist_ok=True)
    depth, conf, Ks, Ts, names, src = [], [], [], [], [], []
    for name, idx, img, K, k90 in items:
        save_jpeg(dst / "images" / name, img, K)
        d, c = gt_for(cap, idx, k90)
        depth.append(d), conf.append(c), Ks.append(K), Ts.append(upright_pose(cap.poses[idx], k90)), names.append(name), src.append(idx)
    np.savez_compressed(dst / "gt.npz", depth=np.stack(depth), conf=np.stack(conf), K=np.stack(Ks),
                        T_world_cam=np.stack(Ts), names=np.array(names), source_frame=np.array(src))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("capture", type=Path)
    ap.add_argument("-o", "--out", type=Path, default=ROOT / "data/derived/eval_bundle")
    ap.add_argument("--video-frames", type=int, nargs="+", default=[60, 120])
    ap.add_argument("--video-only", action="store_true", help="add/refresh video sets, keep existing photo sets")
    args = ap.parse_args()

    cap = StrayCapture.open(args.capture)
    name = args.capture.resolve().name
    if not args.video_only:
        shutil.rmtree(args.out, ignore_errors=True)
    _, _, upx = frame_angles(cap.poses)

    sets = [] if not args.video_only else [
        s for s in json.loads((args.out / "index.json").read_text())["sets"] if s.startswith("photos_")]
    photo_dirs = [] if args.video_only else sorted((ROOT / "data/derived/photos_gt").glob(f"{name}_*_n*"))
    for gt_dir in photo_dirs:
        mode, n = gt_dir.name[len(name) + 1:].rsplit("_n", 1)
        for room_json in sorted(gt_dir.glob("*.json")):
            meta = json.loads(room_json.read_text())
            raw = read_frames(args.capture, [f["source_frame"] for f in meta["frames"]])
            items = []
            for f in meta["frames"]:
                img, K, k90 = upright(raw[f["source_frame"]], cap.K, upx[f["source_frame"]])
                items.append((f["file"], f["source_frame"], img, K, k90))
            set_name = f"photos_{mode}_{room_json.stem}_n{n}"
            write_set(args.out / set_name, cap, items)
            sets.append(set_name)

    for nv in args.video_frames:
        # rgb.mp4 has one frame fewer than odometry.csv
        idxs = np.linspace(0, len(cap) - 2, nv).round().astype(int)
        raw = read_frames(args.capture, idxs)
        clip_upx = float(np.median(upx[idxs]))  # a real clip has ONE orientation: majority vote, not per frame
        items = []
        for j, idx in enumerate(idxs, 1):
            img, K, k90 = upright(raw[int(idx)], cap.K, clip_upx)
            items.append((f"FRAME_{j:04d}.jpg", int(idx), img, K, k90))
        write_set(args.out / f"video{nv}", cap, items)
        sets.append(f"video{nv}")

    (args.out / "index.json").write_text(json.dumps({"source_capture": name, "sets": sets}, indent=1))
    print(f"{len(sets)} sets -> {args.out}")


if __name__ == "__main__":
    main()
