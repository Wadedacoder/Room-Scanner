"""Derive a *proxy* photo tier from a Stray Scanner capture (dev data until real iPhone stills exist).

For each requested N in 2..8 we write upright, sharp, mildly down-tilted JPEG stills per room, chosen in one of two modes:
  spread  greedy farthest-point over (yaw, position): views as different as possible, so little overlap. This is the
          adversarial case; E1 showed it breaks pose registration. N=2 is a subset of N=4, of N=6, and so on.
  sweep   closest proxy for the protocol's "turn in place" ring: N stills spread evenly in viewing direction and
          ORDERED by direction, so consecutive stills are neighbours. Real protocol photos are 0.5x landscape (~100 deg
          wide) and overlap ~50%; these proxy frames are 1x portrait (~48 deg), so overlap here is a worst case.

Only the JPEGs (with EXIF focal length, like a real iPhone photo) go into the pipeline-facing folder:
    data/derived/photos/<capture>_<mode>_n<N>/<room>/IMG_0001.jpg
Poses/intrinsics are written to a separate ground-truth folder the pipeline never reads:
    data/derived/photos_gt/<capture>_<mode>_n<N>/<room>.json

Disclosed limitation: these frames come from a Pro iPhone's video stream (motion blur, video ISP, 1920x1440),
not from the Camera app on a non-Pro iPhone 15. Real stills will replace them.

Usage: python scripts/make_photo_tier.py data/raw/lidar/c00a170fe1 --rooms bench/annotations/c00a170fe1_rooms.yaml
"""

from __future__ import annotations

import argparse
import sys
import json
from pathlib import Path

import cv2
import numpy as np
import yaml
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))  # run from a checkout without relying on the editable install's .pth
from roomscan.io.stray import StrayCapture  # noqa: E402


def frame_angles(T: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """yaw (rad, around world +Y), pitch (rad, + is up), roll indicator for camera-to-world poses (N,4,4)."""
    fwd = T[:, :3, 2]  # OpenCV camera looks down +Z
    yaw = np.arctan2(fwd[:, 0], -fwd[:, 2])
    pitch = np.arcsin(np.clip(fwd[:, 1], -1, 1))
    up_cam_x = T[:, 1, 0]  # world-up component of the camera +X axis: large |.| => phone held in portrait
    return yaw, pitch, up_cam_x


def sharpness(cap_path: Path, step: int) -> dict[int, float]:
    vid = cv2.VideoCapture(str(cap_path / "rgb.mp4"))
    out, i = {}, 0
    while True:
        ok = vid.grab()
        if not ok:
            break
        if i % step == 0:
            _, f = vid.retrieve()
            g = cv2.cvtColor(cv2.resize(f, (480, 360)), cv2.COLOR_BGR2GRAY)
            out[i] = float(cv2.Laplacian(g, cv2.CV_64F).var())
        i += 1
    return out


def read_frames(cap_path: Path, idxs) -> dict[int, np.ndarray]:
    """Decode the requested frames by reading sequentially. Seeking (CAP_PROP_POS_FRAMES) on this HEVC stream
    lands on the wrong frame (measured: up to 38 grey levels mean diff), which would pair images with wrong poses."""
    want, out, i = set(int(x) for x in idxs), {}, 0
    vid = cv2.VideoCapture(str(cap_path / "rgb.mp4"))
    while len(out) < len(want):
        ok, f = vid.read()
        if not ok:
            raise RuntimeError(f"video ended at frame {i}; missing {sorted(want - set(out))}")
        if i in want:
            out[i] = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
        i += 1
    return out


def select_order(cap: StrayCapture, sharp: dict[int, float], max_n: int, ranges: list[list[int]],
                 mode: str = "spread") -> list[int]:
    T = cap.poses
    yaw, pitch, _ = frame_angles(T)
    pos = T[:, [0, 2], 3]
    cand = np.array([i for i in sorted(sharp) if any(a <= i <= b for a, b in ranges)])
    s = np.array([sharp[i] for i in cand])
    # protocol-like: phone tilted slightly down, not at the floor/ceiling; top 50% sharpest;
    # wide corner-to-corner views rather than wall close-ups (median depth in the top 60% for this room)
    view = np.array([np.median(cap.depth(i)[cap.confidence(i) >= 1]) for i in cand])
    ok = ((pitch[cand] > np.radians(-35)) & (pitch[cand] < np.radians(10)) & (s >= np.median(s))
          & (view >= np.percentile(view, 40)))
    cand = cand[ok]
    if len(cand) < max_n:
        raise RuntimeError("not enough usable frames")
    if mode == "sweep":  # even coverage of viewing direction, ordered by direction so neighbours are closest
        order = [int(cand[np.argmax([sharp[i] for i in cand])])]
        while len(order) < max_n:
            d = np.abs(np.angle(np.exp(1j * (yaw[cand][:, None] - yaw[order][None, :])))).min(1)
            order.append(int(cand[np.argmax(d)]))
        return sorted(order, key=lambda i: yaw[i])
    # spread: start from the sharpest frame, then greedily maximise min distance in (yaw, position)
    order = [int(cand[np.argmax([sharp[i] for i in cand])])]
    while len(order) < max_n:
        dyaw = np.abs(np.angle(np.exp(1j * (yaw[cand][:, None] - yaw[order][None, :]))))  # wrapped
        dpos = np.linalg.norm(pos[cand][:, None] - pos[order][None, :], axis=-1)
        d = (dyaw / np.pi + 0.25 * dpos).min(1)
        order.append(int(cand[np.argmax(d)]))
    return order


def upright(img: np.ndarray, K: np.ndarray, up_cam_x: float) -> tuple[np.ndarray, np.ndarray, int]:
    """Rotate a landscape sensor frame so world-up points to the image top, adjusting K. Returns (img, K, k90)."""
    h, w = img.shape[:2]
    if abs(up_cam_x) < 0.7:
        return img, K, 0
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    if up_cam_x > 0:  # camera +X points up: rotate 90 deg counter-clockwise
        img2 = np.rot90(img, 1)
        K2 = np.array([[fy, 0, cy], [0, fx, w - 1 - cx], [0, 0, 1]])
        return np.ascontiguousarray(img2), K2, 1
    img2 = np.rot90(img, -1)
    K2 = np.array([[fy, 0, h - 1 - cy], [0, fx, cx], [0, 0, 1]])
    return np.ascontiguousarray(img2), K2, -1


def save_jpeg(path: Path, img: np.ndarray, K: np.ndarray) -> None:
    h, w = img.shape[:2]
    f35 = int(round(K[0, 0] * np.hypot(36, 24) / np.hypot(w, h)))  # 35 mm-equivalent focal, as iPhone EXIF has
    exif = Image.Exif()
    exif[0x010F] = "Apple"
    exif[0x0110] = "derived-from-StrayScanner-video"
    exif.get_ifd(0x8769)[0xA405] = f35
    Image.fromarray(img).save(path, quality=92, exif=exif)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("capture", type=Path)
    ap.add_argument("--n", type=int, nargs="+", default=[2, 4, 6, 8])
    ap.add_argument("--rooms", type=Path, help="yaml mapping room -> frame ranges (else one room, all frames)")
    ap.add_argument("--step", type=int, default=3, help="score every k-th frame for sharpness")
    ap.add_argument("--mode", nargs="+", default=["sweep", "spread"], choices=["sweep", "spread"])
    args = ap.parse_args()
    assert all(2 <= n <= 8 for n in args.n), "photo tier is 2..8 stills per room"

    cap = StrayCapture.open(args.capture)
    name = args.capture.resolve().name
    rooms = yaml.safe_load(args.rooms.read_text())["rooms"] if args.rooms else {"room1": [[0, len(cap) - 1]]}
    sharp = sharpness(args.capture, args.step)
    _, _, upx = frame_angles(cap.poses)

    yaw, _, _ = frame_angles(cap.poses)
    for mode in args.mode:
        for room, ranges in rooms.items():
            if mode == "spread":
                full = select_order(cap, sharp, max(args.n), ranges, mode)
                orders = {n: full[:n] for n in args.n}
            else:
                orders = {n: select_order(cap, sharp, n, ranges, mode) for n in args.n}
            needed = sorted({i for o in orders.values() for i in o})
            raw = read_frames(args.capture, needed)
            frames = {idx: upright(raw[idx], cap.K, upx[idx]) for idx in needed}
            for n in sorted(args.n):
                order = orders[n]
                out = ROOT / "data/derived/photos" / f"{name}_{mode}_n{n}" / room
                gt = ROOT / "data/derived/photos_gt" / f"{name}_{mode}_n{n}"
                out.mkdir(parents=True, exist_ok=True)
                gt.mkdir(parents=True, exist_ok=True)
                meta = []
                for j, idx in enumerate(order, 1):
                    img, K, k90 = frames[idx]
                    fname = f"IMG_{j:04d}.jpg"
                    save_jpeg(out / fname, img, K)
                    meta.append(
                        {"file": fname, "source_frame": idx, "rot90": k90, "K": K.tolist(),
                         "T_world_cam_opencv": cap.poses[idx].tolist(), "sharpness": sharp[idx]}
                    )
                (gt / f"{room}.json").write_text(json.dumps(
                    {"source_capture": name, "room": room, "mode": mode,
                     "derived": "proxy photo tier from LiDAR-capture video frames", "frames": meta}, indent=1))
                steps = np.degrees(np.abs(np.angle(np.exp(1j * np.diff(yaw[order])))))
                print(f"{mode:6s} {room:9s} n={n}: frames={order}  consecutive yaw step max={steps.max():.0f} deg")

if __name__ == "__main__":
    main()
