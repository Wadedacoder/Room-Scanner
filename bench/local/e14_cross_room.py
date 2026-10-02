"""E14: can photos from different room folders be linked? Cross-room feature matching, SIFT vs DISK+LightGlue.

For every pair of photos from different rooms: matches verified by a fundamental-matrix RANSAC (MAGSAC). A real shared
view (a doorway looking into the next room) should give far more verified matches than coincidences (repeated doors,
fans, tiles). Prints, per room pair, the best photo pair.

Usage: python bench/local/e14_cross_room.py [house_b]
"""

from __future__ import annotations

import itertools
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps

import pillow_heif

pillow_heif.register_heif_opener()
ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "runs/e14"


def load(capture: str, long_side: int = 1024):
    items = []
    for rd in sorted((ROOT / "data/raw/photos" / capture).iterdir()):
        if not rd.is_dir():
            continue
        for p in sorted(rd.glob("*.HEIC")) + sorted(rd.glob("*.jpg")):
            im = np.asarray(ImageOps.exif_transpose(Image.open(p)).convert("RGB"))
            s = long_side / max(im.shape[:2])
            items.append((rd.name, p.stem[-4:], cv2.resize(im, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)))
    return items


def verify(pi: np.ndarray, pj: np.ndarray) -> int:
    if len(pi) < 12:
        return 0
    F, mask = cv2.findFundamentalMat(pi, pj, cv2.USAC_MAGSAC, 1.5, 0.999, 10000)
    return int(mask.sum()) if mask is not None else 0


def lightglue_matches(items):
    import torch
    import kornia.feature as KF

    # DISK at 1024 px exceeded the 3.2 GB MPS cap; on CPU it took ~1 min per image. The user allows a 6 GB total
    # budget, so MPS with a 4.5 GB cap (0.85 of the 5.33 GB working set), leaving ~1.5 GB for the process + macOS headroom.
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    if dev == "mps":
        torch.mps.set_per_process_memory_fraction(0.85)
    # torch and OpenCV each start an OpenMP pool; together they stalled at ~7% CPU for 8 min. Pin both.
    torch.set_num_threads(4)
    cv2.setNumThreads(1)
    disk = KF.DISK.from_pretrained("depth").to(dev).eval()
    lg = KF.LightGlueMatcher("disk").to(dev).eval()
    feats = []
    with torch.inference_mode():
        for n, (_, _, im) in enumerate(items):
            print(f"  disk image {n}/{len(items)}", flush=True)
            t = torch.from_numpy(im).permute(2, 0, 1).float()[None].to(dev) / 255.0
            f = disk(t, n=2048, pad_if_not_divisible=True)[0]
            feats.append((f.keypoints, f.descriptors, t.shape[-2:]))
        out = {}
        pairs = list(itertools.combinations(range(len(items)), 2))
        t0 = time.time()
        for n, (i, j) in enumerate(pairs):
            if n % 40 == 0:
                print(f"  lightglue pair {n}/{len(pairs)} {time.time() - t0:.0f}s", flush=True)
            ki, di, si = feats[i]
            kj, dj, sj = feats[j]
            lafi = KF.laf_from_center_scale_ori(ki[None], torch.ones(1, len(ki), 1, 1, device=dev))
            lafj = KF.laf_from_center_scale_ori(kj[None], torch.ones(1, len(kj), 1, 1, device=dev))
            _, idx = lg(di, dj, lafi, lafj, hw1=si, hw2=sj)
            idx = idx.cpu().numpy()
            out[(i, j)] = verify(ki.cpu().numpy()[idx[:, 0]].astype(np.float32),
                                 kj.cpu().numpy()[idx[:, 1]].astype(np.float32))
    return out


def sift_matches(items):
    sift = cv2.SIFT_create(nfeatures=4000)
    feats = [sift.detectAndCompute(cv2.cvtColor(im, cv2.COLOR_RGB2GRAY), None) for _, _, im in items]
    bf = cv2.BFMatcher(cv2.NORM_L2)
    out = {}
    for i, j in itertools.combinations(range(len(items)), 2):
        (ki, di), (kj, dj) = feats[i], feats[j]
        m = bf.knnMatch(di, dj, k=2)
        good = [a for a, b in (x for x in m if len(x) == 2) if a.distance < 0.75 * b.distance]
        out[(i, j)] = verify(np.float32([ki[a.queryIdx].pt for a in good]), np.float32([kj[a.trainIdx].pt for a in good]))
    return out


def main():
    capture = sys.argv[1] if len(sys.argv) > 1 else "house_b"
    OUT.mkdir(parents=True, exist_ok=True)
    items = load(capture)
    result = {}
    which = sys.argv[2:] or ["sift", "disk_lightglue"]
    for name, fn in (("sift", sift_matches), ("disk_lightglue", lightglue_matches)):
        if name not in which:
            continue
        t = time.time()
        m = fn(items)
        within = [v for (i, j), v in m.items() if items[i][0] == items[j][0]]
        best = {}
        for (i, j), v in m.items():
            a, b = items[i][0], items[j][0]
            if a == b:
                continue
            key = "|".join(sorted((a, b)))
            if v > best.get(key, (0,))[0]:
                best[key] = (v, items[i][1], items[j][1])
        result[name] = {"seconds": round(time.time() - t, 1),
                        "within_room_median": float(np.median(within)) if within else None,
                        "cross_room_best": dict(sorted(best.items(), key=lambda x: -x[1][0]))}
        print(name, json.dumps(result[name]))
    (OUT / f"{capture}_cross_room.json").write_text(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
