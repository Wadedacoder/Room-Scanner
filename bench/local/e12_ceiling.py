"""E12: why is the photo-tier ceiling wrong? Diagnosis + ablation on house_b (study has tape: ceiling 274.3 cm).

Step 1 (diagnose): reconstruct each room once (cached in runs/e12/), then report the height profile of the points,
the detected floor and the camera height above it, and render the photos with pixels coloured by 3D height so we can
see which surfaces form each layer.
Step 2 (ablate): ceiling estimators, one change at a time (see VARIANTS), scored on the study.

Usage: python bench/local/e12_ceiling.py [diagnose|ablate]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from roomscan.config import load_config
from roomscan.geometry.cloud import find_levels
from roomscan.io.photos import load_room
from roomscan.recon.learned import Recon, backproject, gravity_align, voxelize

OUT = ROOT / "runs/e12"
ROOMS = ["study", "kitchen", "living", "bedroom", "bathroom", "store"]
TAPE_CEIL = {"study": 2.743}


def get_recon(room: str):
    cache = OUT / f"{room}.npz"
    photos = load_room(ROOT / "data/raw/photos/house_b" / room)
    if cache.exists():
        z = np.load(cache)
        rec = Recon(z["c2w"], z["depth"], z["conf"], z["K"], float(z["scale"]), float(z["fperr"]))
    else:
        from roomscan.recon.learned import LearnedRecon

        rec = LearnedRecon(load_config("lite")).reconstruct([p.image for p in photos], np.stack([p.K for p in photos]))
        np.savez_compressed(cache, c2w=rec.c2w, depth=rec.depth, conf=rec.conf, K=rec.K, scale=rec.scale,
                            fperr=rec.focal_pred_err)
    return rec, photos


def per_pixel_world(rec: Recon, G: np.ndarray, conf_pct: float = 30):
    """World points for every pixel of every view (NaN where invalid), gravity-aligned, plus normals' up component."""
    thr = np.percentile(rec.conf, conf_pct)
    out = []
    for i in range(len(rec.depth)):
        D = rec.depth[i]
        h, w = D.shape
        v, u = np.mgrid[0:h, 0:w]
        K = rec.K[i]
        X = np.stack([(u + 0.5 - K[0, 2]) * D / K[0, 0], (v + 0.5 - K[1, 2]) * D / K[1, 1], D], -1)
        P = X @ rec.c2w[i, :3, :3].T + rec.c2w[i, :3, 3]
        P = P @ G.T
        P[(D <= 0.2) | (D > 8) | (rec.conf[i] < thr)] = np.nan
        # surface normal from neighbouring points; its y component: -1 floor (faces up), +1 ceiling (faces down)
        dx = np.gradient(P, axis=1)
        dy = np.gradient(P, axis=0)
        n = np.cross(dx, dy)
        n /= np.linalg.norm(n, axis=-1, keepdims=True) + 1e-12
        cam = (rec.c2w[i, :3, 3] @ G.T)
        to_cam = cam - P
        n *= np.sign((n * to_cam).sum(-1, keepdims=True))  # orient normals toward the camera
        out.append((P, n[..., 1]))
    return out


def diagnose():
    import cv2

    OUT.mkdir(parents=True, exist_ok=True)
    report = {}
    for room in ROOMS:
        rec, photos = get_recon(room)
        pts, cams = backproject(rec)
        G = gravity_align(pts, rec.c2w)
        P = np.concatenate(pts) @ G.T
        cam_y = rec.c2w[:, :3, 3] @ G.T
        lev = find_levels(voxelize(P, 0.03), float(np.median(cam_y[:, 1])))
        h = P[:, 1] - lev.floor
        hist, e = np.histogram(h, np.arange(-0.5, 4.0, 0.05))
        peaks = sorted([(int(hist[i]), round(float(e[i]), 2)) for i in range(len(hist))], reverse=True)[:6]
        report[room] = {"camera_height_above_floor_m": np.round(cam_y[:, 1] - lev.floor, 2).tolist(),
                        "floor_y": round(lev.floor, 3), "top_height_bins_m(count,height)": peaks,
                        "share_points_above_1p9m": round(float((h > 1.9).mean()), 3), "scale_k": rec.scale}
        print(room, json.dumps(report[room]))
        # colour pixels by height: red 1.9-2.3 m, green 2.5-3.0 m, blue < 0.2 m (floor)
        pp = per_pixel_world(rec, G)
        tiles = []
        for i, (Pi, ny) in enumerate(pp):
            img = cv2.resize(photos[i].image, (Pi.shape[1], Pi.shape[0]))
            hh = Pi[..., 1] - lev.floor
            ov = img.copy()
            ov[(hh > 1.9) & (hh < 2.3)] = [230, 40, 40]
            ov[(hh >= 2.5) & (hh < 3.0)] = [40, 200, 60]
            ov[hh < 0.2] = [40, 90, 230]
            tiles.append(cv2.addWeighted(img, 0.45, ov, 0.55, 0))
        sheet = np.concatenate([cv2.resize(t, (378, 284)) for t in tiles], axis=1)
        cv2.imwrite(str(OUT / f"{room}_heights.jpg"), cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR))
    (OUT / "diagnose.json").write_text(json.dumps(report, indent=1))


def ablate():
    """Ceiling = (ceiling level - floor level). Variants change only how the floor is chosen; ceiling level is the same."""
    rows = []
    for room in ROOMS:
        rec, _ = get_recon(room)
        pts, _ = backproject(rec)
        G = gravity_align(pts, rec.c2w)
        P = voxelize(np.concatenate(pts) @ G.T, 0.03)
        cy = float(np.median((rec.c2w[:, :3, 3] @ G.T)[:, 1]))
        row = {"room": room}
        for name, kw in (("A_densest(0.5.1)", {"floor_rule": "densest"}), ("B_lowest_40pct", {"floor_rule": "lowest"}),
                         ("C_lowest_20pct", {"floor_rule": "lowest", "support": 0.2}),
                         ("D_lowest_60pct", {"floor_rule": "lowest", "support": 0.6})):
            lv = find_levels(P, cy, **kw)
            row[name] = {"camera_h": round(cy - lv.floor, 2),
                         "ceiling_m": None if lv.ceiling is None else round(lv.ceiling - lv.floor, 3)}
        if room in TAPE_CEIL:
            row["tape_ceiling_m"] = TAPE_CEIL[room]
        rows.append(row)
        print(json.dumps(row))
    (OUT / "ablate.json").write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    {"diagnose": diagnose, "ablate": ablate}.get(sys.argv[1] if len(sys.argv) > 1 else "diagnose", diagnose)()
