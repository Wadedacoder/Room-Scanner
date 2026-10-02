"""E6: why does the photo tier under-measure rooms? Ablation per room on the proxy sets (c00a170fe1, 8 sweep photos).

  A  pipeline as shipped: learned depth + learned poses (DA3-Base, lite profile), metric from DA3Metric + true focal
  B  learned depth, TRUE poses (ARKit, upright camera)       -> isolates pose error
  C  LiDAR depth, TRUE poses, same 8 frames                  -> upper bound: what these 8 views can show at all
Reference: the LiDAR tier's room area from the full 37 s walk (room mapping by camera positions, see the build log).

Usage: python bench/local/e6_photo_tier_ablation.py [--set sweep] [--n 8]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from roomscan.config import load_config  # noqa: E402
from roomscan.geometry.cloud import find_levels  # noqa: E402
from roomscan.pipeline.backend import PHOTO_ERR, rooms_from_cloud  # noqa: E402
from roomscan.pipeline.photos import VOXEL, wall_rays  # noqa: E402
from roomscan.recon.learned import Recon, backproject, gravity_align, voxelize  # noqa: E402

LIDAR_AREA = {"living": 7.41, "corridor": 8.35, "bathroom": 5.62}  # LiDAR tier, runs/tier_compare (r1, r2, r3)
OUT = ROOT / "runs/e6"


MODE = "carve"


def area_from(pts, cams, align: bool, c2w=None) -> float:
    if align:
        G = gravity_align(pts, c2w)
        pts, cams = [P @ G.T for P in pts], cams @ G.T
    P = voxelize(np.concatenate(pts), VOXEL)
    floor = find_levels(P, float(np.median(cams[:, 1]))).floor
    res = rooms_from_cloud(P, cams, wall_rays(pts, cams, floor), PHOTO_ERR, VOXEL, split=False, id_prefix="x",
                           interior_mode=MODE)
    return res.rooms[0]["floor_area"]["value"] if res.rooms else 0.0


def gt_depth_points(g, conf_min=2):
    pts = []
    for i in range(len(g["depth"])):
        D, C = g["depth"][i], g["conf"][i]
        K = g["K"][i].copy()
        K[:2] *= D.shape[1] / 1440.0  # bundle K is at 1440-wide upright image resolution
        v, u = np.nonzero((C >= conf_min) & (D > 0.1) & (D < 6))
        z = D[v, u]
        x, y = (u + 0.5 - K[0, 2]) * z / K[0, 0], (v + 0.5 - K[1, 2]) * z / K[1, 1]
        T = g["T_world_cam"][i]
        pts.append(np.stack([x, y, z], 1) @ T[:3, :3].T + T[:3, 3])
    return pts


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", default="sweep")
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--interior", default="carve", choices=["carve", "walls", "box"])
    args = ap.parse_args()
    global MODE
    MODE = args.interior
    OUT.mkdir(parents=True, exist_ok=True)
    cfg = load_config("lite")
    model = None
    rows = []
    for room in ("living", "corridor", "bathroom"):
        d = ROOT / f"data/derived/eval_bundle/photos_{args.set}_{room}_n{args.n}"
        g = np.load(d / "gt.npz")
        cache = OUT / f"{args.set}_{room}_n{args.n}.npz"
        if cache.exists():
            z = np.load(cache)
            rec = Recon(z["c2w"], z["depth"], z["conf"], z["K"], float(z["scale"]), float(z["fperr"]))
        else:
            if model is None:
                from roomscan.recon.learned import LearnedRecon

                model = LearnedRecon(cfg)
            imgs = [np.asarray(Image.open(d / "images" / n).convert("RGB")) for n in g["names"]]
            rec = model.reconstruct(imgs, g["K"].copy())
            np.savez_compressed(cache, c2w=rec.c2w, depth=rec.depth, conf=rec.conf, K=rec.K, scale=rec.scale,
                                fperr=rec.focal_pred_err)
        # A: as shipped
        pts, cams = backproject(rec)
        a = area_from(pts, cams, align=True, c2w=rec.c2w)
        # B: learned depth, true poses. Same back-projection, but with the ARKit cameras (already gravity-aligned).
        recB = Recon(g["T_world_cam"].copy(), rec.depth, rec.conf, rec.K, rec.scale, rec.focal_pred_err)
        ptsB, camsB = backproject(recB)
        b = area_from(ptsB, camsB, align=False)
        # C: LiDAR depth, true poses
        ptsC = gt_depth_points(g)
        c = area_from(ptsC, g["T_world_cam"][:, :3, 3], align=False)
        ref = LIDAR_AREA[room]
        row = {"room": room, "lidar_full_walk_m2": ref, "A_shipped": round(a, 2), "B_true_poses": round(b, 2),
               "C_lidar_depth_true_poses": round(c, 2), "scale_k": round(rec.scale, 4),
               "focal_pred_err": round(rec.focal_pred_err, 3)}
        rows.append(row)
        print(json.dumps(row), flush=True)
    (OUT / f"results_{args.set}_n{args.n}_{args.interior}.json").write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
