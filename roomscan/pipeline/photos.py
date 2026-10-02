"""Photo tier: folder of per-room photo folders -> per-room metric reconstruction -> shared back-end -> plan dict."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from roomscan.geometry.cloud import find_levels
from roomscan.io.photos import load_capture
from roomscan.pipeline.backend import PHOTO_ERR, empty_plan, meas, rooms_from_cloud
from roomscan.recon.learned import LearnedRecon, backproject, gravity_align, voxelize

VOXEL = 0.03  # photo clouds are sparser and noisier than LiDAR


def wall_rays(pts: list[np.ndarray], cams: np.ndarray, floor_y: float, per_view: int = 3000, seed: int = 0):
    rng = np.random.default_rng(seed)
    starts, ends = [], []
    for P, c in zip(pts, cams):
        h = P[:, 1] - floor_y
        P = P[(h > 0.3) & (h < 1.8)]
        if len(P) > per_view:
            P = P[rng.choice(len(P), per_view, replace=False)]
        ends.append(P[:, [0, 2]])
        starts.append(np.repeat(c[None, [0, 2]], len(P), 0))
    return np.concatenate(starts), np.concatenate(ends)


def reconstruct_room(model: LearnedRecon, photos):
    rec = model.reconstruct([p.image for p in photos], np.stack([p.K for p in photos]))
    pts, cams = backproject(rec)
    G = gravity_align(pts, rec.c2w)
    pts = [P @ G.T for P in pts]
    cams = cams @ G.T
    return rec, pts, cams, G


def run_photos(path: Path, cfg) -> tuple[dict, dict]:
    t0 = time.time()
    capture = load_capture(Path(path))
    model = LearnedRecon(cfg)
    plan = empty_plan("photos", str(path))
    warnings, debug = [], {}
    x_cursor, total_area, total_var = 0.0, 0.0, 0.0
    for name, photos in capture.items():
        rec, pts, cams, Gr = reconstruct_room(model, photos)
        P = voxelize(np.concatenate(pts), VOXEL)
        floor = find_levels(P, float(np.median(cams[:, 1]))).floor
        rays = wall_rays(pts, cams, floor)
        # doorways: views INTO the next room are low-confidence depth, which the 30th-percentile filter removes
        # (E15: max distance 2.7 m with it, 6.6 m without), so openings get their own lightly filtered points
        far_pts, _ = backproject(rec, conf_pct=5)
        far_pts = [Q @ Gr.T for Q in far_pts]
        sl_rng = np.random.default_rng(0)
        sl = [(np.repeat(c[None, [0, 2]], min(len(Q), 20000), 0), Q[sel][:, [0, 2]], Q[sel][:, 1])
              for c, Q in zip(cams, far_pts) for sel in [sl_rng.choice(len(Q), min(len(Q), 20000), replace=False)]]
        sightlines = tuple(np.concatenate([x[i] for x in sl]) for i in range(3))
        fwd = (rec.c2w[:, :3, 2] @ Gr.T)[:, [0, 2]]  # camera viewing directions, gravity frame
        hfov = float(2 * np.arctan(rec.depth.shape[2] / 2 / rec.K[0, 0, 0]))
        res = rooms_from_cloud(P, cams, rays, PHOTO_ERR, VOXEL, split=False, id_prefix=name, label=name,
                               sightlines=sightlines, cam_fwd_xz=fwd, hfov=hfov,
                               interior_mode=cfg["recon"]["photo_outline"])  # sparse views can't carve (E6)
        warnings += res.warnings
        debug[name] = {"scale": rec.scale, "focal_pred_err": rec.focal_pred_err, "n_photos": len(photos),
                       "focal_source": photos[0].focal_source, "backend": res.debug}
        for room in res.rooms:
            # rooms are not stitched yet: shift each into its own slot along +x so the drawing doesn't overlap
            poly = np.array(room["polygon"])
            shift = x_cursor - poly[:, 0].min()
            room["polygon"] = (poly + [shift, 0]).round(4).tolist()
            for w in room["walls"]:
                w["start"] = [round(w["start"][0] + shift, 4), w["start"][1]]
                w["end"] = [round(w["end"][0] + shift, 4), w["end"][1]]
            x_cursor = poly[:, 0].max() + shift + 1.0
            plan["rooms"].append(room)
            total_area += room["floor_area"]["value"]
            total_var += ((room["floor_area"]["hi"] - room["floor_area"]["lo"]) / (2 * 1.645)) ** 2
    plan["property"] = {"footprint_area": meas(total_area, float(np.sqrt(total_var)), "m2")}
    plan["capture"]["runtime_s"] = round(time.time() - t0, 1)
    plan["capture"]["models"] = {"geometry": cfg["recon"]["model"], "metric": cfg["recon"]["metric_model"]}
    plan["warnings"] = warnings + [
        "rooms are NOT stitched yet (doorway matching not implemented): they are drawn side by side and adjacency "
        "is empty",
        "photo-tier intervals use a 5% scale term from proxy-data experiments E1-E2; not yet calibrated on real photos",
        "openings, damage and scope are not implemented yet"]
    return plan, debug
