"""Video tier: phone video -> frames -> COLMAP poses (+ focal) -> metric scale from DA3Metric -> shared back-end.

v1 scope (0.5.0): uses COLMAP's largest reconstructed piece and reports how much of the video it covers. Joining
pieces with DA3 through fast turns (E5) is the next step.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
from PIL import Image

from roomscan.geometry.cloud import find_levels
from roomscan.io.video import FOV_1X_DEG, extract_frames, probe
from roomscan.pipeline.backend import VIDEO_ERR, ErrorModel, empty_plan, rooms_from_cloud
from roomscan.pipeline.photos import wall_rays
from roomscan.recon.learned import gravity_align, voxelize
from roomscan.recon.sfm import run_sfm

VOXEL = 0.03
SFM_LONG_SIDE = 960


def metric_depths(cfg, images: list[np.ndarray], f_px: float):
    """DA3Metric depth per image (one per call, memory-capped), converted with the given focal (pixels at the image's
    own resolution). Returns depth maps at the model's processing resolution and the scale factor to that size."""
    import torch

    from roomscan import models
    from roomscan.recon.da3_loader import load_da3_class

    dev = cfg["runtime"]["device"]
    dev = dev if (dev != "mps" or torch.backends.mps.is_available()) else "cpu"
    frac = float(cfg["runtime"].get("gpu_memory_fraction", 0.0) or 0.0)
    if dev == "mps" and frac > 0:
        torch.mps.set_per_process_memory_fraction(frac)
    from roomscan.recon.da3_loader import load_pretrained

    model = load_pretrained(load_da3_class(), models.get(cfg["recon"]["metric_model"]).hf_repo).to(dev).eval()
    out = []
    with torch.inference_mode():
        for img in images:
            d = model.inference([img], process_res=cfg["recon"]["image_long_side"]).depth[0]
            s = d.shape[1] / img.shape[1]
            out.append(d * (f_px * s) / 300.0)  # DA3Metric: depth = focal(px, processing res) * out / 300
    del model
    if dev == "mps":
        torch.mps.empty_cache()
    return out, out[0].shape[1] / images[0].shape[1]


def run_video(path: Path, cfg) -> tuple[dict, dict]:
    t0 = time.time()
    info = probe(path)
    work = Path(cfg["runtime"]["cache_dir"]) / "video" / Path(path).stem
    frames_dir = work / "frames"
    rc = cfg["recon"]
    names = extract_frames(path, frames_dir, rc["video_fps"], SFM_LONG_SIDE)
    w, h = Image.open(frames_dir / names[0]).size
    f_guess = (max(w, h) / 2) / np.tan(np.radians(FOV_1X_DEG / 2))
    sfm = run_sfm(frames_dir, names, f_guess, work, rc["video_matcher"], rc["video_seq_overlap"],
                  rc["video_abs_pose_min_inliers"])
    t_sfm = time.time() - t0
    f_px = float(sfm.K[0, 0])
    fov = float(np.degrees(2 * np.arctan(max(w, h) / 2 / f_px)))
    lens = "0.5x ultra-wide" if fov > 90 else ("1x wide" if fov > 45 else "tele")
    idx_of = {n: i for i, n in enumerate(names)}

    # pieces in time order; the fix loop's "before" used only the largest one (recon.video_join_pieces=false)
    pieces = sorted(sfm.all_pieces, key=lambda p: min(idx_of[n] for n in p.c2w))
    if not rc.get("video_join_pieces", True):
        pieces = [max(sfm.all_pieces, key=lambda p: len(p.c2w))]

    # keyframes spread over all pieces in proportion to their size (>= 3 each), for metric scale + dense depth
    total = sum(len(p.c2w) for p in pieces)
    keys_per_piece = []
    for p in pieces:
        reg = sorted(p.c2w, key=idx_of.get)
        k = min(len(reg), max(3, round(rc["video_keyframes"] * len(reg) / total)))
        keys_per_piece.append([reg[i] for i in np.linspace(0, len(reg) - 1, k).round().astype(int)])
    all_keys = [k for ks in keys_per_piece for k in ks]
    imgs = {k: np.asarray(Image.open(frames_dir / k).convert("RGB")) for k in all_keys}
    depth_list, s = metric_depths(cfg, [imgs[k] for k in all_keys], f_px)
    depths = dict(zip(all_keys, depth_list))

    # metric scale PER PIECE: DA3Metric depth vs that piece's own sparse COLMAP depths (pieces have unrelated scales)
    pieces_metric, piece_scales, all_ratios, piece_n = [], [], [], []
    for p, ks in zip(pieces, keys_per_piece):
        ratios = []
        for k in ks:
            uv, idx = p.obs[k]
            if len(idx) < 10:
                continue
            T = p.c2w[k]
            z = ((p.points[idx] - T[:3, 3]) @ T[:3, :3])[:, 2]
            px = np.floor(uv * s).astype(int)
            D = depths[k]
            ok = (z > 1e-6) & (px[:, 0] >= 0) & (px[:, 0] < D.shape[1]) & (px[:, 1] >= 0) & (px[:, 1] < D.shape[0])
            if ok.sum() >= 10:
                ratios.append(np.median(D[px[ok, 1], px[ok, 0]] / z[ok]))
        if not ratios:
            continue
        sc = float(np.median(ratios))
        all_ratios += [r / sc for r in ratios]
        piece_scales.append(sc)
        piece_n.append((len(ratios), len(p.c2w)))
        m = {}
        for n, T in p.c2w.items():
            T = T.copy()
            T[:3, 3] *= sc
            m[idx_of[n]] = T
        pieces_metric.append(m)
    if not pieces_metric:
        raise RuntimeError("could not fix metric scale: no keyframe sees enough COLMAP points")
    scale_spread = float(np.std(all_ratios)) if len(all_ratios) > 2 else 0.1
    # E21: the fixed 4% scale term was overconfident (per-keyframe scale spread 20-29%). Each piece's scale is a median
    # of n keyframe ratios (standard error ~1.25 * spread / sqrt(n)); combine pieces weighted by frames, then add the
    # DA3Metric systematic term (E1) in quadrature.
    w = np.array([f for _, f in piece_n], float)
    se = np.array([1.25 * scale_spread / np.sqrt(n) for n, _ in piece_n])
    scale_se = float(np.sqrt(np.sum(w * se ** 2) / w.sum()))
    err = ErrorModel(VIDEO_ERR.abs_m, float(np.hypot(VIDEO_ERR.rel, scale_se)))
    join_log = []
    if len(pieces_metric) > 1:
        from roomscan.recon.bridge import join_pieces

        world, join_log = join_pieces(pieces_metric, lambda i: np.asarray(Image.open(frames_dir / names[i])
                                                                          .convert("RGB")), cfg)
    else:
        world = pieces_metric[0]
    reg = [names[i] for i in sorted(world)]
    keys = [k for k in all_keys if idx_of[k] in world]

    # dense metric cloud: each keyframe's metric depth placed with its joined metric pose
    K = sfm.K.copy()
    K[:2] *= s
    pts, cams = [], []
    for k in keys:
        D = depths[k]
        T = world[idx_of[k]]
        v, u = np.nonzero((D > 0.2) & (D < 6.0))
        v, u = v[::2], u[::2]
        z = D[v, u]
        X = np.stack([(u + 0.5 - K[0, 2]) * z / K[0, 0], (v + 0.5 - K[1, 2]) * z / K[1, 1], z], 1)
        pts.append(X @ T[:3, :3].T + T[:3, 3])
        cams.append(T)
    cams = np.stack(cams)
    G = gravity_align(pts, cams)
    pts = [P @ G.T for P in pts]
    cam_xyz = cams[:, :3, 3] @ G.T
    P = voxelize(np.concatenate(pts), VOXEL)
    floor = find_levels(P, float(np.median(cam_xyz[:, 1]))).floor
    # door/window evidence: points seen through wall lines, plus camera viewing directions for "gap" openings
    rng = np.random.default_rng(0)
    sel = [rng.choice(len(Q), min(len(Q), 3000), replace=False) for Q in pts]
    sightlines = (np.concatenate([np.repeat(c[None, [0, 2]], len(i), 0) for c, i in zip(cam_xyz, sel)]),
                  np.concatenate([Q[i][:, [0, 2]] for Q, i in zip(pts, sel)]),
                  np.concatenate([Q[i][:, 1] for Q, i in zip(pts, sel)]))
    fwd = (cams[:, :3, 2] @ G.T)[:, [0, 2]]
    hfov = float(2 * np.arctan(depths[keys[0]].shape[1] / 2 / K[0, 0]))
    res = rooms_from_cloud(P, cam_xyz, wall_rays(pts, cam_xyz, floor), err, VOXEL, split=True,
                           interior_mode="walls", sightlines=sightlines, cam_fwd_xz=fwd, hfov=hfov)

    plan = empty_plan("video", str(path))
    plan["rooms"] = res.rooms
    plan["adjacency"] = res.adjacency or []
    total = sum(r["floor_area"]["value"] for r in res.rooms)
    var = sum(((r["floor_area"]["hi"] - r["floor_area"]["lo"]) / 3.29) ** 2 for r in res.rooms)
    from roomscan.pipeline.backend import meas

    plan["property"] = {"footprint_area": meas(total, float(np.sqrt(var)), "m2")}
    dmg_warn = []
    _video_damage(plan, res, keys, imgs, depths, K, cams, G, err, cfg, dmg_warn)
    coverage = len(reg) / len(names)
    plan["capture"] |= {"device": info.model or "unknown", "drift_correction": "off",
                        "runtime_s": round(time.time() - t0, 1),
                        "video": {"duration_s": round(info.duration_s, 1), "hdr": info.hdr, "frames": len(names),
                                  "registered": len(reg), "pieces": sfm.pieces[:5], "keyframes": len(keys),
                                  "pieces_joined": sum(1 for g in join_log if g.get("joined")) + 1,
                                  "join_log": join_log,
                                  "focal_px": round(f_px, 1), "fov_deg": round(fov, 1), "lens_estimate": lens,
                                  "metric_scale_spread": round(scale_spread, 3),
                                  "scale_rel_sigma": round(err.rel, 4), "sfm_s": round(t_sfm, 1),
                                  "sfm": {"fps": rc["video_fps"], "matcher": rc["video_matcher"],
                                          "overlap": rc["video_seq_overlap"]}}}
    warns = list(res.warnings) + dmg_warn
    if coverage < 0.8:
        warns.append(f"camera tracking covered only {coverage:.0%} of the video ({len(sfm.pieces)} pieces); "
                     "rooms seen only in the untracked part are missing")
    if lens != "0.5x ultra-wide":
        warns.append(f"estimated lens {lens} ({fov:.0f}° wide); the protocol asks for 0.5x")
    if info.hdr:
        warns.append("HDR video: converted to standard range for processing (the protocol asks for HDR off)")
    plan["warnings"] = warns + [(f"video-tier intervals use a {100 * err.rel:.1f}% scale term (4% DA3Metric + "
                                 "measured per-piece scale error); not yet calibrated against tape"),
                                "video openings use learned-depth tolerances (E15); widths are untaped"]
    return plan, {"backend": res.debug, "piece_scales": piece_scales, "join_log": join_log}


def _video_damage(plan, res, keys, imgs, depths, K, cams, G, err, cfg, warnings) -> None:
    """Per room: keyframes taken from inside it, their metric depth and gravity-aligned joined poses."""
    from roomscan.damage.stage import finalize, room_damage, views_in_room
    from roomscan.geometry import plan2d as p2

    if cfg["damage"]["backend"] == "off" or not plan["rooms"]:
        finalize(plan, [])
        return
    c2w = cams.copy()
    c2w[:, :3, :3] = G @ cams[:, :3, :3]
    c2w[:, :3, 3] = cams[:, :3, 3] @ G.T
    theta, floor = res.debug["theta"], res.debug["levels"].floor
    cam_uv = c2w[:, :3, 3][:, [0, 2]] @ p2.rot2(-theta).T
    regions = []
    for room in plan["rooms"]:
        idx = views_in_room(room, cam_uv, min(8, cfg["damage"]["keyframes_per_room"]))
        if not idx:
            continue
        regions += room_damage(room, [imgs[keys[i]] for i in idx], np.stack([depths[keys[i]] for i in idx]),
                               np.stack([K] * len(idx)), c2w[idx], floor, theta, err, cfg,
                               Path(cfg["runtime"]["cache_dir"]), warnings, [keys[i] for i in idx])
    finalize(plan, regions)
