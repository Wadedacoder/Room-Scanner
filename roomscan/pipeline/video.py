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
from roomscan.pipeline.backend import VIDEO_ERR, empty_plan, rooms_from_cloud
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
    sfm = run_sfm(frames_dir, names, f_guess, work, rc["video_matcher"], rc["video_seq_overlap"])
    t_sfm = time.time() - t0
    reg = sorted(sfm.c2w)
    f_px = float(sfm.K[0, 0])
    fov = float(np.degrees(2 * np.arctan(max(w, h) / 2 / f_px)))
    lens = "0.5x ultra-wide" if fov > 90 else ("1x wide" if fov > 45 else "tele")

    # keyframes for dense depth: evenly spaced over the registered frames
    n_key = min(cfg["recon"]["video_keyframes"], len(reg))
    keys = [reg[i] for i in np.linspace(0, len(reg) - 1, n_key).round().astype(int)]
    imgs = [np.asarray(Image.open(frames_dir / k).convert("RGB")) for k in keys]
    depths, s = metric_depths(cfg, imgs, f_px)

    # metric scale: DA3Metric depth vs COLMAP depth of the sparse points each keyframe observes
    ratios = []
    for k, D in zip(keys, depths):
        uv, idx = sfm.obs[k]
        if len(idx) < 10:
            continue
        T = sfm.c2w[k]
        Xc = (sfm.points[idx] - T[:3, 3]) @ T[:3, :3]  # world -> camera
        z = Xc[:, 2]
        px = np.floor(uv * s).astype(int)
        ok = (z > 1e-6) & (px[:, 0] >= 0) & (px[:, 0] < D.shape[1]) & (px[:, 1] >= 0) & (px[:, 1] < D.shape[0])
        if ok.sum() < 10:
            continue
        ratios.append(np.median(D[px[ok, 1], px[ok, 0]] / z[ok]))
    if not ratios:
        raise RuntimeError("could not fix metric scale: no keyframe sees enough COLMAP points")
    scale = float(np.median(ratios))
    scale_spread = float(np.std(ratios) / scale) if len(ratios) > 2 else 0.1

    # dense metric cloud: each keyframe's metric depth placed with COLMAP's (scaled) pose
    K = sfm.K.copy()
    K[:2] *= s
    pts, cams = [], []
    for k, D in zip(keys, depths):
        T = sfm.c2w[k].copy()
        T[:3, 3] *= scale
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
    res = rooms_from_cloud(P, cam_xyz, wall_rays(pts, cam_xyz, floor), VIDEO_ERR, VOXEL, split=True,
                           interior_mode="walls")

    plan = empty_plan("video", str(path))
    plan["rooms"] = res.rooms
    total = sum(r["floor_area"]["value"] for r in res.rooms)
    var = sum(((r["floor_area"]["hi"] - r["floor_area"]["lo"]) / 3.29) ** 2 for r in res.rooms)
    from roomscan.pipeline.backend import meas

    plan["property"] = {"footprint_area": meas(total, float(np.sqrt(var)), "m2")}
    coverage = len(reg) / len(names)
    plan["capture"] |= {"device": info.model or "unknown", "drift_correction": "off",
                        "runtime_s": round(time.time() - t0, 1),
                        "video": {"duration_s": round(info.duration_s, 1), "hdr": info.hdr, "frames": len(names),
                                  "registered": len(reg), "pieces": sfm.pieces[:5], "keyframes": len(keys),
                                  "focal_px": round(f_px, 1), "fov_deg": round(fov, 1), "lens_estimate": lens,
                                  "metric_scale_spread": round(scale_spread, 3), "sfm_s": round(t_sfm, 1),
                                  "sfm": {"fps": rc["video_fps"], "matcher": rc["video_matcher"],
                                          "overlap": rc["video_seq_overlap"]}}}
    warns = list(res.warnings)
    if coverage < 0.8:
        warns.append(f"camera tracking covered only {coverage:.0%} of the video ({len(sfm.pieces)} pieces); "
                     "rooms seen only in the untracked part are missing")
    if lens != "0.5x ultra-wide":
        warns.append(f"estimated lens {lens} ({fov:.0f}° wide); the protocol asks for 0.5x")
    if info.hdr:
        warns.append("HDR video: converted to standard range for processing (the protocol asks for HDR off)")
    plan["warnings"] = warns + ["video-tier intervals use a 4% scale term from proxy experiments; not yet calibrated",
                                "openings, adjacency, damage and scope are not implemented yet"]
    return plan, {"backend": res.debug, "scale": scale, "ratios": ratios}
