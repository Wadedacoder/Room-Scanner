"""LiDAR tier: Stray Scanner capture -> fused cloud -> shared back-end -> plan dict (schema/plan.schema.json)."""

from __future__ import annotations

import time

import numpy as np

from roomscan.geometry.cloud import find_levels, fuse, wall_band_rays
from roomscan.io.stray import StrayCapture
from roomscan.pipeline.backend import LIDAR_ERR, empty_plan, rooms_from_cloud


def run_lidar(path, cfg) -> tuple[dict, dict]:
    t0 = time.time()
    timing = {}
    lc = cfg["lidar"]
    cap = StrayCapture.open(path)
    drift = None
    if cfg["drift"]["method"] == "posegraph":
        from roomscan.stitch.drift import CorrectedCapture, correct_drift

        drift = correct_drift(cap)
        cap = CorrectedCapture(cap, drift.corrections)
        timing["drift"] = time.time() - t0
    P = fuse(cap, lc["frame_stride"], lc["min_confidence"], lc["max_depth_m"], lc["voxel_m"])
    timing["fuse"] = time.time() - t0
    cam = cap.poses[:, :3, 3]
    floor = find_levels(P, float(np.median(cam[:, 1]))).floor
    # carving only needs coverage, not density: ~150 rays from every 3rd fused frame keeps it to seconds
    rays = wall_band_rays(cap, lc["frame_stride"] * 3, floor, max_depth=lc["max_depth_m"], per_frame=150)
    res = rooms_from_cloud(P, cam, rays, LIDAR_ERR, lc["voxel_m"], split=True)
    timing["geometry"] = time.time() - t0

    plan = empty_plan("lidar", str(path))
    plan["capture"]["drift_correction"] = cfg["drift"]["method"]
    if drift is not None:
        plan["capture"]["drift"] = {"fragments": len(drift.fragments), "revisits_matched": len(drift.loops),
                                    "revisits_rejected": drift.rejected,
                                    "max_shift_m": round(drift.max_shift_m, 3),
                                    "max_yaw_deg": round(drift.max_yaw_deg, 2),
                                    "revisit_misalignment_cm": [round(drift.residual_before_cm, 1),
                                                                round(drift.residual_after_cm, 1)]}
    plan["rooms"] = res.rooms
    plan["property"] = {"footprint_area": res.footprint}
    plan["warnings"] = res.warnings + [
        "openings, adjacency, damage and scope are not implemented yet",
        "intervals are propagated, not yet calibrated against ground truth"]
    timing["total"] = time.time() - t0
    plan["capture"]["runtime_s"] = round(timing["total"], 1)
    plan["capture"]["timing_s"] = {k: round(v, 1) for k, v in timing.items()}
    res.debug["cloud"] = P
    return plan, res.debug
