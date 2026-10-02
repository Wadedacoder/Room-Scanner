"""LiDAR tier, capture -> plan dict (schema/plan.schema.json).

v0.1 scope: rooms, walls, floor area, ceiling height and property footprint, each with an interval.
Not yet: openings, adjacency, damage, scope. Those fields are emitted empty and listed in `warnings`.
Intervals are propagated from fit statistics plus a fixed LiDAR systematic term, and are NOT yet calibrated.
"""

from __future__ import annotations

import time

import numpy as np
from matplotlib.path import Path as MplPath

from roomscan.geometry import plan2d as p2
from roomscan.geometry.cloud import find_levels, fuse, wall_band_rays
from roomscan.io.stray import StrayCapture

Z90 = 1.645  # two-sided 90% normal quantile
LIDAR_SYS_M = 0.01  # per-surface systematic (ARKit depth bias), until the benchmark calibrates it


def meas(value: float, sigma: float, unit: str) -> dict:
    return {"value": round(value, 4), "lo": round(value - Z90 * sigma, 4), "hi": round(value + Z90 * sigma, 4),
            "unit": unit, "coverage": 0.9}


def room_ceiling(P_room_y: np.ndarray, floor: float, room_area: float, res: float = 0.01):
    """Ceiling = densest level 1.9-4 m above the floor, accepted only if it is a real surface over the room.

    Guard against confident garbage: a scan that never looked up still has shelf and cabinet tops up there.
    """
    h = P_room_y - floor
    up = h[(h > 1.9) & (h < 4.0)]
    if len(up) < 500:
        return None
    hist, e = np.histogram(up, np.arange(1.9, 4.0 + res, res))
    lev = e[np.argmax(hist)] + res / 2
    layer = up[np.abs(up - lev) < 0.02]
    # ~1 point per 2 cm voxel: the layer must cover >= 40% of the floor area
    if len(layer) * 0.02 * 0.02 < 0.4 * room_area:
        return None
    return float(np.median(layer)), float(1.2533 * layer.std() / np.sqrt(len(layer)))


def run_lidar(path, cfg) -> tuple[dict, dict]:
    t0 = time.time()
    timing = {}
    lc = cfg["lidar"]
    cap = StrayCapture.open(path)
    P = fuse(cap, lc["frame_stride"], lc["min_confidence"], lc["max_depth_m"], lc["voxel_m"])
    timing["fuse"] = time.time() - t0
    cam = cap.poses[:, :3, 3]
    lev = find_levels(P, float(np.median(cam[:, 1])))
    y = P[:, 1] - lev.floor
    wall_pts = P[(y > 0.3) & (y < 2.0)]
    theta = p2.dominant_angle(wall_pts[:, [0, 2]])
    # carving only needs coverage, not density: ~150 rays from every 3rd fused frame keeps it to seconds
    rays = wall_band_rays(cap, lc["frame_stride"] * 3, lev.floor, max_depth=lc["max_depth_m"], per_frame=150)
    maps = p2.build_maps(P, lev.floor, theta, cam[:, [0, 2]], rays)
    timing["maps"] = time.time() - t0
    masks = p2.split_rooms(maps.interior, maps.grid.res, maps.traj_px)
    R = p2.rot2(-theta)
    wall_uv = wall_pts[:, [0, 2]] @ R.T
    P_uv = P[:, [0, 2]] @ R.T

    rooms, warnings = [], []
    total_area, total_var = 0.0, 0.0
    for k, mask in enumerate(masks):
        g = p2.room_geometry(mask, maps.grid, wall_uv)
        if len(g.polygon) < 3:
            warnings.append(f"room {k}: could not trace a closed outline; skipped")
            continue
        rid = f"r{k + 1}"
        poly = g.polygon
        walls = []
        perim, perim_var = 0.0, 0.0
        n = len(poly)
        for i in range(n):
            a, b = poly[i], poly[(i + 1) % n]
            L = float(np.hypot(*(b - a)))
            # each corner is the intersection of two fitted walls; a wall's length depends on its 2 neighbours
            f_prev, f_next = g.walls[i % len(g.walls)], g.walls[(i + 2) % len(g.walls)]
            sig = float(np.sqrt(f_prev.sigma ** 2 + f_next.sigma ** 2 + 2 * LIDAR_SYS_M ** 2))
            walls.append({"id": f"{rid}.w{i + 1}", "start": a.round(4).tolist(), "end": b.round(4).tolist(),
                          "length": meas(L, sig, "m")})
            perim += L
            perim_var += sig ** 2
        area = p2.polygon_area(poly)
        area_sig = float(np.sqrt(perim_var) * np.sqrt(area) / 2 + perim * LIDAR_SYS_M / 2)
        inside = MplPath(poly).contains_points(P_uv)
        ceil = room_ceiling(P[inside, 1], lev.floor, area)
        room = {"id": rid, "label": f"room {k + 1}", "polygon": poly.round(4).tolist(), "walls": walls,
                "openings": [], "floor_area": meas(area, area_sig, "m2"),
                "perimeter": meas(perim, float(np.sqrt(perim_var)), "m")}
        if ceil is None:
            room["ceiling_height"] = None
            warnings.append(f"{rid}: ceiling not observed well enough to measure; no value reported")
        else:
            c, cs = ceil
            room["ceiling_height"] = meas(c, float(np.sqrt(cs ** 2 + lev.floor_sigma ** 2 + 2 * LIDAR_SYS_M ** 2)), "m")
        rooms.append(room)
        total_area += area
        total_var += area_sig ** 2

    warnings += ["openings, adjacency, damage and scope are not implemented in pipeline v0.1",
                 "intervals are propagated, not yet calibrated against ground truth"]
    plan = {
        "schema_version": "0.1",
        "capture": {"tier": "lidar", "source": str(path), "pipeline_version": "0.1",
                    "drift_correction": "off", "runtime_s": round(time.time() - t0, 1)},
        "rooms": rooms, "adjacency": [],
        "property": {"footprint_area": meas(total_area, float(np.sqrt(total_var)), "m2")},
        "damage": [], "concealed_flags": [], "scope": [], "warnings": warnings,
    }
    timing["total"] = time.time() - t0
    plan["capture"]["timing_s"] = {k: round(v, 1) for k, v in timing.items()}
    debug = {"theta": theta, "maps": maps, "masks": masks, "levels": lev}
    return plan, debug
