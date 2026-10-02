"""Shared geometry back-end: gravity-aligned point cloud + camera rays -> rooms with dimensioned walls.

Every tier (LiDAR, video, photos) ends here; the tiers differ only in how they produce the cloud and in their error
model, passed as `ErrorModel`. Intervals: sigma^2 = fit^2 + abs^2 + (rel * value)^2, reported at 90%.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from matplotlib.path import Path as MplPath

from roomscan.geometry import plan2d as p2
from roomscan.geometry.cloud import find_levels

Z90 = 1.645  # two-sided 90% normal quantile


@dataclass(frozen=True)
class ErrorModel:
    abs_m: float  # per-surface systematic, metres (LiDAR depth bias)
    rel: float  # relative scale uncertainty (monocular metric scale); 0 for LiDAR


LIDAR_ERR = ErrorModel(abs_m=0.01, rel=0.0)
# Photo / video scale: E1-E2 measured -0.7% .. -9.3% with the true focal (mean about -4%); sigma 5% covers that
# spread at 90% without correcting a bias measured on one device only. Recalibrate on the tape benchmark.
VIDEO_ERR = ErrorModel(abs_m=0.02, rel=0.04)
PHOTO_ERR = ErrorModel(abs_m=0.03, rel=0.05)


def meas(value: float, sigma: float, unit: str) -> dict:
    return {"value": round(value, 4), "lo": round(value - Z90 * sigma, 4), "hi": round(value + Z90 * sigma, 4),
            "unit": unit, "coverage": 0.9}


def room_ceiling(P_room_y: np.ndarray, floor: float, room_area: float, voxel: float, res: float = 0.01):
    """Ceiling = densest level 1.9-4 m above the floor, accepted only if it is a real surface over the room.

    Guard against confident garbage: a scan that never looked up still has shelf and cabinet tops up there.
    """
    h = P_room_y - floor
    up = h[(h > 1.9) & (h < 4.0)]
    if len(up) < 200:
        return None
    hist, e = np.histogram(up, np.arange(1.9, 4.0 + res, res))
    lev = e[np.argmax(hist)] + res / 2
    layer = up[np.abs(up - lev) < 0.02]
    # ~1 point per voxel: the layer must cover >= 40% of the floor area
    if len(layer) * voxel * voxel < 0.4 * room_area:
        return None
    return float(np.median(layer)), float(1.2533 * layer.std() / np.sqrt(len(layer)))


@dataclass
class BackendResult:
    rooms: list[dict]
    warnings: list[str]
    footprint: dict
    debug: dict


def rooms_from_cloud(P: np.ndarray, cam_xyz: np.ndarray, rays, err: ErrorModel, voxel: float,
                     split: bool = True, id_prefix: str = "r", label: str | None = None,
                     interior_mode: str = "carve") -> BackendResult:
    """P: (N,3) world points, +Y up, metres. cam_xyz: camera centres. rays: (start_xz, end_xz) for carving.

    split=False treats everything as ONE room (photo tier: each folder is one room by protocol).
    """
    lev = find_levels(P, float(np.median(cam_xyz[:, 1])))
    y = P[:, 1] - lev.floor
    wall_pts = P[(y > 0.3) & (y < 2.0)]
    theta = p2.dominant_angle(wall_pts[:, [0, 2]])
    maps = p2.build_maps(P, lev.floor, theta, cam_xyz[:, [0, 2]], rays, interior_mode=interior_mode)
    if split:
        masks = p2.split_rooms(maps.interior, maps.grid.res, maps.traj_px)
    else:
        masks = [p2.largest_component(maps.interior)] if maps.interior.any() else []
    R = p2.rot2(-theta)
    wall_uv = wall_pts[:, [0, 2]] @ R.T
    P_uv = P[:, [0, 2]] @ R.T

    rooms, warnings = [], []
    total_area, total_var = 0.0, 0.0
    for k, mask in enumerate(masks):
        others = np.zeros_like(mask)
        for j, m in enumerate(masks):
            if j != k:
                others |= m
        g = p2.room_geometry(mask, maps.grid, wall_uv, others)
        rid = f"{id_prefix}{k + 1}" if split or id_prefix == "r" else id_prefix
        if len(g.polygon) < 3:
            warnings.append(f"{rid}: could not trace a closed outline; skipped")
            continue
        poly = g.polygon
        walls, perim, perim_var = [], 0.0, 0.0
        n = len(poly)
        for i in range(n):
            a, b = poly[i], poly[(i + 1) % n]
            L = float(np.hypot(*(b - a)))
            # each corner is the intersection of two fitted walls; a wall's length depends on its 2 neighbours
            f_prev, f_next = g.walls[i % len(g.walls)], g.walls[(i + 2) % len(g.walls)]
            sig = float(np.sqrt(f_prev.sigma ** 2 + f_next.sigma ** 2 + 2 * err.abs_m ** 2 + (err.rel * L) ** 2))
            walls.append({"id": f"{rid}.w{i + 1}", "start": a.round(4).tolist(), "end": b.round(4).tolist(),
                          "length": meas(L, sig, "m")})
            perim += L
            perim_var += sig ** 2
        area = p2.polygon_area(poly)
        area_sig = float(np.sqrt(perim_var) * np.sqrt(area) / 2 + perim * err.abs_m / 2 + 2 * err.rel * area)
        inside = MplPath(poly).contains_points(P_uv)
        ceil = room_ceiling(P[inside, 1], lev.floor, area, voxel)
        room = {"id": rid, "label": label or f"room {k + 1}", "polygon": poly.round(4).tolist(), "walls": walls,
                "openings": [], "floor_area": meas(area, area_sig, "m2"),
                "perimeter": meas(perim, float(np.sqrt(perim_var)), "m")}
        if ceil is None:
            room["ceiling_height"] = None
            warnings.append(f"{rid}: ceiling not observed well enough to measure; no value reported")
        else:
            c, cs = ceil
            room["ceiling_height"] = meas(c, float(np.sqrt(cs ** 2 + lev.floor_sigma ** 2 + 2 * err.abs_m ** 2
                                                            + (err.rel * c) ** 2)), "m")
        rooms.append(room)
        total_area += area
        total_var += area_sig ** 2
    footprint = meas(total_area, float(np.sqrt(total_var)), "m2")
    return BackendResult(rooms, warnings, footprint,
                         {"theta": theta, "maps": maps, "masks": masks, "levels": lev})


def empty_plan(tier: str, source: str) -> dict:
    return {"schema_version": "0.1",
            "capture": {"tier": tier, "source": source, "pipeline_version": "0.2"},
            "rooms": [], "adjacency": [], "property": {}, "damage": [], "concealed_flags": [], "scope": [],
            "warnings": []}
