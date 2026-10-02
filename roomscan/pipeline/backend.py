"""Shared geometry back-end: gravity-aligned point cloud + camera rays -> rooms with dimensioned walls.

Every tier (LiDAR, video, photos) ends here; the tiers differ only in how they produce the cloud and in their error
model, passed as `ErrorModel`. Intervals: sigma^2 = fit^2 + abs^2 + (rel * value)^2, reported at 90%.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from matplotlib.path import Path as MplPath

from roomscan import __version__
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
                     interior_mode: str = "carve", sightlines=None) -> BackendResult:
    """P: (N,3) world points, +Y up, metres. cam_xyz: camera centres. rays: (start_xz, end_xz) for carving.

    split=False treats everything as ONE room (photo tier: each folder is one room by protocol).
    sightlines: optional (cam_xz, pt_xz, pt_y) samples (world frame) used to find doors/windows: points the camera saw
    through a wall line.
    """
    lev = find_levels(P, float(np.median(cam_xyz[:, 1])))
    # Plausibility (E12): a handheld phone is 1.0-1.95 m above the floor. Outside that, the "floor" is a furniture top
    # or clutter (the store room: 0.77 m), so ceiling heights measured from it would be confident garbage.
    cam_h = float(np.median(cam_xyz[:, 1]) - lev.floor)
    floor_ok = 1.0 <= cam_h <= 1.95
    y = P[:, 1] - lev.floor
    wall_pts = P[(y > 0.3) & (y < 2.0)]
    theta = p2.dominant_angle(wall_pts[:, [0, 2]])
    maps = p2.build_maps(P, lev.floor, theta, cam_xyz[:, [0, 2]], rays,
                         interior_mode="walls" if interior_mode == "box" else interior_mode)
    if split:
        masks = p2.split_rooms(maps.interior, maps.grid.res, maps.traj_px)
    else:
        masks = [p2.largest_component(maps.interior)] if maps.interior.any() else []
    R = p2.rot2(-theta)
    wall_uv = wall_pts[:, [0, 2]] @ R.T
    P_uv = P[:, [0, 2]] @ R.T

    rooms, warnings = [], []
    total_area, total_var = 0.0, 0.0
    if interior_mode == "box" and not split:
        # sparse photos (E13): outermost long wall lines -> rectangle; falls back to the wall-enclosed outline
        band = (y > 0.2) & (y < 1.8)
        box = p2.manhattan_box(P[band][:, [0, 2]] @ R.T, y[band], cam_xyz[:, [0, 2]] @ R.T)
        if box is not None:
            u0, u1, v0, v1, _ = box
            g = maps.grid
            m = np.zeros(g.shape, bool)
            a, b = g.to_px(np.array([[u0, v0], [u1, v1]]))
            m[max(a[1], 0):b[1] + 1, max(a[0], 0):b[0] + 1] = True
            masks = [m]
        else:
            warnings.append(f"{id_prefix}: no box outline found (too few long wall lines); using the wall flood")
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
        ceil = room_ceiling(P[inside, 1], lev.floor, area, voxel) if floor_ok else None
        if not floor_ok:
            warnings.append(f"{rid}: floor not found reliably (camera {cam_h:.2f} m above it; a handheld phone is "
                            f"1.0-1.95 m), so no ceiling height is reported")
        openings = []
        if sightlines is not None:
            from roomscan.openings.detect import detect_openings

            c_uv, p_uv, p_y = sightlines[0] @ R.T, sightlines[1] @ R.T, sightlines[2] - lev.floor
            tol = (0.08, 0.25) if err.rel == 0 else (0.20, 0.40)  # LiDAR vs learned-depth walls (E15)
            for j, o in enumerate(detect_openings(poly, p_uv, p_y, c_uv, wall_tol=tol[0], beyond_min=tol[1])):
                w_sig = float(np.sqrt(2 * (0.025 + err.abs_m) ** 2 + (err.rel * o.width) ** 2))  # +-half a slot per jamb
                openings.append({"id": f"{rid}.o{j + 1}", "wall_id": f"{rid}.w{o.wall + 1}",
                                 "type": o.kind, "width": meas(o.width, w_sig, "m"),
                                 "offset_along_wall": meas(o.offset, w_sig, "m"),
                                 "sill_height": meas(max(o.sill, 0.0), 0.05, "m"), "connects_to": None,
                                 "detection_confidence": round(min(1.0, o.support / 200), 2),
                                 "centre_plan": list(o.centre_uv)})
        room = {"id": rid, "label": label or f"room {k + 1}", "polygon": poly.round(4).tolist(), "walls": walls,
                "openings": openings, "floor_area": meas(area, area_sig, "m2"),
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
                         {"theta": theta, "maps": maps, "masks": masks, "levels": lev, "camera_height_m": cam_h})


def empty_plan(tier: str, source: str) -> dict:
    return {"schema_version": "0.1",
            "capture": {"tier": tier, "source": source, "pipeline_version": __version__},
            "rooms": [], "adjacency": [], "property": {}, "damage": [], "concealed_flags": [], "scope": [],
            "warnings": []}
