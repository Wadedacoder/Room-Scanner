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


def second_ceiling(P_room_y: np.ndarray, floor: float, room_area: float, voxel: float, main: float,
                   res: float = 0.05, min_frac: float = 0.25) -> float | None:
    """Another flat level >= 15 cm from the main ceiling covering >= min_frac of the room (E27: a 27 m2 LiDAR room
    with most of its ceiling at 2.40 m and ~8 m2 at 3.05 m: a lowered section, or the outline reaching under a
    neighbour's higher ceiling). The schema holds one ceiling per room, so this becomes a warning."""
    h = P_room_y - floor
    up = h[(h > 1.9) & (h < 4.0) & (np.abs(h - main) > 0.15)]
    if len(up) < 200:
        return None
    hist, e = np.histogram(up, np.arange(1.9, 4.0 + res, res))
    k = int(np.argmax(hist))
    lev = e[k] + res / 2
    layer = up[np.abs(up - lev) < 0.03]
    return float(np.median(layer)) if len(layer) * voxel * voxel >= min_frac * room_area else None


@dataclass
class BackendResult:
    rooms: list[dict]
    warnings: list[str]
    footprint: dict
    debug: dict
    adjacency: list = None


def rooms_from_cloud(P: np.ndarray, cam_xyz: np.ndarray, rays, err: ErrorModel, voxel: float,
                     split: bool = True, id_prefix: str = "r", label: str | None = None,
                     interior_mode: str = "carve", sightlines=None, cam_fwd_xz=None,
                     hfov: float | None = None) -> BackendResult:
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
        walls, perim, perim_var, fit_var = [], 0.0, 0.0, 0.0
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
            fit_var += f_prev.sigma ** 2 + f_next.sigma ** 2 + 2 * err.abs_m ** 2  # wall noise without the scale term
        area = p2.polygon_area(poly)
        # E26: the scale term was counted twice (inside every wall sigma, then again as 2*rel*area, added linearly).
        # Area = wall-position noise (independent per wall) + a common scale factor (area scales with its square).
        area_sig = float(np.hypot(np.sqrt(fit_var) * np.sqrt(area) / 2, 2 * err.rel * area))
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
            vis = {}
            if cam_fwd_xz is not None and hfov is not None:
                vis = {"cams_uv": cam_xyz[:, [0, 2]] @ R.T, "cams_fwd": cam_fwd_xz @ R.T, "hfov": hfov}
            for j, o in enumerate(detect_openings(poly, p_uv, p_y, c_uv, wall_tol=tol[0], beyond_min=tol[1], **vis)):
                w_sig = float(np.sqrt(2 * (0.025 + err.abs_m) ** 2 + (err.rel * o.width) ** 2))  # +-half a slot per jamb
                openings.append({"id": f"{rid}.o{j + 1}", "wall_id": f"{rid}.w{o.wall + 1}",
                                 "type": o.kind, "width": meas(o.width, w_sig, "m"),
                                 "offset_along_wall": meas(o.offset, w_sig, "m"),
                                 "sill_height": meas(max(o.sill, 0.0), 0.05, "m"), "connects_to": None,
                                 "detection_confidence": round(min(1.0, o.support / 200), 2) if o.evidence ==
                                 "see_through" else 0.3,
                                 "evidence": o.evidence,
                                 "centre_plan": list(o.centre_uv)})
        room = {"id": rid, "label": label or f"room {k + 1}", "polygon": poly.round(4).tolist(), "walls": walls,
                "openings": openings, "floor_area": meas(area, area_sig, "m2"),
                "perimeter": meas(perim, float(np.sqrt(perim_var)), "m")}
        if ceil is None:
            room["ceiling_height"] = None
            warnings.append(f"{rid}: ceiling not observed well enough to measure; no value reported")
        else:
            c, cs = ceil
            c2 = second_ceiling(P[inside, 1], lev.floor, area, voxel, c)
            if c2 is not None:
                warnings.append(f"{rid}: two ceiling levels, {c:.2f} m (reported, most of the room) and {c2:.2f} m "
                                "over part of it (a lowered section, or the outline reaches under a neighbour's ceiling)")
            room["ceiling_height"] = meas(c, float(np.sqrt(cs ** 2 + lev.floor_sigma ** 2 + 2 * err.abs_m ** 2
                                                            + (err.rel * c) ** 2)), "m")
        rooms.append(room)
        total_area += area
        total_var += area_sig ** 2
    footprint = meas(total_area, float(np.sqrt(total_var)), "m2")
    adjacency = _adjacency(rooms, masks if split else [], maps.grid, maps.interior) if len(rooms) > 1 else []
    return BackendResult(rooms, warnings, footprint,
                         {"theta": theta, "maps": maps, "masks": masks, "levels": lev, "camera_height_m": cam_h},
                         adjacency)


def _adjacency(rooms: list[dict], masks: list, grid, interior=None, door_match_m: float = 0.6) -> list[dict]:
    """Room pairs that connect, in one shared frame (LiDAR, video):
    door  : an opening detected from both rooms at the same place (within door_match_m); both get connects_to;
    open  : the rooms' floor areas touch (the watershed cut an open passage; no wall between them)."""
    import cv2

    out, seen = [], set()
    for i, a in enumerate(rooms):
        for j in range(i + 1, len(rooms)):
            b = rooms[j]
            best = None
            for oa in a["openings"]:
                for ob in b["openings"]:
                    if oa["type"] == "window" or ob["type"] == "window":
                        continue
                    d = float(np.hypot(*(np.array(oa["centre_plan"]) - np.array(ob["centre_plan"]))))
                    if d < door_match_m and (best is None or d < best[0]):
                        best = (d, oa, ob)
            if best:
                _, oa, ob = best
                oa["connects_to"], ob["connects_to"] = b["id"], a["id"]
                out.append({"a": a["id"], "b": b["id"], "via": oa["id"]})
                seen.add((i, j))
    if masks and len(masks) == len(rooms) and interior is not None:
        # open passage = one continuous stretch of floor the LiDAR rays crossed, from one room into the other.
        # Across a wall there is always a break (rays stop short of each wall face), even where the wall itself was
        # not detected (E19: "touching" masks linked living and bathroom across their shared wall).
        union_all = np.zeros_like(masks[0], bool)
        for m in masks:
            union_all |= m
        loose = interior & ~union_all  # carved floor the watershed cleanup left unassigned (the cut zones)
        for i in range(len(rooms)):
            for j in range(i + 1, len(rooms)):
                if (i, j) in seen:
                    continue
                # only floor directly between the two rooms (within ~20 cm of both) may bridge them; the full loose
                # set connected nearly every room pair through other rooms' surroundings
                k20 = np.ones((21, 21), np.uint8)
                near = (cv2.dilate(masks[i].astype(np.uint8), k20).astype(bool)
                        & cv2.dilate(masks[j].astype(np.uint8), k20).astype(bool))
                region = (masks[i] | masks[j] | (loose & near)).astype(np.uint8)
                region = cv2.dilate(region, np.ones((2, 2), np.uint8))  # bridge 1-cell cracks only
                n, lab = cv2.connectedComponents(region, connectivity=4)
                li, lj = np.unique(lab[masks[i]]), np.unique(lab[masks[j]])
                if set(li[li > 0]) & set(lj[lj > 0]):
                    out.append({"a": rooms[i]["id"], "b": rooms[j]["id"], "via": "open passage"})
    return out


def empty_plan(tier: str, source: str) -> dict:
    return {"schema_version": "0.1",
            "capture": {"tier": tier, "source": source, "pipeline_version": __version__},
            "rooms": [], "adjacency": [], "property": {}, "damage": [], "concealed_flags": [], "scope": [],
            "warnings": []}
