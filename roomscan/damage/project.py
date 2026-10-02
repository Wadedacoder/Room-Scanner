"""Turn a 2D damage detection (box in one photo/keyframe) into a metric region on a room surface.

Inputs per detection: the view's metric depth map, intrinsics and gravity-aligned camera pose (the same data every tier
already has), the room's plan rotation and walls. Steps:
  1. back-project the pixels inside the box (confident depth only) to 3D;
  2. decide the surface: floor (height < 0.15 m), ceiling (within 0.15 m of the ceiling level), else the nearest wall
     line in the plan;
  3. express the points in that surface's own 2D frame (wall: along-wall u, height v; floor/ceiling: plan u, v) and
     take a robust extent (5th-95th percentile) as the region; area = extent_w x extent_h for a box-shaped stain.
Uncertainty: depth noise of the tier plus half a pixel-footprint at the box edges.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from roomscan.geometry import plan2d as p2


@dataclass
class Detection:
    view: int  # index into the tier's views (photos / keyframes)
    cls: str  # water_stain | mold | crack | hole | peeling_paint | fire_smoke | other
    box: tuple[float, float, float, float]  # x0, y0, x1, y1 in the view's depth-map pixels
    confidence: float
    source: str = ""  # backend that produced it (e.g. "claude:<model>", "owlv2", "synthetic")


@dataclass
class SurfaceRegion:
    surface_id: str
    kind: str  # wall | floor | ceiling
    polygon: list[list[float]]  # surface-local metres
    width: float
    height: float
    area: float
    height_above_floor: float  # bottom of the region (walls), 0 for floor
    n_points: int


def project_detection(det: Detection, depth: np.ndarray, K: np.ndarray, c2w: np.ndarray, floor_y: float,
                      ceiling_y: float | None, theta: float, walls: list[dict], room_id: str,
                      min_points: int = 30) -> SurfaceRegion | None:
    x0, y0, x1, y1 = (round(v) for v in det.box)
    h, w = depth.shape
    x0, x1 = max(0, x0), min(w, x1)
    y0, y1 = max(0, y0), min(h, y1)
    if x1 - x0 < 2 or y1 - y0 < 2:
        return None
    v, u = np.mgrid[y0:y1, x0:x1]
    z = depth[y0:y1, x0:x1]
    ok = (z > 0.2) & (z < 8.0)
    if ok.sum() < min_points:
        return None
    u, v, z = u[ok], v[ok], z[ok]
    Xc = np.stack([(u + 0.5 - K[0, 2]) * z / K[0, 0], (v + 0.5 - K[1, 2]) * z / K[1, 1], z], 1)
    X = Xc @ c2w[:3, :3].T + c2w[:3, 3]
    hgt = X[:, 1] - floor_y
    uv = X[:, [0, 2]] @ p2.rot2(-theta).T
    med_h = float(np.median(hgt))
    if med_h < 0.15:
        return _flat_region(f"{room_id}.floor", "floor", uv, 0.0, len(z))
    if ceiling_y is not None and abs(np.median(X[:, 1]) - ceiling_y) < 0.15:
        return _flat_region(f"{room_id}.ceiling", "ceiling", uv, med_h, len(z))
    # nearest wall line (distance from the region's median plan point to each wall segment)
    c = np.median(uv, axis=0)
    best, best_d = None, np.inf
    for wl in walls:
        a, b = np.array(wl["start"]), np.array(wl["end"])
        d = b - a
        L = float(np.hypot(*d))
        if L < 0.2:
            continue
        t = float(np.clip(np.dot(c - a, d) / (L * L), 0, 1))
        dist = float(np.hypot(*(c - (a + t * d))))
        if dist < best_d:
            best, best_d = (wl, a, d / L, L), dist
    if best is None or best_d > 0.5:
        return None
    wl, a, dirv, L = best
    along = (uv - a) @ dirv
    lo_u, hi_u = np.percentile(along, [5, 95])
    lo_v, hi_v = np.percentile(hgt, [5, 95])
    wdt, hgt_ = float(hi_u - lo_u), float(hi_v - lo_v)
    poly = [[round(lo_u, 3), round(lo_v, 3)], [round(hi_u, 3), round(lo_v, 3)], [round(hi_u, 3), round(hi_v, 3)],
            [round(lo_u, 3), round(hi_v, 3)]]
    return SurfaceRegion(wl["id"], "wall", poly, round(wdt, 3), round(hgt_, 3), round(wdt * hgt_, 4),
                         round(float(lo_v), 3), len(z))


def _flat_region(sid, kind, uv, height, n):
    lo, hi = np.percentile(uv, 5, axis=0), np.percentile(uv, 95, axis=0)
    wdt, dep = float(hi[0] - lo[0]), float(hi[1] - lo[1])
    poly = [[round(lo[0], 3), round(lo[1], 3)], [round(hi[0], 3), round(lo[1], 3)], [round(hi[0], 3), round(hi[1], 3)],
            [round(lo[0], 3), round(hi[1], 3)]]
    return SurfaceRegion(sid, kind, poly, round(wdt, 3), round(dep, 3), round(wdt * dep, 4), round(height, 3), n)
