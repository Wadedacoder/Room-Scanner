"""Top-down geometry: dominant wall axes, interior mask, room split, per-room wall polygons.

Plan coordinates: (u, v) in metres after rotating world (x, z) by the dominant wall angle, so walls are axis-aligned.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np


@dataclass
class Grid:
    origin: np.ndarray  # (2,) plan coords of pixel (0,0) corner
    res: float
    shape: tuple[int, int]  # (rows=v, cols=u)

    def to_px(self, uv: np.ndarray) -> np.ndarray:
        return np.floor((uv - self.origin) / self.res).astype(int)  # (col, row)

    def to_uv(self, px: np.ndarray) -> np.ndarray:
        return (px + 0.5) * self.res + self.origin


def rot2(theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]])


def dominant_angle(xz: np.ndarray, res: float = 0.02) -> float:
    """Angle that makes wall points most axis-aligned (sharpest projections), in [0, 90) deg -> radians."""
    best, best_a = -1.0, 0.0
    sub = xz[:: max(1, len(xz) // 200000)]
    for coarse in (np.arange(0, 90, 1.0), None):
        angles = coarse if coarse is not None else np.arange(best_a - 1, best_a + 1, 0.05)
        for a in angles:
            uv = sub @ rot2(np.radians(-a)).T
            score = 0.0
            for k in (0, 1):
                h, _ = np.histogram(uv[:, k], np.arange(uv[:, k].min(), uv[:, k].max() + res, res))
                score += float((h.astype(np.float64) ** 2).sum())
            if score > best:
                best, best_a = score, float(a)
    return np.radians(best_a)


def rasterize(uv: np.ndarray, grid: Grid) -> np.ndarray:
    px = grid.to_px(uv)
    ok = (px[:, 0] >= 0) & (px[:, 0] < grid.shape[1]) & (px[:, 1] >= 0) & (px[:, 1] < grid.shape[0])
    img = np.zeros(grid.shape, np.float32)
    np.add.at(img, (px[ok, 1], px[ok, 0]), 1)
    return img


def vertical_coverage(uv: np.ndarray, y: np.ndarray, grid: Grid, lo: float = 0.2, hi: float = 2.2,
                      step: float = 0.1) -> np.ndarray:
    """Fraction of `step`-high bands between lo and hi (above floor) that contain points, per cell."""
    m = (y > lo) & (y < hi)
    px = grid.to_px(uv[m])
    b = ((y[m] - lo) / step).astype(int)
    nb = int(round((hi - lo) / step))
    ok = (px[:, 0] >= 0) & (px[:, 0] < grid.shape[1]) & (px[:, 1] >= 0) & (px[:, 1] < grid.shape[0])
    cell = px[ok, 1] * grid.shape[1] + px[ok, 0]
    uniq = np.unique(cell.astype(np.int64) * nb + b[ok])
    cov = np.bincount(uniq // nb, minlength=grid.shape[0] * grid.shape[1]).reshape(grid.shape)
    # a wall seen at a slant smears over 2 cells; take the max over a 3x3 neighbourhood of the band union
    return cov / nb


def carve(start: np.ndarray, end: np.ndarray, grid: Grid, stop_short: float = 0.05) -> np.ndarray:
    """Count, per cell, the rays that pass through it (2D, plan coords)."""
    d = end - start
    L = np.linalg.norm(d, axis=1)
    keep = L > stop_short + grid.res
    start, d, L = start[keep], d[keep], L[keep]
    steps = np.ceil((L - stop_short) / grid.res).astype(int)
    counts = np.zeros(grid.shape[0] * grid.shape[1], np.int64)
    for chunk in range(0, len(start), 20000):
        sl = slice(chunk, chunk + 20000)
        n = steps[sl].max()
        t = np.arange(n)[None, :] * grid.res / L[sl, None]
        valid = np.arange(n)[None, :] < steps[sl, None]
        pts = start[sl, None, :] + t[..., None] * d[sl, None, :]
        px = grid.to_px(pts[valid])
        ok = (px[:, 0] >= 0) & (px[:, 0] < grid.shape[1]) & (px[:, 1] >= 0) & (px[:, 1] < grid.shape[0])
        cell = px[ok, 1].astype(np.int64) * grid.shape[1] + px[ok, 0]  # ~1 sample per ray per crossed cell
        counts += np.bincount(cell, minlength=counts.size)
    return counts.reshape(grid.shape)


def enclosed_by_walls(band_uv: np.ndarray, walls: np.ndarray, grid: Grid, traj_px: np.ndarray,
                      gap_close_m: float = 0.3) -> np.ndarray:
    """Sparse-view interior: flood from the cameras through non-wall space, bounded by the convex hull of all
    observed wall-band points. Walls are dilated by `gap_close_m` first so small holes between observed wall pieces
    do not leak; the dilation is given back afterwards."""
    hull_img = np.zeros(grid.shape, np.uint8)
    px = grid.to_px(band_uv)
    ok = (px[:, 0] >= 0) & (px[:, 0] < grid.shape[1]) & (px[:, 1] >= 0) & (px[:, 1] < grid.shape[0])
    if ok.sum() < 3:
        return np.zeros(grid.shape, bool)
    hull = cv2.convexHull(px[ok].astype(np.int32))
    cv2.fillConvexPoly(hull_img, hull, 1)
    k = max(1, int(gap_close_m / grid.res))
    thick = cv2.dilate(walls.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * k + 1, 2 * k + 1)))
    free = (hull_img.astype(bool) & ~thick.astype(bool)).astype(np.uint8)
    region = np.zeros((grid.shape[0] + 2, grid.shape[1] + 2), np.uint8)
    for c, r in traj_px:
        if 0 <= r < grid.shape[0] and 0 <= c < grid.shape[1] and free[r, c] and not region[r + 1, c + 1]:
            cv2.floodFill(free, region, (int(c), int(r)), 1, flags=4 | cv2.FLOODFILL_MASK_ONLY | (255 << 8))
    inside = region[1:-1, 1:-1] > 0
    inside = cv2.dilate(inside.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * k + 1, 2 * k + 1)))
    return inside.astype(bool) & hull_img.astype(bool) & ~walls


@dataclass
class PlanMaps:
    grid: Grid
    walls: np.ndarray  # bool, wall evidence
    floor: np.ndarray  # bool, floor seen
    interior: np.ndarray  # bool
    traj_px: np.ndarray


def build_maps(P: np.ndarray, floor_y: float, theta: float, cam_xz: np.ndarray, rays, res: float = 0.02,
               interior_mode: str = "carve") -> PlanMaps:
    """interior_mode: "carve" (dense LiDAR: space rays crossed) or "walls" (sparse photos: space enclosed by the
    observed walls, reached from the cameras, kept inside the hull of everything observed)."""
    R = rot2(-theta)
    uv = P[:, [0, 2]] @ R.T
    y = P[:, 1] - floor_y
    pad = 0.5
    origin = uv.min(0) - pad
    shape = tuple((np.ceil((uv.max(0) + pad - origin) / res)).astype(int)[::-1])
    grid = Grid(origin, res, shape)
    floor_img = rasterize(uv[np.abs(y) < 0.04], grid)
    walls = vertical_coverage(uv, y, grid, 0.2, 1.8) >= 0.35  # tall & continuous: walls yes, sofas/tables no
    walls = cv2.morphologyEx(walls.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8)).astype(bool)
    floor = cv2.morphologyEx((floor_img >= 1).astype(np.uint8), cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8)).astype(bool)

    # interior = space that LiDAR rays crossed (free-space carving), minus walls. Rays stop 5 cm short of their
    # surface so the wall itself is not carved.
    traj_px = grid.to_px(cam_xz @ R.T)
    if interior_mode == "carve":
        free = carve(rays[0] @ R.T, rays[1] @ R.T, grid) >= 3  # >= ~3 independent rays
        free &= ~walls
    else:
        free = enclosed_by_walls(uv[(y > 0.2) & (y < 1.8)], walls, grid, traj_px)
        if free.sum() * res * res < 0.5:  # cameras never landed in open space (noisy walls): fall back to carving
            free = (carve(rays[0] @ R.T, rays[1] @ R.T, grid) >= 3) & ~walls
    interior = cv2.morphologyEx(free.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    interior = cv2.morphologyEx(interior, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8)).astype(bool)
    return PlanMaps(grid, walls, floor, interior, traj_px)


def split_rooms(interior: np.ndarray, res: float, traj_px: np.ndarray, door_half_width: float = 0.38,
                min_room_m2: float = 1.0):
    """Watershed on the distance transform: cores further than ~half a door from any wall become room seeds."""
    dist = cv2.distanceTransform(interior.astype(np.uint8), cv2.DIST_L2, 5) * res
    core = (dist > door_half_width).astype(np.uint8)
    n, markers = cv2.connectedComponents(core)
    # drop seeds that are too small to be a room (furniture gaps); they get absorbed by neighbours
    for k in range(1, n):
        if (markers == k).sum() * res * res < 0.15:
            markers[markers == k] = 0
    markers = markers.astype(np.int32)
    markers[~interior] = n + 1  # background label
    img = cv2.cvtColor((interior * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    cv2.watershed(img, markers)
    rooms = []
    visited = set()
    for c, r in traj_px:
        if 0 <= r < markers.shape[0] and 0 <= c < markers.shape[1]:
            visited.add(int(markers[r, c]))
    for k in range(1, n):
        m = markers == k
        if k not in visited:  # only glimpsed through a doorway: partial, so not measured
            continue
        if m.sum() * res * res >= min_room_m2:
            m = cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_OPEN, np.ones((5, 5), np.uint8)).astype(bool)
            rooms.append(m)
    return rooms


@dataclass
class WallFit:
    axis: int  # 0: wall at u = c (vertical in plan), 1: wall at v = c
    c: float
    sigma: float  # std error of the fitted position (m)
    n: int


@dataclass
class RoomGeom:
    polygon: np.ndarray  # (k,2) plan coords, CCW
    walls: list[WallFit] = field(default_factory=list)


def _axis_polygon(mask: np.ndarray, grid: Grid, eps: float = 0.08, close_m: float = 0.4,
                  others: np.ndarray | None = None) -> np.ndarray:
    # fill furniture-shaped bites out of the floor before tracing; walls are re-fitted to wall points afterwards.
    # The fill may not grow into space another room owns (rooms must not overlap).
    k = max(3, int(close_m / grid.res) | 1)
    mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (k, k)))
    if others is not None:
        mask &= ~cv2.dilate(others.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
    cnts, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cnt = max(cnts, key=cv2.contourArea)
    approx = cv2.approxPolyDP(cnt, eps / grid.res, True)[:, 0, :].astype(float)
    return grid.to_uv(approx)


def rectilinear(poly: np.ndarray, min_len: float = 0.3) -> list[tuple[int, float, float, float]]:
    """Polygon -> list of axis-aligned edges (axis, c, start, end); short diagonal bits are dropped."""
    edges = []
    for a, b in zip(poly, np.roll(poly, -1, 0)):
        d = b - a
        if np.hypot(*d) < min_len:
            continue
        axis = 0 if abs(d[0]) < abs(d[1]) else 1  # axis 0: runs along v at u = c
        c = (a[0] + b[0]) / 2 if axis == 0 else (a[1] + b[1]) / 2
        s, e = (a[1], b[1]) if axis == 0 else (a[0], b[0])
        if edges and edges[-1][0] == axis:  # merge collinear-ish consecutive edges
            pa, pc, ps, pe = edges[-1]
            edges[-1] = (axis, (pc + c) / 2, ps, e)
        else:
            edges.append((axis, c, s, e))
    if len(edges) > 1 and edges[0][0] == edges[-1][0]:
        a0, c0, s0, e0 = edges.pop()
        _, c1, _, e1 = edges[0]
        edges[0] = (a0, (c0 + c1) / 2, s0, e1)
    return edges


def fit_wall(axis: int, c: float, s: float, e: float, wall_uv: np.ndarray, inside_sign: float,
             search: float = 0.25) -> WallFit:
    """Re-fit an edge's position to the wall surface: the inner-most dense layer of wall points near it."""
    lo, hi = sorted((s, e))
    along = wall_uv[:, 1 - axis]
    across = wall_uv[:, axis]
    # The traced edge is the free-space boundary, i.e. just inside this room's face of the wall. Search a little
    # outward (to the face) and a little inward, never through the wall to the neighbouring room's face.
    rel = (across - c) * inside_sign  # > 0 is into the room
    m = (along > lo + 0.1) & (along < hi - 0.1) & (rel > -0.15) & (rel < 0.10)
    pts = across[m]
    if len(pts) < 50:
        return WallFit(axis, c, 0.05, int(len(pts)))
    # wall surface = densest 1 cm layer in that window
    h, edges = np.histogram(pts, np.arange(c - search, c + search + 0.01, 0.01))
    peak = edges[np.argmax(h)] + 0.005
    layer = pts[np.abs(pts - peak) < 0.02]
    pos = float(np.median(layer))
    sigma = float(1.2533 * layer.std() / np.sqrt(len(layer)))
    return WallFit(axis, pos, sigma, int(len(layer)))


def room_geometry(mask: np.ndarray, grid: Grid, wall_uv: np.ndarray, others: np.ndarray | None = None) -> RoomGeom:
    poly = _axis_polygon(mask, grid, others=others)
    edges = rectilinear(poly)
    centroid = poly.mean(0)
    fits = []
    for axis, c, s, e in edges:
        sign = 1.0 if centroid[axis] > c else -1.0
        fits.append(fit_wall(axis, c, s, e, wall_uv, sign))
    # corners: intersection of consecutive (perpendicular) walls
    corners = []
    for f0, f1 in zip(fits, fits[1:] + fits[:1]):
        if f0.axis == f1.axis:
            continue
        u = f0.c if f0.axis == 0 else f1.c
        v = f0.c if f0.axis == 1 else f1.c
        corners.append((u, v))
    corners = np.array(corners)
    if len(corners) >= 3 and _signed_area(corners) < 0:
        corners = corners[::-1]
    return RoomGeom(corners, fits)


def _signed_area(p: np.ndarray) -> float:
    x, y = p[:, 0], p[:, 1]
    return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def polygon_area(p: np.ndarray) -> float:
    return abs(_signed_area(p))


def largest_component(mask: np.ndarray) -> np.ndarray:
    n, lab = cv2.connectedComponents(mask.astype(np.uint8))
    if n <= 1:
        return mask.astype(bool)
    sizes = np.bincount(lab.ravel())
    sizes[0] = 0
    return lab == int(np.argmax(sizes))
