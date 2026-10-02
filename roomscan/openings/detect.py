"""Doors, open passages and windows in a room's walls, from points the cameras saw THROUGH the wall line.

For each straight wall of a room outline (plan frame, wall-aligned) we look along the wall in 5 cm slots:
  wall    points within +-wall_tol of the wall line at 0.3-1.8 m height (solid wall, or a closed door leaf)
  beyond  points more than beyond_min outside the wall line whose camera ray crosses the wall inside this slot
Tolerances are per tier: LiDAR walls are ~1 cm thick in the cloud (8 / 25 cm), photo walls ~20 cm (20 / 40 cm, E15).
An opening is a run of slots with little wall but with "beyond" points: the camera saw into the next room.
  door / open passage: the see-through reaches down to the floor (< 0.3 m)
  window:              see-through only above a sill (lowest beyond point > 0.5 m)
Width = run length, refined to the nearest wall points on either side (the jambs).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

SLOT = 0.05


@dataclass
class Opening:
    wall: int  # index into the room's wall list
    offset: float  # metres from the wall's start to the opening's near edge
    width: float
    kind: str  # door | open_passage | window
    sill: float  # lowest see-through height above the floor
    support: int  # number of see-through points
    centre_uv: tuple[float, float]
    evidence: str = "see_through"  # see_through | gap (no wall surface where a camera was looking; E15)


def _segment_crossings(cam_uv, pts_uv, a, b):
    """Parameter t in [0,1] along wall a->b where each camera->point ray crosses it (nan where it doesn't)."""
    d = b - a
    r = pts_uv - cam_uv
    den = r[:, 0] * d[1] - r[:, 1] * d[0]
    with np.errstate(divide="ignore", invalid="ignore"):
        s = ((a[0] - cam_uv[:, 0]) * d[1] - (a[1] - cam_uv[:, 1]) * d[0]) / den  # along the ray
        t = ((a[0] - cam_uv[:, 0]) * r[:, 1] - (a[1] - cam_uv[:, 1]) * r[:, 0]) / den  # along the wall
    ok = (s > 0) & (s < 1) & (t >= 0) & (t <= 1)
    return np.where(ok, t, np.nan)


def _visible_slots(a, u, nslot, cams_uv, cams_fwd, hfov, max_dist=6.0):
    """Slots of wall a + u*t that lie inside at least one camera's horizontal field of view."""
    t = (np.arange(nslot) + 0.5) * SLOT
    centres = a[None] + t[:, None] * u[None]
    vis = np.zeros(nslot, bool)
    for c, f in zip(cams_uv, cams_fwd):
        d = centres - c
        dist = np.linalg.norm(d, axis=1)
        cosang = (d @ f) / np.maximum(dist, 1e-9) / max(np.linalg.norm(f), 1e-9)
        vis |= (dist < max_dist) & (cosang > np.cos(hfov / 2))
    return vis


def detect_openings(polygon: np.ndarray, pts_uv: np.ndarray, pts_h: np.ndarray, pts_cam_uv: np.ndarray,
                    min_w: float = 0.55, max_w: float = 2.4, wall_tol: float = 0.08,
                    beyond_min: float = 0.25, cams_uv=None, cams_fwd=None, hfov: float | None = None) -> list[Opening]:
    """polygon: (n,2) room outline, CCW, plan frame. pts_uv/pts_h: all points (plan position, height above floor).
    pts_cam_uv: for each point, the plan position of the camera that saw it."""
    out: list[Opening] = []
    n = len(polygon)
    centroid = polygon.mean(0)
    for wi in range(n):
        a, b = polygon[wi], polygon[(wi + 1) % n]
        L = float(np.linalg.norm(b - a))
        if L < min_w + 0.1:
            continue
        u = (b - a) / L
        nrm = np.array([-u[1], u[0]])
        if np.dot(centroid - a, nrm) > 0:  # make nrm point OUT of the room
            nrm = -nrm
        rel = pts_uv - a
        along, across = rel @ u, rel @ nrm
        nslot = int(np.ceil(L / SLOT))
        band = (pts_h > 0.3) & (pts_h < 1.8)
        wall_sel = band & (np.abs(across) < wall_tol) & (along > 0) & (along < L)
        wall_cnt = np.bincount(np.clip((along[wall_sel] / SLOT).astype(int), 0, nslot - 1), minlength=nslot)
        beyond = across > beyond_min
        t = _segment_crossings(pts_cam_uv[beyond], pts_uv[beyond], a, b)
        hit = np.isfinite(t)
        bslot = np.clip((t[hit] * L / SLOT).astype(int), 0, nslot - 1)
        bh = pts_h[beyond][hit]
        see_cnt = np.bincount(bslot, minlength=nslot)
        low = np.full(nslot, np.inf)
        np.minimum.at(low, bslot, bh)
        wall_ref = np.percentile(wall_cnt[wall_cnt > 0], 75) if (wall_cnt > 0).any() else 1.0
        see_slot = (see_cnt >= 3) & (wall_cnt < 0.15 * wall_ref)
        gap_slot = np.zeros(nslot, bool)
        if cams_uv is not None and hfov is not None and (wall_cnt > 0).mean() >= 0.4:
            # sparse photos (E15): a doorway often shows only as missing wall where a camera WAS looking
            gap_slot = (wall_cnt < 0.15 * wall_ref) & _visible_slots(a, u, nslot, cams_uv, cams_fwd, hfov)
            gap_slot[:2] = gap_slot[-2:] = False  # corners: wall-fit noise, not doors
        open_slot = see_slot | gap_slot
        # close 1-slot holes, then take runs
        s = open_slot.copy()
        s[1:-1] |= open_slot[:-2] & open_slot[2:]
        i = 0
        while i < nslot:
            if not s[i]:
                i += 1
                continue
            j = i
            while j + 1 < nslot and s[j + 1]:
                j += 1
            w = (j - i + 1) * SLOT
            if min_w <= w <= max_w:
                sill = float(np.min(low[i:j + 1]))
                kind = "window" if sill > 0.5 else ("door" if w <= 1.2 else "open_passage")
                mid = a + u * ((i + j + 1) / 2 * SLOT)
                seen = see_slot[i:j + 1].mean() >= 0.3
                if not np.isfinite(sill):
                    sill = 0.0  # gap with no see-through: assume it reaches the floor (door), low confidence
                    kind = "door" if w <= 1.2 else "open_passage"
                out.append(Opening(wi, round(i * SLOT, 3), round(w, 3), kind, round(sill, 2),
                                   int(see_cnt[i:j + 1].sum()), (round(float(mid[0]), 3), round(float(mid[1]), 3)),
                                   "see_through" if seen else "gap"))
            i = j + 1
    return out
