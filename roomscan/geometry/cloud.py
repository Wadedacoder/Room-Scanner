"""Fuse LiDAR frames into a voxelised world point cloud, and find floor / ceiling levels."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from roomscan.io.stray import StrayCapture


def fuse(cap: StrayCapture, stride: int, min_conf: int, max_depth: float, voxel: float,
         frames: np.ndarray | None = None) -> np.ndarray:
    """Average all confident points per voxel. Returns (M,3) float32, gravity-aligned (+Y up)."""
    idx = np.arange(0, len(cap) - 1, stride) if frames is None else frames
    keys_all, sums, counts = [], [], []
    for start in range(0, len(idx), 64):  # bounded memory: reduce every 64 frames
        P = np.concatenate([cap.points_world(i, min_conf, max_depth) for i in idx[start:start + 64]])
        k = np.floor(P / voxel).astype(np.int32)
        uk, inv = np.unique(k, axis=0, return_inverse=True)
        s = np.zeros((len(uk), 3))
        np.add.at(s, inv.ravel(), P)
        keys_all.append(uk), sums.append(s), counts.append(np.bincount(inv.ravel(), minlength=len(uk)))
    uk, inv = np.unique(np.concatenate(keys_all), axis=0, return_inverse=True)
    s = np.zeros((len(uk), 3))
    np.add.at(s, inv.ravel(), np.concatenate(sums))
    c = np.bincount(inv.ravel(), weights=np.concatenate(counts))
    keep = c >= 2  # a voxel seen once is likely a flying pixel
    return (s[keep] / c[keep, None]).astype(np.float32)


@dataclass
class Levels:
    floor: float
    ceiling: float | None
    floor_sigma: float
    ceiling_sigma: float | None


def _refine_level(y: np.ndarray, guess: float, band: float = 0.04) -> tuple[float, float]:
    near = y[np.abs(y - guess) < band]
    med = float(np.median(near))
    near = near[np.abs(near - med) < 0.02]
    return float(np.median(near)), float(1.2533 * near.std() / np.sqrt(max(len(near), 1)))


def find_levels(P: np.ndarray, cam_y: float, floor_rule: str = "lowest", support: float = 0.4) -> Levels:
    """Floor and ceiling levels from a gravity-aligned cloud (+Y up).

    floor_rule="lowest" (default since 0.5.2): the LOWEST flat level below the camera whose point count reaches
    `support` x the densest one. "densest" (before 0.5.2) took the densest level, which picked the desk top in the study
    and the bed in the bedroom (E12): their tops can hold as many points as the partly hidden floor.
    Levels are found on 5 cm bins smoothed over 3 bins, so one level split across two bins still counts as one.
    """
    y = P[:, 1]
    bins = np.arange(y.min(), y.max() + 0.01, 0.01)
    h, e = np.histogram(y, bins)
    centers = (e[:-1] + e[1:]) / 2
    below = centers < cam_y - 0.6
    if floor_rule == "densest":
        floor_guess = centers[below][np.argmax(h[below])]
    else:
        hb = np.convolve(h, np.ones(5), "same")  # 5 cm window
        cand = np.nonzero(below)[0]
        thr = support * hb[cand].max()
        strong = cand[hb[cand] >= thr]
        # local maxima only, then the lowest of them
        peaks = [i for i in strong if hb[i] >= hb[max(i - 5, 0):i + 6].max()]
        floor_guess = centers[min(peaks)] if peaks else centers[below][np.argmax(h[below])]
    floor, fs = _refine_level(y, floor_guess)
    above = centers > cam_y + 0.3
    ceiling = cs = None
    if above.any():
        i = np.argmax(h[above])
        # a real ceiling is a big flat level; a few shelf tops are not
        if h[above][i] > 0.15 * h[below].max():
            ceiling, cs = _refine_level(y, centers[above][i])
    return Levels(floor, ceiling, fs, cs)


def wall_band_rays(cap: StrayCapture, stride: int, floor_y: float, min_conf: int = 1, max_depth: float = 5.0,
                   lo: float = 0.3, hi: float = 1.8, per_frame: int = 400, seed: int = 0):
    """Camera->surface rays whose endpoint is 0.3-1.8 m above the floor, projected to (x, z).

    Everything a ray crosses was empty space; this is what defines a room's interior (free-space carving).
    Returns (start_xz, end_xz) arrays of shape (R, 2).
    """
    rng = np.random.default_rng(seed)
    starts, ends = [], []
    for i in range(0, len(cap) - 1, stride):
        P = cap.points_world(i, min_conf, max_depth)
        h = P[:, 1] - floor_y
        P = P[(h > lo) & (h < hi)]
        if len(P) > per_frame:
            P = P[rng.choice(len(P), per_frame, replace=False)]
        ends.append(P[:, [0, 2]])
        starts.append(np.repeat(cap.poses[i, [0, 2], 3][None], len(P), 0))
    return np.concatenate(starts), np.concatenate(ends)
