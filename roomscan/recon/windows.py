"""Run a multi-view model on more views than fit in memory: overlapping windows chained by similarity transforms.

Every window shares `overlap` views with the window before it. A shared view's camera pose and depth are known in both
windows' frames, which fixes the transform between them:
    scale s     median over shared pixels of depth_prev / depth_new  (model outputs are scale-ambiguous per window)
    rotation R  chordal mean of R_prev_i @ R_new_i^T over shared views
    translation t  mean of C_prev_i - R @ (s * C_new_i) over shared views
Poses are camera-to-world (OpenCV). numpy only, so the same code runs locally and on Kaggle.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np


def make_windows(n: int, size: int, overlap: int) -> list[list[int]]:
    """Index windows of `size` covering 0..n-1, consecutive windows sharing `overlap` views."""
    if size >= n:
        return [list(range(n))]
    if not 0 < overlap < size:
        raise ValueError("need 0 < overlap < size")
    step = size - overlap
    starts = list(range(0, n - size + 1, step))
    if starts[-1] + size < n:
        starts.append(n - size)  # last window flush with the end (overlaps more than `overlap`)
    return [list(range(s, s + size)) for s in starts]


def project_rotation(M: np.ndarray) -> np.ndarray:
    U, _, Vt = np.linalg.svd(M)
    return U @ np.diag([1.0, 1.0, np.linalg.det(U @ Vt)]) @ Vt


def sim3_from_shared(T_prev: np.ndarray, T_new: np.ndarray, d_prev: np.ndarray, d_new: np.ndarray):
    """Similarity (s, R, t) with  X_prev = s * R @ X_new + t, from shared views.

    T_*: (k,4,4) c2w of the shared views in each frame; d_*: (k,h,w) their depth maps (0 = invalid).
    """
    m = (d_prev > 0) & (d_new > 0) & np.isfinite(d_prev) & np.isfinite(d_new)
    if m.sum() < 100:
        raise ValueError("shared views have too few valid depth pixels to fix scale")
    s = float(np.median(d_prev[m] / d_new[m]))
    R = project_rotation(sum(T_prev[i, :3, :3] @ T_new[i, :3, :3].T for i in range(len(T_prev))))
    t = np.mean([T_prev[i, :3, 3] - s * R @ T_new[i, :3, 3] for i in range(len(T_prev))], axis=0)
    return s, R, t


def apply_sim3(T: np.ndarray, s: float, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    out = T.copy()
    out[:, :3, :3] = R @ T[:, :3, :3]
    out[:, :3, 3] = (s * (R @ T[:, :3, 3].T)).T + t
    return out


@dataclass
class WindowPrediction:
    """What a model returns for one window, all for the window's own views in order."""

    c2w: np.ndarray  # (k,4,4)
    depth: np.ndarray  # (k,h,w)
    conf: np.ndarray  # (k,h,w)
    intrinsics: np.ndarray  # (k,3,3)


@dataclass
class Stitched:
    c2w: np.ndarray
    depth: np.ndarray
    conf: np.ndarray
    intrinsics: np.ndarray
    windows: list[list[int]]
    scales: list[float]  # per-window scale applied (1.0 for the first)


def run_windowed(n: int, size: int, overlap: int, predict: Callable[[list[int]], WindowPrediction]) -> Stitched:
    """Call `predict` on each window and chain the results into window 0's frame and scale."""
    windows = make_windows(n, size, overlap)
    c2w = depth = conf = K = None
    placed = np.zeros(n, bool)
    scales: list[float] = []
    for w in windows:
        p = predict(w)
        if c2w is None:
            h, wd = p.depth.shape[1:]
            c2w, depth = np.zeros((n, 4, 4)), np.zeros((n, h, wd), np.float32)
            conf, K = np.zeros((n, h, wd), np.float32), np.zeros((n, 3, 3))
            s, R, t = 1.0, np.eye(3), np.zeros(3)
        else:
            shared = [j for j, g in enumerate(w) if placed[g]]
            gs = [w[j] for j in shared]
            s, R, t = sim3_from_shared(c2w[gs], p.c2w[shared], depth[gs], p.depth[shared])
        scales.append(s)
        new = [j for j, g in enumerate(w) if not placed[g]]
        gn = [w[j] for j in new]
        c2w[gn] = apply_sim3(p.c2w[new], s, R, t)
        depth[gn] = p.depth[new] * s
        conf[gn] = p.conf[new]
        K[gn] = p.intrinsics[new]
        placed[gn] = True
    return Stitched(c2w, depth, conf, K, windows, scales)
