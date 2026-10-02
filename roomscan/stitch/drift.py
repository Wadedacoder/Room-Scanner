"""Drift correction for long LiDAR walks: 2D wall scan-matching between revisits + a small pose graph.

ARKit's up axis is reliable (accelerometer); what drifts over a multi-room walk is heading and position. So each
fragment of consecutive frames gets a correction (yaw about its centroid + a translation), estimated from places the
walk revisits:

  1. Cut the walk into fragments (~4 s each); each is locally consistent.
  2. For fragment pairs that are close in space but far apart in time, match their top-down wall images: brute-force
     yaw in +-3 deg, translation by phase correlation. Keep a match only if it raises wall overlap and stays small
     (a large "correction" means a wrong match, not drift).
  3. Solve a linear least-squares pose graph: rotations first, then translations. Consecutive fragments are tied
     together (ARKit is locally right), fragment 0 is fixed.

E7 showed Open3D point-to-plane ICP on these fragments diverges (metre-scale "corrections"), hence this approach.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from roomscan.io.stray import StrayCapture

RES = 0.04  # m per pixel for wall images (phase correlation is sub-pixel; 2 cm was 4x slower for the same result)


@dataclass
class DriftResult:
    corrections: np.ndarray  # (n_frames, 4, 4) world_corrected <- world_arkit
    fragments: list[tuple[int, int]]
    loops: list[dict] = field(default_factory=list)  # accepted revisit matches
    rejected: int = 0
    max_shift_m: float = 0.0
    max_yaw_deg: float = 0.0
    residual_before_cm: float = 0.0  # median misalignment over accepted revisits, ARKit poses
    residual_after_cm: float = 0.0  # same, after correction


def _rot(theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]])


def _image(xz: np.ndarray, origin: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    img = np.zeros(shape, np.float32)
    px = np.floor((xz - origin) / RES).astype(int)
    ok = (px[:, 0] >= 0) & (px[:, 0] < shape[1]) & (px[:, 1] >= 0) & (px[:, 1] < shape[0])
    np.add.at(img, (px[ok, 1], px[ok, 0]), 1.0)
    img = np.minimum(img, 3.0)
    return cv2.GaussianBlur(img, (0, 0), 1.0)


def _overlap(a: np.ndarray, b: np.ndarray) -> float:
    """Share of b's wall pixels that land within ~4 cm of a's walls."""
    am = cv2.dilate((a > 0.2).astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    bm = b > 0.2
    return float((am & bm).sum() / max(bm.sum(), 1))


def match_fragments(xz_i: np.ndarray, xz_j: np.ndarray, max_yaw_deg: float = 3.0, max_shift: float = 0.3):
    """Correction (yaw about j's centroid, shift d) moving fragment j's walls onto fragment i's, or None."""
    cj = xz_j.mean(0)
    lo = np.minimum(xz_i.min(0), xz_j.min(0)) - 0.5
    hi = np.maximum(xz_i.max(0), xz_j.max(0)) + 0.5
    shape = tuple(int(x) for x in np.ceil((hi - lo) / RES)[::-1])
    if shape[0] * shape[1] > 6_000_000:
        return None
    A = _image(xz_i, lo, shape)
    base = _overlap(A, _image(xz_j, lo, shape))
    win = cv2.createHanningWindow(shape[::-1], cv2.CV_32F)
    best = None

    def try_yaw(yaw):
        nonlocal best
        B = _image((xz_j - cj) @ _rot(yaw).T + cj, lo, shape)
        (dx, dy), _ = cv2.phaseCorrelate(B, A, win)
        d = np.array([dx, dy]) * RES
        if np.linalg.norm(d) > max_shift:
            return
        ov = _overlap(A, _image((xz_j - cj) @ _rot(yaw).T + cj + d, lo, shape))
        if best is None or ov > best[2]:
            best = (yaw, d, ov)

    for yaw in np.radians(np.arange(-max_yaw_deg, max_yaw_deg + 1e-9, 1.0)):  # coarse
        try_yaw(yaw)
    if best is not None:  # refine around the coarse winner
        for dyaw in np.radians([-0.75, -0.5, -0.25, 0.25, 0.5, 0.75]):
            try_yaw(best[0] + dyaw)
    if best is None or best[2] < 0.35 or best[2] < base + 0.02:
        return None
    return {"yaw": float(best[0]), "d": best[1], "overlap_before": base, "overlap_after": best[2], "cj": cj}


def correct_drift(cap: StrayCapture, frag_len: int = 240, stride: int = 6, max_depth: float = 4.0,
                  revisit_radius: float = 2.0, min_gap: int = 3) -> DriftResult:
    n = len(cap) - 1
    frags = [(s, min(s + frag_len, n)) for s in range(0, n, frag_len)]
    if len(frags) > 1 and frags[-1][1] - frags[-1][0] < frag_len // 3:
        frags[-2] = (frags[-2][0], frags[-1][1])
        frags.pop()
    floor = None
    walls = []
    for a, b in frags:
        P = np.concatenate([cap.points_world(i, 2, max_depth) for i in range(a, b, stride)])
        if floor is None:
            floor = np.percentile(P[:, 1], 2)
        h = P[:, 1] - floor
        walls.append(P[(h > 0.4) & (h < 1.8)][:, [0, 2]])
    cams = np.array([cap.poses[a:b, [0, 2], 3].mean(0) for a, b in frags])
    cent = np.array([w.mean(0) if len(w) else c for w, c in zip(walls, cams)])

    # revisit edges: (i, j, yaw_ij, d_ij, cj)
    edges, loops, rejected = [], [], 0
    for i in range(len(frags)):
        for j in range(i + min_gap, len(frags)):
            if np.linalg.norm(cams[i] - cams[j]) > revisit_radius or len(walls[i]) < 200 or len(walls[j]) < 200:
                continue
            m = match_fragments(walls[i], walls[j])
            if m is None:
                rejected += 1
                continue
            edges.append((i, j, m["yaw"], m["d"], m["cj"]))
            loops.append({"i": i, "j": j, "yaw_deg": round(np.degrees(m["yaw"]), 2),
                          "shift_cm": round(float(np.linalg.norm(m["d"])) * 100, 1),
                          "overlap": [round(m["overlap_before"], 3), round(m["overlap_after"], 3)]})

    K = len(frags)
    theta = np.zeros(K)
    t = np.zeros((K, 2))
    if edges:
        w_odo = 4.0  # consecutive fragments: ARKit is locally right, so their relative correction ~ 0
        # 1) rotations: theta_j - theta_i = -yaw_ij (j must turn by yaw_ij relative to i), theta_0 = 0
        rows, rhs, wts = [], [], []
        for i in range(K - 1):
            r = np.zeros(K); r[i], r[i + 1] = -1, 1; rows.append(r); rhs.append(0.0); wts.append(w_odo)
        for i, j, yaw, _, _ in edges:
            r = np.zeros(K); r[i], r[j] = -1, 1; rows.append(r); rhs.append(yaw); wts.append(1.0)
        A = np.array(rows) * np.sqrt(wts)[:, None]
        b = np.array(rhs) * np.sqrt(wts)
        theta[1:] = np.linalg.lstsq(A[:, 1:], b, rcond=None)[0]
        # 2) translations, rotations fixed. Correction of fragment k: x' = R(theta_k)(x - cent_k) + cent_k + t_k.
        #    Revisit: the point cj in j's raw frame matches q = R(yaw)(cj - cj) + cj + d = cj + d in i's raw frame,
        #    so the corrected positions must agree: C_j(cj) = C_i(cj + d).
        rows, rhs, wts = [], [], []
        for i in range(K - 1):
            for ax in (0, 1):
                r = np.zeros(2 * K); r[2 * i + ax], r[2 * (i + 1) + ax] = -1, 1; rows.append(r)
            # C_{i+1}(c) = C_i(c) at the shared boundary point c (their common centroid)
            c = (cent[i] + cent[i + 1]) / 2
            lhs = (_rot(theta[i]) @ (c - cent[i]) + cent[i]) - (_rot(theta[i + 1]) @ (c - cent[i + 1]) + cent[i + 1])
            rhs += list(lhs); wts += [w_odo, w_odo]
        for i, j, yaw, d, cj in edges:
            q = cj + d
            lhs = (_rot(theta[i]) @ (q - cent[i]) + cent[i]) - (_rot(theta[j]) @ (cj - cent[j]) + cent[j])
            for ax in (0, 1):
                r = np.zeros(2 * K); r[2 * i + ax], r[2 * j + ax] = -1, 1; rows.append(r)
            rhs += list(lhs); wts += [1.0, 1.0]
        A = np.array(rows) * np.sqrt(wts)[:, None]
        b = np.array(rhs) * np.sqrt(wts)
        sol = np.linalg.lstsq(A[:, 2:], b, rcond=None)[0]
        t[1:] = sol.reshape(-1, 2)

    # per-fragment 4x4 world corrections (rotation about +Y through the fragment's wall centroid)
    corr = np.tile(np.eye(4), (len(cap), 1, 1))
    node = []
    for k, (a, b) in enumerate(frags):
        R2 = _rot(theta[k])
        T = np.eye(4)
        # world (x, z) plane: x' = R2 (x - c) + c + t  ->  3x3 rotation about +Y
        T[0, 0], T[0, 2], T[2, 0], T[2, 2] = R2[0, 0], R2[0, 1], R2[1, 0], R2[1, 1]
        off = cent[k] - R2 @ cent[k] + t[k]
        T[0, 3], T[2, 3] = off
        corr[a:b] = T
        node.append(T)
    corr[frags[-1][1]:] = node[-1]

    def residual(apply: bool) -> float:
        res = []
        for i, j, _, d, cj in edges:
            q = cj + d
            if apply:
                pi = _rot(theta[i]) @ (q - cent[i]) + cent[i] + t[i]
                pj = _rot(theta[j]) @ (cj - cent[j]) + cent[j] + t[j]
            else:
                pi, pj = q, cj
            res.append(np.linalg.norm(pi - pj))
        return float(np.median(res) * 100) if res else 0.0

    shifts = [float(np.linalg.norm(T[[0, 2], 3])) for T in node]
    return DriftResult(corr, frags, loops, rejected, max(shifts) if shifts else 0.0,
                       float(np.degrees(np.abs(theta).max())) if K else 0.0, residual(False), residual(True))


class CorrectedCapture:
    """A StrayCapture whose poses are drift-corrected; drop-in for fuse() / wall_band_rays()."""

    def __init__(self, cap: StrayCapture, corrections: np.ndarray):
        self._cap, self._corr = cap, corrections
        self.poses = corrections @ cap.poses

    def __len__(self):
        return len(self._cap)

    def points_world(self, i: int, min_conf: int = 2, max_depth: float = 6.0) -> np.ndarray:
        P = self._cap.points_world(i, min_conf, max_depth)
        C = self._corr[i]
        return P @ C[:3, :3].T + C[:3, 3]
