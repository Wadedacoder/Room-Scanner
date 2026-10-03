"""Photo tier: place per-room reconstructions into one property frame.

For a strong cross-room link (photo i of room A and photo j of room B see the same surfaces, e.g. through a doorway):
  1. lift A's matched pixels to 3D with A's metric depth (A's gravity-aligned frame);
  2. gravity-aware PnP (4 DoF: yaw + translation; both frames already know "up", E28): where photo j must have been,
     in A's frame, to see those 3D points at B's matched pixels;
  3. B already knows where photo j was in B's own frame, so T_A<-B = pose_j_in_A * inverse(pose_j_in_B).
Both frames are gravity-aligned, so T is restricted to yaw + translation; the yaw is then snapped to the 90-degree
steps between the two rooms' wall directions (walls are square to each other).
Rooms are attached along a maximum spanning tree of links (strongest first) starting from the best-connected room.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from roomscan.geometry import plan2d as p2

MIN_INLIERS = 30  # candidates; acceptance is the two-way agreement test below (E18)
MAX_YAW_DISAGREE_DEG = 10.0
MAX_TILT_DEG = 15.0
MAX_T_DISAGREE_M = 1.0
MAX_OVERLAP_FRAC = 0.15  # of the smaller room's floor; allows wall-position error, not a room on top of another


@dataclass
class RoomFrame:
    name: str
    c2w: np.ndarray  # (n,4,4) photo cameras in the room's GRAVITY-ALIGNED metric frame
    depth: np.ndarray  # (n,h,w) metric depth at processing resolution
    K: np.ndarray  # (n,3,3) intrinsics at processing resolution
    photo_hw: tuple[int, int]  # original photo size (h, w)
    theta: float  # room's wall angle: plan uv = xz @ rot2(-theta).T
    polygon_uv: np.ndarray  # room outline in its own plan frame


@dataclass
class Placement:
    yaw: float  # rotation about +Y from room frame to property frame
    t: np.ndarray  # (2,) translation in the property x-z plane
    parent: str | None = None
    link: dict = field(default_factory=dict)


def _relative_pose(A: RoomFrame, ia: int, B: RoomFrame, jb: int, pts_a: np.ndarray, pts_b: np.ndarray):
    """yaw, t (x-z) mapping room B's gravity frame into room A's, plus PnP inlier count."""
    D = A.depth[ia]
    h, w = D.shape
    s = w / A.photo_hw[1]
    u, v = pts_a[:, 0] * s, pts_a[:, 1] * s
    ok = (u >= 0) & (u < w) & (v >= 0) & (v < h)
    u, v, pb = u[ok], v[ok], pts_b[ok]
    z = D[v.astype(int), u.astype(int)]
    ok = (z > 0.2) & (z < 8.0)
    u, v, z, pb = u[ok], v[ok], z[ok], pb[ok]
    if len(z) < 12:
        return None
    K = A.K[ia]
    Xc = np.stack([(u + 0.5 - K[0, 2]) * z / K[0, 0], (v + 0.5 - K[1, 2]) * z / K[1, 1], z], 1)
    T = A.c2w[ia]
    Xw = Xc @ T[:3, :3].T + T[:3, 3]  # 3D in A's frame
    sb = B.depth.shape[2] / B.photo_hw[1]
    Kb = B.K[jb].copy()
    Kb[:2] /= sb  # back to original-photo pixels (pts_b are original pixels)
    # E28: 6-DoF EPnP on 15-40 mostly coplanar points (a door, one wall) returned 40-60 deg tilts for links whose yaw
    # agreed both ways, so the links were rejected. Both rooms are gravity-aligned: only yaw + 3D offset are unknown.
    sol = _pnp_4dof(Xw, pb, Kb, B.c2w[jb][:3, :3])
    if sol is None:
        return None
    yaw, t_cam, n_inl = sol
    R = _yaw_mat(yaw)
    pose_j_in_A = np.eye(4)
    pose_j_in_A[:3, :3] = R @ B.c2w[jb][:3, :3]
    pose_j_in_A[:3, 3] = t_cam
    T_AB = pose_j_in_A @ np.linalg.inv(B.c2w[jb])
    return yaw, T_AB[[0, 2], 3].copy(), n_inl, 0.0


def _yaw_mat(yaw: float) -> np.ndarray:
    """Rotation about +Y by yaw (same convention as arctan2(R[0,2], R[0,0]))."""
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, 0, s], [0, 1.0, 0], [-s, 0, c]])


def _pnp_4dof(Xw: np.ndarray, px: np.ndarray, K: np.ndarray, R_cam_b: np.ndarray, thr_px: float = 8.0,
              seed: int = 0) -> tuple[float, np.ndarray, int] | None:
    """Camera pose in A's frame with known gravity: rotation = Ry(yaw) @ R_cam_b, unknown yaw and centre C.
    For each yaw (1 deg grid) the projection is linear in C; 2-point RANSAC + least squares on inliers.
    Returns (yaw, camera centre in A, inliers)."""
    rng = np.random.default_rng(seed)
    n = len(Xw)
    if n < 8:
        return None
    rays = np.stack([(px[:, 0] - K[0, 2]) / K[0, 0], (px[:, 1] - K[1, 2]) / K[1, 1], np.ones(n)], 1)
    pairs = rng.integers(0, n, size=(64, 2))
    pairs = pairs[pairs[:, 0] != pairs[:, 1]]
    best = (0, None)
    for yaw in np.radians(np.arange(0, 360, 1.0)):
        Rcw = (_yaw_mat(yaw) @ R_cam_b).T  # world (A) -> camera
        # camera coords: Xc = Rcw (X - C); projection constraint x*Xc_z - Xc_x = 0, y*Xc_z - Xc_y = 0 (linear in C)
        P = Xw @ Rcw.T  # Rcw X
        r0, r1, r2 = Rcw
        A_rows = np.concatenate([rays[:, [0]] * r2 - r0, rays[:, [1]] * r2 - r1])  # (2n,3) coefficients of -C
        b = np.concatenate([rays[:, 0] * P[:, 2] - P[:, 0], rays[:, 1] * P[:, 2] - P[:, 1]])
        for i, j in pairs:
            idx = [i, j, n + i, n + j]
            C, *_ = np.linalg.lstsq(A_rows[idx], b[idx], rcond=None)
            Xc = P - C @ Rcw.T
            front = Xc[:, 2] > 0.1
            err = np.full(n, np.inf)
            u = K[0, 0] * Xc[front, 0] / Xc[front, 2] + K[0, 2]
            v = K[1, 1] * Xc[front, 1] / Xc[front, 2] + K[1, 2]
            err[front] = np.hypot(u - px[front, 0], v - px[front, 1])
            k = int((err < thr_px).sum())
            if k > best[0]:
                best = (k, (yaw, C, err < thr_px))
    if best[1] is None or best[0] < 8:
        return None
    yaw, C, inl = best[1]
    Rcw = (_yaw_mat(yaw) @ R_cam_b).T
    P = Xw[inl] @ Rcw.T
    r0, r1, r2 = Rcw
    A_rows = np.concatenate([rays[inl, 0:1] * r2 - r0, rays[inl, 1:2] * r2 - r1])
    b = np.concatenate([rays[inl, 0] * P[:, 2] - P[:, 0], rays[inl, 1] * P[:, 2] - P[:, 1]])
    C, *_ = np.linalg.lstsq(A_rows, b, rcond=None)
    return float(yaw), C, int(inl.sum())


def _snap_yaw(yaw: float, theta_a: float, theta_b: float) -> float:
    """Snap so B's walls are parallel or perpendicular to A's. Rotating B by yaw (about +Y) turns its wall angle
    theta_b into theta_b - yaw; square to A means theta_b - yaw = theta_a + k*90deg."""
    rel = yaw - (theta_b - theta_a)
    snapped = np.round(rel / (np.pi / 2)) * (np.pi / 2)
    return float(snapped + theta_b - theta_a)


def _xz_rot(yaw: float) -> np.ndarray:
    """Rotation about +Y acting on (x, z) column vectors: x' = c x + s z, z' = -s x + c z."""
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, s], [-s, c]])


def place_rooms(frames: dict[str, RoomFrame], links, min_inliers: int = MIN_INLIERS):
    """Returns placements (room -> Placement in the frame of the root room) and the accepted edges."""
    cand = []
    for L in links:
        if L.inliers < min_inliers or L.room_a not in frames or L.room_b not in frames:
            continue
        # E18: a raw match count is noisy (47 vs 42 for the same pair across runs on the Mac GPU) and some 25-40-match
        # pairs are coincidences. Accept a link only if the pose computed from A's depth and the one from B's depth
        # agree (mirror-image yaw within 10 deg) and both are level (the two rooms share gravity).
        r1 = _relative_pose(frames[L.room_a], L.photo_a, frames[L.room_b], L.photo_b, L.pts_a, L.pts_b)
        r2 = _relative_pose(frames[L.room_b], L.photo_b, frames[L.room_a], L.photo_a, L.pts_b, L.pts_a)
        if r1 is None or r2 is None:
            continue
        (y1, t1, n1, tilt1), (y2, t2, n2, tilt2) = r1, r2
        disagree = abs(np.degrees(np.angle(np.exp(1j * (y1 + y2)))))
        if disagree > MAX_YAW_DISAGREE_DEG or max(tilt1, tilt2) > MAX_TILT_DEG:
            continue
        # E28: the offsets must agree too (B->A computed from B's side is the mirror of A->B)
        t_gap = float(np.linalg.norm(t2 + _xz_rot(-y1) @ t1))
        if t_gap > max(MAX_T_DISAGREE_M, 0.5 * float(np.linalg.norm(t1))):  # rooms differ in metric scale by ~5-15%
            continue
        yaw = float(np.angle((np.exp(1j * y1) + np.exp(-1j * y2)) / 2))
        cand.append({"a": L.room_a, "b": L.room_b, "yaw": yaw, "t": t1, "matches": L.inliers, "pnp_inliers": n1 + n2,
                     "yaw_disagree_deg": round(float(disagree), 1), "tilt_deg": round(max(tilt1, tilt2), 1)})
    if not cand:
        return {}, []
    degree: dict[str, int] = {}
    for c in cand:
        degree[c["a"]] = degree.get(c["a"], 0) + c["pnp_inliers"]
        degree[c["b"]] = degree.get(c["b"], 0) + c["pnp_inliers"]
    root = max(degree, key=degree.get)
    placed = {root: Placement(0.0, np.zeros(2))}
    edges, rejected = [], []
    changed = True
    while changed:
        changed = False
        for c in sorted(cand, key=lambda x: -x["pnp_inliers"]):
            a, b = c["a"], c["b"]
            for (p, q, yaw, t) in ((a, b, c["yaw"], c["t"]), (b, a, -c["yaw"], -_xz_rot(-c["yaw"]) @ c["t"])):
                if p in placed and q not in placed:
                    yaw_s = _snap_yaw(yaw, frames[p].theta, frames[q].theta)
                    P = placed[p]
                    cand_pl = Placement(P.yaw + yaw_s, P.t + _xz_rot(P.yaw) @ t, p, c)
                    # E31: rooms cannot share floor. A link that puts this room on top of a placed one is wrong
                    # (repetitive texture can pass the match and agreement tests); try the room's other links.
                    clash = _worst_overlap(frames, placed, q, cand_pl)
                    if clash > MAX_OVERLAP_FRAC:
                        rejected.append({"a": p, "b": q, "matches": c["matches"], "overlap": round(clash, 2)})
                        continue
                    placed[q] = cand_pl
                    edges.append({"a": p, "b": q, "matches": c["matches"], "pnp_inliers": c["pnp_inliers"],
                                  "yaw_disagree_deg": c["yaw_disagree_deg"], "tilt_deg": c["tilt_deg"],
                                  "yaw_snap_deg": round(float(np.degrees(yaw_s - yaw)), 1)})
                    changed = True
    place_rooms.rejected = rejected  # for debug output
    return placed, edges


def _room_xz(frame: RoomFrame, pl: Placement) -> np.ndarray:
    """Room outline in the root room's x-z frame."""
    xz = frame.polygon_uv @ p2.rot2(-frame.theta)
    return (xz @ _xz_rot(pl.yaw).T) + pl.t


def _worst_overlap(frames: dict, placed: dict, q: str, pl: Placement) -> float:
    """Largest overlap of room q (at placement pl) with any placed room, as a fraction of the smaller room's area."""
    from shapely.geometry import Polygon

    A = Polygon(_room_xz(frames[q], pl)).buffer(0)
    worst = 0.0
    for r, plr in placed.items():
        B = Polygon(_room_xz(frames[r], plr)).buffer(0)
        small = min(A.area, B.area)
        if small > 0:
            worst = max(worst, A.intersection(B).area / small)
    return worst


def polygon_to_property(poly_uv: np.ndarray, frame: RoomFrame, pl: Placement, root_theta: float) -> np.ndarray:
    """Room plan polygon -> property plan coordinates (axis-aligned with the root room's walls)."""
    xz = poly_uv @ p2.rot2(-frame.theta)  # uv = xz @ R.T  =>  xz = uv @ R
    xz_root = (xz @ _xz_rot(pl.yaw).T) + pl.t
    return xz_root @ p2.rot2(-root_theta).T
