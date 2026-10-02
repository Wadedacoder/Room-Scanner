"""Photo tier: place per-room reconstructions into one property frame.

For a strong cross-room link (photo i of room A and photo j of room B see the same surfaces, e.g. through a doorway):
  1. lift A's matched pixels to 3D with A's metric depth (A's gravity-aligned frame);
  2. PnP: where photo j must have been, in A's frame, to see those 3D points at B's matched pixels;
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

MIN_INLIERS = 45  # E14: real links 47-89, coincidences 14-38


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
    okp, rvec, tvec, inl = cv2.solvePnPRansac(Xw.astype(np.float64), pb.astype(np.float64), Kb, None,
                                              reprojectionError=8.0, iterationsCount=2000, confidence=0.999,
                                              flags=cv2.SOLVEPNP_EPNP)
    if not okp or inl is None or len(inl) < 10:
        return None
    Rw2c, _ = cv2.Rodrigues(rvec)
    pose_j_in_A = np.eye(4)
    pose_j_in_A[:3, :3] = Rw2c.T
    pose_j_in_A[:3, 3] = (-Rw2c.T @ tvec).ravel()
    T_AB = pose_j_in_A @ np.linalg.inv(B.c2w[jb])
    yaw = float(np.arctan2(T_AB[0, 2], T_AB[0, 0]))  # rotation about +Y
    tilt = float(np.degrees(np.arccos(np.clip(T_AB[1, 1], -1, 1))))
    return yaw, T_AB[[0, 2], 3].copy(), int(len(inl)), tilt


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
        r = _relative_pose(frames[L.room_a], L.photo_a, frames[L.room_b], L.photo_b, L.pts_a, L.pts_b)
        if r is None:
            continue
        yaw, t, n_pnp, tilt = r
        if tilt > 15:  # both frames are gravity-aligned; a big tilt means a wrong PnP
            continue
        cand.append({"a": L.room_a, "b": L.room_b, "yaw": yaw, "t": t, "matches": L.inliers, "pnp_inliers": n_pnp,
                     "tilt_deg": round(tilt, 1)})
    if not cand:
        return {}, []
    degree: dict[str, int] = {}
    for c in cand:
        degree[c["a"]] = degree.get(c["a"], 0) + c["pnp_inliers"]
        degree[c["b"]] = degree.get(c["b"], 0) + c["pnp_inliers"]
    root = max(degree, key=degree.get)
    placed = {root: Placement(0.0, np.zeros(2))}
    edges = []
    changed = True
    while changed:
        changed = False
        for c in sorted(cand, key=lambda x: -x["pnp_inliers"]):
            a, b = c["a"], c["b"]
            for (p, q, yaw, t) in ((a, b, c["yaw"], c["t"]), (b, a, -c["yaw"], -_xz_rot(-c["yaw"]) @ c["t"])):
                if p in placed and q not in placed:
                    yaw_s = _snap_yaw(yaw, frames[p].theta, frames[q].theta)
                    P = placed[p]
                    placed[q] = Placement(P.yaw + yaw_s, P.t + _xz_rot(P.yaw) @ t, p, c)
                    edges.append({"a": p, "b": q, "matches": c["matches"], "pnp_inliers": c["pnp_inliers"],
                                  "yaw_snap_deg": round(float(np.degrees(yaw_s - yaw)), 1)})
                    changed = True
    return placed, edges


def polygon_to_property(poly_uv: np.ndarray, frame: RoomFrame, pl: Placement, root_theta: float) -> np.ndarray:
    """Room plan polygon -> property plan coordinates (axis-aligned with the root room's walls)."""
    xz = poly_uv @ p2.rot2(-frame.theta)  # uv = xz @ R.T  =>  xz = uv @ R
    xz_root = (xz @ _xz_rot(pl.yaw).T) + pl.t
    return xz_root @ p2.rot2(-root_theta).T
