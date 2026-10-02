"""Loader for Stray Scanner captures (LiDAR tier).

Folder layout (https://docs.strayrobots.io/apps/scanner/format.html):
    rgb.mp4            HEVC video, 1920x1440
    depth/NNNNNN.png   uint16 depth in mm, 256x192
    confidence/NNNNNN.png  ARKit confidence 0/1/2
    odometry.csv       timestamp, frame, x, y, z, qx, qy, qz, qw[, fx, fy, cx, cy, ...]
    camera_matrix.csv  3x3 intrinsics at RGB resolution
    imu.csv

Poses are camera-to-world with an OpenCV camera convention (+X right, +Y down, +Z forward), in an ARKit world
frame that is gravity-aligned with +Y up. (Verified on our captures: with the ARKit camera convention the floor
lands at camera height, which is impossible; with OpenCV it sits ~1.5 m below the camera.)
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

import numpy as np
from PIL import Image

RGB_WH = (1920, 1440)
DEPTH_WH = (256, 192)


def quat_to_rot(q: np.ndarray) -> np.ndarray:
    """(N,4) xyzw quaternions -> (N,3,3) rotation matrices."""
    q = q / np.linalg.norm(q, axis=-1, keepdims=True)
    x, y, z, w = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    return np.stack(
        [
            np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], -1),
            np.stack([2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], -1),
            np.stack([2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], -1),
        ],
        -2,
    )


@dataclass
class StrayCapture:
    root: Path

    @classmethod
    def open(cls, root: str | Path) -> "StrayCapture":
        root = Path(root)
        missing = [n for n in ("odometry.csv", "camera_matrix.csv", "depth") if not (root / n).exists()]
        if missing:
            raise FileNotFoundError(f"{root} is not a Stray Scanner capture (missing {missing})")
        return cls(root)

    @cached_property
    def _odom(self) -> np.ndarray:
        return np.genfromtxt(self.root / "odometry.csv", delimiter=",", skip_header=1)

    @property
    def timestamps(self) -> np.ndarray:
        return self._odom[:, 0]

    @property
    def frame_ids(self) -> np.ndarray:
        return self._odom[:, 1].astype(int)

    @cached_property
    def K(self) -> np.ndarray:
        """Intrinsics at RGB resolution."""
        return np.loadtxt(self.root / "camera_matrix.csv", delimiter=",")

    @cached_property
    def K_depth(self) -> np.ndarray:
        s = DEPTH_WH[0] / RGB_WH[0]
        K = self.K.copy()
        K[:2] *= s
        return K

    @cached_property
    def poses(self) -> np.ndarray:
        """(N,4,4) camera-to-world transforms, OpenCV camera convention, world +Y up."""
        T = np.tile(np.eye(4), (len(self._odom), 1, 1))
        T[:, :3, :3] = quat_to_rot(self._odom[:, 5:9])
        T[:, :3, 3] = self._odom[:, 2:5]
        return T

    def __len__(self) -> int:
        return len(self._odom)

    def depth(self, i: int) -> np.ndarray:
        """Depth in metres, (192,256) float32."""
        return np.asarray(Image.open(self.root / "depth" / f"{self.frame_ids[i]:06d}.png"), np.float32) / 1000.0

    def confidence(self, i: int) -> np.ndarray:
        return np.asarray(Image.open(self.root / "confidence" / f"{self.frame_ids[i]:06d}.png"))

    def points_world(self, i: int, min_conf: int = 2, max_depth: float = 6.0) -> np.ndarray:
        """Back-project frame i's depth to world-frame points (M,3)."""
        D, C = self.depth(i), self.confidence(i)
        v, u = np.nonzero((C >= min_conf) & (D > 0.1) & (D < max_depth))
        z = D[v, u]
        K = self.K_depth
        x = (u + 0.5 - K[0, 2]) * z / K[0, 0]
        y = (v + 0.5 - K[1, 2]) * z / K[1, 1]
        pc = np.stack([x, y, z], 1)
        T = self.poses[i]
        return pc @ T[:3, :3].T + T[:3, 3]

    def summary(self) -> dict:
        p = self.poses[:, :3, 3]
        return {
            "frames": len(self),
            "duration_s": float(self.timestamps[-1] - self.timestamps[0]),
            "path_length_m": float(np.linalg.norm(np.diff(p, axis=0), axis=1).sum()),
            "loop_gap_m": float(np.linalg.norm(p[-1] - p[0])),
            "fx": float(self.K[0, 0]),
        }
