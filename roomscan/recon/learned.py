"""Learned geometry for the photo / video tiers: DA3 depth + relative poses, metric scale from DA3Metric + true focal.

Measured basis (docs/EXPERIMENTS.md): the true (EXIF) focal gives -0.7% .. -9.3% scale error, the model's own focal
guess up to +18% (E1, E2). So the true focal is used both for the metric conversion and for back-projection.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from roomscan import models
from roomscan.recon.windows import WindowPrediction, run_windowed


@dataclass
class Recon:
    c2w: np.ndarray  # (n,4,4) OpenCV camera-to-world, metric
    depth: np.ndarray  # (n,h,w) metric z-depth at processing resolution
    conf: np.ndarray  # (n,h,w)
    K: np.ndarray  # (n,3,3) TRUE intrinsics scaled to processing resolution
    scale: float  # factor applied to the relative reconstruction
    focal_pred_err: float  # mean(predicted / true focal) - 1, reported as a diagnostic


class LearnedRecon:
    """Loads ONE model at a time (geometry, then metric) and frees it after use: on an 8 GB laptop two resident
    models plus activations exhausted shared GPU memory (see lite profile, gpu_memory_fraction)."""

    def __init__(self, cfg):
        import torch

        from roomscan.recon.da3_loader import load_da3_class

        rc = cfg["recon"]
        dev = cfg["runtime"]["device"]
        self.device = dev if (dev != "mps" or torch.backends.mps.is_available()) else "cpu"
        frac = float(cfg["runtime"].get("gpu_memory_fraction", 0.0) or 0.0)
        if self.device == "mps" and frac > 0:
            torch.mps.set_per_process_memory_fraction(frac)
        self.DA3 = load_da3_class()
        self.repos = (models.get(rc["model"]).hf_repo, models.get(rc["metric_model"]).hf_repo)
        self.win, self.overlap, self.res = rc["max_views"], rc["window_overlap"], rc["image_long_side"]
        self.torch = torch

    def _load(self, repo):
        return self.DA3.from_pretrained(repo).to(self.device).eval()

    def _free(self, model):
        del model
        import gc

        gc.collect()
        if self.device == "mps":
            self.torch.mps.empty_cache()
        elif self.device == "cuda":
            self.torch.cuda.empty_cache()

    def _infer(self, model, imgs):
        with self.torch.inference_mode():
            return model.inference(imgs, process_res=self.res)

    def reconstruct(self, images: list[np.ndarray], K_full: np.ndarray) -> Recon:
        n = len(images)
        overlap = min(self.overlap, self.win - 1)

        model = self._load(self.repos[0])

        def predict(idx):
            p = self._infer(model, [images[i] for i in idx])
            E = np.tile(np.eye(4), (len(idx), 1, 1))
            E[:, :3, :4] = p.extrinsics
            return WindowPrediction(np.linalg.inv(E), p.depth, p.conf, p.intrinsics)

        st = run_windowed(n, self.win, overlap, predict)
        self._free(model)
        h, w = st.depth.shape[1:]
        sx = w / images[0].shape[1]
        K = K_full.copy()
        K[:, :2] *= sx
        metric_model = self._load(self.repos[1])
        # DA3Metric is monocular: one image per call gives identical depth with a fraction of the activation memory
        raw = np.concatenate([self._infer(metric_model, [images[i]]).depth for i in range(n)])
        self._free(metric_model)
        metric = raw * K[:, 0, 0][:, None, None] / 300.0  # DA3Metric convention: depth = focal * out / 300
        valid = (st.depth > 1e-4) & np.isfinite(metric) & (st.conf >= np.percentile(st.conf, 30))
        k = float(np.median(metric[valid] / st.depth[valid]))
        c2w = st.c2w.copy()
        c2w[:, :3, 3] *= k
        return Recon(c2w, st.depth * k, st.conf, K, k, float(np.mean(st.intrinsics[:, 0, 0] / K[:, 0, 0] - 1)))


def backproject(rec: Recon, conf_pct: float = 30, max_depth: float = 8.0, stride: int = 2):
    """World points (n_pts,3) and per-view camera centres from a Recon."""
    pts, cams = [], rec.c2w[:, :3, 3]
    thr = np.percentile(rec.conf, conf_pct)
    for i in range(len(rec.depth)):
        D = rec.depth[i][::stride, ::stride]
        C = rec.conf[i][::stride, ::stride]
        v, u = np.nonzero((D > 0.2) & (D < max_depth) & (C >= thr))
        z = D[v, u]
        K = rec.K[i]
        x = (u * stride + 0.5 - K[0, 2]) * z / K[0, 0]
        y = (v * stride + 0.5 - K[1, 2]) * z / K[1, 1]
        P = np.stack([x, y, z], 1) @ rec.c2w[i, :3, :3].T + rec.c2w[i, :3, 3]
        pts.append(P)
    return pts, cams


def gravity_align(pts: list[np.ndarray], c2w: np.ndarray) -> np.ndarray:
    """Rotation G (3x3) taking the reconstruction frame to a +Y-up world.

    Initial "down" = mean image-down axis of the cameras (people hold phones level); refined by fitting the floor plane
    (the densest layer of points along that axis, on the far side from the cameras).
    """
    down = c2w[:, :3, 1].mean(0)
    down /= np.linalg.norm(down)
    P = np.concatenate(pts)
    P = P[:: max(1, len(P) // 200000)]
    for _ in range(2):
        h = P @ down  # larger = lower
        cam_h = np.median(c2w[:, :3, 3] @ down)
        below = h > cam_h + 0.5
        if below.sum() < 500:
            break
        hist, e = np.histogram(h[below], 200)
        lev = e[np.argmax(hist)]
        F = P[below][np.abs(h[below] - lev) < 0.05]
        if len(F) < 300:
            break
        c = F - F.mean(0)
        nrm = np.linalg.svd(c, full_matrices=False)[2][-1]
        if np.dot(nrm, down) < 0:
            nrm = -nrm
        if np.degrees(np.arccos(np.clip(np.dot(nrm, down), -1, 1))) > 25:  # floor fit disagrees: keep camera prior
            break
        down = nrm
    up = -down
    # rotation mapping `up` to +Y (Rodrigues)
    yaxis = np.array([0.0, 1.0, 0.0])
    v = np.cross(up, yaxis)
    c = float(np.dot(up, yaxis))
    if np.linalg.norm(v) < 1e-9:
        return np.eye(3) if c > 0 else np.diag([1.0, -1.0, -1.0])
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx * (1 / (1 + c))


def voxelize(P: np.ndarray, voxel: float) -> np.ndarray:
    k = np.floor(P / voxel).astype(np.int64)
    uk, inv = np.unique(k, axis=0, return_inverse=True)
    s = np.zeros((len(uk), 3))
    np.add.at(s, inv.ravel(), P)
    c = np.bincount(inv.ravel())
    return (s / c[:, None]).astype(np.float32)
