"""Kaggle job: which learned-geometry model should the photo / video tiers use, per hardware profile?

Input: the eval bundle from scripts/make_eval_bundle.py (proxy photo sets per room with N = 2/4/6/8 stills, and a
60-frame "video" set), each image paired with ARKit LiDAR depth + pose as ground truth.

For every (model, set) we measure:
  rot_err_deg      mean pairwise relative-rotation error vs ARKit
  ate_m / sim3_s   camera-centre RMSE after a similarity alignment, and the scale that alignment needed
  focal_err        predicted focal / true focal - 1
  absrel           depth shape error after the best single scale (pure geometry quality)
  scale_*          metric scale error = median(pred / lidar) - 1; this becomes wall-length error 1:1
                   native  : the model's own metric output (nested model only)
                   predf   : relative model + DA3Metric, metric conversion with the *predicted* focal
                   truef   : relative model + DA3Metric, metric conversion with the *true* focal (≈ EXIF focal)
  runtime_s, peak_gb
"""

from __future__ import annotations

import gc
import glob
import importlib
import json
import os
import shutil
import subprocess
import sys
import tarfile
import time
import traceback
from pathlib import Path

DA3_COMMIT = "3d835ec"
WORK = Path("/kaggle/working")
os.environ.setdefault("HF_HOME", "/tmp/hf")


def sh(cmd: str) -> None:
    print("+", cmd, flush=True)
    subprocess.run(cmd, shell=True, check=True)


def install() -> None:
    # --no-deps: DA3 pins numpy<2 etc.; we must not downgrade Kaggle's preinstalled stack.
    sh(f"pip install -q --no-deps git+https://github.com/ByteDance-Seed/depth-anything-3@{DA3_COMMIT}")
    pip_name = {"cv2": "opencv-python-headless", "PIL": "pillow", "sklearn": "scikit-learn"}
    for _ in range(12):  # install whatever the import chain actually needs
        try:
            importlib.invalidate_caches()
            importlib.import_module("depth_anything_3.api")
            return
        except ModuleNotFoundError as e:
            sh(f"pip install -q {pip_name.get(e.name, e.name)}")
    raise RuntimeError("could not satisfy depth_anything_3 imports")


def find_bundle() -> Path:
    hits = glob.glob("/kaggle/input/**/index.json", recursive=True)
    if not hits:
        for arc in glob.glob("/kaggle/input/**/*.tar*", recursive=True):
            with tarfile.open(arc) as t:
                t.extractall(WORK / "bundle")
        hits = glob.glob(str(WORK / "bundle/**/index.json"), recursive=True)
    return Path(hits[0]).parent


install()
import cv2  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from depth_anything_3.api import DepthAnything3  # noqa: E402

MULTIVIEW = {
    "da3-small": "depth-anything/DA3-SMALL",
    "da3-base": "depth-anything/DA3-BASE",
    "da3-large-1.1": "depth-anything/DA3-LARGE-1.1",
    "da3-giant-1.1": "depth-anything/DA3-GIANT-1.1",
    "da3-nested-giant-large": "depth-anything/DA3NESTED-GIANT-LARGE",
}
METRIC = "depth-anything/DA3METRIC-LARGE"
PROCESS_RES = 504


def load(repo: str):
    return DepthAnything3.from_pretrained(repo).to("cuda").eval()


def unload(model) -> None:
    del model
    gc.collect()
    torch.cuda.empty_cache()


def timed(model, images):
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    t = time.time()
    pred = model.inference(images, process_res=PROCESS_RES)
    torch.cuda.synchronize()
    return pred, time.time() - t, torch.cuda.max_memory_allocated() / 2**30


def to_gt_res(d: np.ndarray, shape) -> np.ndarray:
    return np.stack([cv2.resize(x.astype(np.float32), (shape[1], shape[0]), interpolation=cv2.INTER_LINEAR) for x in d])


def rot_angle(R: np.ndarray) -> float:
    return float(np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))))


def umeyama(src: np.ndarray, dst: np.ndarray):
    mu_s, mu_d = src.mean(0), dst.mean(0)
    xs, xd = src - mu_s, dst - mu_d
    U, S, Vt = np.linalg.svd(xd.T @ xs / len(src))
    D = np.eye(3)
    if np.linalg.det(U @ Vt) < 0:
        D[2, 2] = -1
    R = U @ D @ Vt
    s = np.trace(np.diag(S) @ D) / xs.var(0).sum()
    return s, R, mu_d - s * R @ mu_s


def pose_metrics(ext_w2c: np.ndarray, T_gt: np.ndarray) -> dict:
    n = len(T_gt)
    T_pred = np.tile(np.eye(4), (n, 1, 1))
    T_pred[:, :3, :4] = ext_w2c
    T_pred = np.linalg.inv(T_pred)  # -> c2w
    errs = [rot_angle((T_gt[i, :3, :3].T @ T_gt[j, :3, :3]).T @ (T_pred[i, :3, :3].T @ T_pred[j, :3, :3]))
            for i in range(n) for j in range(i + 1, n)]
    out = {"rot_err_deg": float(np.mean(errs))}
    if n >= 3:
        s, R, t = umeyama(T_pred[:, :3, 3], T_gt[:, :3, 3])
        aligned = (s * (R @ T_pred[:, :3, 3].T)).T + t
        out |= {"ate_m": float(np.sqrt(((aligned - T_gt[:, :3, 3]) ** 2).sum(1).mean())), "sim3_s": float(s)}
    return out


def depth_metrics(pred: np.ndarray, gt: np.ndarray, conf: np.ndarray) -> dict:
    m = (conf == 2) & (gt > 0.2) & (gt < 5.0)
    p, g = pred[m], gt[m]
    s = np.median(g / p)
    return {"absrel": float(np.mean(np.abs(p * s - g) / g)), "raw_ratio": float(np.median(p / g)), "valid_px": int(m.sum())}


def scale_err(pred: np.ndarray, gt: np.ndarray, conf: np.ndarray) -> float:
    m = (conf == 2) & (gt > 0.2) & (gt < 5.0)
    return float(np.median(pred[m] / gt[m]) - 1)


def main() -> None:
    bundle = find_bundle()
    sets = json.loads((bundle / "index.json").read_text())["sets"]
    print("GPU:", torch.cuda.get_device_name(0), "| sets:", sets, flush=True)
    data = {}
    for s in sets:
        g = np.load(bundle / s / "gt.npz")
        data[s] = {"images": [str(bundle / s / "images" / n) for n in g["names"]],
                   "depth": g["depth"], "conf": g["conf"], "K": g["K"], "T": g["T_world_cam"]}

    rows: list[dict] = []
    preds_dir = WORK / "preds"
    preds_dir.mkdir(exist_ok=True)

    # 1) metric-depth model once per set (raw network output; conversion needs a focal length)
    metric_raw = {}
    model = load(METRIC)
    for s in sets:
        pred, dt, mem = timed(model, data[s]["images"])
        metric_raw[s] = pred.depth  # (N,h,w) raw
        f_true_proc = data[s]["K"][:, 0, 0] * pred.depth.shape[2] / 1440.0
        mono = to_gt_res(metric_raw[s] * f_true_proc[:, None, None] / 300.0, data[s]["depth"].shape[1:])
        rows.append({"model": "da3metric-large(mono,truef)", "set": s, "n": len(data[s]["images"]), "runtime_s": dt,
                     "peak_gb": mem, "scale_truef": scale_err(mono, data[s]["depth"], data[s]["conf"])})
        print(rows[-1], flush=True)
    unload(model)

    # 2) multi-view models
    for name, repo in MULTIVIEW.items():
        try:
            model = load(repo)
        except Exception as e:  # noqa: BLE001
            rows.append({"model": name, "error": f"load: {e!r}"})
            continue
        for s in sets:
            d = data[s]
            row = {"model": name, "set": s, "n": len(d["images"])}
            try:
                pred, dt, mem = timed(model, d["images"])
                h, w = pred.depth.shape[1:]
                f_true_proc = d["K"][:, 0, 0] * w / 1440.0
                f_pred = pred.intrinsics[:, 0, 0]
                gt_shape = d["depth"].shape[1:]
                dep = to_gt_res(pred.depth, gt_shape)
                row |= {"runtime_s": dt, "peak_gb": mem, "focal_err": float(np.mean(f_pred / f_true_proc - 1)),
                        "is_metric": int(getattr(pred, "is_metric", 0) or 0)}
                row |= pose_metrics(pred.extrinsics, d["T"])
                row |= depth_metrics(dep, d["depth"], d["conf"])
                if row["is_metric"]:
                    row["scale_native"] = scale_err(dep, d["depth"], d["conf"])
                for tag, f in (("predf", f_pred), ("truef", f_true_proc)):
                    metric = metric_raw[s] * f[:, None, None] / 300.0
                    k = np.median(metric / np.maximum(pred.depth, 1e-6))  # one scale for the whole set
                    row[f"scale_{tag}"] = scale_err(dep * k, d["depth"], d["conf"])
                np.savez_compressed(preds_dir / f"{name}__{s}.npz", depth=pred.depth.astype(np.float16),
                                    conf=pred.conf.astype(np.float16), extrinsics=pred.extrinsics,
                                    intrinsics=pred.intrinsics)
            except torch.cuda.OutOfMemoryError:
                row["error"] = "OOM"
                torch.cuda.empty_cache()
            except Exception as e:  # noqa: BLE001
                row["error"] = repr(e)
                traceback.print_exc()
            rows.append(row)
            print(row, flush=True)
        unload(model)
        shutil.rmtree(os.environ["HF_HOME"], ignore_errors=True)  # weights total >15 GB; keep disk free

    (WORK / "results.json").write_text(json.dumps({"gpu": torch.cuda.get_device_name(0), "da3_commit": DA3_COMMIT,
                                                   "process_res": PROCESS_RES, "rows": rows}, indent=1))
    print("DONE", len(rows), "rows")


main()
