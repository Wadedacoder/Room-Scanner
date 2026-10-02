"""Kaggle job E3: strategies for the video tier, plus MapAnything (which failed to install in E2).

E2 found windowed DA3 chains break on video (0-12/60 frames registered). E3 tries:
  * window overlap 4 instead of 2 (more shared evidence per link)
  * denser sampling: 120 frames instead of 60
  * MapAnything on all frames at once (no chaining), with and without the true intrinsics
DA3 runs on video sets only (photo sets <= 8 views fit one window; E2 already measured them).

--- E2 notes kept below ---
Kaggle job E2: windowed DA3 vs MapAnything, and "spread" vs "sweep" photo selection.

Changes from E1 (bench/kaggle/model_eval):
  * DA3 runs through roomscan.recon.windows (the pipeline's own code, shipped as the roomscan-code dataset), with the
    window sizes the hardware profiles use, so 6-8 photo sets and the 60-frame video fit on a T4.
  * DA3Metric runs in chunks (it is monocular; chunking changes nothing but memory).
  * MapAnything (Apache weights) is added, run on all views at once in memory-efficient mode, with and without the
    true intrinsics (MapAnything, unlike DA3, accepts intrinsics without poses, which is what EXIF gives us).
  * Per-view pose check: a view counts as registered if its orientation error < 10 deg after one global rotation.
"""

from __future__ import annotations

import gc
import glob
import importlib
import json
import os
import subprocess
import sys
import tarfile
import time
import traceback
from pathlib import Path

DA3_COMMIT = "3d835ec"
MA_COMMIT = "3d10cf7"
WORK = Path("/kaggle/working")
os.environ.setdefault("HF_HOME", "/tmp/hf")
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
PIP_NAME = {"cv2": "opencv-python-headless", "PIL": "pillow", "sklearn": "scikit-learn", "box": "python-box",
            "hydra": "hydra-core", "omegaconf": "omegaconf", "rerun": "rerun-sdk"}


def sh(cmd: str) -> None:
    print("+", cmd, flush=True)
    subprocess.run(cmd, shell=True, check=True)


def install(spec: str, module: str) -> None:
    sh(f"pip install -q --no-deps {spec}")  # never let a research repo downgrade Kaggle's stack
    for _ in range(15):
        try:
            importlib.invalidate_caches()
            importlib.import_module(module)
            return
        except ModuleNotFoundError as e:
            sh(f"pip install -q {PIP_NAME.get(e.name, e.name)}")
    raise RuntimeError(f"could not satisfy imports for {module}")


def find(pattern: str) -> Path:
    hits = glob.glob(f"/kaggle/input/**/{pattern}", recursive=True)
    if hits:
        return Path(hits[0])
    for arc in glob.glob("/kaggle/input/**/*.tar*", recursive=True):
        dst = WORK / "extracted" / Path(arc).stem
        if not dst.exists():
            with tarfile.open(arc) as t:
                t.extractall(dst)
    return Path(glob.glob(str(WORK / f"extracted/**/{pattern}"), recursive=True)[0])


# ---- code under test: the pipeline's own windowing ----
sys.path.insert(0, str(find("roomscan/__init__.py").parents[1]))
install(f"git+https://github.com/ByteDance-Seed/depth-anything-3@{DA3_COMMIT}", "depth_anything_3.api")

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from depth_anything_3.api import DepthAnything3  # noqa: E402
from PIL import Image  # noqa: E402

from roomscan.recon.windows import WindowPrediction, run_windowed  # noqa: E402

DA3_VARIANTS = [  # (name, repo, window, overlap)
    ("da3-base-w8o2", "depth-anything/DA3-BASE", 8, 2),
    ("da3-base-w8o4", "depth-anything/DA3-BASE", 8, 4),
    ("da3-large-1.1-w8o4", "depth-anything/DA3-LARGE-1.1", 8, 4),
]
OVERLAP = 4
METRIC = "depth-anything/DA3METRIC-LARGE"
RES = 504


# ---------------- metrics ----------------
def ang(R):
    return float(np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))))


def umeyama(src, dst):
    mu_s, mu_d = src.mean(0), dst.mean(0)
    xs, xd = src - mu_s, dst - mu_d
    U, S, Vt = np.linalg.svd(xd.T @ xs / len(src))
    D = np.eye(3)
    if np.linalg.det(U @ Vt) < 0:
        D[2, 2] = -1
    R = U @ D @ Vt
    return np.trace(np.diag(S) @ D) / xs.var(0).sum(), R, mu_d - (np.trace(np.diag(S) @ D) / xs.var(0).sum()) * R @ mu_s


def pose_metrics(Tp, Tg):
    n = len(Tg)
    pair = [ang((Tg[i, :3, :3].T @ Tg[j, :3, :3]).T @ (Tp[i, :3, :3].T @ Tp[j, :3, :3]))
            for i in range(n) for j in range(i + 1, n)]
    M = sum(Tg[i, :3, :3] @ Tp[i, :3, :3].T for i in range(n))
    U, _, Vt = np.linalg.svd(M)
    A = U @ np.diag([1, 1, np.linalg.det(U @ Vt)]) @ Vt
    per = [ang(Tg[i, :3, :3].T @ A @ Tp[i, :3, :3]) for i in range(n)]
    out = {"rot_err_deg": float(np.mean(pair)), "rot_err_med": float(np.median(pair)),
           "views_ok": int(sum(p < 10 for p in per)), "per_view_deg": [round(p, 1) for p in per]}
    if n >= 3:
        s, R, t = umeyama(Tp[:, :3, 3], Tg[:, :3, 3])
        al = (s * (R @ Tp[:, :3, 3].T)).T + t
        out |= {"ate_m": float(np.sqrt(((al - Tg[:, :3, 3]) ** 2).sum(1).mean())), "sim3_s": float(s)}
    return out


def valid(gt, conf):
    return (conf == 2) & (gt > 0.2) & (gt < 5.0)


def depth_metrics(pred, gt, conf):
    m = valid(gt, conf) & (pred > 0) & np.isfinite(pred)
    p, g = pred[m], gt[m]
    s = np.median(g / p)
    return {"absrel": float(np.mean(np.abs(p * s - g) / g)), "valid_px": int(m.sum())}


def scale_err(pred, gt, conf):
    m = valid(gt, conf) & (pred > 0) & np.isfinite(pred)
    return float(np.median(pred[m] / gt[m]) - 1)


def resize_to(d, shape):
    return np.stack([cv2.resize(x.astype(np.float32), (shape[1], shape[0]), interpolation=cv2.INTER_LINEAR) for x in d])


def sample_through_K(pred, K_proc, K_gt, shape):
    """Map GT pixels into the processed image via the intrinsics (handles crop+resize), sample nearest."""
    h, w = shape
    v, u = np.mgrid[0:h, 0:w]
    rays = np.stack([u + 0.5, v + 0.5, np.ones_like(u, float)], -1).reshape(-1, 3) @ np.linalg.inv(K_gt).T
    out = np.zeros((len(pred), h, w), np.float32)
    for i in range(len(pred)):
        p = rays @ K_proc[i].T
        up, vp = (p[:, 0] / p[:, 2]).astype(int), (p[:, 1] / p[:, 2]).astype(int)
        H, W = pred[i].shape
        ok = (up >= 0) & (up < W) & (vp >= 0) & (vp < H)
        flat = np.zeros(len(p), np.float32)
        flat[ok] = pred[i][vp[ok], up[ok]]
        out[i] = flat.reshape(h, w)
    return out


# ---------------- runners ----------------
def gpu_reset():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()


def peak_gb():
    return torch.cuda.max_memory_allocated() / 2**30


def run_metric_mono(sets, data):
    model = DepthAnything3.from_pretrained(METRIC).to("cuda").eval()
    raw, rows = {}, []
    for s in sets:
        gpu_reset()
        t = time.time()
        chunks = [model.inference(data[s]["images"][i:i + 8], process_res=RES).depth
                  for i in range(0, len(data[s]["images"]), 8)]
        raw[s] = np.concatenate(chunks)
        f_true = data[s]["K"][:, 0, 0] * raw[s].shape[2] / 1440.0
        mono = resize_to(raw[s] * f_true[:, None, None] / 300.0, data[s]["depth"].shape[1:])
        rows.append({"model": "da3metric-large(mono,truef)", "set": s, "n": len(data[s]["images"]),
                     "runtime_s": time.time() - t, "peak_gb": peak_gb(),
                     "scale_truef": scale_err(mono, data[s]["depth"], data[s]["conf"])})
        print(rows[-1], flush=True)
    del model
    return raw, rows


def run_da3(name, repo, win, overlap, sets, data, metric_raw):
    model = DepthAnything3.from_pretrained(repo).to("cuda").eval()
    rows = []
    for s in sets:
        d = data[s]
        row = {"model": name, "set": s, "n": len(d["images"]), "window": win, "overlap": overlap}
        try:
            gpu_reset()
            t = time.time()

            def predict(idx):
                p = model.inference([d["images"][i] for i in idx], process_res=RES)
                E = np.tile(np.eye(4), (len(idx), 1, 1))
                E[:, :3, :4] = p.extrinsics
                return WindowPrediction(np.linalg.inv(E), p.depth, p.conf, p.intrinsics)

            st = run_windowed(len(d["images"]), win, overlap, predict)
            row |= {"runtime_s": time.time() - t, "peak_gb": peak_gb(), "n_windows": len(st.windows)}
            w_proc = st.depth.shape[2]
            f_true = d["K"][:, 0, 0] * w_proc / 1440.0
            f_pred = st.intrinsics[:, 0, 0]
            dep = resize_to(st.depth, d["depth"].shape[1:])
            row["focal_err"] = float(np.mean(f_pred / f_true - 1))
            row |= pose_metrics(st.c2w, d["T"])
            row |= depth_metrics(dep, d["depth"], d["conf"])
            for tag, f in (("predf", f_pred), ("truef", f_true)):
                metric = metric_raw[s] * f[:, None, None] / 300.0
                k = np.median(metric / np.maximum(st.depth, 1e-6))
                row[f"scale_{tag}"] = scale_err(dep * k, d["depth"], d["conf"])
            np.savez_compressed(WORK / "preds" / f"{name}__{s}.npz", depth=st.depth.astype(np.float16),
                                c2w=st.c2w, intrinsics=st.intrinsics)
        except torch.cuda.OutOfMemoryError:
            row["error"] = "OOM"
        except Exception as e:  # noqa: BLE001
            row["error"] = repr(e)
            traceback.print_exc()
        rows.append(row)
        print({k: v for k, v in row.items() if k != "per_view_deg"}, flush=True)
    del model
    return rows


def run_mapanything(sets, data):
    install(f"git+https://github.com/facebookresearch/map-anything@{MA_COMMIT}", "mapanything.models")
    from mapanything.models import MapAnything
    from mapanything.utils.image import preprocess_inputs

    model = MapAnything.from_pretrained("facebook/map-anything-apache").to("cuda").eval()
    rows = []
    for s in sets:
        d = data[s]
        K_proc_withK = None
        for tag in ("withK", "noK"):
            row = {"model": f"mapanything-apache({tag})", "set": s, "n": len(d["images"])}
            try:
                views = []
                for i, p in enumerate(d["images"]):
                    v = {"img": Image.open(p).convert("RGB")}
                    if tag == "withK":
                        v["intrinsics"] = torch.tensor(d["K"][i], dtype=torch.float32)
                    views.append(v)
                gpu_reset()
                t = time.time()
                preds = model.infer(preprocess_inputs(views), memory_efficient_inference=True, use_amp=True,
                                    amp_dtype="bf16", apply_mask=True, mask_edges=True)
                row |= {"runtime_s": time.time() - t, "peak_gb": peak_gb()}
                depth = np.stack([p["depth_z"][0, ..., 0].float().cpu().numpy() for p in preds])
                c2w = np.stack([p["camera_poses"][0].float().cpu().numpy() for p in preds])
                Kp = np.stack([p["intrinsics"][0].float().cpu().numpy() for p in preds])
                if tag == "withK":
                    K_proc_withK = Kp
                K_map = K_proc_withK if K_proc_withK is not None else Kp
                gshape = d["depth"].shape[1:]
                K_gt = d["K"][0].copy()
                K_gt[:2] *= gshape[1] / 1440.0
                dep = sample_through_K(depth, K_map, K_gt, gshape)
                row["focal_err"] = float(np.mean(Kp[:, 0, 0] / K_map[:, 0, 0] - 1)) if tag == "noK" else 0.0
                row |= pose_metrics(c2w, d["T"])
                row |= depth_metrics(dep, d["depth"], d["conf"])
                row["scale_native"] = scale_err(dep, d["depth"], d["conf"])
                np.savez_compressed(WORK / "preds" / f"mapanything-{tag}__{s}.npz", depth=depth.astype(np.float16),
                                    c2w=c2w, intrinsics=Kp)
            except torch.cuda.OutOfMemoryError:
                row["error"] = "OOM"
            except Exception as e:  # noqa: BLE001
                row["error"] = repr(e)
                traceback.print_exc()
            rows.append(row)
            print({k: v for k, v in row.items() if k != "per_view_deg"}, flush=True)
    del model
    return rows


def main():
    bundle = find("index.json").parent
    sets = json.loads((bundle / "index.json").read_text())["sets"]
    print("GPU:", torch.cuda.get_device_name(0), "| sets:", len(sets), flush=True)
    data = {}
    for s in sets:
        g = np.load(bundle / s / "gt.npz")
        data[s] = {"images": [str(bundle / s / "images" / n) for n in g["names"]],
                   "depth": g["depth"], "conf": g["conf"], "K": g["K"], "T": g["T_world_cam"]}
    (WORK / "preds").mkdir(exist_ok=True)

    def save(rows):
        (WORK / "results.json").write_text(json.dumps(
            {"gpu": torch.cuda.get_device_name(0), "da3_commit": DA3_COMMIT, "mapanything_commit": MA_COMMIT,
             "process_res": RES, "overlap": OVERLAP, "rows": rows}, indent=1))

    video_sets = [s for s in sets if s.startswith("video")]
    metric_raw, rows = run_metric_mono(video_sets, data)
    save(rows)
    for name, repo, win, ov in DA3_VARIANTS:
        rows += run_da3(name, repo, win, ov, video_sets, data, metric_raw)
        save(rows)
    try:
        rows += run_mapanything(sets, data)
    except Exception as e:  # noqa: BLE001  (a failed install must not lose the DA3 results)
        rows.append({"model": "mapanything-apache", "error": repr(e)})
        traceback.print_exc()
    save(rows)
    print("DONE", len(rows), "rows")


main()
