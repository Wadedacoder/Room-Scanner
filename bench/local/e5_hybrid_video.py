"""E5: hybrid video poses = accurate COLMAP pieces (E4) joined by DA3 across the gaps where tracking broke.

For each gap between consecutive pieces A -> B, DA3 sees 4 frames (lite profile cap): two from the end of A and two
from the start of B, spaced apart so their baselines fix scale. A similarity transform maps B's frame into A's using
those shared cameras. Frames inside gaps get poses interpolated between their neighbours. Everything is scored on
all frames against ARKit, like E1-E4. Runs on the M1 laptop (DA3-Base on MPS).

Usage: python bench/local/e5_hybrid_video.py video480 [--device mps]
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))

BUNDLE = ROOT / "data/derived/eval_bundle"


# pycolmap and torch each ship their own libomp; loading both in one process aborts (OMP Error #15). The
# KMP_DUPLICATE_LIB_OK workaround can silently corrupt results, so COLMAP pieces are exported in a child process.
_EXPORT = r"""
import json, os, sys, numpy as np, pycolmap
sp, names = sys.argv[1], json.loads(sys.argv[2])
out = []
for m in sorted(os.listdir(sp)):
    rec = pycolmap.Reconstruction(os.path.join(sp, m)); p = {}
    for img in rec.images.values():
        if img.has_pose:
            cfw = img.cam_from_world() if callable(img.cam_from_world) else img.cam_from_world
            T = np.eye(4); T[:3, :4] = cfw.matrix(); p[names.index(img.name)] = np.linalg.inv(T).tolist()
    out.append(p)
print(json.dumps(out))
"""


def load_pieces(set_name: str, names: list[str], min_frames: int = 10) -> list[dict[int, np.ndarray]]:
    sp = ROOT / f"runs/e4_colmap/{set_name}/sparse"
    raw = subprocess.run([sys.executable, "-c", _EXPORT, str(sp), json.dumps(names)], capture_output=True,
                         text=True, check=True).stdout
    pieces = [{int(k): np.array(v) for k, v in p.items()} for p in json.loads(raw)]
    pieces = [p for p in pieces if len(p) >= min_frames]
    pieces.sort(key=lambda p: min(p))
    # drop pieces fully contained in an earlier, longer one (COLMAP sometimes re-registers a stretch)
    kept = []
    for p in pieces:
        if kept and max(p) <= max(kept[-1]):
            continue
        kept.append(p)
    return kept


def sim3_from_cams(src: list[np.ndarray], dst: list[np.ndarray]):
    """(s, R, t) with dst = s R src + t, from >= 2 camera poses (c2w) known in both frames."""
    R = sum(d[:3, :3] @ s_[:3, :3].T for s_, d in zip(src, dst))
    U, _, Vt = np.linalg.svd(R)
    R = U @ np.diag([1, 1, np.linalg.det(U @ Vt)]) @ Vt
    cs, cd = np.stack([x[:3, 3] for x in src]), np.stack([x[:3, 3] for x in dst])
    # scale from the spread of all shared camera centres (more stable than one baseline)
    s = float(np.sqrt(((cd - cd.mean(0)) ** 2).sum() / max(((cs - cs.mean(0)) ** 2).sum(), 1e-12)))
    t = cd.mean(0) - s * R @ cs.mean(0)
    return s, R, t


def apply(T: np.ndarray, s, R, t) -> np.ndarray:
    out = np.eye(4)
    out[:3, :3] = R @ T[:3, :3]
    out[:3, 3] = s * R @ T[:3, 3] + t
    return out


def interpolate(poses: dict[int, np.ndarray], n: int) -> np.ndarray:
    known = sorted(poses)
    out = np.zeros((n, 4, 4))
    rots = Rotation.from_matrix(np.stack([poses[k][:3, :3] for k in known]))
    slerp = Slerp(known, rots)
    for i in range(n):
        j = min(max(i, known[0]), known[-1])
        T = np.eye(4)
        T[:3, :3] = slerp([j]).as_matrix()[0]
        T[:3, 3] = [np.interp(j, known, [poses[k][a, 3] for k in known]) for a in range(3)]
        out[i] = T
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("set", nargs="?", default="video480")
    ap.add_argument("--device", default="mps")
    ap.add_argument("--spacing", type=int, default=6, help="frames between the two cameras taken from each piece")
    ap.add_argument("--mode", choices=["direct", "through"], default="through",
                    help="direct: DA3 sees 2+2 frames across the gap; through: DA3 chains every frame through the gap")
    args = ap.parse_args()

    import torch

    sys.path.insert(0, str(ROOT))
    from roomscan.recon.da3_loader import load_da3_class
    from roomscan.recon.windows import WindowPrediction, run_windowed

    DepthAnything3 = load_da3_class()

    from e4_score import score

    d = BUNDLE / args.set
    g = np.load(d / "gt.npz")
    names = [str(n) for n in g["names"]]
    imgs = [str(d / "images" / n) for n in names]
    pieces = load_pieces(args.set, names)
    print(f"{len(pieces)} pieces:", [(min(p), max(p), len(p)) for p in pieces])

    t0 = time.time()
    model = DepthAnything3.from_pretrained("depth-anything/DA3-BASE").to(args.device).eval()
    t_load = time.time() - t0

    world = dict(pieces[0])  # global frame = first piece's frame
    links = []
    t1 = time.time()
    for A, B in zip(pieces, pieces[1:]):
        a = sorted(i for i in A if i in world)  # A already expressed in the global frame
        b = sorted(B)
        fa = [a[-1 - args.spacing], a[-1]]
        fb = [b[0], b[args.spacing]]
        if args.mode == "direct":
            idx = fa + fb
            with torch.inference_mode():
                pred = model.inference([imgs[i] for i in idx], process_res=504)
            E = np.tile(np.eye(4), (4, 1, 1))
            E[:, :3, :4] = pred.extrinsics
            Wd = dict(zip(idx, np.linalg.inv(E)))
        else:
            # every frame from A's 2nd anchor to B's 2nd anchor, chained in windows of 4 consecutive frames (lite cap)
            idx = list(range(fa[0], fb[1] + 1)) if fb[1] > fa[0] else sorted(set(fa + fb))

            def predict(w, idx=idx):
                with torch.inference_mode():
                    p = model.inference([imgs[idx[j]] for j in w], process_res=504)
                E = np.tile(np.eye(4), (len(w), 1, 1))
                E[:, :3, :4] = p.extrinsics
                return WindowPrediction(np.linalg.inv(E), p.depth, p.conf, p.intrinsics)

            st = run_windowed(len(idx), 4, 2, predict)
            Wd = dict(zip(idx, st.c2w))
        shA = [i for i in Wd if i in world and i in A]  # every bridge frame that piece A also posed
        shB = [i for i in Wd if i in B]
        sA = sim3_from_cams([Wd[i] for i in shA], [world[i] for i in shA])  # window -> global
        sB = sim3_from_cams([B[i] for i in shB], [Wd[i] for i in shB])  # piece B -> window
        for i, T in B.items():
            if i not in world:
                world[i] = apply(apply(T, *sB), *sA)
        if args.mode == "through":  # gap frames get the bridge's own poses instead of interpolation
            for i in idx:
                if i not in world:
                    world[i] = apply(Wd[i], *sA)
        links.append({"gap": [max(A), min(B)], "n_views": len(idx), "scale_B_to_window": round(float(sB[0]), 4)})
    t_bridge = time.time() - t1

    n = len(names)
    Tp = interpolate(world, n)
    covered = np.array([i in world for i in range(n)])
    full = score(Tp, g["T_world_cam"])
    measured = score(Tp[covered], g["T_world_cam"][covered])
    res = {"set": args.set, "frames": n, "pieces": len(pieces), "posed_by_colmap_or_bridge": int(covered.sum()),
           "all_frames": full, "posed_frames_only": measured, "links": links,
           "runtime_s": {"model_load": round(t_load, 1), "bridging": round(t_bridge, 1)}, "device": args.device}
    print(json.dumps(res, indent=1))
    out = ROOT / "runs/e5_hybrid"
    out.mkdir(parents=True, exist_ok=True)
    res["mode"] = args.mode
    (out / f"{args.set}_{args.mode}.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
