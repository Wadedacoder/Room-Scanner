"""DISK keypoints + LightGlue matches for sequential video frames, handed to COLMAP in place of SIFT (backlog 2.5).

E22: COLMAP breaks a handheld walk at fast turns because blurred frames get too few SIFT matches; loosening its
registration thresholds admitted wrong poses. E14 showed DISK + LightGlue matching where SIFT can't (0.5x photos:
43 vs 18 median matches). This runs in the parent (torch) process and writes an .npz the COLMAP child imports
(torch and pycolmap can't share a process: OpenMP clash).

GPU matching is not bit-reproducible (E18), so the result is checkpointed by input hash: the same frames replay the
same matches and therefore the same COLMAP solve.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def sequential_matches(frames_dir: Path, names: list[str], overlap: int, cfg, out: Path, n_feat: int = 2048,
                       long_side: int = 640) -> Path:
    import cv2
    import kornia.feature as KF
    import torch
    from PIL import Image

    if out.exists():
        return out
    dev = cfg["runtime"]["device"]
    dev = dev if (dev != "mps" or torch.backends.mps.is_available()) else "cpu"
    frac = float(cfg["runtime"].get("gpu_memory_fraction", 0.0) or 0.0)
    if dev == "mps" and frac > 0:
        torch.mps.set_per_process_memory_fraction(frac)
    torch.set_num_threads(4)
    cv2.setNumThreads(1)
    disk = KF.DISK.from_pretrained("depth").to(dev).eval()
    lg = KF.LightGlueMatcher("disk").to(dev).eval()
    kps, feats = [], []
    with torch.inference_mode():
        # features live on the CPU; only the pair being matched is on the GPU (E23: all 272 frames' features on MPS
        # pushed the run past the 5.5 GB watchdog limit)
        for k, n in enumerate(names):
            im = Image.open(frames_dir / n).convert("RGB")
            sc = min(1.0, long_side / max(im.size))  # E23: DISK at 960 px peaked past 5.5 GB on the 8 GB Mac
            if sc < 1.0:
                im = im.resize((round(im.width * sc), round(im.height * sc)), Image.LANCZOS)
            t = torch.from_numpy(np.array(im)).permute(2, 0, 1).float()[None].to(dev) / 255.0
            f = disk(t, n=n_feat, pad_if_not_divisible=True)[0]
            kp = f.keypoints.cpu()
            kps.append((kp.numpy() / sc).astype(np.float32) + 0.5)  # frame pixels; COLMAP centres at +0.5
            feats.append((kp, f.descriptors.cpu(), tuple(t.shape[-2:])))
            del t, f
            if dev == "mps" and k % 20 == 19:
                torch.mps.empty_cache()

        def on_dev(i):
            kp, desc, hw = feats[i]
            kp = kp.to(dev)
            return desc.to(dev), KF.laf_from_center_scale_ori(kp[None], torch.ones(1, len(kp), 1, 1, device=dev)), hw

        pairs, matches = [], []
        for i in range(len(names)):
            di, li, si = on_dev(i)
            for j in range(i + 1, min(len(names), i + 1 + overlap)):
                dj, lj, sj = on_dev(j)
                _, idx = lg(di, dj, li, lj, hw1=si, hw2=sj)
                idx = idx.cpu().numpy().astype(np.uint32)
                if len(idx) >= 15:
                    pairs.append((i, j))
                    matches.append(idx)
            if dev == "mps":  # E23: each LightGlue call leaves ~90 MB in the MPS cache; clearing every 200 hit 5.5 GB
                torch.mps.empty_cache()
    feats.clear()
    del disk, lg
    if dev == "mps":
        torch.mps.empty_cache()
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, names=np.array(names), pairs=np.array(pairs, np.int32).reshape(-1, 2),
                        n_kp=np.array([len(k) for k in kps]), kp=np.concatenate(kps),
                        n_m=np.array([len(m) for m in matches]),
                        m=np.concatenate(matches) if matches else np.zeros((0, 2), np.uint32))
    tmp.replace(out)
    return out


def import_into_colmap(db_path: Path, npz_path: Path) -> None:
    """Child process: replace the database's SIFT keypoints/matches with the learned ones, then verify geometrically."""
    import pycolmap

    d = np.load(npz_path)
    names = [str(n) for n in d["names"]]
    kp_split = np.split(d["kp"], np.cumsum(d["n_kp"])[:-1])
    m_split = np.split(d["m"], np.cumsum(d["n_m"])[:-1]) if len(d["n_m"]) else []
    db = pycolmap.Database.open(str(db_path))
    ids = {img.name: img.image_id for img in db.read_all_images()}
    db.clear_matches()
    db.clear_two_view_geometries()
    db.clear_descriptors()
    db.clear_keypoints()
    for n, k in zip(names, kp_split):
        db.write_keypoints(ids[n], np.ascontiguousarray(k, np.float32))
    for (i, j), m in zip(d["pairs"], m_split):
        db.write_matches(ids[names[i]], ids[names[j]], np.ascontiguousarray(m, np.uint32))
    db.close()
    pycolmap.geometric_verification(str(db_path))
