"""Cross-room photo links: DISK features + LightGlue matching (kornia), verified by a fundamental-matrix RANSAC.

E14: SIFT cannot link 0.5x ring photos (same-room median 18 verified matches); DISK+LightGlue finds the real
cross-room links (kitchen<->living 89, through a doorway photo) while unrelated pairs stay at 14-38.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class Link:
    room_a: str
    photo_a: int  # index within room A's photo list
    room_b: str
    photo_b: int
    pts_a: np.ndarray  # (n,2) pixel coords in the ORIGINAL photo A
    pts_b: np.ndarray
    inliers: int


def _verify(pa, pb):
    if len(pa) < 12:
        return np.zeros(len(pa), bool)
    F, mask = cv2.findFundamentalMat(pa, pb, cv2.USAC_MAGSAC, 1.5, 0.999, 10000)
    return mask.ravel().astype(bool) if mask is not None else np.zeros(len(pa), bool)


def cross_room_links(rooms: dict[str, list[np.ndarray]], cfg, long_side: int = 1024,
                     n_feat: int = 2048) -> list[Link]:
    """Checkpointed wrapper: DISK + LightGlue on the Mac GPU is not bit-reproducible (47 vs 42 matches for one pair,
    E18), so the same photos must replay the same links (roomscan.recon.checkpoint)."""
    import json

    from roomscan.recon import checkpoint as ck

    names = sorted(rooms)
    matcher = cfg["recon"].get("photo_matcher", "disk_lightglue")
    meta = {"rooms": names, "res": long_side, "n": n_feat} | ({"matcher": matcher} if matcher != "disk_lightglue" else {})
    key = ck.key_for("links", [im for n in names for im in rooms[n]], meta)
    path = ck.path_for(cfg["runtime"]["cache_dir"], "links", key, ".json")
    if ck.enabled() and path.exists():
        return [Link(d["a"], d["pa"], d["b"], d["pb"], np.array(d["xa"], np.float32).reshape(-1, 2),
                     np.array(d["xb"], np.float32).reshape(-1, 2), d["n"]) for d in json.loads(path.read_text())]
    links = (_cross_room_links_mast3r(rooms, cfg) if matcher == "mast3r"
             else _cross_room_links_live(rooms, cfg, long_side, n_feat))
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps([{"a": L.room_a, "pa": L.photo_a, "b": L.room_b, "pb": L.photo_b,
                                "xa": L.pts_a.round(2).tolist(), "xb": L.pts_b.round(2).tolist(), "n": L.inliers}
                               for L in links]))
    tmp.replace(path)
    return links


def _cross_room_links_live(rooms: dict[str, list[np.ndarray]], cfg, long_side: int = 1024,
                           n_feat: int = 2048) -> list[Link]:
    """rooms: name -> list of RGB photos (original resolution). Returns verified matches for every cross-room photo
    pair, strongest first."""
    import kornia.feature as KF
    import torch

    dev = cfg["runtime"]["device"]
    dev = dev if (dev != "mps" or torch.backends.mps.is_available()) else "cpu"
    frac = float(cfg["runtime"].get("gpu_memory_fraction", 0.0) or 0.0)
    if dev == "mps" and frac > 0:
        torch.mps.set_per_process_memory_fraction(frac)
    torch.set_num_threads(4)
    cv2.setNumThreads(1)  # torch + OpenCV thread pools in one process stalled (E14)
    disk = KF.DISK.from_pretrained("depth").to(dev).eval()
    lg = KF.LightGlueMatcher("disk").to(dev).eval()
    items, feats = [], []
    with torch.inference_mode():
        for name, photos in rooms.items():
            for k, im in enumerate(photos):
                s = long_side / max(im.shape[:2])
                small = cv2.resize(im, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
                t = torch.from_numpy(small).permute(2, 0, 1).float()[None].to(dev) / 255.0
                f = disk(t, n=n_feat, pad_if_not_divisible=True)[0]
                items.append((name, k, s))
                feats.append((f.keypoints, f.descriptors, t.shape[-2:]))
        links = []
        for i, j in itertools.combinations(range(len(items)), 2):
            if items[i][0] == items[j][0]:
                continue
            ki, di, si = feats[i]
            kj, dj, sj = feats[j]
            lafi = KF.laf_from_center_scale_ori(ki[None], torch.ones(1, len(ki), 1, 1, device=dev))
            lafj = KF.laf_from_center_scale_ori(kj[None], torch.ones(1, len(kj), 1, 1, device=dev))
            _, idx = lg(di, dj, lafi, lafj, hw1=si, hw2=sj)
            idx = idx.cpu().numpy()
            pa = ki.cpu().numpy()[idx[:, 0]].astype(np.float32)
            pb = kj.cpu().numpy()[idx[:, 1]].astype(np.float32)
            m = _verify(pa, pb)
            if m.sum() == 0:
                continue
            (na, ka, sa), (nb, kb, sb) = items[i], items[j]
            links.append(Link(na, ka, nb, kb, pa[m] / sa, pb[m] / sb, int(m.sum())))
    del disk, lg
    if dev == "mps":
        torch.mps.empty_cache()
    return sorted(links, key=lambda x: -x.inliers)


def _cross_room_links_mast3r(rooms: dict[str, list[np.ndarray]], cfg, size: int = 512) -> list[Link]:
    """MASt3R dense matching for every cross-room pair (E31: 5 of 7 house rooms linked vs 3 with DISK). Needs the
    naver/mast3r repo (path in $MAST3R_REPO, default ~/mast3r) and a GPU to be practical; licence CC BY-NC-SA 4.0
    (non-commercial). Matches are returned in the pixel frame of the given images, like the DISK path."""
    import itertools
    import os
    import sys
    import tempfile

    import torch
    from PIL import Image

    repo = os.path.expanduser(os.environ.get("MAST3R_REPO", "~/mast3r"))
    for p in (repo, os.path.join(repo, "dust3r")):
        if p not in sys.path:
            sys.path.insert(0, p)
    from dust3r.inference import inference
    from dust3r.utils.image import load_images
    from mast3r.fast_nn import fast_reciprocal_NNs
    from mast3r.model import AsymmetricMASt3R

    dev = cfg["runtime"]["device"]
    dev = dev if (dev != "mps" or torch.backends.mps.is_available()) else "cpu"
    model = AsymmetricMASt3R.from_pretrained("naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric").to(dev).eval()
    items = [(name, k, im) for name, ims in rooms.items() for k, im in enumerate(ims)]
    links = []
    with tempfile.TemporaryDirectory() as td:
        paths = []
        for i, (_, _, im) in enumerate(items):
            p = os.path.join(td, f"{i}.jpg")
            Image.fromarray(im).save(p, quality=92)
            paths.append(p)

        def to_img(m, wh, shape):  # load_images: resize long side to `size`, centre-crop to a multiple of 16
            w, h = wh
            s = size / max(w, h)
            ox, oy = (round(w * s) - shape[1]) / 2, (round(h * s) - shape[0]) / 2
            return np.stack([(m[:, 0] + ox) / s, (m[:, 1] + oy) / s], 1).astype(np.float32)

        for i, j in itertools.combinations(range(len(items)), 2):
            (ra, ka, ia), (rb, kb, ib) = items[i], items[j]
            if ra == rb:
                continue
            imgs = load_images([paths[i], paths[j]], size=size, verbose=False)
            with torch.no_grad():
                out = inference([tuple(imgs)], model, dev, batch_size=1, verbose=False)
            d1, d2 = out["pred1"]["desc"].squeeze(0).detach(), out["pred2"]["desc"].squeeze(0).detach()
            m1, m2 = fast_reciprocal_NNs(d1, d2, subsample_or_initxy1=8, device=dev, dist="dot", block_size=2**13)
            (h1, w1), (h2, w2) = (tuple(int(v) for v in im["true_shape"][0]) for im in imgs)
            ok = ((m1[:, 0] >= 3) & (m1[:, 0] < w1 - 3) & (m1[:, 1] >= 3) & (m1[:, 1] < h1 - 3)
                  & (m2[:, 0] >= 3) & (m2[:, 0] < w2 - 3) & (m2[:, 1] >= 3) & (m2[:, 1] < h2 - 3))
            pa = to_img(m1[ok], (ia.shape[1], ia.shape[0]), (h1, w1))
            pb = to_img(m2[ok], (ib.shape[1], ib.shape[0]), (h2, w2))
            m = _verify(pa, pb)
            if m.sum():
                links.append(Link(ra, ka, rb, kb, pa[m], pb[m], int(m.sum())))
    del model
    if dev == "cuda":
        torch.cuda.empty_cache()
    elif dev == "mps":
        torch.mps.empty_cache()
    return sorted(links, key=lambda x: -x.inliers)
