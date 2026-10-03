"""E31: does MASt3R find cross-room links that DISK+LightGlue misses? (photo stitching)

For every cross-room photo pair of one capture: MASt3R (ViT-L, 512, metric; CC BY-NC-SA 4.0) dense matching ->
reciprocal nearest neighbours -> fundamental-matrix MAGSAC verification (same test as roomscan/stitch/links.py).
Writes links.json (counts) and links.npz (verified matched pixels per pair, original-photo coordinates) so the
pipeline's own placement (4-DoF PnP + two-way agreement, E28) can be run on them locally.
"""

import glob
import itertools
import json
import os
import subprocess
import sys
import tarfile
import time
from pathlib import Path

WORK = Path("/kaggle/working")
CAPTURE = os.environ.get("CAPTURE", "house_b_hall")


def sh(c):
    print("+", c, flush=True)
    subprocess.run(c, shell=True, check=True)


for arc in glob.glob("/kaggle/input/**/*.tar", recursive=True):
    with tarfile.open(arc) as t:
        t.extractall(WORK / "extracted")
sh("git clone -q --recursive https://github.com/naver/mast3r /tmp/mast3r")
sh("pip install -q roma einops 'huggingface_hub<1' pillow-heif")
sys.path.insert(0, "/tmp/mast3r")
sys.path.insert(0, "/tmp/mast3r/dust3r")

import cv2
import numpy as np
import pillow_heif
import torch
from PIL import Image, ImageOps

pillow_heif.register_heif_opener()
from mast3r.fast_nn import fast_reciprocal_NNs
from mast3r.model import AsymmetricMASt3R
from dust3r.inference import inference
from dust3r.utils.image import load_images

cap = Path(next(h for root in ("/kaggle/input", str(WORK / "extracted"))
                for h in sorted(glob.glob(f"{root}/**/captures/{CAPTURE}", recursive=True))))
rooms = sorted(d for d in cap.iterdir() if d.is_dir())
jpg = WORK / "jpg"
jpg.mkdir(exist_ok=True)
items = []  # (room, index, path, original (w, h))
for r in rooms:
    for k, p in enumerate(sorted(q for q in r.iterdir() if not q.name.startswith("."))):
        im = ImageOps.exif_transpose(Image.open(p)).convert("RGB")
        out = jpg / f"{r.name}_{k}.jpg"
        im.save(out, quality=92)
        items.append((r.name, k, str(out), im.size))
print(len(items), "photos", flush=True)
model = AsymmetricMASt3R.from_pretrained("naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric").to("cuda").eval()
res, arrays = [], {}
t0 = time.time()
for i, j in itertools.combinations(range(len(items)), 2):
    (ra, ka, pa, sa), (rb, kb, pb, sb) = items[i], items[j]
    if ra == rb:
        continue
    imgs = load_images([pa, pb], size=512, verbose=False)
    with torch.no_grad():
        out = inference([tuple(imgs)], model, "cuda", batch_size=1, verbose=False)
    d1, d2 = out["pred1"]["desc"].squeeze(0).detach(), out["pred2"]["desc"].squeeze(0).detach()
    m1, m2 = fast_reciprocal_NNs(d1, d2, subsample_or_initxy1=8, device="cuda", dist="dot", block_size=2**13)
    H1, W1 = imgs[0]["true_shape"][0]
    H2, W2 = imgs[1]["true_shape"][0]
    ok = (m1[:, 0] >= 3) & (m1[:, 0] < W1 - 3) & (m1[:, 1] >= 3) & (m1[:, 1] < H1 - 3) & \
         (m2[:, 0] >= 3) & (m2[:, 0] < W2 - 3) & (m2[:, 1] >= 3) & (m2[:, 1] < H2 - 3)
    m1, m2 = m1[ok].astype(np.float32), m2[ok].astype(np.float32)
    # back to original-photo pixels (load_images resizes to 512 long side then centre-crops to a multiple of 16)
    def to_orig(m, size, shape):
        w, h = size
        s = 512 / max(w, h)
        rw, rh = round(w * s), round(h * s)
        ox, oy = (rw - shape[1]) / 2, (rh - shape[0]) / 2
        return np.stack([(m[:, 0] + ox) / s, (m[:, 1] + oy) / s], 1)
    p1 = to_orig(m1, sa, (int(H1), int(W1)))
    p2 = to_orig(m2, sb, (int(H2), int(W2)))
    n_ver = 0
    if len(p1) >= 12:
        F, mask = cv2.findFundamentalMat(p1, p2, cv2.USAC_MAGSAC, 1.5, 0.999, 10000)
        if mask is not None:
            v = mask.ravel().astype(bool)
            n_ver = int(v.sum())
            arrays[f"{ra}#{ka}|{rb}#{kb}"] = np.concatenate([p1[v], p2[v]], 1)
    res.append({"a": ra, "pa": ka, "b": rb, "pb": kb, "raw": int(len(p1)), "verified": n_ver})
print(f"{len(res)} cross-room pairs in {time.time() - t0:.0f} s", flush=True)
res.sort(key=lambda r: -r["verified"])
for r in res[:30]:
    print(r, flush=True)
(WORK / "links.json").write_text(json.dumps(res, indent=1))
np.savez_compressed(WORK / "links.npz", **arrays)
