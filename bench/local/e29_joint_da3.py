"""E29: can one multi-view model place every room at once? All photos of a house in ONE DA3 pass (no per-room
reconstruction, no stitching), then a top-down view coloured by room.

    python bench/local/e29_joint_da3.py data/raw/photos/house_b_hall runs/e29 [res]

Runs locally (the home photos stay on this machine): process resolution lowered so all views fit one pass.
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from roomscan.config import load_config
from roomscan.io.photos import load_capture
from roomscan.recon.learned import LearnedRecon, backproject, gravity_align

cap_dir, out = Path(sys.argv[1]), Path(sys.argv[2])
res = int(sys.argv[3]) if len(sys.argv) > 3 else 336
out.mkdir(parents=True, exist_ok=True)
cap = load_capture(cap_dir)
names, images, Ks, room_of = [], [], [], []
for room, photos in cap.items():
    for p in photos:
        names.append(p.path.name); images.append(p.image); Ks.append(p.K); room_of.append(room)
n = len(images)
cfg = load_config(profile="lite", overrides=[f"recon.max_views={n}", f"recon.image_long_side={res}",
                                              "runtime.gpu_memory_fraction=0.85"])
rec = LearnedRecon(cfg).reconstruct(images, np.stack(Ks))
pts, cams = backproject(rec, conf_pct=30)
G = gravity_align(pts, rec.c2w)
pts = [P @ G.T for P in pts]
cams = cams @ G.T
rooms = list(cap)
colors = plt.cm.tab10(np.arange(len(rooms)))
fig, ax = plt.subplots(figsize=(10, 10))
for i, P in enumerate(pts):
    c = colors[rooms.index(room_of[i])]
    # wall band only (0.3-1.8 m above the floor estimate) so the outline reads clearly
    h = P[:, 1]
    floor = np.percentile(np.concatenate([q[:, 1] for q in pts]), 2)
    band = (h - floor > 0.3) & (h - floor < 1.8)
    Q = P[band][:: 4]
    ax.scatter(Q[:, 0], Q[:, 2], s=0.2, color=c, alpha=0.4)
for r_i, r in enumerate(rooms):
    idx = [i for i in range(n) if room_of[i] == r]
    ax.scatter(cams[idx, 0], cams[idx, 2], s=60, color=colors[r_i], edgecolor="k", label=r, zorder=5)
ax.set_aspect("equal"); ax.legend(); ax.set_title(f"{cap_dir.name}: all {n} photos in one DA3 pass (res {res})")
fig.savefig(out / f"{cap_dir.name}_joint.png", dpi=110)
np.savez(out / f"{cap_dir.name}_joint.npz", cams=cams, room_of=np.array(room_of), scale=rec.scale)
print(f"{n} views, metric scale {rec.scale:.3f}; camera spread per room (m):",
      {r: round(float(np.ptp(cams[[i for i in range(n) if room_of[i] == r]][:, [0, 2]], axis=0).max()), 2) for r in rooms})
