"""E34: why do the video rooms not match the photo rooms of the same flat?

Runs the video tier on house_b_walk.MOV (COLMAP checkpointed) and plots, top-down: the dense cloud's wall band
coloured by COLMAP piece, the keyframe cameras by piece, the plan rooms, and the photo-tier rooms for scale.
"""
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from roomscan.config import load_config
from roomscan.pipeline.video import run_video

vid = Path(sys.argv[1] if len(sys.argv) > 1 else "data/raw/video/house_b_walk.MOV")
out = Path("runs/e34"); out.mkdir(parents=True, exist_ok=True)
cfg = load_config(profile="lite"); cfg["damage"]["backend"] = "off"
plan, dbg = run_video(vid, cfg)
(out / "plan.json").write_text(json.dumps(plan, indent=1))
P, floor, cams = dbg["cloud"], dbg["floor"], dbg["cam_xyz"]
kp = np.array(dbg["key_piece"])
np.savez(out / "video_debug.npz", cloud=P, floor=floor, cams=cams, key_piece=kp)
h = P[:, 1] - floor
band = P[(h > 0.4) & (h < 1.8)][::3]
fig, ax = plt.subplots(figsize=(12, 12))
ax.scatter(band[:, 0], band[:, 2], s=0.15, c="0.6", alpha=0.4)
cm = plt.cm.tab20(np.arange(20))
for pi in sorted(set(kp)):
    m = kp == pi
    ax.plot(cams[m, 0], cams[m, 2], "-o", ms=3, color=cm[pi % 20], label=f"piece {pi} ({m.sum()} kf)")
th = dbg["backend"]["theta"]
from roomscan.geometry import plan2d as p2
for r in plan["rooms"]:
    xz = np.array(r["polygon"]) @ p2.rot2(-th)
    ax.plot(*np.vstack([xz, xz[:1]]).T, "k-", lw=2)
    ax.text(*xz.mean(0), f"{r['id']} {r['floor_area']['value']:.1f} m²", fontsize=11, weight="bold")
ax.set_aspect("equal"); ax.legend(fontsize=7, loc="upper right")
ax.set_title(f"{vid.name}: wall band (grey), keyframe cameras by COLMAP piece, plan rooms (black)")
fig.savefig(out / "video_topdown.png", dpi=100)
print("pieces:", dbg["piece_scales"][:20])
print("join log:", [(g.get("piece_frames"), g.get("joined"), g.get("chain_scale")) for g in dbg["join_log"]])
print("rooms:", [(r["id"], r["floor_area"]["value"]) for r in plan["rooms"]])
