"""E31 (local half): place photo rooms using MASt3R cross-room matches computed on Kaggle (bench/kaggle/mast3r_links).

    python bench/local/e31_mast3r_place.py data/raw/photos/house_b_hall runs/e31_mast3r [--with-disk]

The matches replace (or, with --with-disk, are added to) the DISK+LightGlue links; placement is the pipeline's own:
4-DoF gravity-aware PnP per link + two-way yaw/offset agreement (E28). Writes plan.json/plan.svg next to the matches.
"""

import json
import sys
from pathlib import Path

import numpy as np

import roomscan.stitch.links as links_mod
from roomscan.config import load_config
from roomscan.pipeline.photos import run_photos
from roomscan.render.svg import render_svg
from roomscan.stitch.links import Link

cap, run_dir = Path(sys.argv[1]), Path(sys.argv[2])
with_disk = "--with-disk" in sys.argv
arr = np.load(run_dir / "links.npz")
mast3r = []
for key in arr.files:
    a, b = key.split("|")
    (ra, pa), (rb, pb) = a.split("#"), b.split("#")
    m = arr[key]
    mast3r.append(Link(ra, int(pa), rb, int(pb), m[:, :2].astype(np.float32), m[:, 2:].astype(np.float32), len(m)))
mast3r.sort(key=lambda L: -L.inliers)
orig = links_mod.cross_room_links


def patched(room_photos, cfg):
    return sorted(mast3r + (orig(room_photos, cfg) if with_disk else []), key=lambda L: -L.inliers)


links_mod.cross_room_links = patched
cfg = load_config(profile="lite")
cfg["damage"]["backend"] = "off"
plan, dbg = run_photos(cap, cfg)
tag = "mast3r+disk" if with_disk else "mast3r"
(run_dir / f"plan_{tag}.json").write_text(json.dumps(plan, indent=1))
(run_dir / f"plan_{tag}.svg").write_text(render_svg(plan))
print(tag, "adjacency:", [(e["a"], e["b"], e["via"][:40]) for e in plan["adjacency"]])
print([w for w in plan["warnings"] if "connected" in w or "linked" in w])
print("edges:", json.dumps(dbg.get("edges", []), default=str)[:1500])
