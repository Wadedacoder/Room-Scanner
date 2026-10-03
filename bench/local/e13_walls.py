"""E13: why are photo-tier wall sizes off in rooms other than the study? Top-down diagnosis per room.

Uses the reconstructions cached by E12 (runs/e12/<room>.npz, house_b). For each room draws: wall-band points (grey),
detected wall cells (black), camera positions + view directions (red), the interior used for the outline (blue tint)
and the final polygon (orange); and reports per wall how much point support it has (observed vs inferred).

Usage: python bench/local/e13_walls.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "bench/local"))

from e12_ceiling import ROOMS, get_recon

from roomscan.geometry import plan2d as p2
from roomscan.geometry.cloud import find_levels
from roomscan.pipeline.backend import PHOTO_ERR, rooms_from_cloud
from roomscan.pipeline.photos import VOXEL, wall_rays
from roomscan.recon.learned import backproject, gravity_align, voxelize

OUT = ROOT / "runs/e13"


def analyse(room: str, ax):
    rec, photos = get_recon(room)
    pts, cams = backproject(rec)
    G = gravity_align(pts, rec.c2w)
    pts = [P @ G.T for P in pts]
    cams = cams @ G.T
    P = voxelize(np.concatenate(pts), VOXEL)
    floor = find_levels(P, float(np.median(cams[:, 1]))).floor
    res = rooms_from_cloud(P, cams, wall_rays(pts, cams, floor), PHOTO_ERR, VOXEL, split=False, id_prefix=room,
                           interior_mode="walls")
    d = res.debug
    th, maps = d["theta"], d["maps"]
    R = p2.rot2(-th)
    g = maps.grid
    ext = [g.origin[0], g.origin[0] + g.shape[1] * g.res, g.origin[1], g.origin[1] + g.shape[0] * g.res]
    h = P[:, 1] - floor
    band = P[(h > 0.3) & (h < 1.8)][:, [0, 2]] @ R.T
    ax.scatter(band[:, 0], band[:, 1], s=0.2, c="0.75")
    ax.imshow(np.ma.masked_where(~maps.interior, maps.interior), origin="lower", extent=ext, cmap="Blues", alpha=0.35,
              vmin=0, vmax=2)
    ax.imshow(np.ma.masked_where(~maps.walls, maps.walls), origin="lower", extent=ext, cmap="Greys", vmin=0, vmax=1)
    c = cams[:, [0, 2]] @ R.T
    fwd = (rec.c2w[:, :3, 2] @ G.T)[:, [0, 2]] @ R.T
    for i in range(len(c)):
        ax.arrow(c[i, 0], c[i, 1], fwd[i, 0] * 0.6, fwd[i, 1] * 0.6, color="r", width=0.02)
    info = {"room": room, "photos": len(photos), "walls": []}
    if res.rooms:
        poly = np.array(res.rooms[0]["polygon"])
        loop = np.vstack([poly, poly[:1]])
        ax.plot(loop[:, 0], loop[:, 1], "-", color="darkorange", lw=2)
        geo = p2.room_geometry(res.debug["masks"][0], g, band)
        for w, f in zip(res.rooms[0]["walls"], geo.walls):
            info["walls"].append({"len": round(w["length"]["value"], 2), "support_pts": f.n})
        info["area"] = res.rooms[0]["floor_area"]["value"]
        info["bbox"] = np.round(np.sort(np.ptp(poly, 0)), 2).tolist()
    info["camera_spread_m"] = round(float(np.ptp(c, 0).max()), 2)
    info["band_extent_m"] = np.round(np.ptp(band, 0), 2).tolist() if len(band) else None
    ax.set_title(f"{room}: {info.get('bbox')}", fontsize=9)
    ax.set_aspect("equal")
    return info


def main():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    OUT.mkdir(parents=True, exist_ok=True)
    fig, axs = plt.subplots(2, 3, figsize=(15, 10))
    report = []
    for room, ax in zip(ROOMS, axs.ravel()):
        info = analyse(room, ax)
        report.append(info)
        print(json.dumps(info))
    fig.tight_layout()
    fig.savefig(OUT / "rooms_topdown.png", dpi=70)
    (OUT / "report.json").write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
