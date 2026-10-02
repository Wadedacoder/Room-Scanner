"""E18: why do near-threshold cross-room links fail to place? PnP from both directions for the strongest links."""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from roomscan.config import load_config  # noqa: E402
from roomscan.geometry.cloud import find_levels  # noqa: E402
from roomscan.io.photos import load_room  # noqa: E402
from roomscan.pipeline.backend import PHOTO_ERR, rooms_from_cloud  # noqa: E402
from roomscan.pipeline.photos import VOXEL, reconstruct_room, wall_rays  # noqa: E402
from roomscan.recon.learned import LearnedRecon, voxelize  # noqa: E402
from roomscan.stitch.links import cross_room_links  # noqa: E402
from roomscan.stitch.photo_graph import RoomFrame, _relative_pose  # noqa: E402

capture = sys.argv[1] if len(sys.argv) > 1 else "house_b_hall"
rooms = sys.argv[2:] or ["bathroom", "living", "hall", "study"]
cfg = load_config("lite", overrides=["runtime.gpu_memory_fraction=0.85"])
m = LearnedRecon(cfg)
frames, photos = {}, {}
for r in rooms:
    ph = load_room(ROOT / f"data/raw/photos/{capture}/{r}")
    rec, pts, cams, G = reconstruct_room(m, ph)
    P = voxelize(np.concatenate(pts), VOXEL)
    fl = find_levels(P, float(np.median(cams[:, 1]))).floor
    res = rooms_from_cloud(P, cams, wall_rays(pts, cams, fl), PHOTO_ERR, VOXEL, split=False, id_prefix=r,
                           interior_mode="walls")
    c2w = rec.c2w.copy()
    c2w[:, :3, :3] = G @ rec.c2w[:, :3, :3]
    c2w[:, :3, 3] = rec.c2w[:, :3, 3] @ G.T
    frames[r] = RoomFrame(r, c2w, rec.depth, rec.K, ph[0].image.shape[:2], res.debug["theta"], None)
    photos[r] = [p.image for p in ph]
links = cross_room_links(photos, cfg)


def fmt(x):
    return None if x is None else dict(yaw=round(float(np.degrees(x[0])), 1), t=np.round(x[1], 2).tolist(), pnp=x[2],
                                       tilt=round(x[3], 1))


for L in links[:6]:
    r1 = _relative_pose(frames[L.room_a], L.photo_a, frames[L.room_b], L.photo_b, L.pts_a, L.pts_b)
    r2 = _relative_pose(frames[L.room_b], L.photo_b, frames[L.room_a], L.photo_a, L.pts_b, L.pts_a)
    print(f"{L.room_a}->{L.room_b} matches {L.inliers}: depth-from-{L.room_a} {fmt(r1)} | depth-from-{L.room_b} {fmt(r2)}",
          flush=True)
