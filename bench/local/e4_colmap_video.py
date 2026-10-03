"""E4: classical SfM (COLMAP) for video-tier camera poses, scored against ARKit ground truth.

E2/E3 showed learned multi-view models chained over windows lose most video frames. A walkthrough video is the case
COLMAP's sequential matcher was built for, and it runs on a laptop CPU. Lens: the true focal (stand-in for the
photo's EXIF focal), held fixed. Scale is not COLMAP's job (DA3Metric + focal gives it, E1-E3); poses are scored
after a similarity alignment, exactly like the Kaggle runs.

Usage: python bench/local/e4_colmap_video.py video60 video120 [--max-size 1024]
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pycolmap

sys.path.insert(0, str(Path(__file__).resolve().parent))

ROOT = Path(__file__).resolve().parents[2]
BUNDLE = ROOT / "data/derived/eval_bundle"
OUT = ROOT / "runs/e4_colmap"


from e4_score import ang, score, umeyama  # noqa: F401


def run(set_name: str, max_size: int) -> dict:
    d = BUNDLE / set_name
    g = np.load(d / "gt.npz")
    names = [str(n) for n in g["names"]]
    K = g["K"][0]
    work = OUT / set_name
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True)
    db = work / "database.db"
    t0 = time.time()

    reader = pycolmap.ImageReaderOptions()
    reader.camera_model = "PINHOLE"
    reader.camera_params = f"{K[0, 0]},{K[1, 1]},{K[0, 2]},{K[1, 2]}"
    ext = pycolmap.FeatureExtractionOptions()
    if hasattr(ext, "max_image_size"):
        ext.max_image_size = max_size
    elif hasattr(ext, "sift"):
        ext.sift.max_image_size = max_size
    pycolmap.extract_features(db, d / "images", image_names=names, camera_mode=pycolmap.CameraMode.SINGLE,
                              reader_options=reader, extraction_options=ext, device=pycolmap.Device.cpu)
    pair = pycolmap.SequentialPairingOptions()
    pair.overlap = 10
    pycolmap.match_sequential(db, pairing_options=pair, device=pycolmap.Device.cpu)
    opts = pycolmap.IncrementalPipelineOptions()
    for k in ("ba_refine_focal_length", "ba_refine_principal_point", "ba_refine_extra_params"):
        if hasattr(opts, k):
            setattr(opts, k, False)  # the lens is known (EXIF); don't let BA drift it
    maps = pycolmap.incremental_mapping(db, d / "images", work / "sparse", options=opts)
    elapsed = time.time() - t0

    if not maps:
        return {"set": set_name, "n": len(names), "registered": 0, "runtime_s": elapsed, "error": "no model"}
    rec = max(maps.values(), key=lambda r: r.num_reg_images())
    c2w = {}
    for img in rec.images.values():
        if not img.has_pose:
            continue
        cfw = img.cam_from_world() if callable(img.cam_from_world) else img.cam_from_world
        T = np.eye(4)
        T[:3, :4] = cfw.matrix()
        c2w[img.name] = np.linalg.inv(T)
    idx = [i for i, n in enumerate(names) if n in c2w]
    Tp = np.stack([c2w[names[i]] for i in idx])
    Tg = g["T_world_cam"][idx]
    out = {"set": set_name, "n": len(names), "registered": len(idx), "models": len(maps), "runtime_s": elapsed}
    out |= score(Tp, Tg)
    out["views_ok_of_all"] = f"{out['views_ok']}/{len(names)}"
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("sets", nargs="+")
    ap.add_argument("--max-size", type=int, default=1024)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    rows = [run(s, args.max_size) for s in args.sets]
    for r in rows:
        print(json.dumps(r))
    (OUT / "results.json").write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    sys.exit(main())
