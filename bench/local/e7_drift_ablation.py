"""E7: drift correction ablation on the multi-room LiDAR walks (brief Part 2: footprint with correction on and off).

No ground truth yet, so three internal measures:
  wall_sharpness   share of wall-band points within 2 cm of the strongest wall lines (drift smears walls -> lower)
  footprint / rooms  what the plan reports
  repeatability    the two walks (c7d28f72c6, 1a8384c3f6) cover the same flat; their footprints should agree

Usage: python bench/local/e7_drift_ablation.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from roomscan.config import load_config
from roomscan.geometry import plan2d as p2
from roomscan.pipeline.lidar import run_lidar
from roomscan.render.svg import render_svg

OUT = ROOT / "runs/e7_drift"
CAPTURES = ["c7d28f72c6", "1a8384c3f6"]


def wall_sharpness(P: np.ndarray, floor: float, theta: float) -> float:
    y = P[:, 1] - floor
    uv = P[(y > 0.3) & (y < 2.0)][:, [0, 2]] @ p2.rot2(-theta).T
    score = []
    for k in (0, 1):
        h, _ = np.histogram(uv[:, k], np.arange(uv[:, k].min(), uv[:, k].max() + 0.01, 0.01))
        top = np.sort(h)[::-1][: max(1, len(h) // 20)]  # strongest 5% of 1 cm slices = wall lines
        # count each top slice with its +-2 cm neighbours via a 5-wide box filter
        hb = np.convolve(h, np.ones(5), "same")
        score.append(np.sort(hb)[::-1][: len(top)].sum() / max(h.sum(), 1))
    return float(np.mean(score))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for cap in CAPTURES:
        for method in ("off", "posegraph"):
            cfg = load_config("lite", overrides=[f"drift.method={method}"])
            t = time.time()
            plan, dbg = run_lidar(ROOT / "data/raw/lidar" / cap, cfg)
            (OUT / f"{cap}_{method}.svg").write_text(render_svg(plan))
            (OUT / f"{cap}_{method}.json").write_text(json.dumps(plan, indent=1))
            row = {"capture": cap, "drift": method, "rooms": len(plan["rooms"]),
                   "areas_m2": [r["floor_area"]["value"] for r in plan["rooms"]],
                   "footprint_m2": plan["property"]["footprint_area"]["value"],
                   "wall_sharpness": round(wall_sharpness(dbg["cloud"], dbg["levels"].floor, dbg["theta"]), 4),
                   "runtime_s": round(time.time() - t, 1), "drift_stats": plan["capture"].get("drift")}
            rows.append(row)
            print(json.dumps(row), flush=True)
    (OUT / "results.json").write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
