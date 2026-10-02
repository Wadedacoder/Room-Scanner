"""E8: score photo-tier runs of the real study capture against the tape ground truth.

Ground truth (bench/ground_truth/home_tape.yaml, study): four wall readings not yet assigned to walls
(350.5, 330.2, 312.4, 312.4 cm; the 350.5 vs 330.2 pair differs because of a cove) and ceiling 274.3 cm.
Until wall order is recorded we score what is order-free:
  short side   vs 312.4 cm
  long side    vs the band 330.2-350.5 cm (inside the band = 0 error; outside = distance to the nearer edge)
  floor area   vs 3.124 x 3.302 .. 3.124 x 3.505 m^2 (cove makes the true value somewhere in between)
  ceiling      vs 274.3 cm
Interval check: does the reported 90% interval contain the truth (for the long side and area, overlap the band)?

Usage: python bench/score_study.py runs/e8_study            (scores every <variant>/plan.json under it)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

SHORT, LONG_LO, LONG_HI, CEIL = 3.124, 3.302, 3.505, 2.743
AREA_LO, AREA_HI = SHORT * LONG_LO, SHORT * LONG_HI


def band_err(v: float, lo: float, hi: float) -> float:
    return 0.0 if lo <= v <= hi else (v - hi if v > hi else v - lo)


def score(plan: dict) -> dict:
    rooms = [r for r in plan["rooms"] if r["id"].startswith("study")] or plan["rooms"]
    if not rooms:
        return {"error": "no room"}
    r = rooms[0]
    poly = np.array(r["polygon"])
    ext = np.sort(np.ptp(poly, axis=0))  # plan frame is wall-aligned, so extents = room sides
    short, long_ = float(ext[0]), float(ext[1])
    a = r["floor_area"]
    out = {
        "short_m": round(short, 3), "short_err_pct": round(100 * (short - SHORT) / SHORT, 1),
        "long_m": round(long_, 3), "long_err_pct": round(100 * band_err(long_, LONG_LO, LONG_HI) / LONG_LO, 1),
        "area_m2": round(a["value"], 2), "area_err_pct": round(100 * band_err(a["value"], AREA_LO, AREA_HI) / AREA_LO, 1),
        "area_interval_hits": bool(a["lo"] <= AREA_HI and a["hi"] >= AREA_LO),
        "area_interval_halfwidth_pct": round(100 * (a["hi"] - a["lo"]) / 2 / max(a["value"], 1e-9), 1),
        "walls": len(r["walls"]),
    }
    c = r.get("ceiling_height")
    if c:
        out |= {"ceiling_m": c["value"], "ceiling_err_cm": round(100 * (c["value"] - CEIL), 1),
                "ceiling_interval_hits": bool(c["lo"] <= CEIL <= c["hi"])}
    else:
        out["ceiling_m"] = None
    return out


def main() -> None:
    root = Path(sys.argv[1] if len(sys.argv) > 1 else "runs/e8_study")
    rows = []
    for p in sorted(root.glob("*/plan.json")):
        plan = json.loads(p.read_text())
        row = {"variant": p.parent.name, "pipeline": plan["capture"].get("pipeline_version")} | score(plan)
        rows.append(row)
        print(json.dumps(row))
    (root / "scores.json").write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
