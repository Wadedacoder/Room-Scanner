"""Score a plan.json against tape ground truth, gate by gate, using the brief's per-tier thresholds.

    python bench/gates.py <plan.json> [--room study] [--gt bench/ground_truth/home_tape.yaml]

Gates (brief Part 2 + inferred Round 1 baselines, docs/PLAN.md):
  wall length   photo ±8%, video ±3%, LiDAR ±2 cm or 1%      (opposite walls paired, order-free)
  floor area    photo ±10%, video ±5%, LiDAR ±2%
  ceiling       ≤ 1.5 cm (all tiers; the brief's gate is a LiDAR-grade figure)
  opening width ≤ 2 cm on ≥ 85% of openings (missed and phantom openings count as misses)
  calibration   each reported 90% interval should contain the tape value
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
WALL_TOL = {"photos": ("rel", 0.08), "video": ("rel", 0.03), "lidar": ("lidar", None)}
AREA_TOL = {"photos": 0.10, "video": 0.05, "lidar": 0.02}
CEIL_TOL_M = 0.015
OPENING_TOL_M = 0.02


def _mean(v):
    vals = [x for x in (v or []) if x is not None]
    return float(np.mean(vals)) if vals else None


def ground_truth(gt: dict, room: str) -> dict | None:
    r = next((x for x in gt.get("rooms", []) if x["id"] == room), None)
    if r is None:
        return None
    walls = [_mean(w.get("length_cm")) for w in r.get("walls", [])]
    out = {"ceiling_m": (_mean(r.get("ceiling_height_cm")) or 0) / 100 or None,
           "openings_m": [x / 100 for x in (_mean(o.get("width_cm")) for o in r.get("openings", [])) if x]}
    if len(walls) >= 4 and all(w is not None for w in walls[:4]):
        a, b = sorted([(walls[0] + walls[2]) / 2, (walls[1] + walls[3]) / 2])
        out |= {"short_m": a / 100, "long_m": b / 100,
                "long_range_m": (min(walls[1], walls[3], walls[0], walls[2], key=lambda x: abs(x - b)) / 100,
                                 max(walls[1], walls[3]) / 100 if b == (walls[1] + walls[3]) / 2 else
                                 max(walls[0], walls[2]) / 100),
                "area_m2": a * b / 1e4}
    return out


def wall_pass(tier: str, pred: float, truth: float) -> bool:
    kind, tol = WALL_TOL[tier]
    if kind == "rel":
        return abs(pred - truth) <= tol * truth
    return abs(pred - truth) <= max(0.02, 0.01 * truth)


def score(plan: dict, gt_room: dict, room_id: str) -> list[dict]:
    tier = plan["capture"]["tier"]
    room = next((r for r in plan["rooms"] if r["id"] == room_id or r["label"] == room_id), None)
    if room is None and len(plan["rooms"]) == 1:
        room = plan["rooms"][0]
    if room is None:
        return [{"gate": "room found", "pass": False, "detail": f"no room '{room_id}' in plan"}]
    rows = []
    poly = np.array(room["polygon"])
    short, long_ = np.sort(np.ptp(poly, axis=0))
    if "short_m" in gt_room:
        for name, pred, truth in (("wall: short side", short, gt_room["short_m"]),
                                  ("wall: long side", long_, gt_room["long_m"])):
            rows.append({"gate": name, "pred": round(float(pred), 3), "truth": round(truth, 3),
                         "err_pct": round(100 * (pred - truth) / truth, 1), "pass": wall_pass(tier, pred, truth)})
        a = room["floor_area"]
        t = gt_room["area_m2"]
        rows.append({"gate": "floor area", "pred": a["value"], "truth": round(t, 2),
                     "err_pct": round(100 * (a["value"] - t) / t, 1),
                     "pass": abs(a["value"] - t) <= AREA_TOL[tier] * t,
                     "interval_holds": a["lo"] <= t <= a["hi"]})
    c, tc = room.get("ceiling_height"), gt_room.get("ceiling_m")
    if tc:
        if c is None:
            rows.append({"gate": "ceiling", "pred": None, "truth": tc, "pass": False, "detail": "not reported"})
        else:
            rows.append({"gate": "ceiling", "pred": c["value"], "truth": tc, "err_cm": round(100 * (c["value"] - tc), 1),
                         "pass": abs(c["value"] - tc) <= CEIL_TOL_M, "interval_holds": c["lo"] <= tc <= c["hi"]})
    if gt_room.get("openings_m"):
        pred = sorted(o["width"]["value"] for o in room.get("openings", []) if o["type"] != "window")
        truth = sorted(gt_room["openings_m"])
        hits = sum(1 for t in truth if any(abs(p - t) <= OPENING_TOL_M for p in pred))
        n = max(len(truth), len(pred))  # missed AND phantom openings both count against
        rows.append({"gate": "opening widths", "pred": pred, "truth": truth, "pass": n > 0 and hits / n >= 0.85,
                     "detail": f"{hits}/{n} within 2 cm"})
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("plan", type=Path)
    ap.add_argument("--room", default="study")
    ap.add_argument("--gt", type=Path, default=ROOT / "bench/ground_truth/home_tape.yaml")
    a = ap.parse_args()
    plan = json.loads(a.plan.read_text())
    g = ground_truth(yaml.safe_load(a.gt.read_text()), a.room)
    if g is None:
        print(f"no ground truth for room '{a.room}'")
        return 2
    rows = score(plan, g, a.room)
    print(f"{plan['capture']['tier']} · pipeline {plan['capture'].get('pipeline_version')} · room {a.room}")
    for r in rows:
        extra = {k: v for k, v in r.items() if k not in ("gate", "pass")}
        print(f"  {'PASS' if r['pass'] else 'FAIL'}  {r['gate']:16s} {json.dumps(extra)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
