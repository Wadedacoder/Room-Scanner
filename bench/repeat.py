"""Repeatability: run one capture N times LIVE (checkpoints bypassed) and report the spread of every gated number.

    python bench/repeat.py <capture> --runs 3 [--room study] [--out runs/repeat/<name>]

Each run is a separate `roomscan run` process under scripts/run_guarded.sh with ROOMSCAN_NO_CACHE=1, so model
inference, COLMAP and matching all execute again. Reports per quantity: every run's value, mean, max-min range and
std, plus the gate results of each run against the tape (bench/gates.py). Writes <out>/repeat.json.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bench"))
from gates import ground_truth, score  # noqa: E402


def run_once(capture: Path, out: Path, limit_mb: int) -> dict:
    env = os.environ | {"ROOMSCAN_NO_CACHE": "1", "PYTHONPATH": str(ROOT), "HF_HUB_OFFLINE": "1",
                        "LIMIT_MB": str(limit_mb)}
    cmd = [str(ROOT / "scripts/run_guarded.sh"), str(ROOT / ".venv/bin/python"), "-m", "roomscan.cli", "run",
           str(capture), "-o", str(out)]
    p = subprocess.run(cmd, env=env, capture_output=True, text=True, cwd=ROOT)
    if p.returncode != 0:
        raise RuntimeError(f"run failed ({p.returncode}): {p.stdout[-800:]}{p.stderr[-800:]}")
    return json.loads((out / capture.name / "plan.json").read_text())


def summarize(plans: list[dict], room_id: str, gt: dict | None) -> dict:
    rows = {}
    for plan in plans:
        room = next((r for r in plan["rooms"] if r["id"] == room_id or r["label"] == room_id), plan["rooms"][0])
        short, long_ = np.sort(np.ptp(np.array(room["polygon"]), axis=0))
        c = room.get("ceiling_height")
        for k, v in (("short side m", short), ("long side m", long_), ("floor area m2", room["floor_area"]["value"]),
                     ("ceiling m", c["value"] if c else None), ("openings", len(room.get("openings", []))),
                     ("runtime s", plan["capture"].get("runtime_s"))):
            rows.setdefault(k, []).append(None if v is None else round(float(v), 4))
    out = {}
    for k, vals in rows.items():
        v = np.array([x for x in vals if x is not None], float)
        out[k] = {"runs": vals, "mean": round(float(v.mean()), 4) if len(v) else None,
                  "range": round(float(np.ptp(v)), 4) if len(v) else None,
                  "std": round(float(v.std(ddof=1)), 4) if len(v) > 1 else None}
    if gt is not None:
        out["gates"] = [[(r["gate"], bool(r["pass"])) for r in score(p, gt, room_id)] for p in plans]
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("capture", type=Path)
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--room", default="study")
    ap.add_argument("--gt", type=Path, default=ROOT / "bench/ground_truth/home_tape.yaml")
    ap.add_argument("--out", type=Path)
    ap.add_argument("--limit-mb", type=int, default=5500)
    ap.add_argument("--summarize-only", action="store_true", help="re-score existing runs in --out, no new runs")
    a = ap.parse_args()
    out = a.out or ROOT / "runs/repeat" / a.capture.name
    plans = []
    for i in range(a.runs):
        if a.summarize_only:
            plans.append(json.loads((out / f"run{i + 1}" / a.capture.name / "plan.json").read_text()))
            continue
        plans.append(run_once(a.capture.resolve(), out / f"run{i + 1}", a.limit_mb))
        print(f"run {i + 1}/{a.runs}: {plans[-1]['capture'].get('runtime_s')} s", flush=True)
    gt = ground_truth(yaml.safe_load(a.gt.read_text()), a.room) if a.gt.exists() else None
    res = {"capture": str(a.capture), "tier": plans[0]["capture"]["tier"],
           "pipeline_version": plans[0]["capture"].get("pipeline_version"), "runs": a.runs,
           "summary": summarize(plans, a.room, gt)}
    (out / "repeat.json").write_text(json.dumps(res, indent=1))
    for k, v in res["summary"].items():
        print(f"  {k:14s} {v}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
