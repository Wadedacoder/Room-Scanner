"""Interval calibration on the one taped room: every plan of the study, scored against tape (backlog 2.3).

    python bench/calibration.py runs/sweep/*/plan.json runs/sweep2/*/plan.json

For each plan that contains the study: short side, long side (longest wall along each plan axis, with that wall's
interval), floor area and ceiling, each with its error and whether the reported 90% interval contains the tape value.
Coverage per quantity should be ~90% if the intervals are calibrated. Captures of the same room are not independent
(several variant sets reuse the same photos), so this is evidence, not a calibration fit.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bench"))
from gates import ground_truth  # noqa: E402


def side_walls(room: dict) -> tuple[dict, dict]:
    """Longest wall roughly along each plan axis -> (short-side wall, long-side wall)."""
    best = {}
    for w in room["walls"]:
        d = np.array(w["end"]) - np.array(w["start"])
        axis = int(abs(d[1]) > abs(d[0]))
        if axis not in best or w["length"]["value"] > best[axis]["length"]["value"]:
            best[axis] = w
    if len(best) < 2:
        return None, None
    a, b = best[0], best[1]
    return (a, b) if a["length"]["value"] <= b["length"]["value"] else (b, a)


def score(plan: dict, gt: dict) -> list[dict] | None:
    room = next((r for r in plan["rooms"] if r["id"] == "study" or r["label"] == "study"), None)
    if room is None and plan["capture"]["tier"] != "lidar" and len(plan["rooms"]) == 1:
        room = plan["rooms"][0]
    if room is None:
        return None
    rows = []
    s, l = side_walls(room)
    for name, m, truth in (("short side", s and s["length"], gt["short_m"]), ("long side", l and l["length"], gt["long_m"]),
                           ("floor area", room["floor_area"], gt["area_m2"]),
                           ("ceiling", room.get("ceiling_height"), gt.get("ceiling_m"))):
        if m is None or truth is None:
            rows.append({"q": name, "pred": None})
            continue
        rows.append({"q": name, "pred": m["value"], "truth": truth, "err_pct": 100 * (m["value"] - truth) / truth,
                     "holds": bool(m["lo"] <= truth <= m["hi"]), "half_pct": 100 * (m["hi"] - m["lo"]) / 2 / m["value"]})
    return rows


def main(paths: list[str]) -> int:
    gt = ground_truth(yaml.safe_load((ROOT / "bench/ground_truth/home_tape.yaml").read_text()), "study")
    table = {}
    for p in paths:
        plan = json.loads(Path(p).read_text())
        rows = score(plan, gt)
        if rows is None:
            continue
        name = f"{Path(p).parent.name} ({plan['capture']['tier']})"
        table[name] = rows
        cells = []
        for r in rows:
            cells.append(f"{r['q']}: n/a" if r["pred"] is None else
                         f"{r['q']} {r['err_pct']:+.1f}% ±{r['half_pct']:.0f}% {'✓' if r['holds'] else '✗'}")
        print(f"{name:34s} " + " | ".join(cells))
    print()
    for q in ("short side", "long side", "floor area", "ceiling"):
        for tier in ("photos", "video"):
            hs = [r["holds"] for n, rows in table.items() if f"({tier})" in n for r in rows if r["q"] == q and r["pred"]]
            if hs:
                print(f"coverage {tier:6s} {q:10s}: {sum(hs)}/{len(hs)}")
    out = ROOT / "bench/results/calibration_study.json"
    out.write_text(json.dumps(table, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
