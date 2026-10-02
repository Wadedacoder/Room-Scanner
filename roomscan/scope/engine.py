"""Damage regions -> concealed-damage flags and scope line items, by the explicit rules in rules.yaml.

Every flag and line item records the rule id that produced it (the brief: "concealed-damage flags with the rule that
fired, scope line items keyed to surfaces"). Quantities carry intervals propagated from the region and wall intervals.
"""

from __future__ import annotations

from pathlib import Path

import yaml

from roomscan.pipeline.backend import meas

RULES = Path(__file__).resolve().parent / "rules.yaml"


def _matches(when: dict, region: dict, length: float) -> bool:
    if "class" in when and region["class"] != when["class"]:
        return False
    if "surface" in when and region["surface_kind"] != when["surface"]:
        return False
    bottom = region.get("height_above_floor_m", 0.0)
    if "bottom_below_m" in when and not bottom < when["bottom_below_m"]:
        return False
    if "bottom_above_m" in when and not bottom > when["bottom_above_m"]:
        return False
    if "min_length_m" in when and not length >= when["min_length_m"]:
        return False
    return True


def _wall(room: dict, surface_id: str):
    return next((w for w in room["walls"] if w["id"] == surface_id), None)


def apply_rules(damage: list[dict], rooms: list[dict], rel_err: float = 0.05) -> tuple[list[dict], list[dict]]:
    """damage: plan.json DamageRegion dicts plus 'surface_kind', 'room_id', 'height_above_floor_m'."""
    rules = yaml.safe_load(RULES.read_text())
    by_room = {r["id"]: r for r in rooms}
    flags, items = [], []
    for d in damage:
        room = by_room.get(d["room_id"])
        if room is None:
            continue
        length = max(d["extent_w"]["value"], d["extent_h"]["value"])
        for r in rules["concealed_rules"]:
            if _matches(r["when"], d, length):
                flags.append({"id": f"F{len(flags) + 1}", "surface_id": d["surface_id"], "rule_id": r["id"],
                              "rationale": " ".join(r["flag"].split()), "triggered_by": [d["id"]]})
        wall = _wall(room, d["surface_id"])
        ceil = room.get("ceiling_height") or {"value": 2.5}
        quantities = {
            "region_area_m2": d["area"]["value"],
            "region_length_m": length,
            "wall_length_m": wall["length"]["value"] if wall else None,
            "wall_area_m2": wall["length"]["value"] * ceil["value"] if wall else None,
            "flood_cut_area_m2": wall["length"]["value"] * 0.6 if wall else None,
            "floor_area_m2": room["floor_area"]["value"],
        }
        units = {"region_area_m2": "m2", "region_length_m": "m", "wall_length_m": "m", "wall_area_m2": "m2",
                 "flood_cut_area_m2": "m2", "floor_area_m2": "m2"}
        for r in rules["scope_rules"]:
            if not _matches(r["when"], d, length):
                continue
            for it in r["items"]:
                q = quantities.get(it["qty"])
                if q is None:
                    continue
                q *= 1 + it.get("margin", 0.0)
                # a wall-wide item is listed once per surface, however many regions trigger it
                key = (it["code"], d["surface_id"])
                if it["qty"] in ("wall_area_m2", "floor_area_m2", "wall_length_m", "flood_cut_area_m2") and any(
                        (x["code"], x["surface_id"]) == key for x in items):
                    continue
                items.append({"id": f"S{len(items) + 1}", "surface_id": d["surface_id"], "code": it["code"],
                              "description": it["description"],
                              "quantity": meas(round(q, 3), abs(q) * rel_err, units[it["qty"]]),
                              "source": [d["id"], r["id"]]})
    return flags, items
