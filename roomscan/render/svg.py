"""plan.json -> plan.svg: rooms, dimensioned walls, openings, damage, area and ceiling labels."""

from __future__ import annotations

import numpy as np

PALETTE = ["#4C78A8", "#F58518", "#54A24B", "#B279A2", "#E45756", "#72B7B2", "#EECA3B", "#9D755D"]


def _fmt(m: dict | None, unit: str = "m") -> str:
    if not m:
        return "n/a"
    half = (m["hi"] - m["lo"]) / 2
    return f"{m['value']:.2f} ±{half:.2f} {unit}"


def render_svg(plan: dict, px_per_m: float = 80.0, margin: float = 1.0) -> str:
    pts = np.array([p for r in plan["rooms"] for p in r["polygon"]]) if plan["rooms"] else np.zeros((1, 2))
    lo, hi = pts.min(0) - margin, pts.max(0) + margin
    W, H = (hi - lo) * px_per_m

    def xy(p):  # plan v grows up, SVG y grows down
        return (p[0] - lo[0]) * px_per_m, H - (p[1] - lo[1]) * px_per_m

    out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W:.0f}" height="{H:.0f}" viewBox="0 0 {W:.0f} {H:.0f}" '
           'font-family="Helvetica, Arial, sans-serif">', f'<rect width="100%" height="100%" fill="#ffffff"/>']
    for k, r in enumerate(plan["rooms"]):
        c = PALETTE[k % len(PALETTE)]
        poly = " ".join(f"{x:.1f},{y:.1f}" for x, y in map(xy, r["polygon"]))
        out.append(f'<polygon points="{poly}" fill="{c}" fill-opacity="0.12" stroke="#222" stroke-width="3"/>')
        for w in r["walls"]:
            (x1, y1), (x2, y2) = xy(w["start"]), xy(w["end"])
            mx, my = (x1 + x2) / 2, (y1 + y2) / 2
            ang = np.degrees(np.arctan2(y2 - y1, x2 - x1))
            if ang > 90 or ang < -90:
                ang += 180
            out.append(f'<text x="{mx:.1f}" y="{my:.1f}" font-size="11" fill="#333" text-anchor="middle" '
                       f'transform="rotate({ang:.1f} {mx:.1f} {my:.1f}) translate(0 -5)">{w["length"]["value"]:.2f}</text>')
        walls_by_id = {w["id"]: w for w in r["walls"]}
        for o in r.get("openings", []):
            w = walls_by_id.get(o["wall_id"])
            if not w:
                continue
            a, b = np.array(w["start"]), np.array(w["end"])
            u = (b - a) / max(np.linalg.norm(b - a), 1e-9)
            p0 = a + u * o["offset_along_wall"]["value"]
            p1 = p0 + u * o["width"]["value"]
            (x0, y0), (x1, y1) = xy(p0), xy(p1)
            col = {"window": "#4C78A8", "door": "#B4531F"}.get(o["type"], "#54A24B")
            out.append(f'<line x1="{x0:.1f}" y1="{y0:.1f}" x2="{x1:.1f}" y2="{y1:.1f}" stroke="#ffffff" stroke-width="5"/>')
            out.append(f'<line x1="{x0:.1f}" y1="{y0:.1f}" x2="{x1:.1f}" y2="{y1:.1f}" stroke="{col}" stroke-width="2" '
                       f'stroke-dasharray="4 3"/>')
            out.append(f'<text x="{(x0 + x1) / 2:.1f}" y="{(y0 + y1) / 2 - 7:.1f}" font-size="10" text-anchor="middle" '
                       f'fill="{col}">{o["type"]} {o["width"]["value"]:.2f}</text>')
        # damage: wall regions as a red band at their measured span along the wall; floor/ceiling regions are listed
        # in the room (their surface coordinates are room-local, so drawing them would misplace them in a stitched plan)
        flat = []
        for d in plan.get("damage", []):
            w = walls_by_id.get(d["surface_id"])
            if w is None:
                if d["surface_id"].startswith(r["id"] + "."):
                    flat.append(d)
                continue
            us = [q[0] for q in d.get("polygon_on_surface", [])]
            if not us:
                continue
            a, b = np.array(w["start"]), np.array(w["end"])
            u = (b - a) / max(np.linalg.norm(b - a), 1e-9)
            (x0, y0), (x1, y1) = xy(a + u * min(us)), xy(a + u * max(us))
            out.append(f'<line x1="{x0:.1f}" y1="{y0:.1f}" x2="{x1:.1f}" y2="{y1:.1f}" stroke="#D62728" stroke-width="7" '
                       f'stroke-opacity="0.8"/>')
            out.append(f'<text x="{(x0 + x1) / 2:.1f}" y="{(y0 + y1) / 2 + 16:.1f}" font-size="10" text-anchor="middle" '
                       f'fill="#D62728">{d["id"]} {d["class"].replace("_", " ")}</text>')
        cx, cy = xy(np.mean(r["polygon"], axis=0))
        for i, d in enumerate(flat):
            surf = d["surface_id"].rsplit(".", 1)[-1]
            out.append(f'<text x="{cx:.1f}" y="{cy + 46 + 13 * i:.1f}" font-size="10" text-anchor="middle" fill="#D62728">'
                       f'{d["id"]} {d["class"].replace("_", " ")} on {surf}, {d["area"]["value"]:.2f} m²</text>')
        ceil = _fmt(r.get("ceiling_height"))
        out.append(f'<text x="{cx:.1f}" y="{cy:.1f}" font-size="14" font-weight="bold" text-anchor="middle" fill="{c}">'
                   f'{r["label"]}</text>')
        out.append(f'<text x="{cx:.1f}" y="{cy + 16:.1f}" font-size="11" text-anchor="middle" fill="#333">'
                   f'{_fmt(r["floor_area"], "m²")}</text>')
        out.append(f'<text x="{cx:.1f}" y="{cy + 30:.1f}" font-size="11" text-anchor="middle" fill="#333">'
                   f'ceiling {ceil}</text>')
    fp = plan["property"]["footprint_area"]
    out.append(f'<text x="12" y="20" font-size="13" fill="#222">{plan["capture"]["tier"]} · footprint '
               f'{_fmt(fp, "m²")} · {len(plan["rooms"])} rooms</text>')
    out.append(f'<text x="12" y="37" font-size="11" fill="#D62728">{len(plan.get("damage", []))} damage regions · '
               f'{len(plan.get("concealed_flags", []))} concealed-damage flags · {len(plan.get("scope", []))} '
               'scope items</text>')
    out.append("</svg>")
    return "\n".join(out)
