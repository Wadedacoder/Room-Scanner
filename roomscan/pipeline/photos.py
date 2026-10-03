"""Photo tier: folder of per-room photo folders -> per-room metric reconstruction -> shared back-end -> plan dict."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from roomscan.damage.stage import finalize, room_damage
from roomscan.geometry.cloud import find_levels
from roomscan.io.photos import load_capture
from roomscan.pipeline.backend import PHOTO_ERR, ErrorModel, empty_plan, meas, rooms_from_cloud
from roomscan.recon.learned import LearnedRecon, backproject, gravity_align, voxelize

VOXEL = 0.03  # photo clouds are sparser and noisier than LiDAR


def wall_rays(pts: list[np.ndarray], cams: np.ndarray, floor_y: float, per_view: int = 3000, seed: int = 0):
    rng = np.random.default_rng(seed)
    starts, ends = [], []
    for P, c in zip(pts, cams):
        h = P[:, 1] - floor_y
        P = P[(h > 0.3) & (h < 1.8)]
        if len(P) > per_view:
            P = P[rng.choice(len(P), per_view, replace=False)]
        ends.append(P[:, [0, 2]])
        starts.append(np.repeat(c[None, [0, 2]], len(P), 0))
    return np.concatenate(starts), np.concatenate(ends)


def reconstruct_room(model: LearnedRecon, photos):
    rec = model.reconstruct([p.image for p in photos], np.stack([p.K for p in photos]))
    pts, cams = backproject(rec)
    G = gravity_align(pts, rec.c2w)
    pts = [P @ G.T for P in pts]
    cams = cams @ G.T
    return rec, pts, cams, G


def run_photos(path: Path, cfg) -> tuple[dict, dict]:
    t0 = time.time()
    warnings = []
    capture = load_capture(Path(path), warnings)
    model = LearnedRecon(cfg)
    plan = empty_plan("photos", str(path))
    debug = {}
    total_area, total_var = 0.0, 0.0
    frames, room_photos, per_room = {}, {}, {}
    damage, damage_inputs = [], []
    from roomscan.stitch.photo_graph import RoomFrame

    for name, photos in capture.items():
        rec, pts, cams, Gr = reconstruct_room(model, photos)
        P = voxelize(np.concatenate(pts), VOXEL)
        floor = find_levels(P, float(np.median(cams[:, 1]))).floor
        rays = wall_rays(pts, cams, floor)
        # doorways: views INTO the next room are low-confidence depth, which the 30th-percentile filter removes
        # (E15: max distance 2.7 m with it, 6.6 m without), so openings get their own lightly filtered points
        far_pts, _ = backproject(rec, conf_pct=5)
        far_pts = [Q @ Gr.T for Q in far_pts]
        sl_rng = np.random.default_rng(0)
        sl = [(np.repeat(c[None, [0, 2]], min(len(Q), 20000), 0), Q[sel][:, [0, 2]], Q[sel][:, 1])
              for c, Q in zip(cams, far_pts) for sel in [sl_rng.choice(len(Q), min(len(Q), 20000), replace=False)]]
        sightlines = tuple(np.concatenate([x[i] for x in sl]) for i in range(3))
        fwd = (rec.c2w[:, :3, 2] @ Gr.T)[:, [0, 2]]  # camera viewing directions, gravity frame
        hfov = float(2 * np.arctan(rec.depth.shape[2] / 2 / rec.K[0, 0, 0]))
        err, why = _photo_error(photos)
        if why:
            warnings.append(f"{name}: off-protocol photos ({why}); scale interval widened to ±{100 * err.rel:.0f}% (1σ)")
        res = rooms_from_cloud(P, cams, rays, err, VOXEL, split=False, id_prefix=name, label=name,
                               sightlines=sightlines, cam_fwd_xz=fwd, hfov=hfov,
                               interior_mode=cfg["recon"]["photo_outline"])  # sparse views can't carve (E6)
        warnings += res.warnings
        debug[name] = {"scale": rec.scale, "focal_pred_err": rec.focal_pred_err, "n_photos": len(photos),
                       "focal_source": photos[0].focal_source, "backend": res.debug}
        if not res.rooms:
            continue
        c2w_g = rec.c2w.copy()
        c2w_g[:, :3, :3] = Gr @ rec.c2w[:, :3, :3]
        c2w_g[:, :3, 3] = rec.c2w[:, :3, 3] @ Gr.T
        frames[name] = RoomFrame(name, c2w_g, rec.depth, rec.K, photos[0].image.shape[:2], res.debug["theta"],
                                 np.array(res.rooms[0]["polygon"]))
        room_photos[name] = [p.image for p in photos]
        per_room[name] = res.rooms[0]
        # damage inputs kept for after the loop (sweep, 2026-10-03: running OWLv2 between rooms left its GPU cache
        # behind and the next room's DA3 ran out of MPS memory; only one model on the GPU at a time)
        damage_inputs.append((res.rooms[0], [p.image for p in photos], rec.depth, rec.K, c2w_g,
                              res.debug["levels"].floor, res.debug["theta"], [p.path.name for p in photos]))

    # damage before rooms move into the property frame: regions live in surface-local coordinates
    for room, imgs, depth, K, c2w, floor_y, theta, names in damage_inputs:
        damage += room_damage(room, imgs, depth, K, c2w, floor_y, theta, PHOTO_ERR, cfg,
                              Path(cfg["runtime"]["cache_dir"]), warnings, names)
    damage_inputs.clear()

    placed, edges, root_theta = {}, [], 0.0
    if cfg["recon"].get("photo_stitch", True) and len(frames) > 1:
        from roomscan.stitch.links import cross_room_links
        from roomscan.stitch.photo_graph import place_rooms

        links = cross_room_links(room_photos, cfg)
        placed, edges = place_rooms(frames, links)
        debug["links"] = [{"a": L.room_a, "b": L.room_b, "photos": [L.photo_a, L.photo_b], "matches": L.inliers}
                          for L in links[:15]]
        if placed:
            root_theta = frames[next(r for r, p in placed.items() if p.parent is None)].theta
    x_cursor = 0.0
    if placed:
        allp = np.concatenate([_to_property(per_room[r]["polygon"], frames[r], placed[r], root_theta) for r in placed])
        x_cursor = float(allp[:, 0].max()) + 2.0
    for name, room in per_room.items():
        room["connected"] = name in placed or len(per_room) == 1  # False: no reliable link; drawn apart
        if name in placed:
            _transform_room(room, frames[name], placed[name], root_theta)
        else:
            # not linked to the others: drawn to the side, and said so
            poly = np.array(room["polygon"])
            shift = np.array([x_cursor - poly[:, 0].min(), 0.0])
            _shift_room(room, shift)
            x_cursor = float(poly[:, 0].max() + shift[0] + 1.2)
        plan["rooms"].append(room)
        total_area += room["floor_area"]["value"]
        total_var += ((room["floor_area"]["hi"] - room["floor_area"]["lo"]) / (2 * 1.645)) ** 2
    if edges:
        debug["wall_snaps"] = _snap_shared_walls({r: per_room[r] for r in placed}, edges)
    plan["adjacency"] = [{"a": e["a"], "b": e["b"], "via": f"visual-link ({e['matches']} matches)"} for e in edges]
    _pair_doors(per_room, plan["adjacency"])
    unplaced = [r for r in per_room if r not in placed]
    if placed and unplaced:
        warnings.append(f"connected {len(placed)} of {len(per_room)} rooms; not linked (drawn to the side): "
                        f"{', '.join(unplaced)}. Add a photo per doorway looking through it into the next room.")
    elif not placed and len(per_room) > 1:
        warnings.append("rooms could not be linked: no photo sees into another room; drawn side by side")
    debug["placements"] = {r: {"parent": p.parent, "yaw_deg": round(float(np.degrees(p.yaw)), 1),
                               "t": np.round(p.t, 3).tolist()} for r, p in placed.items()}
    debug["edges"] = edges
    finalize(plan, damage)
    plan["property"] = {"footprint_area": meas(total_area, float(np.sqrt(total_var)), "m2")}
    plan["capture"]["runtime_s"] = round(time.time() - t0, 1)
    plan["capture"]["models"] = {"geometry": cfg["recon"]["model"], "metric": cfg["recon"]["metric_model"]}
    plan["warnings"] = warnings + [
        "photo-tier intervals use a 5% scale term from proxy-data experiments E1-E2; not yet calibrated on real photos",
        "damage areas use a box-shaped region (5-95% extent on the surface); scope quantities carry a 5% term"]
    return plan, debug


def _to_property(poly_uv, frame, pl, root_theta):
    from roomscan.stitch.photo_graph import polygon_to_property

    return polygon_to_property(np.asarray(poly_uv, float), frame, pl, root_theta)


def _transform_room(room: dict, frame, pl, root_theta: float) -> None:
    room["polygon"] = _to_property(room["polygon"], frame, pl, root_theta).round(4).tolist()
    for w in room["walls"]:
        se = _to_property([w["start"], w["end"]], frame, pl, root_theta).round(4)
        w["start"], w["end"] = se[0].tolist(), se[1].tolist()
    for o in room.get("openings", []):
        if "centre_plan" in o:
            o["centre_plan"] = _to_property([o["centre_plan"]], frame, pl, root_theta).round(4)[0].tolist()


def _shift_room(room: dict, shift) -> None:
    room["polygon"] = (np.array(room["polygon"]) + shift).round(4).tolist()
    for w in room["walls"]:
        w["start"] = (np.array(w["start"]) + shift).round(4).tolist()
        w["end"] = (np.array(w["end"]) + shift).round(4).tolist()
    for o in room.get("openings", []):
        if "centre_plan" in o:
            o["centre_plan"] = (np.array(o["centre_plan"]) + shift).round(4).tolist()


WALL_THICKNESS = 0.15  # typical interior wall; rooms linked through a doorway share a wall


def _edges_axis(room: dict):
    """Axis-aligned polygon edges as (axis, coord, lo, hi, outward_sign, door_centres_along)."""
    poly = np.array(room["polygon"])
    cen = poly.mean(0)
    doors: dict[str, list] = {}
    for o in room.get("openings", []):
        if o["type"] != "window" and "centre_plan" in o:
            doors.setdefault(o["wall_id"], []).append(o["centre_plan"])
    out = []
    for w in room["walls"]:
        a, b = np.array(w["start"]), np.array(w["end"])
        d = b - a
        if np.hypot(*d) < 0.3:
            continue
        axis = 0 if abs(d[0]) < abs(d[1]) else 1  # axis 0: wall at u = c, runs along v
        c = (a[axis] + b[axis]) / 2
        lo, hi = sorted((a[1 - axis], b[1 - axis]))
        out.append((axis, c, lo, hi, 1.0 if c > cen[axis] else -1.0,
                    [float(pc[1 - axis]) for pc in doors.get(w["id"], [])]))
    return out


def _snap_shared_walls(rooms: dict, edges: list[dict]) -> list[dict]:
    """For each placed link a->b: find the facing wall pair (parallel, opposite outward normals, overlapping along the
    wall, preferring a wall with a door) and slide b (with everything attached through it) so the two walls are
    WALL_THICKNESS apart. E17: the visual link fixes direction well but distance poorly (depth seen through a doorway
    is the least reliable part of the image), leaving kitchen and living ~2.8 m apart."""
    children: dict[str, list[str]] = {}
    for e in edges:
        children.setdefault(e["a"], []).append(e["b"])

    def subtree(r):
        out = [r]
        for c in children.get(r, []):
            out += subtree(c)
        return out

    log = []
    for e in edges:
        A, B = rooms[e["a"]], rooms[e["b"]]
        best = None
        for (ax, ca, la, ha, sa, da) in _edges_axis(A):
            for (bx, cb, lb, hb, sb, db) in _edges_axis(B):
                if ax != bx or sa != -sb:  # parallel walls whose outsides face each other
                    continue
                gap = (cb - ca) * sa  # positive: B's wall lies outside A's wall, as it should
                overlap = min(ha, hb) - max(la, lb)
                # E18: the along-wall position from a visual link can be 1-2 m off, so allow near-misses
                if overlap < -1.5 or gap < -1.0:
                    continue
                door_pair = None
                if da and db:  # rooms linked through a doorway connect THROUGH a door: line the doors up
                    dd = [(x - y) for x in da for y in db]
                    k = int(np.argmin(np.abs(dd)))
                    if abs(dd[k]) < 2.5:
                        door_pair = dd[k]
                score = abs(gap - WALL_THICKNESS) + max(0.0, -overlap) - (1.0 if door_pair is not None else 0.0)
                if best is None or score < best[0]:
                    best = (score, ax, sa, gap, door_pair)
        if best is None:
            log.append({"a": e["a"], "b": e["b"], "snapped": False})
            continue
        _, ax, sa, gap, door_pair = best
        delta = np.zeros(2)
        delta[ax] = (WALL_THICKNESS - gap) * sa
        if door_pair is not None:
            delta[1 - ax] = door_pair
        for r in subtree(e["b"]):
            _shift_room(rooms[r], delta)
        log.append({"a": e["a"], "b": e["b"], "snapped": True, "moved_m": round(float(np.hypot(*delta)), 3),
                    "doors_aligned": door_pair is not None})
    return log


def _pair_doors(rooms: dict, adjacency: list[dict], max_d: float = 0.8) -> None:
    """For each linked room pair, the two doors that coincide after placement (closest pair within max_d; photo
    door centres are +-20 cm, E15) get connects_to, and the adjacency names the door. Unlinked rooms are not placed
    by door geometry: on house_b the only candidate (study, one weak 0.70 m door) fits two free doors equally (E25)."""
    for a in adjacency:
        ra, rb = rooms[a["a"]], rooms[a["b"]]
        best = None
        for oa in ra["openings"]:
            for ob in rb["openings"]:
                if "window" in (oa["type"], ob["type"]) or "centre_plan" not in oa or "centre_plan" not in ob:
                    continue
                d = float(np.hypot(*(np.array(oa["centre_plan"]) - np.array(ob["centre_plan"]))))
                if d < max_d and (best is None or d < best[0]):
                    best = (d, oa, ob)
        if best:
            _, oa, ob = best
            oa["connects_to"], ob["connects_to"] = rb["id"], ra["id"]
            a["via"] += f", door {oa['id']}↔{ob['id']} ({best[0]:.2f} m apart)"


def _photo_error(photos) -> tuple[ErrorModel, str]:
    """Scale term by protocol compliance (E26). On the taped study, the protocol capture (landscape 0.5x ring) is within
    2% and its intervals hold; 8 off-protocol sets of the same room (1x lens, portrait, mixed lenses, 2-3 photos) were
    off by 9-21% on the short side (RMS ~14%, plus one -79% outlier) while claiming +-9%. The 15% term is fitted on those
    same captures (in-sample)."""
    if any(p.focal_source.startswith("assumed") for p in photos):  # the assumed 0.5x lens may be wrong (1x = 2x off)
        return ErrorModel(PHOTO_ERR.abs_m, 0.25), "no EXIF focal length"
    reasons = []
    f35 = [float(p.focal_source.split("=")[1]) for p in photos if p.focal_source.startswith("exif f35=")]
    if any(f >= 18 for f in f35):
        reasons.append("not all 0.5x lens")
    if any(p.image.shape[0] > p.image.shape[1] for p in photos):
        reasons.append("portrait photos")
    if len(photos) < 4:
        reasons.append(f"only {len(photos)} photos")
    return (ErrorModel(PHOTO_ERR.abs_m, 0.15), ", ".join(reasons)) if reasons else (PHOTO_ERR, "")
