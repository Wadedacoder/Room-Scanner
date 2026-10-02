import numpy as np

from roomscan.damage.project import Detection, project_detection
from roomscan.pipeline.backend import meas
from roomscan.scope.engine import apply_rules


def _wall_view():
    """Camera 2 m in front of a flat wall (plane z = 2 in camera, wall at world x..., gravity +Y up)."""
    K = np.array([[200.0, 0, 100], [0, 200.0, 100], [0, 0, 1]])
    depth = np.full((200, 200), 2.0)
    c2w = np.eye(4)
    c2w[:3, :3] = np.diag([1.0, -1.0, 1.0]) @ np.eye(3)  # image down = world down
    c2w[1, 3] = 1.5  # camera 1.5 m above the floor (floor_y = 0)
    walls = [{"id": "r.w1", "start": [-3.0, 2.0], "end": [3.0, 2.0], "length": meas(6.0, 0.01, "m")}]
    return K, depth, c2w, walls


def test_stain_box_projects_to_wall_with_metric_size():
    K, depth, c2w, walls = _wall_view()
    # box 40 x 40 px at 2 m with f = 200 px -> 0.40 m x 0.40 m, centred on the image -> 1.5 m above floor
    det = Detection(0, "water_stain", (80, 80, 120, 120), 0.9, "synthetic")
    r = project_detection(det, depth, K, c2w, floor_y=0.0, ceiling_y=None, theta=0.0, walls=walls, room_id="r")
    assert r.surface_id == "r.w1" and r.kind == "wall"
    assert abs(r.width - 0.36) < 0.06 and abs(r.height - 0.36) < 0.06  # 5-95% extent of a 0.40 m box
    assert 1.2 < r.height_above_floor < 1.4


def test_low_stain_fires_cavity_rule_and_flood_cut_scope():
    room = {"id": "r", "walls": [{"id": "r.w1", "length": meas(4.0, 0.02, "m")}],
            "floor_area": meas(12.0, 0.3, "m2"), "ceiling_height": meas(2.7, 0.02, "m")}
    dmg = [{"id": "D1", "room_id": "r", "surface_id": "r.w1", "surface_kind": "wall", "class": "water_stain",
            "height_above_floor_m": 0.1, "area": meas(0.2, 0.02, "m2"), "extent_w": meas(0.5, 0.02, "m"),
            "extent_h": meas(0.4, 0.02, "m")}]
    flags, items = apply_rules(dmg, [room])
    assert [f["rule_id"] for f in flags] == ["R-WATER-BASE"]
    codes = {i["code"] for i in items}
    assert {"DRY-SEAL", "PNT-WALL", "DRY-FLOOD", "DRY-RPL"} <= codes
    paint = next(i for i in items if i["code"] == "PNT-WALL")
    assert abs(paint["quantity"]["value"] - 4.0 * 2.7) < 1e-6
    assert all(i["surface_id"] == "r.w1" for i in items)


def test_room_damage_stage_projects_merges_and_finalizes():
    from types import SimpleNamespace

    from roomscan.damage.stage import finalize, room_damage

    K, depth, c2w, walls = _wall_view()
    room = {"id": "r", "walls": walls, "floor_area": meas(12.0, 0.3, "m2"), "ceiling_height": meas(2.7, 0.02, "m")}
    img = np.zeros((200, 200, 3), np.uint8)
    # the same stain seen in two views -> one region with both frames as evidence
    raw = [{"image": i, "class": "water_stain", "x0": 400, "y0": 400, "x1": 600, "y1": 600, "confidence": 0.8,
            "note": ""} for i in (0, 1)]
    cfg = {"damage": {"backend": "vlm", "keyframes_per_room": 10}}
    err = SimpleNamespace(abs_m=0.03, rel=0.05)
    regions = room_damage(room, [img, img], np.stack([depth, depth]), np.stack([K, K]), np.stack([c2w, c2w]), 0.0,
                          0.0, err, cfg, None, [], ["a.jpg", "b.jpg"], detector=lambda ims: (raw, "synthetic"))
    assert len(regions) == 1 and regions[0]["evidence_frames"] == ["a.jpg", "b.jpg"]
    w = regions[0]["extent_w"]
    assert w["lo"] < 0.36 < w["hi"]
    plan = {"rooms": [room]}
    finalize(plan, regions)
    assert plan["damage"][0]["id"] == "D1" and "source" not in plan["damage"][0]
    assert {s["code"] for s in plan["scope"]} == {"DRY-SEAL", "PNT-WALL"}


def test_missing_sdk_or_key_degrades_to_warning(monkeypatch):
    import builtins

    from roomscan.damage.stage import room_damage

    real_import = builtins.__import__

    def no_anthropic(name, *a, **k):
        if name == "anthropic":
            raise ImportError("no anthropic")
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", no_anthropic)
    K, depth, c2w, walls = _wall_view()
    room = {"id": "r", "walls": walls, "ceiling_height": None}
    cfg = {"damage": {"backend": "vlm", "keyframes_per_room": 4, "vlm_model": "m", "cache": "replay_or_live"}}
    warns = []
    out = room_damage(room, [np.zeros((200, 200, 3), np.uint8)], depth[None], K[None], c2w[None], 0.0, 0.0, None,
                      cfg, "/nonexistent-cache", warns)
    assert out == [] and "detector unavailable" in warns[0] and cfg["damage"]["backend"] == "off"
