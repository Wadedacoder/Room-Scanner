"""Pipeline stage: per-room views -> damage regions on surfaces -> concealed flags + scope (rules.yaml).

Every tier calls `room_damage` once per room with the views it already has (image, metric depth, depth-resolution K,
gravity-aligned camera-to-world pose), then `finalize` once per plan. The detector is pluggable: `damage.backend`
= vlm (Claude, detect_claude.py) | off. With no credentials the stage warns and reports no damage rather than failing,
so geometry still ships.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from roomscan.damage.project import Detection, project_detection
from roomscan.pipeline.backend import meas
from roomscan.scope.engine import apply_rules

Z90 = 1.645


def _pick_views(n: int, k: int) -> list[int]:
    return list(range(n)) if n <= k else np.linspace(0, n - 1, k).round().astype(int).tolist()


def room_damage(room: dict, images, depth: np.ndarray, K: np.ndarray, c2w: np.ndarray, floor_y: float, theta: float,
                err, cfg, cache_root: Path, warnings: list[str], view_names: list[str] | None = None,
                detector=None) -> list[dict]:
    """Damage regions (plan.json DamageRegion + internal fields) for one room; walls in the room's own plan frame.

    detector: optional callable(images) -> (raw detections on a 0-1000 grid, source); defaults to the configured backend.
    """
    if cfg["damage"]["backend"] == "off" and detector is None:
        return []
    idx = _pick_views(len(images), min(8, cfg["damage"]["keyframes_per_room"]))
    imgs = [images[i] for i in idx]
    dh, dw = depth.shape[1:]
    ih, iw = imgs[0].shape[:2]
    if abs(dw / dh - iw / ih) > 0.02:
        warnings.append(f"{room['id']}: depth aspect {dw}x{dh} differs from the image {iw}x{ih}; damage skipped")
        return []
    if detector is None:
        from roomscan.damage import detect_claude as dc

        try:
            raw, source = dc.detect_room(imgs, cfg, cache_root)
        except dc.NoCredentials as e:
            warnings.append(f"damage: detector unavailable ({str(e)[:120]}; set ANTHROPIC_API_KEY); damage, concealed "
                            "flags and scope are empty for this run")
            cfg["damage"]["backend"] = "off"  # don't retry for every room
            return []
        except FileNotFoundError as e:
            warnings.append(f"{room['id']}: {e}")
            return []
    else:
        raw, source = detector(imgs)
    ceil = room.get("ceiling_height")
    ceiling_y = floor_y + ceil["value"] if ceil else None
    regions = []
    for det in _to_view(raw, idx, (dh, dw), source):
        i = det.view
        r = project_detection(det, depth[i], K[i], c2w[i], floor_y, ceiling_y, theta, room["walls"], room["id"])
        if r is None:
            continue
        z = float(np.median(depth[i][depth[i] > 0])) if (depth[i] > 0).any() else 2.0
        px = z / K[i][0, 0]  # one depth pixel's footprint at the region
        sw = float(np.hypot(err.abs_m + px, err.rel * r.width))
        sh = float(np.hypot(err.abs_m + px, err.rel * r.height))
        regions.append({"room_id": room["id"], "surface_id": r.surface_id, "surface_kind": r.kind, "class": det.cls,
                        "polygon_on_surface": r.polygon, "area": meas(r.area, np.hypot(r.width * sh, r.height * sw),
                                                                      "m2"),
                        "extent_w": meas(r.width, sw, "m"), "extent_h": meas(r.height, sh, "m"),
                        "confidence": round(det.confidence, 3), "height_above_floor_m": r.height_above_floor,
                        "evidence_frames": [view_names[i] if view_names else str(i)], "source": source})
    return _merge(regions)


def _to_view(raw, idx, hw, source) -> list[Detection]:
    h, w = hw
    out = []
    for d in raw:
        if d["image"] < 0 or d["image"] >= len(idx) or d["confidence"] < 0.5:
            continue
        box = (d["x0"] / 1000 * w, d["y0"] / 1000 * h, d["x1"] / 1000 * w, d["y1"] / 1000 * h)
        out.append(Detection(idx[d["image"]], d["class"], box, float(d["confidence"]), source))
    return out


def _rect(r):
    p = np.array(r["polygon_on_surface"])
    return p[:, 0].min(), p[:, 1].min(), p[:, 0].max(), p[:, 1].max()


def _iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def _merge(regions: list[dict], min_iou: float = 0.2) -> list[dict]:
    """The same stain seen in several views -> one region (most confident view's geometry, evidence from all)."""
    out = []
    for r in sorted(regions, key=lambda r: -r["confidence"]):
        same = next((o for o in out if o["surface_id"] == r["surface_id"] and o["class"] == r["class"]
                     and _iou(_rect(o), _rect(r)) >= min_iou), None)
        if same is None:
            out.append(r)
        else:
            same["evidence_frames"] = sorted(set(same["evidence_frames"]) | set(r["evidence_frames"]))
    return out


def finalize(plan: dict, regions: list[dict]) -> None:
    """Number the regions, run the rules, and fill plan['damage'|'concealed_flags'|'scope'] (schema fields only)."""
    for k, r in enumerate(regions):
        r["id"] = f"D{k + 1}"
    flags, items = apply_rules(regions, plan["rooms"])
    keep = ("id", "surface_id", "class", "polygon_on_surface", "area", "extent_w", "extent_h", "confidence",
            "evidence_frames")
    plan["damage"] = [{k: r[k] for k in keep} for r in regions]
    plan["concealed_flags"] = flags
    plan["scope"] = items


def views_in_room(room: dict, cam_uv: np.ndarray, k: int) -> list[int]:
    """Up to k view indices whose camera stands inside the room outline (plan frame), spread over the capture."""
    from matplotlib.path import Path as MplPath

    inside = np.nonzero(MplPath(np.array(room["polygon"])).contains_points(cam_uv))[0]
    return [int(inside[i]) for i in _pick_views(len(inside), k)] if len(inside) else []


def stray_rgb_frames(root: Path, frame_numbers: list[int], long_side: int = 1568) -> dict[int, np.ndarray]:
    """Decode chosen frames of a Stray capture's rgb.mp4 (one sequential ffmpeg pass; HEVC seeking is unreliable)."""
    import subprocess
    import tempfile

    from PIL import Image

    from roomscan.io.video import ffmpeg_exe

    if not frame_numbers:
        return {}
    sel = "+".join(f"eq(n\\,{n})" for n in sorted(frame_numbers))
    with tempfile.TemporaryDirectory() as td:
        subprocess.run([ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-i", str(Path(root) / "rgb.mp4"),
                        "-vf", f"select='{sel}',scale='if(gt(iw,ih),{long_side},-2)':'if(gt(iw,ih),-2,{long_side})'",
                        "-vsync", "0", "-q:v", "3", f"{td}/%05d.jpg"], check=True)
        files = sorted(Path(td).glob("*.jpg"))
        return {n: np.asarray(Image.open(f).convert("RGB")) for n, f in zip(sorted(frame_numbers), files)}
