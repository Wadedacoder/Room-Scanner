"""Damage detection with Claude (vision), cached for deterministic replay.

One request per room: up to 8 photos/keyframes, each downscaled to <= 1568 px, asking for every visible damage region
as a class + box on a 0-1000 grid per image. Structured output (JSON schema) guarantees parseable results.

Replay: each response is stored in cache/vlm/<sha256 of images + prompt + model>.json, which IS committed, so a
benchmark rerun on another machine replays the same detections (the brief accepts cached model outputs when the live
path also runs). `damage.cache`: replay (cache only, error on a miss) | live (always call) | replay_or_live (default).

Credentials: anthropic.Anthropic() resolves ANTHROPIC_API_KEY / ANTHROPIC_AUTH_TOKEN / an `ant auth login` profile.
With none, NoCredentials is raised and the pipeline skips damage detection with a warning.
"""

from __future__ import annotations

import base64
import hashlib
import io
import json
from pathlib import Path

import numpy as np
from PIL import Image

CLASSES = ["water_stain", "mold", "crack", "hole", "peeling_paint", "fire_smoke", "other"]
PROMPT_VERSION = "1"
PROMPT = """You are inspecting interior photos of ONE room for building damage, for an insurance / restoration survey.

For every image, list each visible damage region: water stains or tide marks, mould, cracks, holes, peeling or
bubbling paint, fire or smoke marks. Only real damage on walls, ceiling or floor. Do not report normal wear, shadows,
reflections, furniture, pictures, wiring, or dirt that would wipe off.

Give each region a tight bounding box on a 0-1000 grid of THAT image (x0, y0 = top-left; x1, y1 = bottom-right;
0,0 is the image's top-left corner). Use class "other" only if it is clearly damage but none of the named classes.
If an image has no damage, return no detections for it. Confidence is your probability that this is real damage of
that class (0-1)."""

SCHEMA = {
    "type": "object",
    "properties": {
        "detections": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "image": {"type": "integer", "description": "0-based index of the image"},
                    "class": {"type": "string", "enum": CLASSES},
                    "x0": {"type": "integer"}, "y0": {"type": "integer"},
                    "x1": {"type": "integer"}, "y1": {"type": "integer"},
                    "confidence": {"type": "number"},
                    "note": {"type": "string"},
                },
                "required": ["image", "class", "x0", "y0", "x1", "y1", "confidence", "note"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["detections"],
    "additionalProperties": False,
}


class NoCredentials(RuntimeError):
    pass


def _jpeg_b64(img: np.ndarray, long_side: int = 1568) -> str:
    im = Image.fromarray(img)
    s = min(1.0, long_side / max(im.size))
    if s < 1.0:
        im = im.resize((round(im.width * s), round(im.height * s)), Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=88)
    return base64.standard_b64encode(buf.getvalue()).decode("ascii")


def detect_room(images: list[np.ndarray], cfg, cache_root: Path) -> tuple[list[dict], str]:
    """Returns (detections with boxes on the 0-1000 grid, source tag). Cached by content."""
    model = cfg["damage"]["vlm_model"]
    encoded = [_jpeg_b64(im) for im in images[:8]]
    h = hashlib.sha256((model + PROMPT_VERSION + PROMPT).encode())
    for e in encoded:
        h.update(e.encode())
    path = Path(cache_root) / "vlm" / f"{h.hexdigest()[:24]}.json"
    mode = cfg["damage"]["cache"]
    if mode != "live" and path.exists():
        d = json.loads(path.read_text())
        return d["detections"], f"claude:{d['model']} (cached)"
    if mode == "replay":
        raise FileNotFoundError(f"damage.cache=replay but no cached response at {path}")

    try:
        import anthropic
    except ImportError:
        raise NoCredentials("the anthropic package is not installed (pip install 'roomscan[vlm]')") from None
    try:
        client = anthropic.Anthropic()
        content = [{"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": e}}
                   for e in encoded]
        content.append({"type": "text", "text": f"There are {len(encoded)} images, indexed 0-{len(encoded) - 1}."})
        resp = client.beta.messages.create(
            model=model,
            max_tokens=16000,
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",  # on a safety decline the request is retried on a fallback model in the same call
            system=PROMPT,
            output_config={"format": {"type": "json_schema", "schema": SCHEMA}},
            messages=[{"role": "user", "content": content}],
        )
    except anthropic.AuthenticationError as e:
        raise NoCredentials(str(e)) from None
    except TypeError as e:  # the SDK found no credentials at all
        raise NoCredentials(str(e)) from None
    if resp.stop_reason == "refusal":
        dets = []
    else:
        text = next(b.text for b in resp.content if b.type == "text")
        dets = json.loads(text)["detections"]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"model": resp.model, "prompt_version": PROMPT_VERSION, "stop_reason": resp.stop_reason,
                                "request_id": getattr(resp, "_request_id", None), "detections": dets}, indent=1))
    return dets, f"claude:{resp.model}"

