"""Damage detection with a local open-vocabulary detector (OWLv2, Apache-2.0): no API key, no network.

google/owlv2-base-patch16-ensemble is prompted with text descriptions of each damage class; boxes above
`damage.owlv2_threshold` are kept after per-image non-maximum suppression. Same output as the Claude backend
(boxes on a 0-1000 grid per image), so projection, rules and scope are shared.

Accuracy is NOT measured: there are no staged-damage captures yet. OWLv2 scores are not probabilities; the
`confidence` field carries the raw score, and the threshold is a setting, not a calibrated value.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

MODEL = "google/owlv2-base-patch16-ensemble"
PROMPT_VERSION = "3"
QUERIES = {
    "water_stain": ["a brown water stain", "yellow tide marks from a leak", "a damp patch"],
    "mold": ["black mould spots", "patches of mold growth"],
    "crack": ["a thin crack line in plaster", "a jagged crack"],
    "hole": ["a hole punched in drywall"],
    "peeling_paint": ["flaking peeling paint", "bubbling blistered paint"],
    "fire_smoke": ["black soot marks from smoke"],
}
# E24: thin straight lines (tile grout, skirting edges) score up to 0.28 as "crack" on clean rooms, so cracks need a
# higher score than the global threshold. 0.30 was chosen on the same 17 clean photos it is reported on (in-sample).
CLASS_MIN = {"crack": 0.30}
# E24: a damage query that names a surface ("a cracked ceiling") matches the whole clean surface. Undamaged things are
# queried too; a box whose best label is one of these is dropped, and it suppresses overlapping damage boxes in NMS.
NEGATIVES = ["a plain white wall", "a painted wall", "a ceiling", "a floor", "a window", "a door", "a shadow",
             "a light reflection", "a picture frame", "furniture", "a curtain", "an electrical switch", "a cable",
             # E24 round 2: the 4 false positives on 17 clean house photos were these
             "a decorative tile border", "wall tiles", "a door frame", "a skirting board", "a tap"]


def _nms(boxes: np.ndarray, scores: np.ndarray, iou: float = 0.4) -> list[int]:
    keep, order = [], np.argsort(-scores)
    while len(order):
        i = order[0]
        keep.append(int(i))
        x0 = np.maximum(boxes[i, 0], boxes[order[1:], 0])
        y0 = np.maximum(boxes[i, 1], boxes[order[1:], 1])
        x1 = np.minimum(boxes[i, 2], boxes[order[1:], 2])
        y1 = np.minimum(boxes[i, 3], boxes[order[1:], 3])
        inter = np.clip(x1 - x0, 0, None) * np.clip(y1 - y0, 0, None)
        area = lambda b: (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
        u = area(boxes[[i]]) + area(boxes[order[1:]]) - inter
        order = order[1:][inter / np.maximum(u, 1e-9) < iou]
    return keep


def detect_room(images: list[np.ndarray], cfg, cache_root: Path) -> tuple[list[dict], str]:
    thr = float(cfg["damage"]["owlv2_threshold"])
    h = hashlib.sha256((MODEL + PROMPT_VERSION + json.dumps([QUERIES, NEGATIVES, CLASS_MIN], sort_keys=True) + str(thr)).encode())
    for im in images:
        h.update(np.ascontiguousarray(im).tobytes())
    path = Path(cache_root) / "vlm" / f"owlv2_{h.hexdigest()[:24]}.json"
    if cfg["damage"]["cache"] != "live" and path.exists():
        return json.loads(path.read_text())["detections"], f"owlv2:{MODEL} (cached)"
    if cfg["damage"]["cache"] == "replay":
        raise FileNotFoundError(f"damage.cache=replay but no cached response at {path}")

    import torch
    from PIL import Image
    from transformers import Owlv2ForObjectDetection, Owlv2Processor

    dev = cfg["runtime"]["device"]
    dev = dev if (dev != "mps" or torch.backends.mps.is_available()) else "cpu"
    try:
        proc = Owlv2Processor.from_pretrained(MODEL, local_files_only=True)
        model = Owlv2ForObjectDetection.from_pretrained(MODEL, local_files_only=True)
    except OSError:  # not cached yet: download once (scripts/setup_models.sh prefetches it)
        proc = Owlv2Processor.from_pretrained(MODEL)
        model = Owlv2ForObjectDetection.from_pretrained(MODEL)
    model = model.to(dev).eval()
    texts = [q for qs in QUERIES.values() for q in qs] + NEGATIVES
    cls_of = [c for c, qs in QUERIES.items() for _ in qs] + [None] * len(NEGATIVES)
    dets = []
    with torch.inference_mode():
        for k, im in enumerate(images):
            pil = Image.fromarray(im)
            inputs = proc(text=[texts], images=pil, return_tensors="pt").to(dev)
            out = model(**inputs)
            # OWLv2 pads to a square: boxes are relative to the padded square of side max(w, h)
            side = max(pil.size)
            post = getattr(proc, "post_process_grounded_object_detection", None) or proc.post_process_object_detection
            res = post(out, threshold=thr, target_sizes=[(side, side)])[0]
            b = res["boxes"].cpu().numpy()
            sc = res["scores"].cpu().numpy()
            lb = res["labels"].cpu().numpy()
            del inputs, out, res  # this image's device tensors (sweep: leftovers starved the next DA3 run)
            if not len(b):
                continue
            w, hh = pil.size
            b[:, [0, 2]] = np.clip(b[:, [0, 2]], 0, w)
            b[:, [1, 3]] = np.clip(b[:, [1, 3]], 0, hh)
            for i in _nms(b, sc):
                if cls_of[int(lb[i])] is None:  # best explained as something undamaged
                    continue
                if sc[i] < CLASS_MIN.get(cls_of[int(lb[i])], 0.0):
                    continue
                x0, y0, x1, y1 = b[i]
                if (x1 - x0) * (y1 - y0) > 0.6 * w * hh:  # a box covering most of the frame is the wall, not a stain
                    continue
                dets.append({"image": k, "class": cls_of[int(lb[i])], "x0": int(1000 * x0 / w), "y0": int(1000 * y0 / hh),
                             "x1": int(1000 * x1 / w), "y1": int(1000 * y1 / hh), "confidence": round(float(sc[i]), 4),
                             "note": texts[int(lb[i])]})
    del model, proc
    import gc

    gc.collect()
    if dev == "mps":
        torch.mps.empty_cache()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"model": MODEL, "prompt_version": PROMPT_VERSION, "threshold": thr,
                                "detections": dets}, indent=1))
    return dets, f"owlv2:{MODEL}"
