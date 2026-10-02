"""Checkpoints for model outputs, keyed by a hash of everything that determines them.

A step's output (DA3 reconstruction of a room, COLMAP solve of a video) is stored under cache/<kind>/<key>.npz where
key = sha256(inputs, settings, CODE_VERSION). Effects:
  * a rerun or a resumed run after a crash reuses finished steps (a 6-room house redoes only the room that failed);
  * identical inputs replay identical outputs (the brief accepts cached model outputs if the live path also runs);
  * any change to inputs, settings or the step's code version misses the cache, so stale results are never reused.
Set ROOMSCAN_NO_CACHE=1 to force the live path.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np

# bump when a step's code changes what it outputs
CODE_VERSION = {"da3_recon": "2", "sfm": "3", "links": "1", "video_matches": "1"}


def enabled() -> bool:
    return os.environ.get("ROOMSCAN_NO_CACHE", "") not in ("1", "true", "yes")


def key_for(kind: str, arrays: list[np.ndarray] = (), meta: dict | None = None, files: list[Path] = ()) -> str:
    h = hashlib.sha256()
    h.update(kind.encode())
    h.update(CODE_VERSION.get(kind, "0").encode())
    h.update(json.dumps(meta or {}, sort_keys=True, default=str).encode())
    for a in arrays:
        a = np.ascontiguousarray(a)
        h.update(str(a.shape).encode())
        h.update(str(a.dtype).encode())
        h.update(a.tobytes())
    for f in files:
        with open(f, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
    return h.hexdigest()[:24]


def path_for(cache_dir: str | Path, kind: str, key: str, suffix: str = ".npz") -> Path:
    p = Path(cache_dir) / kind / f"{key}{suffix}"
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def load_npz(path: Path) -> dict | None:
    if not enabled() or not path.exists():
        return None
    try:
        with np.load(path, allow_pickle=False) as z:
            return {k: z[k] for k in z.files}
    except Exception:  # noqa: BLE001  (a half-written file from a crash: ignore and recompute)
        return None


def save_npz(path: Path, **arrays) -> None:
    tmp = path.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, **arrays)
    tmp.replace(path)  # atomic: a crash never leaves a truncated checkpoint under the real name
