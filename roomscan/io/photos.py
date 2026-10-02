"""Photo tier input: a folder with one sub-folder of 2-8 stills per room. Intrinsics come from EXIF."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import ExifTags, Image, ImageOps

try:
    import pillow_heif

    pillow_heif.register_heif_opener()
except ImportError:  # pragma: no cover
    pass

IMG_EXT = {".jpg", ".jpeg", ".heic", ".heif", ".png"}
DIAG_35MM = float(np.hypot(36.0, 24.0))  # 43.27 mm: 35 mm-equivalent focal is defined on this diagonal


@dataclass
class Photo:
    path: Path
    image: np.ndarray  # upright RGB (EXIF orientation applied)
    K: np.ndarray  # 3x3 at this image's resolution
    focal_source: str


def intrinsics_from_exif(img: Image.Image, w: int, h: int) -> tuple[np.ndarray, str]:
    sub = img.getexif().get_ifd(ExifTags.IFD.Exif)
    f35 = sub.get(0xA405)
    if not f35:
        raise ValueError("no 35 mm-equivalent focal length in EXIF; metric scale needs it (was the photo re-saved?)")
    f_px = float(f35) * float(np.hypot(w, h)) / DIAG_35MM
    return np.array([[f_px, 0, w / 2], [0, f_px, h / 2], [0, 0, 1.0]]), f"exif f35={f35}"


def load_room(folder: Path, max_side: int = 1920) -> list[Photo]:
    out = []
    for p in sorted(f for f in folder.iterdir() if f.suffix.lower() in IMG_EXT):
        raw = Image.open(p)
        up = ImageOps.exif_transpose(raw).convert("RGB")
        s = min(1.0, max_side / max(up.size))
        if s < 1.0:
            up = up.resize((round(up.width * s), round(up.height * s)), Image.LANCZOS)
        K, src = intrinsics_from_exif(raw, up.width, up.height)
        out.append(Photo(p, np.asarray(up), K, src))
    if not 2 <= len(out) <= 8:
        raise ValueError(f"{folder.name}: {len(out)} photos; the photo tier takes 2-8 per room")
    return out


def load_capture(root: Path) -> dict[str, list[Photo]]:
    rooms = sorted(d for d in Path(root).iterdir() if d.is_dir() and not d.name.startswith((".", "_")))
    if not rooms:
        raise ValueError(f"{root}: no room folders")
    return {d.name: load_room(d) for d in rooms}
