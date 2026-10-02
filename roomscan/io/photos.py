"""Photo tier input: a folder with one sub-folder of 2-8 stills per room. Intrinsics come from EXIF.

Walk-in robustness (2026-10-03): an off-protocol capture degrades with a warning instead of failing the run:
photos straight in the capture folder = one room; > 8 photos = 8 evenly spaced; < 2 = room skipped; unreadable file =
skipped; no focal length in EXIF = the protocol's 0.5x lens is assumed and the room's scale interval is widened.
"""

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


ASSUMED_F35 = 13.0  # iPhone 0.5x ultra-wide, the protocol's lens


def intrinsics_from_exif(img: Image.Image, w: int, h: int) -> tuple[np.ndarray, str]:
    sub = img.getexif().get_ifd(ExifTags.IFD.Exif)
    f35 = sub.get(0xA405)
    if not f35:  # re-saved / messaged photos lose EXIF: assume the protocol lens (flagged; interval widened)
        f_px = ASSUMED_F35 * float(np.hypot(w, h)) / DIAG_35MM
        return np.array([[f_px, 0, w / 2], [0, f_px, h / 2], [0, 0, 1.0]]), "assumed 0.5x (no EXIF focal)"
    f_px = float(f35) * float(np.hypot(w, h)) / DIAG_35MM
    return np.array([[f_px, 0, w / 2], [0, f_px, h / 2], [0, 0, 1.0]]), f"exif f35={f35}"


def load_room(folder: Path, max_side: int = 1920, warnings: list[str] | None = None,
              files: list[Path] | None = None) -> list[Photo]:
    warnings = warnings if warnings is not None else []
    out = []
    for p in files or sorted(f for f in folder.iterdir() if f.suffix.lower() in IMG_EXT):
        try:
            raw = Image.open(p)
            up = ImageOps.exif_transpose(raw).convert("RGB")
        except Exception as e:  # noqa: BLE001  (corrupt / unsupported file: skip it, keep the room)
            warnings.append(f"{folder.name}: {p.name} unreadable ({str(e)[:60]}); skipped")
            continue
        s = min(1.0, max_side / max(up.size))
        if s < 1.0:
            up = up.resize((round(up.width * s), round(up.height * s)), Image.LANCZOS)
        K, src = intrinsics_from_exif(raw, up.width, up.height)
        out.append(Photo(p, np.asarray(up), K, src))
    if len(out) > 8:
        keep = np.linspace(0, len(out) - 1, 8).round().astype(int)
        warnings.append(f"{folder.name}: {len(out)} photos; the photo tier uses 8 per room, kept 8 evenly spaced")
        out = [out[i] for i in keep]
    if any(ph.focal_source.startswith("assumed") for ph in out):
        warnings.append(f"{folder.name}: no focal length in the photos' EXIF (re-saved or messaged?); assumed the "
                        "protocol's 0.5x lens, so this room's scale is uncertain (interval widened)")
    return out


def load_capture(root: Path, warnings: list[str] | None = None) -> dict[str, list[Photo]]:
    warnings = warnings if warnings is not None else []
    root = Path(root)
    rooms = sorted(d for d in root.iterdir() if d.is_dir() and not d.name.startswith((".", "_")))
    loose = sorted(f for f in root.iterdir() if f.suffix.lower() in IMG_EXT)
    out = {}
    if loose:
        if rooms:
            warnings.append(f"{len(loose)} photos sit directly in {root.name}/ next to room folders; ignored")
        else:
            warnings.append(f"no room folders in {root.name}/: treated all {len(loose)} photos as one room")
            rooms = []
            name = root.name if root.name else "room1"
            got = load_room(root, warnings=warnings, files=loose)
            if len(got) >= 2:
                out[name] = got
    for d in rooms:
        got = load_room(d, warnings=warnings)
        if len(got) < 2:
            warnings.append(f"{d.name}: {len(got)} usable photo(s); a room needs at least 2; skipped")
            continue
        out[d.name] = got
    if not out:
        raise ValueError(f"{root}: no room with at least 2 readable photos")
    return out
