"""Check a photo-folder or video capture against the protocol before running the pipeline.

    python scripts/check_capture.py data/raw/photos/home_tape_a
    python scripts/check_capture.py data/raw/video/home_tape_walk1.MOV

Reports, per photo: lens (0.5x / 1x / other, from the 35 mm-equivalent focal in EXIF), orientation, size, and whether
the focal length needed for metric scale is present. Exit code 1 if anything blocks the pipeline.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

from PIL import ExifTags, Image

try:
    import pillow_heif

    pillow_heif.register_heif_opener()
except ImportError:  # HEIC unreadable without it; reported below
    pillow_heif = None

IMG_EXT = {".jpg", ".jpeg", ".heic", ".heif", ".png"}
VID_EXT = {".mov", ".mp4", ".m4v"}


def lens_name(f35: float | None) -> str:
    if f35 is None:
        return "unknown"
    if f35 < 18:
        return "0.5x ultra-wide"
    if f35 < 35:
        return "1x wide"
    return f"{f35 / 26:.0f}x tele"


def check_photo(p: Path) -> tuple[list[str], list[str], str]:
    errs, warns = [], []
    try:
        im = Image.open(p)
        w, h = im.size
        exif = im.getexif()
        sub = exif.get_ifd(ExifTags.IFD.Exif)
    except Exception as e:  # noqa: BLE001
        return [f"{p.name}: cannot read ({e})"], [], ""
    f35 = sub.get(0xA405)
    f_mm = sub.get(0x920A)
    model = exif.get(0x0110, "?")
    orient = exif.get(0x0112, 1)
    landscape = (w > h) != (orient in (5, 6, 7, 8))  # EXIF orientation 5-8 = rotated 90 deg
    if f35 is None and f_mm is None:
        errs.append(f"{p.name}: no focal length in EXIF (metric scale needs it). Was it edited or screenshotted?")
    lens = lens_name(float(f35) if f35 else None)
    if lens != "0.5x ultra-wide":
        warns.append(f"{p.name}: lens {lens}; protocol asks for 0.5x")
    if not landscape:
        warns.append(f"{p.name}: portrait; protocol asks for landscape (sideways)")
    return errs, warns, f"{p.name:22s} {w}x{h} {'landscape' if landscape else 'portrait '} {lens:16s} f35={f35} {model}"


def check_photos(root: Path) -> int:
    rooms = sorted(d for d in root.iterdir() if d.is_dir() and not d.name.startswith("."))
    loose = [f for f in root.iterdir() if f.suffix.lower() in IMG_EXT]
    errs, warns = [], []
    if loose:
        errs.append(f"{len(loose)} photos sit directly in {root.name}/; put them in one sub-folder per room")
    if not rooms:
        errs.append("no room folders found")
    for d in rooms:
        imgs = sorted(f for f in d.iterdir() if f.suffix.lower() in IMG_EXT)
        print(f"\n[{d.name}] {len(imgs)} photos")
        if not 2 <= len(imgs) <= 8:
            errs.append(f"{d.name}: {len(imgs)} photos; the photo tier takes 2-8 per room")
        elif len(imgs) != 8:
            warns.append(f"{d.name}: {len(imgs)} photos; protocol asks for 8")
        if any(f.suffix.lower() in {".heic", ".heif"} for f in imgs) and pillow_heif is None:
            errs.append("HEIC photos but pillow-heif is not installed")
        for f in imgs:
            e, w, line = check_photo(f)
            errs += e
            warns += w
            if line:
                print("  " + line)
    return report(errs, warns)


def check_video(p: Path) -> int:
    import imageio_ffmpeg

    out = subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-hide_banner", "-i", str(p)], capture_output=True,
                         text=True).stderr
    errs, warns = [], []
    dur = re.search(r"Duration: (\d+):(\d+):([\d.]+)", out)
    stream = re.search(r"Video: (\w+).*?, (\d{3,5})x(\d{3,5}).*?, ([\d.]+) fps", out)
    rot = re.search(r"rotat\w* of (-?[\d.]+)|rotate\s*:\s*(-?\d+)", out)
    f35 = re.search(r"focal_length\.35mm_equivalent\s*:\s*([\d.]+)", out)
    model = re.search(r"com\.apple\.quicktime\.model\s*:\s*(.+)", out)
    secs = int(dur.group(1)) * 3600 + int(dur.group(2)) * 60 + float(dur.group(3)) if dur else 0
    print(f"{p.name}: {secs:.0f} s, " + (f"{stream.group(1)} {stream.group(2)}x{stream.group(3)} @ {stream.group(4)} fps"
                                          if stream else "no video stream found"))
    print(f"  rotation: {rot.group(1) or rot.group(2) if rot else 0}   device: {model.group(1).strip() if model else '?'}"
          f"   35mm-equivalent focal: {f35.group(1) if f35 else 'not in metadata'}")
    if not stream:
        errs.append("no readable video stream")
    if secs and secs < 20:
        warns.append(f"only {secs:.0f} s; protocol expects ~30-45 s per room")
    if not f35:
        warns.append("no focal length in the file metadata: metric scale will need the lens looked up from the phone model (not built yet)")
    elif lens_name(float(f35.group(1))) != "0.5x ultra-wide":
        warns.append(f"lens {lens_name(float(f35.group(1)))}; protocol asks for 0.5x")
    return report(errs, warns)


def report(errs: list[str], warns: list[str]) -> int:
    print()
    for w in warns:
        print(f"  WARN  {w}")
    for e in errs:
        print(f"  ERROR {e}")
    print("\nOK to run." if not errs else "\nFix the errors above before running.")
    return 1 if errs else 0


def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__)
        return 2
    p = Path(sys.argv[1])
    if p.is_dir():
        return check_photos(p)
    if p.suffix.lower() in VID_EXT:
        return check_video(p)
    print(f"{p}: expected a folder of room folders or a video file")
    return 2


if __name__ == "__main__":
    sys.exit(main())
