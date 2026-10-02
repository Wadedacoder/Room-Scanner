"""Video tier input: decode a phone video into upright frames with ffmpeg (bundled via imageio-ffmpeg)."""

from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass
class VideoInfo:
    path: Path
    duration_s: float
    width: int  # as displayed (after rotation)
    height: int
    fps: float
    model: str | None
    hdr: bool


def ffmpeg_exe() -> str:
    import imageio_ffmpeg

    return imageio_ffmpeg.get_ffmpeg_exe()


def probe(path: Path) -> VideoInfo:
    txt = subprocess.run([ffmpeg_exe(), "-hide_banner", "-i", str(path)], capture_output=True, text=True).stderr
    dur = re.search(r"Duration: (\d+):(\d+):([\d.]+)", txt)
    st = re.search(r"Video: .*?, (\d{3,5})x(\d{3,5})[^,]*, .*?([\d.]+) fps", txt)
    rot = re.search(r"rotation of (-?[\d.]+)", txt)
    model = re.search(r"com\.apple\.quicktime\.model\s*:\s*(.+)", txt)
    w, h = int(st.group(1)), int(st.group(2))
    if rot and abs(abs(float(rot.group(1))) - 90) < 1:
        w, h = h, w
    secs = int(dur.group(1)) * 3600 + int(dur.group(2)) * 60 + float(dur.group(3))
    return VideoInfo(Path(path), secs, w, h, float(st.group(3)), model.group(1).strip() if model else None,
                     "arib-std-b67" in txt or "smpte2084" in txt)


def extract_frames(path: Path, out_dir: Path, fps: float, long_side: int) -> list[str]:
    """Upright frames (ffmpeg applies the rotation flag) at `fps`, scaled so the long side is `long_side`.
    HDR (HLG/PQ) is squeezed to 8-bit SDR; good enough for geometry, colours look flat."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for f in out_dir.glob("f_*.jpg"):
        f.unlink()
    vf = f"fps={fps},scale='if(gt(iw,ih),{long_side},-2)':'if(gt(iw,ih),-2,{long_side})',format=yuvj420p"
    subprocess.run([ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-i", str(path), "-vf", vf, "-q:v", "3",
                    str(out_dir / "f_%05d.jpg")], check=True)
    return sorted(p.name for p in out_dir.glob("f_*.jpg"))


# Main (1x) camera horizontal field of view across the long side of a video frame, by phone family. Only a starting
# guess for COLMAP, which refines it; 0.5x is roughly double the angle.
FOV_1X_DEG = 65.0
