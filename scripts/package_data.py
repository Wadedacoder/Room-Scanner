"""Package the benchmark captures as release zips + a checksummed manifest (bench/datasets/release_manifest.json).

    python scripts/package_data.py            # -> dist/data/*.zip, manifest updated
    # then upload dist/data/*.zip as assets of the GitHub release `data-v1`; scripts/fetch_data.sh downloads and verifies

Privacy check: refuses to package a photo or video that carries GPS location metadata (these are rooms in a home).
"""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "dist/data"
MANIFEST = ROOT / "bench/datasets/release_manifest.json"
# release name -> (source path, path inside data/raw/ it unpacks to)
CAPTURES = {
    "photos_house_b": ("data/raw/photos/house_b", "photos/house_b"),
    "video_study_walk1": ("data/raw/video/study_walk1.MOV", "video/study_walk1.MOV"),
    "video_house_b_walk": ("data/raw/video/house_b_walk.MOV", "video/house_b_walk.MOV"),
}
# the LiDAR scans provided with the case study: shipped as the original zips (each holds one <scan id>/ folder)
LIDAR_ZIPS = ["single_room", "single_scan_floor_only", "single_scan_with_ceiling"]


def has_gps(path: Path) -> bool:
    if path.suffix.lower() in (".heic", ".jpg", ".jpeg"):
        import pillow_heif
        from PIL import Image

        pillow_heif.register_heif_opener()
        return bool(Image.open(path).getexif().get_ifd(0x8825))
    if path.suffix.lower() in (".mov", ".mp4"):
        import imageio_ffmpeg

        txt = subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-hide_banner", "-i", str(path)], capture_output=True,
                             text=True).stderr.lower()
        return "location" in txt or "iso6709" in txt
    return False


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = {"release": "data-v1", "assets": {}}
    for name, (src, dest) in CAPTURES.items():
        src = (ROOT / src).resolve()  # folders may be symlinks into data/raw/photos/house_b_src
        files = [src] if src.is_file() else sorted(p for p in src.rglob("*") if p.is_file()
                                                   and not p.name.startswith("."))
        flagged = [str(p) for p in files if has_gps(p)]
        if flagged:
            print(f"REFUSED {name}: GPS metadata in {flagged[:3]}; strip it first", file=sys.stderr)
            return 1
        zpath = OUT / f"{name}.zip"
        with zipfile.ZipFile(zpath, "w", zipfile.ZIP_STORED) as z:  # media is already compressed
            for p in files:
                z.write(p, str(Path(dest) / p.relative_to(src)) if src.is_dir() else dest)
        manifest["assets"][f"{name}.zip"] = {"sha256": sha256(zpath), "bytes": zpath.stat().st_size,
                                             "files": len(files), "unpacks_to": f"data/raw/{dest}"}
        print(f"{zpath.name}: {len(files)} files, {zpath.stat().st_size / 1e6:.1f} MB, no GPS")
    for name in LIDAR_ZIPS:
        src = ROOT / "dataset" / f"{name}.zip"
        zpath = OUT / f"lidar_{name}.zip"
        if not zpath.exists() or zpath.stat().st_size != src.stat().st_size:
            shutil.copyfile(src, zpath)
        with zipfile.ZipFile(zpath) as z:
            scan = z.namelist()[0].split("/")[0]
            n = len(z.namelist())
        manifest["assets"][zpath.name] = {"sha256": sha256(zpath), "bytes": zpath.stat().st_size, "files": n,
                                          "unpacks_to": f"data/raw/lidar/{scan}"}
        print(f"{zpath.name}: scan {scan}, {zpath.stat().st_size / 1e6:.1f} MB")
    MANIFEST.write_text(json.dumps(manifest, indent=1) + "\n")
    print(f"manifest -> {MANIFEST.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
