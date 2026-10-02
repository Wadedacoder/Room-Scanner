#!/usr/bin/env bash
# Fetch the benchmark captures into data/raw/ and verify them against bench/datasets/release_manifest.json.
# Large binaries are never committed (case-study constraint); they are assets of the GitHub release `data-v1`,
# built by scripts/package_data.py.
#   scripts/fetch_data.sh                 # everything (~1.6 GB)
#   scripts/fetch_data.sh photos video    # only assets whose name starts with these prefixes
set -euo pipefail
cd "$(dirname "$0")/.."
REL="${ROOMSCAN_DATA_URL:-https://github.com/Wadedacoder/Room-Scanner/releases/download/data-v1}"
mkdir -p data/downloads data/raw/lidar
python3 - "$@" <<'PY' > data/downloads/.want
import json, sys
m = json.load(open("bench/datasets/release_manifest.json"))["assets"]
pre = sys.argv[1:]
for name, a in m.items():
    if not pre or any(name.startswith(p) for p in pre):
        print(name, a["sha256"], a["unpacks_to"])
PY
while read -r name sha dest; do
  zip="data/downloads/$name"
  if [ -e "$dest" ]; then echo "have  $dest"; continue; fi
  [ -f "$zip" ] || curl -fL --retry 3 "$REL/$name" -o "$zip"
  got=$(shasum -a 256 "$zip" | cut -d' ' -f1)
  if [ "$got" != "$sha" ]; then echo "CHECKSUM MISMATCH: $name (got $got)"; rm -f "$zip"; exit 1; fi
  case "$name" in
    lidar_*) unzip -nq "$zip" -d data/raw/lidar ;;
    *)       unzip -nq "$zip" -d data/raw ;;
  esac
  echo "ok    $dest"
done < data/downloads/.want
