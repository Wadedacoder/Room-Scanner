#!/usr/bin/env bash
# Fetch raw benchmark captures into data/raw/. Large binaries are never committed (case-study constraint).
# TODO: upload the capture zips as GitHub release assets and list them here.
set -euo pipefail
cd "$(dirname "$0")/.."
REL="https://github.com/Wadedacoder/Room-Scanner/releases/download/data-v1"
mkdir -p data/raw/lidar
for name in single_room single_scan_floor_only single_scan_with_ceiling; do
  zip="data/raw/${name}.zip"
  [ -f "$zip" ] || curl -fL "$REL/${name}.zip" -o "$zip"
  unzip -nq "$zip" -d data/raw/lidar
done
echo "captures in data/raw/lidar:"; ls data/raw/lidar
