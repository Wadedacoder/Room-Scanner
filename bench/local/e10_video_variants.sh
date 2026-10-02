#!/usr/bin/env bash
# E10: video tracking strategies on the study video, each under the memory watchdog, then scored against tape.
set -uo pipefail
cd "$(dirname "$0")/../.."
VID=data/raw/video/study_walk1.MOV
run() {  # name, overrides...
  local name=$1; shift
  local args=(); for o in "$@"; do args+=(-s "$o"); done
  echo "== $name $(date +%H:%M:%S)"
  LIMIT_MB=3500 PYTHONPATH=. scripts/run_guarded.sh .venv/bin/python -m roomscan.cli run "$VID" -o "runs/e10_video/$name" "${args[@]}" 2>&1 \
    | sed 's/\x1b\[[0-9;]*m//g' | grep -E "run_guarded|wrote|Error:"
}
run seq8_o10  recon.video_fps=8  recon.video_matcher=sequential recon.video_seq_overlap=10
run seq15_o20 recon.video_fps=15 recon.video_matcher=sequential recon.video_seq_overlap=20
run exh8      recon.video_fps=8  recon.video_matcher=exhaustive
