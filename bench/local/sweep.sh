#!/usr/bin/env bash
# Robustness sweep: every capture on disk through the current pipeline; one line per capture (exit, runtime, rooms,
# schema validation is done by the CLI itself). Walk-in readiness: nothing may crash.
cd "$(dirname "$0")/../.."
OUT=runs/sweep; mkdir -p $OUT
for cap in data/raw/photos/* dataset/1a8384c3f6 dataset/c00a170fe1 dataset/c7d28f72c6 data/raw/video/*; do
  name=$(basename "$cap"); [[ "$name" == *_src ]] && continue
  t0=$(date +%s)
  LIMIT_MB=5500 PYTHONPATH=. HF_HUB_OFFLINE=1 scripts/run_guarded.sh .venv/bin/python -m roomscan.cli run "$cap" -o $OUT > "$OUT/$name.log" 2>&1
  rc=$?
  rooms=$(sed 's/\x1b\[[0-9;]*m//g' "$OUT/$name.log" | grep -cE "^  [a-z0-9_]+: [0-9.]+ m2")
  err=$(sed 's/\x1b\[[0-9;]*m//g' "$OUT/$name.log" | grep -E "Error|Traceback|KILLED" | tail -1 | cut -c1-120)
  echo "$name: exit $rc, $(( $(date +%s) - t0 )) s, rooms $rooms ${err}"
done
