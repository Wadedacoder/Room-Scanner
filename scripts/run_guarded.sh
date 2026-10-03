#!/usr/bin/env bash
# Run a command with a memory watchdog: kill it if its memory (incl. children) exceeds LIMIT_MB.
# Memory = macOS phys_footprint (`footprint`), which counts GPU (MPS / IOAccelerator), compressed and swapped pages;
# plain RSS misses all three (2026-10-03: a video run showed 200 MB RSS while thrashing 2.1 GB of swap, and a 1.6 GB
# MPS tensor showed as 295 MB RSS). Linux falls back to RSS.
# macOS ignores `ulimit -v`, and on an 8 GB laptop a runaway process freezes the whole machine (2026-10-02 crash),
# so long benchmark runs go through this. Logs peak memory every 2 s to stderr.
#   LIMIT_MB=3500 scripts/run_guarded.sh .venv/bin/python -m roomscan.cli run <capture> -o out/
# macOS on battery idle-sleeps after 1 min: a sleeping Mac froze runs mid-way (2026-10-03 "hangs" were sleeps, see
# pmset -g log). Keep the system awake for as long as this script runs.
if [ -z "${CAFFEINATED:-}" ] && command -v caffeinate >/dev/null 2>&1; then CAFFEINATED=1 exec caffeinate -i "$0" "$@"; fi
set -uo pipefail
LIMIT_MB="${LIMIT_MB:-5500}"  # user cap: 6 GB total
mem_mb() {  # total MB of the given pids
  if command -v footprint >/dev/null; then
    local t=0 m
    for p in "$@"; do
      m=$(footprint -p "$p" 2>/dev/null | awk '/Footprint:/ {for (i = 1; i < NF; i++) if ($i == "Footprint:") {v = $(i+1); u = $(i+2)}
                                       if (u == "GB") v *= 1024; else if (u == "KB") v /= 1024; print int(v); exit}')
      t=$((t + ${m:-0}))
    done
    echo "$t"
  else
    ps -o rss= -p "$@" 2>/dev/null | awk '{s+=$1} END {print int(s/1024)}'
  fi
}
"$@" &
PID=$!
PEAK=0
while kill -0 "$PID" 2>/dev/null; do
  RSS=$(mem_mb "$PID" $(pgrep -P "$PID" 2>/dev/null))
  RSS=${RSS:-0}
  [ "$RSS" -gt "$PEAK" ] && PEAK=$RSS
  if [ "$RSS" -gt "$LIMIT_MB" ]; then
    echo "[run_guarded] KILLED: ${RSS} MB > ${LIMIT_MB} MB limit" >&2
    pkill -9 -P "$PID" 2>/dev/null; kill -9 "$PID" 2>/dev/null
    wait "$PID" 2>/dev/null
    echo "[run_guarded] peak ${PEAK} MB" >&2
    exit 137
  fi
  sleep 2
done
wait "$PID"; CODE=$?
echo "[run_guarded] exit ${CODE}, peak ${PEAK} MB" >&2
exit $CODE
