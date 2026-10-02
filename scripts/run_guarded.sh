#!/usr/bin/env bash
# Run a command with a memory watchdog: kill it if its resident memory (incl. children) exceeds LIMIT_MB.
# macOS ignores `ulimit -v`, and on an 8 GB laptop a runaway process freezes the whole machine (2026-10-02 crash),
# so long benchmark runs go through this. Logs peak memory every 2 s to stderr.
#   LIMIT_MB=3500 scripts/run_guarded.sh .venv/bin/python -m roomscan.cli run <capture> -o out/
set -uo pipefail
LIMIT_MB="${LIMIT_MB:-3500}"
"$@" &
PID=$!
PEAK=0
while kill -0 "$PID" 2>/dev/null; do
  # sum RSS (KB) of the process and its children
  RSS=$(ps -o rss= -p "$PID" $(pgrep -P "$PID" 2>/dev/null) 2>/dev/null | awk '{s+=$1} END {print int(s/1024)}')
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
