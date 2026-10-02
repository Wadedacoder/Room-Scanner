#!/usr/bin/env bash
# Health check (run every 30 min by the session watch; also by hand). Prints one line per check, then a verdict,
# and appends a timestamped summary to runs/health.log. Read-only: it never starts or kills a pipeline job.
set -uo pipefail
cd "$(dirname "$0")/.."
PASS=0; FAIL=0; WARN=0
ok()   { echo "  ok    $1"; PASS=$((PASS+1)); }
bad()  { echo "  FAIL  $1"; FAIL=$((FAIL+1)); }
warn() { echo "  warn  $1"; WARN=$((WARN+1)); }
echo "health $(date '+%Y-%m-%d %H:%M:%S')"

# 1. unit tests (fast; the end-to-end LiDAR test is skipped if data is missing)
if PYTHONPATH=. .venv/bin/pytest -q -x >/tmp/health_pytest.txt 2>&1; then
  ok "tests: $(tail -1 /tmp/health_pytest.txt | sed 's/\x1b\[[0-9;]*m//g')"
else
  bad "tests: $(grep -E 'FAILED|Error' /tmp/health_pytest.txt | head -2 | tr '\n' ' ')"
fi

# 2. test website answers
if curl -s -m 5 localhost:8765/api/info >/tmp/health_web.txt 2>/dev/null && grep -q version /tmp/health_web.txt; then
  ok "website up ($(.venv/bin/python -c "import json;print('v'+json.load(open('/tmp/health_web.txt'))['version'])"))"
else
  warn "website not answering on :8765 (start: PYTHONPATH=. .venv/bin/python -m roomscan.cli serve)"
fi

# 3. memory headroom (the 8 GB laptop froze once at 1% free)
FREE=$(memory_pressure -Q 2>/dev/null | grep -oE '[0-9]+%' | head -1 | tr -d '%')
if [ -z "$FREE" ]; then warn "memory: unknown"
elif [ "$FREE" -lt 15 ]; then bad "memory: only ${FREE}% free"
elif [ "$FREE" -lt 30 ]; then warn "memory: ${FREE}% free"
else ok "memory: ${FREE}% free"; fi

# 4. stuck pipeline jobs (anything running > 20 min is suspect; the house_b hang ran 15+ min at ~36% CPU)
# macOS ps has no etimes: convert etime ([[dd-]hh:]mm:ss) to seconds
STUCK=$(ps -o etime=,pid=,command= -ax | awk '
  /roomscan\.cli run|roomscan\.recon\.sfm|bench\/local\// && !/awk/ {
    n=split($1,a,/[-:]/); s=0
    if (n==4) s=a[1]*86400+a[2]*3600+a[3]*60+a[4]; else if (n==3) s=a[1]*3600+a[2]*60+a[3]; else s=a[1]*60+a[2]
    if (s>1200) print s, $2 }')
if [ -n "$STUCK" ]; then bad "long-running jobs: $(echo "$STUCK" | awk '{printf "pid %s %ds; ", $2, $1}')"
else ok "no pipeline job running > 20 min"; fi

# 5. disk space
AVAIL=$(df -g . | awk 'NR==2{print $4}')
[ "${AVAIL:-0}" -lt 5 ] && bad "disk: ${AVAIL} GB free" || ok "disk: ${AVAIL} GB free"

# 6. git: uncommitted work and unpushed commits
DIRTY=$(git status --porcelain | grep -v '^??' | wc -l | tr -d ' ')
AHEAD=$(git rev-list --count origin/main..main 2>/dev/null || echo "?")
[ "$DIRTY" -gt 0 ] && warn "git: $DIRTY modified tracked files not committed" || ok "git: working tree clean"
[ "$AHEAD" != "0" ] && warn "git: $AHEAD commits not pushed" || ok "git: in sync with origin"

# 7. models usable offline (cached weights)
if HF_HUB_OFFLINE=1 .venv/bin/python -c "
from huggingface_hub import try_to_load_from_cache as t
import sys
missing=[r for r in ('depth-anything/DA3-BASE','depth-anything/DA3METRIC-LARGE') if not isinstance(t(r,'model.safetensors'),str)]
sys.exit(1 if missing else 0)" 2>/dev/null; then ok "model weights cached (offline-ready)"
else warn "model weights not fully cached: run scripts/setup_models.sh with network"; fi

VERDICT=$([ $FAIL -gt 0 ] && echo FAIL || ([ $WARN -gt 0 ] && echo WARN || echo OK))
echo "verdict: $VERDICT ($PASS ok, $WARN warn, $FAIL fail)"
mkdir -p runs && echo "$(date '+%F %T') $VERDICT ok=$PASS warn=$WARN fail=$FAIL" >> runs/health.log
