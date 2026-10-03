#!/usr/bin/env bash
# Run the pipeline on a Kaggle GPU (T4, 16 GB) and bring the plans back. Code and captures go up as PRIVATE datasets.
#   scripts/kaggle_run.sh [-p cuda-16gb] [-o out/kaggle] <capture> [<capture> ...] [-- extra roomscan args]
# e.g. scripts/kaggle_run.sh data/raw/photos/house_b data/raw/photos/house_b_study -- -s damage.backend=off
# Needs ~/.kaggle/access_token (or kaggle.json). Typical wall time: ~5 min setup + the runs.
set -euo pipefail
cd "$(dirname "$0")/.."
KAGGLE=.venv/bin/kaggle; USER=devmsjsj; KERNEL="$USER/roomscan-remote-run"
PROFILE=cuda-16gb; OUT=out/kaggle; CAPS=(); EXTRA=()
while [ $# -gt 0 ]; do
  case "$1" in
    -p) PROFILE="$2"; shift 2 ;;
    -o) OUT="$2"; shift 2 ;;
    --) shift; EXTRA=("$@"); break ;;
    *) CAPS+=("$1"); shift ;;
  esac
done
[ ${#CAPS[@]} -gt 0 ] || { echo "usage: $0 [-p profile] [-o out] <capture>... [-- args]"; exit 2; }
STAGE=data/derived/kaggle_remote; rm -rf "$STAGE"; mkdir -p "$STAGE/code" "$STAGE/caps/captures"

stage() {  # $1 slug, $2 dir containing the tar
  printf '{"title": "%s", "id": "%s/%s", "licenses": [{"name": "other"}]}\n' "$1" "$USER" "$1" > "$2/dataset-metadata.json"
  if $KAGGLE datasets status "$USER/$1" >/dev/null 2>&1; then
    $KAGGLE datasets version -p "$2" -m "kaggle_run $(date +%F_%T)" -q
  else
    $KAGGLE datasets create -p "$2" -q   # new datasets are private by default
  fi
  until $KAGGLE datasets status "$USER/$1" 2>&1 | grep -q ready; do sleep 10; done
}

# code: package + the configs and schema it reads at the repo root
export COPYFILE_DISABLE=1  # no macOS ._ AppleDouble files in the tars
tar --exclude='__pycache__' --exclude='._*' -cf "$STAGE/code/roomscan-code.tar" roomscan configs schema
# captures (symlinks followed) + the job list
python3 - "$PROFILE" "$STAGE/caps/captures" "${CAPS[@]}" -- "${EXTRA[@]+"${EXTRA[@]}"}" <<'PY'
import json, shutil, sys
from pathlib import Path
profile, dest, rest = sys.argv[1], Path(sys.argv[2]), sys.argv[3:]
sep = rest.index("--")
caps, extra = rest[:sep], [a for a in rest[sep + 1:] if a]
jobs = []
for c in caps:
    src = Path(c).resolve()
    tgt = dest / src.name
    (shutil.copytree(src, tgt, symlinks=False) if src.is_dir() else shutil.copy2(src, tgt))
    jobs.append({"capture": src.name, "args": ["-p", profile, *extra]})
(dest / "jobs.json").write_text(json.dumps(jobs, indent=1))
print(f"staged {len(jobs)} capture(s), profile {profile}, extra {extra}")
PY
tar --exclude='._*' --exclude='.DS_Store' -chf "$STAGE/caps/roomscan-captures.tar" -C "$STAGE/caps" captures && rm -rf "$STAGE/caps/captures"
stage roomscan-code "$STAGE/code"
stage roomscan-captures "$STAGE/caps"

$KAGGLE kernels push -p bench/kaggle/remote_run --accelerator NvidiaTeslaT4
echo "pushed $KERNEL; waiting…"
sleep 30
while true; do
  st=$($KAGGLE kernels status "$KERNEL" 2>&1 | tr '[:upper:]' '[:lower:]')
  case "$st" in
    *complete*) break ;;
    *error*|*cancel*) echo "kernel failed: $st"; break ;;
  esac
  sleep 30
done
mkdir -p "$OUT"
$KAGGLE kernels output "$KERNEL" -p "$OUT" -q || $KAGGLE kernels output "$KERNEL" -p "$OUT"
echo "results in $OUT:"; cat "$OUT/summary.json" 2>/dev/null || tail -40 "$OUT"/*.log
