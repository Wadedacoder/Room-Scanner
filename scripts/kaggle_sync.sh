#!/usr/bin/env bash
# Package the pipeline code + eval bundle as private Kaggle datasets (new version each call), then push a job.
# Usage: scripts/kaggle_sync.sh bench/kaggle/model_eval_v2 "message"
set -euo pipefail
cd "$(dirname "$0")/.."
JOB="${1:?job folder}"; MSG="${2:-update}"
KAGGLE=.venv/bin/kaggle; USER=devmsjsj
stage() {  # $1 = dataset slug, $2 = tar source dir, $3 = tar member
  local d="data/derived/kaggle_$1"; rm -rf "$d"; mkdir -p "$d"
  tar --exclude='__pycache__' -cf "$d/$1.tar" -C "$2" "$3"
  printf '{"title": "%s", "id": "%s/%s", "licenses": [{"name": "other"}]}\n' "$1" "$USER" "$1" > "$d/dataset-metadata.json"
  if $KAGGLE datasets status "$USER/$1" >/dev/null 2>&1; then
    $KAGGLE datasets version -p "$d" -m "$MSG" -q
  else
    $KAGGLE datasets create -p "$d" -q
  fi
  until $KAGGLE datasets status "$USER/$1" 2>&1 | grep -q ready; do sleep 10; done
}
stage roomscan-code . roomscan
stage roomscan-eval-bundle data/derived eval_bundle
$KAGGLE kernels push -p "$JOB" --accelerator NvidiaTeslaT4
