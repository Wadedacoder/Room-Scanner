#!/usr/bin/env bash
# Demo dry run through the website (local-path jobs, as in the showcase): one capture per tier, wait for each,
# print status, runtime and the headline numbers. Needs `roomscan serve` on :8765.
#   bench/local/demo_dryrun.sh [extra captures...]
set -uo pipefail
cd "$(dirname "$0")/../.."
URL=${URL:-http://localhost:8765}
CAPS=("$PWD/data/raw/photos/house_b" "$PWD/dataset/c00a170fe1" "$PWD/data/raw/video/study_walk1.MOV" "$@")
for cap in "${CAPS[@]}"; do
  t0=$(date +%s)
  id=$(curl -s -X POST "$URL/api/jobs/local" -H 'Content-Type: application/json' \
        -d "{\"path\": \"$cap\", \"label\": \"dryrun $(basename "$cap")\"}" | python3 -c "import json,sys; print(json.load(sys.stdin)['id'])")
  until curl -s "$URL/api/jobs/$id" | grep -qE '"status": ?"(done|failed)"'; do sleep 5; done
  curl -s "$URL/api/jobs/$id" | python3 -c "
import json, sys
j = json.load(sys.stdin); p = j.get('plan_json') or {}
rooms = [(r['label'], r['floor_area']['value']) for r in p.get('rooms', [])]
print(f\"{j['label']:32s} {j['status']:6s} tier={j.get('tier')} rooms={len(rooms)} {rooms[:6]} {(j.get('error') or '')[:100]}\")"
  echo "  wall time $(( $(date +%s) - t0 )) s"
done
