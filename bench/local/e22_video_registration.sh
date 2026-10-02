#!/usr/bin/env bash
# E22: can COLMAP register more of a handheld walk? Ablation on the taped study video: sequential overlap x
# registration threshold. Each variant is a full run; COLMAP results are checkpointed per setting.
set -uo pipefail
cd "$(dirname "$0")/../.."
VID=${1:-data/raw/video/study_walk1.MOV}
for v in "10 30" "20 30" "10 15" "20 15"; do
  set -- $v
  out=runs/e22/ov$1_in$2
  LIMIT_MB=5500 PYTHONPATH=. HF_HUB_OFFLINE=1 scripts/run_guarded.sh .venv/bin/python -m roomscan.cli run "$VID" -o "$out" \
    -s recon.video_seq_overlap=$1 -s recon.video_abs_pose_min_inliers=$2 > "$out.log" 2>&1
  plan="$out/$(basename "$VID")/plan.json"
  .venv/bin/python - "$plan" "$1" "$2" <<'PY'
import json, sys
p = json.load(open(sys.argv[1])); v = p["capture"]["video"]
print(f"overlap {sys.argv[2]:>2} min_inliers {sys.argv[3]}: registered {v['registered']}/{v['frames']}, pieces {v['pieces']}, "
      f"joined {v['pieces_joined']}, rooms {len(p['rooms'])}, areas {[r['floor_area']['value'] for r in p['rooms']]}, "
      f"scale sigma {v.get('scale_rel_sigma')}, runtime {p['capture']['runtime_s']} s", flush=True)
PY
  .venv/bin/python bench/gates.py "$plan" 2>&1 | grep -E "PASS|FAIL"
done
