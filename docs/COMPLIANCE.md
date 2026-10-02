# Compliance matrix

Status: ⬜ not started · 🟨 in progress · ✅ done · ❌ fails gate (reported honestly)

| # | Requirement (brief §) | File path | Artifact | Status |
|---|---|---|---|---|
| 1 | Capture route: stock protocol (P1) | docs/CAPTURE_PROTOCOL.md | one-page protocol | 🟨 |
| 2 | Device matrix (P1) | docs/DEVICE_MATRIX.md | table | ⬜ |
| 3 | Photo tier → stitched plan (P1) | roomscan/io/photos.py | plan.json/svg | ⬜ |
| 4 | Video tier (P1) | roomscan/io/video.py | plan.json/svg | ⬜ |
| 5 | LiDAR tier (P1) | roomscan/io/stray.py | plan.json/svg | 🟨 |
| 6 | Per-room walls, ceiling height, floor area, openings (P2) | roomscan/geometry/ | plan.json | ⬜ |
| 7 | Stitched multi-room plan + adjacency (P2) | roomscan/stitch/ | plan.svg | ⬜ |
| 8 | Damage regions, class + metric extent (P2) | roomscan/damage/ | plan.json `damage` | ⬜ |
| 9 | Concealed-damage flags with fired rule (P2) | roomscan/scope/rules.yaml | plan.json `concealed_flags` | ⬜ |
| 10 | Scope line items keyed to surfaces (P2) | roomscan/scope/ | plan.json `scope` | ⬜ |
| 11 | Interval on every measurement (P2) | roomscan/calib/ | Measurement.lo/hi | ⬜ |
| 12 | One command per capture, JSON to schema, rendered plan (P2) | roomscan/cli.py, schema/plan.schema.json | `roomscan run` | 🟨 |
| 13 | Benchmark set composition (P2) | bench/ground_truth/ | GT yaml + raw | ⬜ |
| 14 | Opening width gate (P2) | bench/gates.py | report table | ⬜ |
| 15 | Ceiling height gate + bias/unrepeatable verdict (P2) | bench/gates.py | report table | ⬜ |
| 16 | Repeatability gate (P2) | bench/gates.py | repeatability table | ⬜ |
| 17 | Drift handling + on/off ablation (P2) | roomscan/stitch/drift.py | ablation figure | ⬜ |
| 18 | Photo-tier whole-property stitch gate (P2) | roomscan/stitch/photo_graph.py | report | ⬜ |
| 19 | Head-to-head vs magicplan, 2 rooms (P3) | bench/h2h/ | table + export | ⬜ |
| 20 | Fix loop declaration, before/after, diff (P4) | fixloop/ | bundle | ⬜ |
| 21 | Reproduction bundle, cached model outputs (Deliv. 4) | scripts/, cache/vlm/ | `make reproduce` | ⬜ |
| 22 | Benchmark report incl. timing (Deliv. 5) | docs/BENCHMARK.md | report | ⬜ |
| 23 | Technical report ≤ 6 pp (Deliv. 7) | docs/REPORT.pdf | pdf | ⬜ |
| 24 | Raw benchmark data (Deliv. 8) | scripts/fetch_data.sh | release assets | 🟨 |
| 25 | Mirrors / glass / wet-look / low light covered (Constraints) | docs/REPORT | failure-mode section | ⬜ |
| 26 | Weights fetched by script (Constraints) | scripts/setup_models.sh | script | ⬜ |
