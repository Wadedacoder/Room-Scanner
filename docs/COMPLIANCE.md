# Compliance matrix

Status: ⬜ not started · 🟨 in progress · ✅ done · ❌ fails gate (reported honestly)

| # | Requirement (brief §) | File path | Artifact | Status |
|---|---|---|---|---|
| 1 | Capture route: stock protocol (P1) | docs/CAPTURE_PROTOCOL.md | one-page protocol (v2: 0.5× overlapping ring, non-Mac hand-off) | ✅ |
| 2 | Device matrix (P1) | docs/DEVICE_MATRIX.md | table (measured figures only) | ✅ |
| 3 | Photo tier → stitched plan (P1) | roomscan/pipeline/photos.py, roomscan/stitch/ | plan.json/svg; study walls −1.6% / +4.5%, area +0.7% (BENCHMARK §1); 3 of 6 house rooms connected | 🟨 stitch partial (look-through capture ⏸) |
| 4 | Video tier (P1) | roomscan/pipeline/video.py, roomscan/recon/sfm.py, bridge.py | plan.json/svg; study −4.0% / −6.9% | ❌ ±3% gate (fix loop, reported) |
| 5 | LiDAR tier (P1) | roomscan/pipeline/lidar.py | plan.json/svg, 5-room walk | ✅ (no tape for accuracy) |
| 6 | Per-room walls, ceiling height, floor area, openings (P2) | roomscan/pipeline/backend.py, roomscan/openings/detect.py | plan.json rooms[] | ✅ (opening widths untaped) |
| 7 | Stitched multi-room plan + adjacency (P2) | roomscan/pipeline/backend.py `_adjacency`, roomscan/stitch/photo_graph.py | plan.json `adjacency` (+ door `connects_to`), plan.svg | ✅ LiDAR · 🟨 photos 3/6 · 🟨 video (code done; whole-house walk merges rooms, E21) |
| 8 | Damage regions, class + metric extent (P2) | roomscan/damage/ (detect_local.py OWLv2 default, detect_claude.py optional, project.py, stage.py) | plan.json `damage` (cached detector outputs in cache/vlm/) | 🟨 runs offline; 0 false positives on 17 clean photos (E24); recall unmeasured (no staged damage) |
| 9 | Concealed-damage flags with fired rule (P2) | roomscan/scope/rules.yaml, engine.py | plan.json `concealed_flags[].rule_id` | ✅ (tests/test_damage_scope.py) |
| 10 | Scope line items keyed to surfaces (P2) | roomscan/scope/ | plan.json `scope[].surface_id` + quantity interval | ✅ |
| 11 | Interval on every measurement (P2) | roomscan/pipeline/backend.py `meas` | Measurement.lo/hi on every number; coverage 8–9/10 on all 10 photo captures of the taped room (E26) | 🟨 one room only |
| 12 | One command per capture, JSON to schema, rendered plan (P2) | roomscan/cli.py, schema/plan.schema.json, roomscan/render/svg.py | `roomscan run` (schema-validated), all 3 tiers; accepts zips; off-protocol captures run with warnings | ✅ |
| 13 | Benchmark set composition (P2) | bench/ground_truth/home_tape.yaml, bench/datasets/release_manifest.json | tape + checksummed raw captures | 🟨 1 taped room ⏸ |
| 14 | Opening width gate (P2) | bench/gates.py | scorer done | ⏸ door widths untaped |
| 15 | Ceiling height gate + bias/unrepeatable verdict (P2) | bench/gates.py, docs/EXPERIMENTS.md E12 | photos −2.0 cm (fails 1.5 cm, interval holds); bias cause found and fixed (E12) | ❌ reported |
| 16 | Repeatability gate (P2) | bench/repeat.py, bench/results/ | photos and video: 3 live runs each, identical to 4 decimals (bench/results/) | ✅ |
| 17 | Drift handling + on/off ablation (P2) | roomscan/stitch/drift.py, bench/local/e7_drift_ablation.py | E7: revisit misalignment 19.7 → 3.5 cm | ✅ (no GT) |
| 18 | Photo-tier whole-property stitch gate (P2) | roomscan/stitch/photo_graph.py | E17–E18: 3 of 6 rooms | 🟨 |
| 19 | Head-to-head vs magicplan, 2 rooms (P3) | (none) | table + export | ✖ not done: no Pro iPhone available; magicplan needs LiDAR |
| 20 | Fix loop declaration, before/after, diff (P4) | fixloop/ (DECLARATION.md, RESULT.md), tags fixloop-before/after | video short side −35.8% → −4.0% | ✅ |
| 21 | Reproduction bundle, cached model outputs (Deliv. 4) | roomscan/recon/checkpoint.py, cache/vlm/, bench/kaggle/ | checkpoints by input hash; `ROOMSCAN_NO_CACHE=1` reruns live | ✅ |
| 22 | Benchmark report incl. timing (Deliv. 5) | docs/BENCHMARK.md | gates, repeatability, calibration coverage, LiDAR drift, connected plans, timing, clean-machine test, robustness sweep | ✅ |
| 23 | Technical report ≤ 6 pp (Deliv. 7) | docs/TECH_REPORT.md | report (~2.6k words + tables, ≈ 5–6 pages) | ✅ |
| 24 | Raw benchmark data (Deliv. 8) | scripts/package_data.py, scripts/fetch_data.sh | packaging + checksummed fetch done; home captures kept private by the owner's decision; case-study LiDAR scans releasable | 🟨 partial |
| 25 | Mirrors / glass / wet-look / low light covered (Constraints) | docs/TECH_REPORT.md §6 | failure-mode table (expected effect, mitigation, tested?) | 🟨 mostly untested, said so |
| 26 | Weights fetched by script (Constraints) | scripts/setup_models.sh, roomscan/recon/da3_loader.py | prefetch + offline-first loading | ✅ |
| 27 | Hardware profiles, clean-machine install | configs/, scripts/setup.sh | `roomscan hw`; clean-machine test ≈ 10 min from empty caches (BENCHMARK §5b) | ✅ |
| 28 | Walk-in readiness (live test, 30%) | docs/WALKIN.md, bench/local/sweep.sh | runbook; sweep of all 25 captures on disk 25/25 complete (BENCHMARK §5c); fully live 6-room run 109 s, 4.2 GB | ✅ |
