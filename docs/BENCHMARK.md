# Benchmark report

Pipeline 0.7.x on an M1 MacBook Air (8 GB, `lite` profile) unless noted. Every number below can be regenerated from
the commands given; the experiment write-ups behind them are in `docs/EXPERIMENTS.md` (E-numbers).

**What ground truth exists.** Tape is available for one room only so far: the study (walls 312.4 / 330.2 / 312.4 /
~350.5 cm, ceiling 274.3 cm, single readings; `bench/ground_truth/home_tape.yaml`). The other five house rooms, door
widths and second readings are still to be taped. The LiDAR captures are public Stray Scanner scans with no tape, so
LiDAR is reported for self-consistency and drift, not accuracy. Claims are limited to what this supports.

## 1. Gates per tier (study, scored by `bench/gates.py`)

Gates: walls photo ±8%, video ±3%, LiDAR ±2 cm or 1%; area photo ±10%, video ±5%, LiDAR ±2%; ceiling ≤ 1.5 cm;
opening widths ≤ 2 cm on ≥ 85%. Walls are scored as the two side lengths against the mean of the opposite tape walls
(long side 3.404 m: the two long walls differ by a cove).

| Tier · capture | Short side | Long side | Floor area | Ceiling | Openings | 90% intervals hold |
|---|---|---|---|---|---|---|
| Photos · `house_b_study` (protocol-style, 4 × 0.5× landscape) | 3.074 m, **−1.6% PASS** | 3.557 m, **+4.5% PASS** | 10.71 m², **+0.7% PASS** | 2.723 m, −2.0 cm **FAIL** | not taped | area yes, ceiling yes |
| Video · `study_walk1.MOV` (off-protocol 1×, after fix loop) | 3.000 m, −4.0% FAIL | 3.168 m, −6.9% FAIL | 9.48 m², −10.8% FAIL | not reported | not taped | area yes |
| LiDAR | no taped LiDAR capture (needs a Pro iPhone) | | | | | |

```bash
roomscan run data/raw/photos/house_b_study -o runs/bench && python bench/gates.py runs/bench/house_b_study/plan.json
roomscan run data/raw/video/study_walk1.MOV  -o runs/bench && python bench/gates.py runs/bench/study_walk1.MOV/plan.json
```

Reading: the photo tier passes its wall and area gates on the one taped room and misses the ceiling gate by 0.5 cm (the
interval still contains the tape). The video tier fails: see the fix loop (`fixloop/RESULT.md`) for the declared
worst gate and what remains (36% of frames never registered, DA3 bridges through fast turns, off-protocol capture).

## 2. Repeatability (live reruns, caches off)

`python bench/repeat.py <capture> --runs 3`: each run is a separate process with `ROOMSCAN_NO_CACHE=1`, so every model
runs again. Results in `runs/repeat/<capture>/repeat.json` (committed copy: `bench/results/`).

| Capture · tier | Runs | Short side | Long side | Floor area | Ceiling | Openings | Gates (each run) | Runtime |
|---|---|---|---|---|---|---|---|---|
| `house_b_study` · photos | 3 | 3.0737 m, range 0 | 3.5567 m, range 0 | 10.7097 m², range 0 | 2.7229 m, range 0 | 1, 1, 1 | walls ✓ area ✓ ceiling ✗, identical | 27.9 ± 1.1 s (cached replay: 7.5 s) |

The photo tier is **bit-for-bit repeatable** on this machine: DA3 runs on MPS with a fixed input order and no sampling,
and every downstream step is seeded. The runtime gap to a cached replay (28 s vs 7.5 s) confirms the models really ran.
Repeatability across machines (CUDA vs MPS) is not measured; small float differences there are expected. The video tier
is seeded and single-threaded in COLMAP (best of 4 seeds), but a live repeat takes ~5 min per run and has not been
tabulated yet.

LiDAR is deterministic by construction (no learned model; seeded sampling): two runs of `c7d28f72c6` give identical
room areas and ceilings (checked in E12's regression run and again after E19).

## 3. LiDAR self-consistency and drift (no tape)

| Capture | Rooms | Footprint | Ceilings | Adjacency | Drift (revisit misalignment) | Runtime |
|---|---|---|---|---|---|---|
| `c7d28f72c6` (215 s walk) | 5 | 63.52 m² | 2.27–3.07 m | 4 links (3 doors, 1 passage) | 19.7 → **3.5 cm** with correction | 212 s |
| `c00a170fe1` | 3 | 21.38 m² | not observed (floor-only scan) | 3 links (1 probably false) | no revisit passed the test | 25 s |

Drift correction on/off (E7): wall sharpness 1.255 → 1.292; on the short walk no revisit is accepted and nothing moves,
by design.

## 4. Connected plans

| Tier | Capture | Rooms | Connected | How |
|---|---|---|---|---|
| Photos | `house_b` (17 photos, 6 rooms) | 6 | 3 (living, kitchen, bathroom) | DISK+LightGlue links, two-way pose agreement, shared-wall snap (E17–E18) |
| LiDAR | `c7d28f72c6` | 5 | all 5 | one cloud; shared doors + carved passages (E19) |
| Video | `house_b_walk.MOV` (212 s, 6 rooms) | **3** (rooms merge) | all 3 via open passages | 62% of frames tracked, 16/18 pieces joined (E21); not usable |

The three unlinked photo rooms have no photo that sees into a neighbour; the protocol now asks for one look-through
photo per doorway (E14).

## 5. Timing and memory (M1 Air 8 GB, lite profile, live = no cache)

| Run | Runtime | Peak memory (phys_footprint, incl. GPU) |
|---|---|---|
| Photos, one room (4 photos), live | 14.7 s | 2.99 GB |
| Photos, full house (17 photos, 6 rooms, with stitching), live | 135.7 s | **4.22 GB** (cross-room matching at the 4.5 GB GPU cap) |
| Photos, full house, replayed from checkpoints | 10.1 s | 0.44 GB |
| Video, study (64 s), COLMAP replayed, depth + join live | 152.9 s | 3.18 GB |
| LiDAR, 215 s walk with drift correction | 311 s | 0.35 GB |

Memory is measured as macOS `phys_footprint` by `scripts/run_guarded.sh` (2 s sampling). Until 2026-10-03 the watchdog
read RSS, which omits GPU (MPS), compressed and swapped memory: it reported ~0.65 GB for photo runs that really use
3–4 GB, and it let a 1698-frame video run thrash 2.1 GB of swap unnoticed. Earlier "peak" figures in EXPERIMENTS are RSS
and understate memory. The full-house live runtime here (136 s) is longer than the 47.7 s in E17, which was measured
with the cross-room links already checkpointed.

## 5b. Clean-machine test (README → result)

Fresh `git clone`, **empty** uv, Hugging Face and torch caches (everything downloaded: 1.6 GB packages, 1.7 GB
weights), README steps timed on the M1 Air (data from the release zips via `ROOMSCAN_DATA_URL`, not yet uploaded):

| Step | Time |
|---|---|
| `scripts/setup.sh` (+ web extra) | 25 s |
| `scripts/setup_models.sh` (torch, DA3, pycolmap, kornia + weight prefetch) | 320 s |
| `scripts/fetch_data.sh photos lidar_single_room video_study` (checksums verified) | 3 s |
| `roomscan run` LiDAR `c00a170fe1` | 59 s |
| `roomscan run` photos `house_b` (6 rooms, live models) | 203 s |
| `pytest -q` | 38 passed, 65 s |
| **Total, clone to two plans** | **≈ 10 min** |

The first attempt failed and found two bugs: every run crashed without the `anthropic` package, and `setup_models.sh`
did not install `kornia`. Both were fixed (`b9e01b6`) before this run. The fresh install reproduced the same room
areas as the development environment (study 10.71 m², kitchen 6.22 m², …). The photo run took 203 s here vs 48 s
warm, because it shared the CPU with a COLMAP job and was the first model load.

## 6. Damage, flags and scope

The chain runs end to end on all tiers (detector → projection onto floor / ceiling / wall → rules → scope), validated
with a synthetic detector (schema-valid output; regions 0.45–1.1 m on the study walls). No accuracy number yet: it needs
staged damage photographed with tape and an Anthropic API key for the detector.

## Not measured yet (and why)
* Opening widths: door widths not taped.
* LiDAR accuracy and the magicplan head-to-head: need a Pro iPhone.
* Interval calibration (coverage ≈ 90%): one taped room gives 4–5 checks, too few to calibrate.
