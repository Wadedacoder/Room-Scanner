# Backlog: everything still required

Ordered by priority. Each item names the brief requirement it serves (P1–P5 = brief parts, D = deliverable) and its
status. Updated as work lands; history is in git, `docs/CHANGELOG.md` and `docs/EXPERIMENTS.md`.

Legend: ⬜ todo · 🟨 in progress · ✅ done · ⏸ blocked on user

## P0: connected plans and correct photo walls (user priority)
| # | Item | Serves | Status |
|---|---|---|---|
| 0.1 | Photo wall sizes: diagnose untaped rooms; fix once tape for living/kitchen/bedroom/bathroom/store exists | P2 gates | 🟨 study ±2%; ⏸ tape for other rooms |
| 0.2 | Door/opening detection on photo rooms (works on LiDAR) | P2 openings, stitch | ✅ every room has an opening (E15); widths ⏸ door tape |
| 0.3 | Full-house photo run hangs (each room alone 10–13 s) | walk-in | ✅ not reproduced after offline-first loading (47.7 s) |
| 0.4 | Place rooms from strong cross-room visual links (DISK+LightGlue + PnP) | P2 photo stitch | ✅ v0.6.0 (kitchen↔living) |
| 0.5 | Door-geometry matching for rooms without a visual link; adjacency; overlap check | P2 photo stitch | 🟨 shared-wall + door alignment for linked rooms done; unlinked rooms ⏸ look-through capture |
| 0.6 | Connected plan in plan.json (`adjacency`, global polygons) + SVG for photo, video, LiDAR | P2 | 🟨 photo done; LiDAR/video ⬜ |
| 0.7 | Protocol: one photo per doorway looking through it into the next room (E14 evidence) | P1 | ✅ (needs a new capture to verify ⏸) |

## Robustness and reproducibility
| # | Item | Serves | Status |
|---|---|---|---|
| 1.1 | Checkpoint model outputs (DA3 recon, COLMAP) by input hash: resume after crash, deterministic replay | D4, walk-in | ✅ |
| 1.2 | Offline-first model loading; weights fetched by script | D4, constraints | 🟨 loading done; setup_models fetches weights ⬜ |
| 1.3 | Health check every 30 min (tests, website, memory, stuck jobs, git) | process | ✅ session watch at :13/:43 |
| 1.4 | Video tracking repeatable (seeded, single-threaded) + best-of-seeds; commit | P2 repeatability | 🟨 uncommitted |
| 1.5 | `cache/` and stale worktree hygiene (.gitignore, other session's branch) | process | ✅ ignored; branch left untouched |

## Tiers
| # | Item | Serves | Status |
|---|---|---|---|
| 2.1 | Video: join COLMAP pieces (DA3 through turns, E5) | P1 video | ✅ v0.7.0 (fix loop) |
| 2.2 | Video: score house_b_walk.MOV (212 s, 6 rooms) | P2 | ⬜ |
| 2.3 | Photo intervals calibrated (coverage ≈ 90%) | P2 calibration | ⬜ |
| 2.4 | LiDAR open-passage adjacency between split rooms | P2 adjacency | ✅ E19 (one unverified passage) |

## Contract items not started
| # | Item | Serves | Status |
|---|---|---|---|
| 3.1 | Damage regions (VLM, cached) with class + metric extent | P2 | ⬜ |
| 3.2 | Concealed-damage flags with fired rule; scope line items | P2 | ⬜ |
| 3.3 | Opening-width gate scorer; ceiling gate; repeatability table (`bench/gates.py`) | P2, D5 | 🟨 gates.py done; repeatability table ⬜; door widths ⏸ tape |

## Benchmark and deliverables
| # | Item | Serves | Status |
|---|---|---|---|
| 4.1 | Fix loop: declare worst gate (number, cause, prediction) BEFORE fixing; before/after regenerable | P4 (25%) | ✅ fixloop/ (video study −35.8% → −4.0%) |
| 4.2 | Head-to-head vs magicplan on 2 rooms | P3 (10%) | ⏸ needs a Pro iPhone |
| 4.3 | Device matrix | D2 | ⬜ |
| 4.4 | Benchmark report (gates per tier, repeatability, timing) | D5 | ⬜ |
| 4.5 | Technical report ≤ 6 pages | D7 | ⬜ |
| 4.6 | Raw data release + fetch script | D8 | ⬜ |
| 4.7 | Clean-machine test: README to a result in < 15 min | D3 | ⬜ |

## Waiting on the user
* Tape: length × width of living, kitchen, bedroom, bathroom, store; door widths; second readings.
* A Pro iPhone for an hour (LiDAR of our rooms + magicplan).
* A protocol capture with doorway "look-through" photos (after 0.7).
