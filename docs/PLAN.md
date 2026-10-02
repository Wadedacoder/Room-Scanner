# Implementation plan (48 h)

## 0. What is being scored (and therefore where the hours go)

| Weight | Component | Implication |
|---|---|---|
| 30% | Walk-in: cold run on *their* iPhone, tier chosen on the day | All 3 tiers must run end-to-end and degrade honestly. Robustness > peak accuracy. |
| 25% | Fix loop | Pick a failing gate early, declare root cause + predicted number, ship fix, regenerable before/after. Cheapest points in the brief. |
| 15% | Benchmark accuracy, all tiers | Needs our own GT; every number regenerable from raw. |
| 10% | Compliance matrix | `docs/COMPLIANCE.md` — keep it current as we go. |
| 10% | Head-to-head vs consumer app | magicplan Starter (free, exports PDF with dimensions). Polycam free = GLTF only. |
| 5% | Capture route | Route 2: stock protocol (`docs/CAPTURE_PROTOCOL.md`). |
| 5% | Process evidence | Commit every working slice. Never one big commit. |

"Confident garbage on thin input caps your total score" → intervals must widen at photo tier; calibration is scored everywhere.

## 1. Inputs & constraints we have

* **Data on disk**: 3 Stray Scanner LiDAR captures (1920×1440 RGB, 256×192 depth, poses = OpenCV camera-to-world in a +Y-up world — verified, see `roomscan/io/stray.py`):
  * `c00a170fe1` "single_room", 37 s — actually **3 spaces**: living/kitchenette, corridor, bathroom (glass shower + mirror). No ceiling seen. Room frame ranges in `bench/annotations/c00a170fe1_rooms.yaml`.
  * `1a8384c3f6` multi-room "floor only", 115 s, ~9×9 m, loop closes to 0.17 m; essentially no ceiling points
  * `c7d28f72c6` multi-room "with ceiling", 215 s, ~9×10 m, loop closes to 0.39 m; ceiling plane ≈ 2.34 m above floor — likely the same space as above → candidate for repeatability
* **Proxy photo tier** (dev only, disclosed): `scripts/make_photo_tier.py` cuts 2/4/6/8 upright stills per room from `c00a170fe1`'s video into `data/derived/photos/c00a170fe1_n{N}/<room>/`, poses kept apart in `data/derived/photos_gt/`. Limits: Pro-phone video frames (not Camera-app stills), no dedicated doorway shots.
* **Missing**: ground truth, photo-tier & video-tier captures, a staged-damage room, a repeat capture at same tier of a single room, magicplan export.
* **Machine**: M1, 8 GB RAM. Rules out ~1 B-param models (MapAnything, VGGT, DA3-Giant) running comfortably; target DA3-Small/Base (Apache-2.0, 80–120 M, poses + depth) + DA3Metric-Large (350 M) or MoGe-2 for scale, on MPS/CPU with ≤ 8 frames per room.
* **No Round 1 spec** → we publish our own schema (`schema/plan.schema.json`) and infer baseline gates (below). Stated as assumption in the report.

## 2. Gates we test against (`bench/gates.py`)

| Gate | Threshold | Tier |
|---|---|---|
| Wall length (inferred R1) | LiDAR ±2 cm or 1% · video ±3% · photo ±8% | all |
| Floor area (inferred R1) | LiDAR ±2% · video ±5% · photo ±10% | all |
| Ceiling height | ≤ 1.5 cm; multi-capture spread ≤ 1 cm; report bias vs. unrepeatable | all (looser target off-LiDAR) |
| Opening width | ≤ 2 cm on ≥ 85% of openings; missed **and** phantom both count as misses | all |
| Repeatability | 2 captures same room/tier agree ≤ 1 cm or 0.5% per wall | ≥ 1 tier |
| Drift | method stated + footprint ablation on/off; "poses as-is" = fail | LiDAR, video |
| Photo-tier stitch | one plan, correct adjacency, no overlaps, footprint ±8% | photo |
| Calibration | empirical coverage of 90% intervals ≈ 90% | all |
| Head-to-head | beat/tie magicplan on ≥ 70% of shared dims, 2 rooms | LiDAR |

## 3. Architecture

```
capture ─► tier adapter ─► (frames, depth, poses, K, σ-model) ─► gravity align ─► room segmentation
        ─► per-room layout (floor/ceiling planes, wall lines, Manhattan snap, polygon)
        ─► openings (wall-plane free-space gaps + RGB edge refinement)
        ─► stitch (drift correction / door-to-door graph for photos) ─► damage ─► scope + concealed flags
        ─► uncertainty → calibrated intervals ─► plan.json + plan.svg
```
One shared geometry back-end; the three tiers differ only in the adapter and in their noise model. That is what makes "same output contract from each" cheap.

### Tier adapters
* **LiDAR** (`io/stray.py`, done): ARKit depth (conf==2) + poses; gravity is +Y already.
* **Video** (`io/video.py`): native Camera clip → keyframes by optical-flow parallax (~1 per 15–20 cm of motion, cap ~60) → DA3-Base multi-view (poses + depth) in overlapping windows of ≤ 8 → metric scale from DA3Metric median ratio → gravity from vanishing points / floor plane. Fallback: COLMAP/GLOMAP (pycolmap wheels) + metric depth for scale.
* **Photos** (`io/photos.py`): per-room folder of 2–8 HEIC/JPG stills; EXIF focal → K; DA3 on the room's set → per-room metric cloud. Wider σ (scale from monocular prior ≈ 3–5%).

### Geometry back-end
* Floor/ceiling: height histogram peaks + RANSAC plane → ceiling height = plane-to-plane distance at room centre. σ from plane residual / √N plus depth-bias term.
* Walls: points in a 0.5–2.0 m height band → 2D occupancy → RANSAC lines → dominant-direction (Manhattan, with non-Manhattan fallback) snap → line arrangement → room polygon. Wall length = distance between adjacent wall-line intersections (corners are intersections, never raw points — corners are where depth is worst).
* Room segmentation (multi-room capture): free-space map from camera trajectory + depth rays; split at narrow passages (door width 0.6–1.2 m) via distance-transform watershed.
* Openings: for each wall line, a 1D profile of "rays that pass through the wall plane" vs "rays that hit it" → gaps = openings; widths refined on full-res RGB by projecting the gap edges and snapping to strong vertical image edges (depth at 256×192 is ~1 cm/px at 1.5 m — too coarse alone for the 2 cm gate).

### Drift (required, ablation required)
1. Per-room fragments (~ segment the trajectory by room).
2. Fragment-to-fragment ICP with shared wall planes + loop closure (both multi-room captures close their loop) → Open3D pose graph optimisation.
3. Plane-anchored correction: walls that are the same physical plane seen from two rooms (wall thickness ~10–15 cm) constrained parallel & consistent.
Ablation: footprint area/outline with `--drift off` vs `on` overlaid in the report.

### Photo-tier stitching (hardest row)
Protocol forces each room's photo set to include **one photo taken standing in each doorway, looking into the room**. Pipeline: detect openings per room → match openings across rooms by (width, height, appearance of the doorway photo via DINOv2/LightGlue on the shared door frame) → place rooms as a graph: doors coincide, walls axis-aligned to a shared Manhattan frame, solve with a small least-squares + non-overlap check. Adjacency = matched-door edges.

### Damage, concealed flags, scope
* Detection: VLM (Claude, disclosed) on ~10 keyframes per room → class + bbox; responses cached by `sha256(image, prompt, model)` in `cache/vlm/` so benchmark replays are deterministic while the live path still runs.
* Metric extent: bbox → ray cast onto the wall/ceiling plane → polygon in surface-local metres, interval from plane distance σ and bbox jitter.
* Concealed flags: `roomscan/scope/rules.yaml`, explicit rule ids (e.g. `R-WATER-BASE`: stain within 0.3 m of floor → flag cavity moisture behind wall, scope 0.6 m flood cut; `R-CEIL-STAIN` → flag leak source above). The output records which rule fired.
* Scope line items: rules map (damage class, surface) → Xactimate-style codes with quantities from metric extent + standard margins.

### Calibration
Each measurement gets σ from propagated error (plane residuals, corner intersection covariance, tier scale prior). Intervals = value ± k_tier·σ; k_tier fit on the benchmark (leave-one-capture-out) to reach 90% coverage; report reliability (coverage at 50/80/90%) per tier.

### Failure modes to cover (brief explicitly asks)
Mirrors/glass (depth goes *through* the wall → phantom openings: flag planar "rooms behind walls" with mirrored texture; use confidence map), wet-look/specular floors (drop conf<2, rely on wall-base line), low light (warn, widen σ, video tier exposure).

## 4. 48-hour schedule

| Hours | Work | Commit checkpoint |
|---|---|---|
| 0–3 | Repo setup ✅, schema ✅, Stray loader ✅. **You, in parallel:** buy/borrow laser; capture photos + video of the same rooms; tape-label walls; fill GT yaml | `scaffold`, `stray loader` |
| 3–12 | LiDAR single room end-to-end: planes → walls → polygon → ceiling → JSON + SVG. `bench/gates.py` + `bench/run.py` | `lidar single-room`, `gates harness` |
| 12–18 | Multi-room: segmentation, openings, drift correction + ablation | `room seg`, `drift on/off` |
| 18–26 | Video + photo adapters (DA3), photo stitching via door graph | `video tier`, `photo tier`, `photo stitch` |
| 26–31 | Damage (VLM + cache), rules, scope | `damage+scope` |
| 31–35 | Calibration, full benchmark run, magicplan head-to-head | `bench v1` |
| 35–41 | **Fix loop**: declare worst gate (write the declaration *before* fixing), ship, rerun | `fix: before`, `fix: after` |
| 41–46 | Tech report (≤ 6 pp), compliance matrix, protocol polish, device matrix | `report` |
| 46–48 | Clean-machine README test (fresh clone → run < 15 min), tag release | `release` |

## 5. Things only you can do (blocking)

1. **Ground truth**: a laser measurer (~$30). Tape alone cannot verify ≤ 1.5 cm ceiling / ≤ 2 cm openings. Measure each value 3×.
2. **Same rooms at photo & video tier** with your non-Pro iPhone (needs physical access to the space in the LiDAR captures, or re-capture LiDAR elsewhere with a borrowed Pro).
3. **Staged damage room**: 2 classes (e.g. tea-stain "water stain" on paper taped to wall + a printed/painted "crack"/hole patch). Measure extents.
4. **Repeat capture** of one room at the same tier (photo tier is fine and you can do it yourself).
5. **magicplan** scan of 2 rooms on a LiDAR device (Starter plan), export PDF; record app version.
6. **Device matrix** honesty: you have no Pro device → LiDAR accuracy rows come only from the provided captures; say so.
