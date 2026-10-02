# Room-Scanner: technical report

Pipeline 0.8.0 · M1 MacBook Air 8 GB (`lite` profile) · iPhone 14 Plus (no LiDAR) · 2026-10-03.
Companion documents: `docs/BENCHMARK.md` (numbers), `docs/EXPERIMENTS.md` (E1–E21, how each was found),
`docs/CHANGELOG.md` (versioned history), `fixloop/` (declared fix), `docs/COMPLIANCE.md` (requirement → file).

## 1. Problem and what was built

A phone capture in one of three tiers (photos, video, LiDAR) goes in. Out comes `plan.json` (validated against
`schema/plan.schema.json`) and `plan.svg`. They contain per-room walls, ceiling height, floor area and openings, the
rooms stitched into one plan with adjacency, damage regions on surfaces, concealed-damage flags and scope line items.
Every number carries a 90% interval. One command per capture: `roomscan run <capture> -o out/` (the tier is detected
from the input).

**Capture route:** Route 2, the stock protocol (`docs/CAPTURE_PROTOCOL.md`). No custom app: a 0.5× overlapping ring
of landscape stills per room, a slow 0.5× video walk, or a Stray Scanner LiDAR scan. Plus one photo per doorway looking
into the next room, which is what lets photo rooms be linked (E14).

**Constraint that shaped everything:** an 8 GB laptop with a shared GPU. It crashed once at 4.46 GB of GPU memory
(E8). Since then the GPU is capped, one model is loaded at a time, and a watchdog (`scripts/run_guarded.sh`) kills a
run before the machine swaps. Stronger machines get bigger models through hardware profiles (`configs/profiles/`).

## 2. Architecture

```
tier adapter ──► metric 3D points + camera poses (gravity-aligned, +Y up)
                     │
                     ▼
shared back-end (roomscan/pipeline/backend.py)
  floor/ceiling levels → wall detection → room outline(s) → walls, area, ceiling, openings, adjacency, intervals
                     │
                     ▼
stitching (photo: visual links · LiDAR: one cloud · video: one joined trajectory)
                     │
                     ▼
damage (detector → surface projection) → rules.yaml → concealed flags + scope → plan.json + plan.svg
```

### 2.1 Tier adapters
* **LiDAR** (`roomscan/io/stray.py`, `pipeline/lidar.py`). Stray Scanner depth (256×192, confidence ≥ 2) fused with
  ARKit poses. ARKit's poses are camera-to-world in the OpenCV convention. Getting that convention wrong put the floor
  at camera height until it was fixed.
* **Photos** (`pipeline/photos.py`, `recon/learned.py`). Depth Anything 3 Base gives multi-view depth and relative
  poses. DA3Metric-Large fixes metric scale with the photo's **EXIF focal length**. E1 showed the predicted focal
  length was off by up to +24% and scale errors reached +18%. The true focal length gives −1 to −9% with the small
  model, as good as the 1.4 B Giant model, which is why the 8 GB profile is viable. Views are processed in
  overlapping windows chained by Sim(3) transforms on shared views, because memory grows steeply with view count (E1).
* **Video** (`pipeline/video.py`, `recon/sfm.py`, `recon/bridge.py`). ffmpeg decodes frames sequentially (HEVC seeking
  returned wrong frames). COLMAP (pycolmap) tracks cameras. It runs in a child process because its OpenMP runtime
  aborts alongside torch's. It is seeded and single-threaded so results repeat, keeping the best of 4 seeds. COLMAP
  splits a handheld walk into pieces at fast turns. Each piece gets its own metric scale from DA3Metric depth against
  its sparse points, and pieces are joined by DA3 bridges at their nearest frames (the fix loop, §5). Keyframe metric
  depth placed at the joined poses gives the dense cloud.

### 2.2 Geometry back-end (shared by all tiers)
* **Floor and ceiling:** height histogram of the cloud. The floor is the *lowest* flat level with ≥ 40% of the
  densest level's support, and is accepted only if the camera is a plausible 1.0–1.95 m above it. Otherwise no ceiling
  is reported, rather than a wrong one (E12). Before this rule, desks and beds became the "floor" and the ceiling was
  wrong by up to 1.5 m. The ceiling is the median of the top layer inside the room outline.
* **Walls:** points covering ≥ 35% of the 0.2–1.8 m height band in a top-down grid, rotated to the dominant wall
  direction.
* **Room outline:** LiDAR and video carve free space with camera→point rays (thousands of viewpoints) and split rooms
  with a watershed. Photos can't carve with only 4–8 viewpoints (E6: 28% of a room even with perfect depth), so the
  outline is the region enclosed by observed walls. A Manhattan-box alternative was tested and rejected (E13).
* **Openings** (`openings/detect.py`): 5 cm slots along each wall. A *see-through* opening is a run with little wall
  surface where points exist beyond the wall line. A *gap* opening (photos only, confidence 0.3) is a run with no wall
  surface inside a camera's view while the rest of the wall is covered. Tolerances depend on the tier: LiDAR walls are
  ±1 cm thick, learned-depth walls ±20 cm (E15).
* **Adjacency:** rooms from one cloud are adjacent through a door detected from both sides (within 0.6 m), or through
  an open passage where the carved floor directly between them connects them (E19).

### 2.3 Stitching
* **LiDAR** is one cloud, so stitching is free, but ARKit drifts (19.7 cm where the 215 s walk revisits places).
  **Drift correction** (`stitch/drift.py`, E7): the walk is cut into 4 s fragments. Revisited fragments are matched
  as top-down wall images (yaw search ±3°, phase correlation, accepted only if wall overlap improves and the shift is
  under 30 cm). A small linear pose graph is then solved. Revisit misalignment fell 19.7 → 3.5 cm and walls got
  sharper. On the short walk no revisit passed the test, and nothing moved, by design. Open3D ICP was tried first and
  diverged (40–90° rotations).
* **Photos** (`stitch/links.py`, `stitch/photo_graph.py`, E14, E17, E18). Each room is reconstructed in its own frame.
  DISK + LightGlue matches photos across rooms (SIFT fails on 0.5× ring photos: 18 median matches even within a room).
  A link gives a relative pose by PnP on the matched points' metric depth. It is accepted only if both directions
  agree (yaw within 10°, tilt within 15°), because match counts on the Mac GPU are not bit-reproducible, so a fixed
  inlier threshold flickered. Yaw is snapped to the wall grid. Then the two rooms' facing walls are snapped to one
  wall thickness apart and their doors aligned (PnP gets direction right but distance poorly). Rooms with no link are
  drawn to the side with a warning, never guessed into place.

### 2.4 Damage, concealed flags, scope
* **Detector** (pluggable). Default: OWLv2 run locally (`damage/detect_local.py`, no key; defect queries plus negative
  queries for clean surfaces, 0 false detections on 17 clean photos, E24). Optional: one Claude `claude-opus-5-5`
  vision request per room (`damage/detect_claude.py`) with up to 8 views, structured JSON-schema output (class + box per image on a 0–1000 grid), server-side fallback enabled.
  Responses are cached by content hash in `cache/vlm/` and committed, so damage replays bit-identically on another
  machine (the brief accepts cached model outputs; `damage.cache=live` forces a fresh call).
* **Projection** (`damage/project.py`): the box's pixels are back-projected with the view's metric depth and
  gravity-aligned pose. The surface is decided by height: floor (< 0.15 m), ceiling (within 0.15 m of it), else the
  nearest wall within 0.5 m. The region is the 5–95% extent in the surface's own (along, up) metres. The same stain
  seen in several views is merged (same surface and class, IoU ≥ 0.2).
* **Rules** (`scope/rules.yaml`, `scope/engine.py`): explicit, readable rules. For example, R-WATER-BASE: a water
  stain whose bottom is < 0.30 m above the floor flags possible wicking into the wall cavity and adds a flood cut and
  lower drywall replacement. R-CEIL-STAIN, R-MOLD-ANY and R-CRACK-LONG (≥ 0.6 m) work the same way. Every flag and
  line item records the rule that produced it and the region ids that triggered it. Quantities come from wall and
  room dimensions, with intervals.

### 2.5 Uncertainty
Each tier has an error model of absolute (m) + relative (scale) terms. LiDAR: 1 cm + 0. Photos: 3 cm + 5%, from
E1–E2 metric-scale spreads. Video: 2 cm + a **measured** scale term, √(4%² + per-piece scale standard error²), where
each COLMAP piece's scale is a median of n keyframe depth ratios (E21). A fixed 4% was overconfident: the per-keyframe
spread is 20–29%, giving 9.6% on the study video and 14.9% on the house walk. A wall's sigma combines both fitted neighbouring wall
lines (corners are intersections) with the tier terms. Area and perimeter propagate from walls. Ceilings combine the
top-layer spread, floor-level spread and tier terms. Intervals are ±1.645σ. **They are propagated, not yet
calibrated:** one taped room gives 4–5 checks. Every interval checked so far contains the tape value, including the
photo ceiling that misses its gate by 0.5 cm.

### 2.6 Engineering for the walk-in test
* Checkpoints keyed by a hash of inputs + settings + code version (`recon/checkpoint.py`) mean an interrupted run
  resumes, and a re-run replays in ~7 s. `ROOMSCAN_NO_CACHE=1` reruns everything live.
* Models load offline-first. A mid-run Hugging Face disconnect killed a job once, and a stalled request hung the
  full-house run.
* Layered config with unknown keys rejected; `config.resolved.yaml` with a digest is written next to every output.
* A local website (`roomscan serve`) runs one guarded job at a time and compares results against typed-in tape.
* A memory watchdog (`scripts/run_guarded.sh`) measures macOS `phys_footprint`, which includes GPU and swap. It used
  to read RSS, which missed both: a video run thrashed 2.1 GB of swap unseen.
* A robustness sweep runs every capture on disk (25: photo variants, LiDAR, videos). It found that any fresh
  multi-room photo capture crashed at room 2, because the damage detector's GPU cache starved the next room's
  reconstruction. That is fixed: damage now runs after all geometry. The sweep is 25/25.

## 3. Results (details: `docs/BENCHMARK.md`)

| Tier | Taped room result | Gate |
|---|---|---|
| Photos (study, 4 stills) | walls −1.6% / +4.5%, area +0.7%, ceiling −2.0 cm (interval holds) | walls ✓ area ✓ ceiling ✗ |
| Video (study, off-protocol 1× walk) | walls −4.0% / −6.9%, area −10.8% (interval holds), no ceiling | ✗ (±3%) |
| LiDAR | no taped LiDAR capture | n/a |

* **Repeatability:** three live runs each of the study photos and the study video are identical to 4 decimals.
* **Whole-house video walk (E21):** 62% of frames tracked, 3 merged rooms instead of 6. Not usable, and reported as
  such: the intervals are wide.
* **LiDAR self-consistency:** 5 rooms with adjacency on the 215 s walk, drift 19.7 → 3.5 cm.
* **Connected plans:** photos connect 3 of 6 house rooms (the three with a look-through photo); LiDAR connects 5/5.
* **Runtime (live):** 15–28 s for one photo room, 136 s for the 6-room house (cross-room matching included), ~5 min for a 64 s video, 3.5 min for a
  215 s LiDAR walk. Peak memory, including GPU: 4.2 GB for the live 6-room photo run, under the 6 GB cap
  (BENCHMARK §5).

## 4. What did not work (and what that taught)
* **Ray carving for photos** gave rooms 43–69% too small (E6). It needs thousands of viewpoints.
* **Proxy "photos" from LiDAR video frames** (E1, E6) were 1× portrait shots facing away from each other. They broke
  pose estimation in every model. The protocol now requires an overlapping 0.5× ring.
* **A fixed match threshold for photo links** was not reproducible on the Mac GPU (47 vs 42 matches for one pair).
  Two-way pose agreement replaced it (E18).
* **LiDAR open-passage adjacency** by mask distance linked rooms through walls. The connected-carved-floor rule fixed
  most cases; one probably-false link remains and is marked unverified (E19).
* **The first fix-loop implementation** joined COLMAP pieces in time order and dropped the largest piece (v1). The
  shipped version anchors on it. Both are kept on record.
* **Registering more video frames** two ways: looser COLMAP thresholds (E22) and DISK+LightGlue matches inside
  COLMAP (E23). Both registered far more frames (79%, 94%) and both made the room worse (−16% and +75% walls), because
  the extra poses were wrong. Fewer correct poses beat more wrong ones.
* **A damage query that names a surface** ("a cracked ceiling") boxed the clean ceiling. Negative queries for
  undamaged things fixed it (E24).

## 5. Fix loop (`fixloop/`)
Declared before any code change (commit `5baa29e`, tag `fixloop-before`). The worst gate was video short side −35.8%
and area −34.5%. The cause: only COLMAP's largest piece was used, 35% of frames. The prediction: within ±10% / ±12%
after joining pieces. After (tag `fixloop-after`): **−4.0% short side, −8.1% area, interval now holds**. Predictions
met, ±3% gate still failed as predicted. What remains: 36% of frames never register, bridges have ~10° rotation error
through fast turns, and the capture is off-protocol.

## 6. Failure modes

| Condition | Expected effect | Mitigation in place | Tested? |
|---|---|---|---|
| **Mirrors** | depth sees a "room" behind the glass. LiDAR/learned depth put points beyond the wall, so a see-through *opening* can appear on a mirror wall, and the room can leak | wall detection requires vertical coverage over 0.2–1.8 m, so a mirror's frame and the wall around it usually keep the wall; a see-through opening needs a run of ≥ 0.55 m with *no* wall surface | **No.** E1: the glass/mirror bathroom made DA3-Base fail pose estimation entirely (0/8), while Large managed 8/8 |
| **Glass (windows, glass doors)** | LiDAR passes through, giving points outside | recorded as an opening: lowest see-through point > 0.5 m is typed *window*, otherwise door (≤ 1.2 m) or open passage | partly: bedroom window found (E15), width untaped |
| **Wet-look / glossy floors** | specular dropouts in LiDAR and DA3; reflections double the floor level | floor = lowest level with ≥ 40% support + camera-height plausibility check, so a reflected floor below the real one has little support | No |
| **Wet-look in damage** | sheen read as a water stain | the prompt excludes reflections and dirt that wipes off; confidence ≥ 0.5; every region names its evidence frames for review | No (no damage captures yet) |
| **Low light** | fewer features (COLMAP pieces split), noisier DA3 depth, more ARKit drift | best-of-4 seeds; piece joining; drift correction. The per-piece metric-scale spread is recorded in plan.json but does not yet widen the intervals | No |
| **Furniture against walls** | outline edges snap to furniture faces | wall band starts at 0.2 m, so low furniture is excluded; tall furniture is not handled | Partly: E13 found outline edges snapping to furniture on sparse photo rooms (open issue) |
| **Off-protocol capture** (1× lens, HDR, mixed orientation) | narrower field of view breaks overlap; mixed shapes broke windows | `scripts/check_capture.py` audits a capture against the protocol; mixed orientations are letterboxed to one shape; the video tier warns on lens and HDR | Yes (E9: all original captures off-protocol, all warned) |

The rule in every case: when evidence is missing, report nothing and say why rather than a confident wrong number.
Examples: no ceiling when the floor is implausible, unlinked rooms drawn apart, empty damage without a detector.

## 7. Limits and next steps
* **Ground truth is thin:** one taped room. Tape for the other five rooms, door widths and second readings would
  allow interval calibration and opening-width scoring.
* **LiDAR accuracy and the magicplan head-to-head** need a Pro iPhone for an hour.
* **Damage recall** needs staged damage photographed with tape. The local detector needs no key (E24). Its 0 false
  positives on 17 clean photos is in-sample.
* **Video:** neither looser registration nor learned matches helped (E22, E23). Next would be DA3 poses throughout,
  with COLMAP only as a scale and loop anchor, plus a protocol emphasis on slow turns.
* **Photos:** rooms without a look-through photo stay unplaced. Door geometry alone was ambiguous on house_b (E25).
