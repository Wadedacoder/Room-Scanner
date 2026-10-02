# Experiments log

## E1 — Learned geometry for the photo / video tiers (2026-10-02)

**Question:** which Depth Anything 3 variant should each hardware profile use, and where does scale come from?
**Setup:** Kaggle T4 (16 GB), DA3 @ `3d835ec`, process_res 504. Proxy photo sets from `c00a170fe1` (3 rooms × N = 2/4/6/8)
and a 60-frame video sample; ground truth = ARKit LiDAR depth + poses for the same frames.
Job: `bench/kaggle/model_eval/`; raw results: `runs/kaggle_model_eval_v1/results.json` (not committed; regenerate with the job).

### Metric scale (the number that becomes wall-length error)
| Method | Scale error range over 12 photo sets | Note |
|---|---|---|
| DA3Metric-Large, **true focal** (mono) | −1.4% … −5.1% (mean ≈ −3.4%) | consistent bias → candidate for calibration |
| Small/Base/Large + DA3Metric, **predicted** focal | −3% … +18% | predicted focal off by up to +24% |
| Small/Base/Large + DA3Metric, **true** focal | −1% … −9.3% | same bias as mono |
| Giant + DA3Metric, predicted focal | −4.6% … +2.6% (n ≤ 4 only) | |
| Nested-Giant native metric | −4.4% … +7.5% (n ≤ 4 only) | |

**Finding 1:** use the photo's EXIF focal, never the predicted one. With it, the 8 GB `lite` profile (Base) has the
same scale accuracy as Giant, inside the ±8% photo gate on every set. The remaining ≈ −3.4% bias is systematic.
Caveat: one capture, one device, one apartment; the bias must be re-measured on real Camera-app stills before we correct for it.

### Poses
Pose-convention check passed (the corridor's views are 60–99° apart and still get 1–4° error).
| Failure | Seen in | Cause |
|---|---|---|
| Both views wrong (45–90°) | living n2, bathroom n2, all models | the two stills face opposite ways and share no content |
| One view flipped 180° | living n4/n6/n8 (fridge, frame 99), all models | an isolated view with no overlap |
| Whole set fails | Base on bathroom n8 (0/8), n4 | small model, glass/mirror bathroom; Large manages 8/8 |

**Finding 2:** the photo protocol must guarantee overlap. Corner-to-corner shots that face away from each other (what our
proxy picker maximised) break registration. Change the protocol to an overlapping sweep (each photo shares ~½ with the
previous one) and make the pipeline detect and drop or flag views that register poorly.

### Memory (T4, 16 GB)
Peak memory grows steeply with the number of views: Small at 8 views uses 3.5 GB; Giant fits 4 views (10.7 GB) and runs
out of memory at 6. Every model ran out of memory on the 60-frame video.
**Finding 3:** process in windows of views. The `cuda-16gb` profile's `max_views: 32` is wrong; it needs ≈ 4 for Giant,
≈ 8 for Large, and video must always be processed in overlapping windows.

### Speed
On a T4, Base takes 0.4–2.1 s per set and Giant 3–11 s. Runtime is not the bottleneck; memory is.

## E2: Windowed DA3, sweep vs spread photo selection (2026-10-02)

Job `bench/kaggle/model_eval_v2/`. DA3 now runs through the pipeline's own `roomscan.recon.windows` (overlapping windows
chained by a similarity transform on the shared views), with T4 window sizes (Base/Large 8, Giant 4) and overlap 2.

| Model | Photo views registered (spread / sweep) | Video60 registered | Scale, true focal | s per set |
|---|---|---|---|---|
| DA3-Base | 41/60 · 40/60 | 12/60 | −1.0% … −6.9% | 1.4 |
| DA3-Large-1.1 | 51/60 · 41/60 | 0/60 | −0.7% … −9.3% | 3.7 |
| DA3-Giant-1.1 | 47/60 · 39/60 | 4/60 | −1.3% … −6.2% | 12 |
| DA3Metric mono | n/a | n/a | −1.4% … −5.9% | 2 |

* Memory: no out-of-memory errors anywhere; windowing makes 60 views fit on 16 GB.
* Sweep vs spread: no gain, but this is **inconclusive**. The proxy "sweep" still has 135–163° gaps, because the walk
  never looked in those directions. Real 0.5× ring photos are needed to settle it.
* Video: chaining 10 windows with overlap 2 drifts; a bad link corrupts every later frame. E3 tries more overlap,
  denser frames, and MapAnything without chaining.
* MapAnything did not run (dependency `hydra` is published on pip as `hydra-core`); fixed in E3.

## E3: Video strategies on Kaggle, plus MapAnything (2026-10-02)

Job `bench/kaggle/model_eval_v3/`. Video sets of 60 and 120 frames (`c00a170fe1`, 37 s walk).

| Strategy | Frames registered (< 10°) | Median rotation error | Position error (m) |
|---|---|---|---|
| DA3-Base, window 8, overlap 2, 60 frames | 12/60 | 28.9° | 0.78 |
| DA3-Base, window 8, overlap 2, **120 frames** | 21/120 | **4.5°** | 0.36 |
| DA3-Base, window 8, overlap 4, 120 frames | 16/120 | 5.3° | 0.35 |
| DA3-Large, window 8, overlap 4, 120 frames | 65/120 | 15.6° | 0.72 |

* Denser frames help a lot; more window overlap does not.
* **MapAnything (Apache) is out.** It runs out of memory on any set of ≥ 4 views on a 16 GB T4, even in
  memory-efficient mode. On 2-photo sets its metric scale is 12–26% low, against −1% … −9% for DA3 with the true focal.
  (Mapping its cropped output to LiDAR pixels goes through its reported intrinsics, which may add some of that error;
  either way it does not fit the hardware.)

## E4: Classical SfM (COLMAP) for video poses, on the M1 (2026-10-02)

`bench/local/e4_colmap_video.py`: pycolmap 4.2, SIFT, sequential matching (overlap 10), lens fixed to the true focal.

| Frames sampled | Largest piece | All pieces together | Accuracy inside pieces | Time (M1 CPU) |
|---|---|---|---|---|
| 120 | 19 | 69/120 | 0.6–1.2° rotation, 1–6 cm position | 22 s |
| 480 | 78 | 359/480 (75%), 7 pieces | ~1° rotation, 1–5 cm position (6 of 7 pieces) | 124 s |

**Finding:** COLMAP is far more accurate than any learned model where it tracks, but the walk breaks into pieces.

## E5: Hybrid video poses: COLMAP pieces joined by DA3 (2026-10-02)

`bench/local/e5_hybrid_video.py`, DA3-Base on the M1 (MPS), lite profile (4 views per pass).

| Bridge | Frames registered (< 10°) | Median rotation error | Position error (m) | Bridging time |
|---|---|---|---|---|
| DA3 sees 2 + 2 frames across each gap | 2/480 | 84° | 1.16 | 17 s |
| DA3 chains every frame through each gap (windows of 4) | 317/480 | 10.7° | 0.67 | 148 s |
| … with scale fitted over all shared frames | **328/480 (68%)** | 10.2° | 0.65 | 146 s |

* Every gap is a fast turn: 65–131° of rotation in about 1 s, so frames across a gap share nothing and a direct bridge
  cannot work. Chaining through the turn, where neighbouring frames are about 9° apart, does work.
* Best video result so far, and it runs on the 8 GB laptop.
* Remaining error comes from those fast turns. Next: loop closure (the protocol now requires ending where you started;
  this proxy walk does not, with a 3.2 m gap) and validating each link before trusting it.
* Engineering notes: pycolmap and torch each bundle OpenMP and abort if loaded in one process, so COLMAP runs in a
  subprocess. DA3's top-level import pulls open3d and pycolmap through its exporters; `roomscan/recon/da3_loader.py`
  stubs that one module.

## E6: Photo tier end to end, and why it under-measures rooms (2026-10-02)

`roomscan run <photo folder>` now runs the photo tier (DA3-Base + DA3Metric with the EXIF focal, shared back-end).
On the proxy set (`c00a170fe1`, 8 sweep stills per room) it first gave rooms 43–69% too small. Ablation in
`bench/local/e6_photo_tier_ablation.py`; reference = LiDAR-tier area of the same room from the full walk.

| Room (LiDAR area) | Room outline method | A: shipped | B: true camera positions | C: LiDAR depth + true positions |
|---|---|---|---|---|
| living (7.41 m²) | carve rays | 2.64 | 2.07 | 2.05 |
| | **enclosed by walls** | 2.64* | 2.07* | **6.18 (−17%)** |
| corridor (8.35 m²) | carve rays | 3.89 | 2.97 | 9.35 |
| | enclosed by walls | 6.33 | 6.27 | 20.77 (leaks through open ends) |
| bathroom (5.62 m²) | carve rays | 3.17 | 4.34 | 3.13 |
| | **enclosed by walls** | 3.17* | **5.16 (−8%)** | **5.25 (−7%)** |

\* the wall flood found no open space around the learned camera positions, so it fell back to carving.

* **Cause 1, fixed:** ray carving needs thousands of viewpoints. With 8 it only covers the view cones (C: perfect
  data still gave 28% of the living room). The photo tier now outlines rooms from the observed walls.
* **Cause 2, open:** learned camera positions on these proxy frames are too wrong for the wall outline (A vs B).
  This is the non-overlap failure from E1: the proxy frames are 1× portrait with 135–163° gaps. The protocol's
  overlapping 0.5× ring is the intended fix and can only be tested on real photos.
* **Cause 3, open:** an open-ended corridor leaks into neighbouring rooms without doorway detection.
* Runtime: 410 s for 3 rooms × 8 photos on the M1, not yet profiled.

## E7: Drift correction, with an on/off comparison (2026-10-02)

`roomscan/stitch/drift.py`, `bench/local/e7_drift_ablation.py`. Method: cut the walk into ~4 s fragments; where the
walk revisits a place, match the two fragments' top-down wall images (yaw search ±3°, shift by phase correlation,
kept only if wall overlap improves and the shift stays under 30 cm); solve a small linear pose graph (rotations,
then translations). Corrections are yaw + translation only; ARKit's gravity axis is trusted.

First attempt, Open3D point-to-plane ICP on fragment clouds: diverged (metre-scale shifts, 40–90° rotations on
revisits), and the graph returned no correction at all. Replaced.

| Walk | Drift | Revisits matched | Revisit misalignment | Wall sharpness | Footprint | Time |
|---|---|---|---|---|---|---|
| c7d28f72c6 (215 s) | off | n/a | 19.7 cm | 1.255 | 65.41 m² | 101 s |
| | **on** | 27 (78 rejected) | **3.5 cm** | **1.292** | 63.52 m² | 210 s |
| 1a8384c3f6 (115 s) | off | n/a | n/a | 1.240 | 58.47 m² | 55 s |
| | on | 0 (18 rejected) | n/a | 1.240 | 58.47 m² (unchanged) | 76 s |

* ARKit drift is real on the long walk (about 20 cm where it revisits places). The correction cuts it to 3.5 cm and
  makes walls sharper.
* Room 4 of the long walk changes from 4.77 to 3.60 m² with correction on. Which is right needs tape ground truth.
* On the short walk no revisit passed the acceptance test, so nothing is changed. This is deliberate: a weak match
  should not move the plan.
* Speed: matching on 4 cm images with a coarse-then-fine yaw search took the step from ~12 min to ~110 s.

## E8: Photo tier on a real capture (study, iPhone 14 Plus), first tape scores (2026-10-02)

13 HEIC stills taken in one turn from the middle of the study, grouped into 8 variants by lens and orientation
(`.claude/worktrees/study-photo-variants/bench/datasets/study_photo_variants.yaml`; not protocol-compliant: only one
landscape 0.5× photo exists). Tape: walls 350.5 / 330.2 / 312.4 / 312.4 cm (not yet assigned to walls; the long-side
pair differs because of a cove), ceiling 274.3 cm. Scorer: `bench/score_study.py` (order-free: short side, long side
against the 330–351 cm band, area against 10.3–10.9 m², interval coverage). Pipeline 0.3.0 + lite profile at 3 views.

| Variant | Photos | Area m² (err) | Short / long side m | 90% interval contains truth |
|---|---|---|---|---|
| portrait_mixed (0.5× + 1×) | 7 | 9.49 (−8%) | 2.83 / 3.35 | yes |
| w1x_landscape | 3 | 13.48 (+25%) | 3.50 / 3.85 | yes |
| landscape_mixed (3 intrinsics) | 6 | 16.53 (+54%) | 3.95 / 4.19 | no |
| uw05_portrait | 4 | 7.15 (−31%) | 2.38 / 3.26 | no |
| uwcrop (2 photos) | 2 | 7.52 (−27%) | 2.54 / 2.96 | no |
| w1x_portrait | 3 | 1.01 (−90%) | 0.65 / 1.56 | no |
| uw05 (portrait + landscape) | 5 | crashed: out of GPU memory | n/a | n/a |
| w1x (portrait + landscape) | 6 | crashed: depth maps of different shapes in one window | n/a | n/a |

* **The photo tier is not reliable yet on real photos:** −90% … +54% area, 2 of 6 intervals contain the truth. That is
  the "confident garbage" the brief penalises; the intervals must widen (or the run must refuse) until this improves.
* Ceiling never reported (none of these photos see enough ceiling). Correct behaviour, but no ceiling score yet.
* Mixed portrait/landscape in one room folder breaks windowed inference. The protocol forbids it, but the walk-in
  test runs whatever is captured: fix by resizing every view to one shape or grouping windows by orientation.
* These sets are not protocol captures; the 8-photo landscape 0.5× ring is still the test that matters.

### The laptop crash during this experiment (17:25–17:44)
The first E8 run froze the 8 GB M1 (the reset report shows a forced power-button restart, not a kernel panic).
Measured afterwards: the photo pipeline held **4.46 GB of GPU (MPS) memory**, invisible in process memory, and free
system memory fell to **1%**. Fixes (0.3.1): GPU memory cap in the lite profile (MPS allocations fail cleanly instead of
starving macOS), one model loaded at a time, the metric model run one image per call, 3 views per window, and
`scripts/run_guarded.sh` for long benchmark loops. Re-run peak: 2.43 GB GPU, < 1.9 GB process, no freeze.

## E9: Capture audit against the protocol (2026-10-02)

Every capture we hold, checked against `docs/CAPTURE_PROTOCOL.md` (pace and turns from ARKit poses; lens, ceiling and
lighting from 12 evenly spaced frames per video; files in `runs/protocol_check/`).

| Capture | Tier | Verdict | Main deviations |
|---|---|---|---|
| `c00a170fe1` (37 s) | LiDAR | furthest off | 3 spaces in 37 s (~12 s each); never tilts above level (86% of frames point >25° down); ends 3.18 m from start; 0.43 m/s |
| `1a8384c3f6` (115 s) | LiDAR | partly | never looks at the ceiling; 0.47 m/s; closes its loop (0.17 m) |
| `c7d28f72c6` (215 s) | LiDAR | closest | looks up (28% of frames); closes its loop (0.39 m); 0.48 m/s and ~43 s per room vs 60–90 s |
| study photos (13) | photos | not a protocol set | mostly portrait, mixed 0.5× / 1× / cropped ultra-wide; one landscape 0.5× photo |
| `study_walk1.MOV` (34 s) | video | close | likely 1× lens (file does not record it); some blur on turns; HDR |
| `house_IMG_4637.MOV` (50 s) | video | off | ~4 rooms in 50 s; heavy blur; does not end at the start; one dark room; HDR |

All three LiDAR walks are ~1.4× faster than the protocol pace and turn at up to 106–133°/s. The LiDAR captures predate
the protocol (2026-09-01). Consequences: ceilings are measurable only on `c7d28f72c6`; drift correction has a loop only
on the two closed walks; all of this data tests robustness, not protocol accuracy.

Protocol changes made from this audit (0.4.1): HDR video off; one floor per capture (the flat has stairs); a check that
0.5× is still selected for video (the file does not record the lens); a concrete LiDAR pace (one step every two
seconds, a quarter turn in at least 2 s) and an explicit ceiling tilt per room.

## E10: Video tier on the real study video (2026-10-02)

`roomscan run study_walk1.MOV` (iPhone 14 Plus, 34 s, HDR, lens not recorded). Pipeline 0.5.0, lite profile, watchdog
peak 2.1 GB, 81 s total (COLMAP 51 s). Scored with `bench/score_study.py`.

| Variant | Frames tracked | Pieces | Short side | Long side | Area | Area interval holds tape |
|---|---|---|---|---|---|---|
| v1: sequential matching, 8 fps, largest piece | 92/272 (34%) | 92, 84, 18, 12, 4 | 2.73 m (−12.6%) | 3.80 m (+8.9%) | 9.97 m² (−3.4%) | yes (±22%) |

* COLMAP's refined focal gives a 57° field of view: the video was shot at **1×**, confirming the capture audit (E9).
* The area is close only because the two side errors cancel; the room shape is wrong and the ±3% video gate fails.
* Two thirds of the video is untracked: the largest piece has 92 frames and the next 84. Joining pieces is next.

## E11: First protocol-style photos (house_b: 19 × landscape 0.5×, 6 rooms) (2026-10-02)

`bench/datasets/house_b.yaml`; pipeline 0.5.1 (model-release fix), lite profile, 64 s, peak 3.0 GB under the watchdog.
Only the study has tape. Rooms are NOT connected: each folder is reconstructed alone and drawn side by side.

| Room | Photos | Area m² (90%) | Ceiling |
|---|---|---|---|
| study | 4 | 11.16 (8.03–14.30) | 2.10 m (1.91–2.28) |
| living | 3 | 12.91 | not observed |
| bedroom | 2 | 8.90 (3.07 × 2.90 m) | not observed |
| kitchen | 3 | 6.22 | 2.67 m |
| bathroom | 2 | 3.84 | not observed |
| store | 3 | 3.43 | not observed |

**Study vs tape:** short side 3.215 m vs 3.124 (+2.9%), long side 3.718 m vs 3.30–3.51 (+6.5%), area 11.16 vs
10.3–10.95 m² (+2.1%, interval holds the truth). Inside the ±8% photo gate. The off-protocol study photos (E8) ranged
−90% … +54%: following the protocol is worth more than any model change so far.

**Ceiling: confident garbage.** 2.10 m against 2.743 m by tape, and the 90% interval (1.91–2.28 m) excludes the truth.
Something else at about 2.1 m passed the ceiling test; to fix before anything is reported as calibrated.

Also fixed here: 0.3.1's "one model at a time" didn't work (the caller and a closure still held the geometry model),
which made `house_a_portrait` run out of the 3.2 GB GPU cap.

## E12: Why the photo-tier ceiling was wrong, and the fix (2026-10-02)

Symptom (E11): study ceiling 2.10 m vs 2.743 m by tape, with a 90% interval (1.91–2.28 m) that excluded the truth.
Script: `bench/local/e12_ceiling.py` (`diagnose`, `ablate`); reconstructions cached in `runs/e12/`.

**Diagnosis.**
* Camera height above the detected floor: 0.82–0.85 m (study), 0.86–0.93 m (bedroom), 0.74–0.79 m (store), but
  1.43–1.54 m in kitchen, living and bathroom. A standing person holds the phone at ~1.4–1.6 m, so the "floor" in the
  first three rooms is ~0.65 m too high.
* Overlay of 3D heights on the photos: the ceiling tiles were found correctly; the "floor" layer covered the desk top
  (study) and the bed (bedroom) as well as the floor.
* Height histogram relative to the camera (study): real floor at −1.50…−1.60 m (~17k points), desk top at −0.85 m
  (~15k points), ceiling at +1.20 m. The detector took the **densest** level below the camera: the desk won narrowly.
* Not the cause: depth model, ceiling test, scale (walls were within +3…+7%), gravity.

**Ablation** (only the floor rule changes; ceiling level identical in every variant):

| Room | A: densest level (0.5.1) | B: lowest level ≥ 40% support | C: ≥ 20% | D: ≥ 60% | Tape |
|---|---|---|---|---|---|
| study | cam 0.85 m → 2.095 m | **cam 1.47 m → 2.722 m** | 2.722 | 2.722 | 2.743 m |
| bedroom | cam 0.89 → 2.172 | **cam 1.36 → 2.641** | 2.641 | 2.174 (miss) | n/a |
| kitchen | 2.671 | 2.670 | 2.670 | 2.670 | n/a |
| living | 2.597 | 2.597 | 2.597 | 2.597 | n/a |
| bathroom | 2.665 | 2.665 | 2.665 | 2.665 | n/a |
| store | cam 0.77 → 1.17 | cam 0.77 → 1.17 | cam 1.48 → 1.879 (box tops) | 1.17 | n/a |

Chosen: B (lowest flat level with ≥ 40% of the densest level's support), plus a plausibility guard: if the camera is
not 1.0–1.95 m above the floor, no ceiling is reported (store).

**Result, full pipeline on house_b (0.5.2):**
* Study ceiling **2.723 m vs 2.743 m (−2.0 cm), interval holds the truth.**
* Study walls also improved, because the wall band is measured from the floor: short side −1.6% (was +2.9%), long side
  +1.6% (was +6.5%), area +0.0% (was +2.1%).
* Store: "floor not found reliably", no ceiling reported (was a confident 1.17 m).
* LiDAR regression check on `c7d28f72c6`: all 5 room areas and ceilings identical to E7.

Note on the gate: the brief's ceiling gate (≤ 1.5 cm) is a LiDAR-tier figure. −2.0 cm on photos is close but
outside it; tape itself is ±0.5 cm at best on a 2.7 m vertical reading.

## E13: Photo-tier wall sizes in the untaped rooms (2026-10-02)

User report: house_b wall sizes look wrong. Only the study has tape, so this uses internal evidence.
Script: `bench/local/e13_walls.py` (top-down per room: wall points, detected wall cells, cameras, outline).

**What the top-down views show.**
* Almost **no wall cells are detected** in photo clouds. The wall test (points over ≥ 35% of the 0.2–1.8 m height band)
  is tuned for dense LiDAR; 2–4 photos give too sparse and noisy a cloud. The outline then comes from the flood inside
  the hull of all points, and edges snap to the densest nearby layer, often furniture.
* Per-wall support is very uneven: some walls rest on 0–22 points (study 3.56 m wall: 3; living 4.64 m wall: 22;
  bedroom 2.96 m wall: 0), i.e. they are inferred, not observed.
* The real walls are visible as thin point lines at the room edges; the study's outline happens to land on them.

**Ablation: outline method** (house_b, pipeline 0.5.2 + option `recon.photo_outline`):

| Room | walls flood (default) | box: outermost long wall lines |
|---|---|---|
| study (tape 3.124 × 3.30–3.51) | 3.07 × 3.56: −1.6% / +1.6%, area 0.0% | 3.35 × 3.51: +7.2% / +0.1%, area +7.7% |
| bedroom | 2.96 × 3.07 | 1.28 × 2.10 (collapsed) |
| kitchen | 2.45 × 2.55 | 2.45 × 2.75 |
| living, store, bathroom | 2.79 × 4.64, 1.48 × 2.38, 1.55 × 2.48 | unchanged (box not found or same) |

Rejected: the box rule overshoots onto lines beyond the wall (door frames, the next room) and collapses the bedroom.
Kept as an option for future tests; default stays "walls".

**Cross-capture repeatability (no tape needed):** the bedroom from `house_a` (7 photos, portrait + one landscape, run
through the 0.5.1 letterbox for the first time) is 2.87 × 3.24 m (9.31 m²); from `house_b` (2 protocol photos)
2.96 × 3.07 m (8.75 m²). Within ~5% of each other.

**Open:** without tape for living, kitchen, store and bathroom, "bad" cannot be quantified. Asked the user for those
dimensions. Candidate causes to test once they exist: per-room metric scale (scale factor varies 1.25–2.39 across
rooms), uncovered walls with too few photos (2–3 per room vs 8 in the protocol), and outline edges snapping to furniture.

## E14: Can photos from different rooms be linked? (2026-10-02)

`bench/local/e14_cross_room.py` on house_b (17 photos in room folders). Matches verified by a fundamental-matrix RANSAC.

| | SIFT | DISK + LightGlue (kornia; Mac GPU, 4.5 GB cap, 55 s) |
|---|---|---|
| same-room photo pairs (median verified matches) | 18 | 43 |
| kitchen ↔ living | 17 | **89** (4643 ↔ 4638: the kitchen photo that looks into the living room) |
| bathroom ↔ living | 21 | **47** (4651 ↔ 4641) |
| every other room pair | 14–44 (noise) | 14–38 (noise) |

* SIFT can't link 0.5× ring photos even within a room (median 18). DISK + LightGlue can, and it finds the real
  cross-room links, but only where a photo sees into the next room: 1–2 of the 5 connections in house_b.
* Engineering: DISK at 1024 px ran out of the 3.2 GB GPU cap; on CPU it took ~1 min per image and stalled with torch
  and OpenCV thread pools in one process. Ran on the GPU with the user-approved 4.5 GB cap.
* Consequence: rooms without a look-through photo need door-geometry matching (E15), and the protocol should ask for
  one photo per doorway looking through it (backlog 0.7).

## E15: Door and window detection on photo rooms (2026-10-02)

Rule (`roomscan/openings/detect.py`): along each wall, 5 cm slots; an opening is a run with little wall surface where
the cameras saw through (see_through), or, for photos, a run with little wall surface inside a camera's field of view
while the rest of the wall is well covered (gap, confidence 0.3).

Iterations on house_b:
1. LiDAR tolerances (wall ±8 cm, beyond 25 cm): **no openings in any photo room.**
2. Diagnosis (study): no points beyond any wall line. The 30th-percentile depth-confidence filter removes the views
   through doorways (furthest point 2.7 m with it, 6.6 m without); photo walls are ~±20 cm thick, not ±1 cm.
   Fix: openings use a lightly filtered (5th percentile) point set; photo tolerances wall ±20 cm, beyond 40 cm.
   Still none.
3. Per-wall profiles (study): one real see-through gap only 0.45 m wide (below the 0.55 m minimum); one 0.65 m stretch
   with no wall surface and nothing visible beyond. Fix: "gap" openings where a camera was looking.
4. Result:

| Room | Openings found |
|---|---|
| kitchen | door 0.95 m (see-through): matches the kitchen↔living visual link (E14) |
| living | door 0.80 m (gap), door 0.55 m (see-through) |
| bathroom | door 0.75 m (gap), door 0.55 m (see-through, weak) |
| study | door 0.70 m (gap) |
| bedroom | window 1.30 m (see-through); no door found |
| store | window 0.85 m (gap); no door found |

No door widths are taped yet, so widths are unverified. LiDAR `c00a170fe1`: the corridor–bathroom door is found from
both rooms at the same place (0.60 / 0.65 m).

## E17: First connected photo plan (house_b) (2026-10-02)

Pipeline 0.6.0: per-room reconstructions are linked by DISK + LightGlue matches between photos of different rooms
(`roomscan/stitch/links.py`); a link with ≥ 45 verified matches is turned into a relative pose by PnP (room A's metric
3D points at the matched pixels → where room B's photo was taken, in A's frame), restricted to yaw + translation, the
yaw snapped to 90° steps between the two rooms' wall directions (`roomscan/stitch/photo_graph.py`).

| Step | kitchen ↔ living |
|---|---|
| visual link | 99 verified matches (kitchen photo looking through the doorway) |
| PnP placement | rotation right (kitchen door wall faces living's wall) but rooms ~2.8 m apart |
| + shared-wall snap | living slid along the wall normal so the facing walls are 15 cm apart; kitchen's 0.95 m door opens into living |

* Why the gap: PnP uses the depth seen through the doorway, the least reliable depth in the image (furthest, lowest
  confidence), so direction is good and distance is not. Rooms linked through a doorway share a wall, which fixes the
  distance; the along-wall position comes from the visual estimate.
* 2 of 6 rooms connected. bathroom, bedroom, store and study have no photo that sees into another room (E14) and are
  drawn to the side with a warning naming them and the fix (a look-through photo per doorway).
* Not yet verified against tape (no living/kitchen measurements).

## E18: More rooms connected: link acceptance by two-way agreement (2026-10-02)

`bench/local/e18_link_debug.py`. The bathroom↔living link had 47 matches in E14 but 42 here: DISK + LightGlue on the
Mac GPU is not bit-reproducible, so a fixed 45-match threshold flipped it in and out. Computing the pose from each
room's depth separately:

| Link | matches | pose from A's depth | pose from B's depth | verdict |
|---|---|---|---|---|
| bathroom → living | 42 | yaw +47.4°, tilt 4.9° | yaw −50.8°, tilt 5.2° | **agree** (mirror yaws within 3.4°, both level) |
| hall → study | 38 | yaw +33.7°, tilt 12.5° | yaw −39.1°, tilt 43.6° | reject (tilt) |
| hall → study (2nd) | 31 | yaw −173.8° | yaw +141.7°, tilt 27.6° | reject |
| bathroom → hall | 27 | yaw +152.5°, tilt 29.7° | tilt 58.5° | reject |

New rule: candidates need ≥ 30 matches, and are accepted only if the two directions agree (yaw within 10°) and both
are level (tilt ≤ 15°). Links are checkpointed so identical photos replay identical links.

Shared-wall snap extended: facing walls may miss each other by up to 1.5 m along the wall (the visual along-wall
position can be 1–2 m off), and when both facing walls carry a door within 2.5 m the doors are lined up.

Result on house_b: **3 of 6 rooms connected** (living, kitchen, bathroom), no overlaps between them. The bathroom's door
does not line up with a living-room door (nearest > 2.5 m away), consistent with it opening onto the hall. Hall
connector photos (4644, 4652) did not add links (best 43 matches, rejected by the agreement test). Bedroom, store and
study have no photo that sees into another room.

## E19: Room adjacency on LiDAR (2026-10-02)

LiDAR rooms come from one cloud, so they are already in one frame; adjacency is filled from:
* **shared door:** an opening detected from both rooms within 0.6 m of each other (both get `connects_to`);
* **open passage:** the two rooms plus the carved (ray-crossed) floor directly between them (within ~20 cm of both)
  form one connected region.

| Attempt for open passages | c00a170fe1 | c7d28f72c6 |
|---|---|---|
| masks within 6 cm | none found | none found |
| masks within 20 cm | + living–bathroom (**false**: across a wall) | + r3–r4 (suspect) |
| … minus detected wall cells | false link remains (that wall wasn't detected) | |
| connected carved floor, any loose floor | links almost every pair (through other rooms' surroundings) | 10 links |
| **connected carved floor directly between the two** | corridor–bathroom (door), living–corridor (passage), living–bathroom (passage, unverified) | r1–r3, r2–r3, r2–r5 (doors), r1–r4 (passage) |

The c00a living–bathroom passage is probably false (the walk went living → corridor → bathroom) but can't be checked
without the space. Kept as a known issue.

## E20: Fix loop: joining COLMAP pieces in the video tier (2026-10-02)

See `fixloop/DECLARATION.md` (written first) and `fixloop/RESULT.md`. Study video, tape ground truth:
before −35.8% short side / −34.5% area (35% of frames) → after −4.0% / −8.1% (64% of frames, 5 of 5 pieces joined).
Prediction met on short side and area, missed on frames used (64% vs ≥ 70%); ±3% gate still fails, as predicted.

## E21: The whole house as one video walk (house_b_walk.MOV, 212 s, 6 rooms) (2026-10-03)

0.5× ultra-wide (estimated 94° FOV, matches the protocol), HDR on (converted), 1698 frames at 8 fps. Ground truth:
no tape for the house rooms, so it is compared with the photo tier of the same house (`house_b`, 6 rooms, 45.95 m²
total) for consistency only. Runs: `runs/e21b/house_b_walk.MOV/` (pipeline 0.7.x + this entry's changes).

| | Result |
|---|---|
| COLMAP (best of 4 seeds, single-threaded) | 50 min; 18 pieces (largest 688 frames), 1047 / 1698 frames registered (62%) |
| Pieces joined by DA3 bridges | 16 of 18; the 169-frame piece was not (no placed frame within the 80-frame bridge limit) |
| Rooms | **3 instead of 6**: 33.8, 8.3, 18.2 m² (total 60.2 m² vs 46.0 m² from photos) |
| Outlines | jagged (14–16 walls per room); room 1 is ~7.6 m long, i.e. several rooms merged |
| Openings / adjacency | 1 door (0.60 m); all three rooms adjacent via open passages |
| Per-keyframe metric-scale spread | 28.6% (study video: 20.4%) |
| Runtime / memory | 887 s after COLMAP (bridging + depth); peak 3.36 GB phys_footprint |

**Verdict: the video tier does not produce a usable plan of a multi-room walk.** Causes, in order of evidence:
1. **38% of the walk never registers**, and the largest unjoined piece holds a whole stretch of it. COLMAP breaks at
   every doorway turn; the bridges are DA3 chains with chain scale factors 0.70–2.50 (E20 saw the same).
2. **Rooms merge** because the joined trajectory misplaces pieces: walls from different pieces don't coincide, so the
   watershed finds no doorway constrictions. Staircase outlines are the same misalignment seen from above.
3. **Metric scale is noisy per keyframe** (29%), so each small piece's scale is uncertain.

**Changes made from this experiment:**
* **Intervals were overconfident.** The video tier used a fixed 4% scale term; the measured per-keyframe spread is
  20–29%. Each piece's scale is a median of n keyframe ratios (standard error ≈ 1.25·spread/√n), so the scale term is
  now √(4%² + frame-weighted mean of piece standard errors²): **9.6% on the study video** (wall intervals now contain
  the tape: short side 3.000 m, interval 2.52–3.48 m vs tape 3.124 m), 14.9% on the house walk. House room-1 area is
  33.8 m² with interval 9.4–58.1 m²: an honest "this capture can't size this room" instead of a confident wrong number.
* **Openings and adjacency on the video tier** (backlog 0.6): sightlines from keyframe depth + camera viewing
  directions, same detector as photos with learned-depth tolerances. Study video: 2 doors (1.15 m gap, 0.85 m
  see-through); the study does have 2 doorways (tape notes), widths untaped.
* **Damage stage skipped every video room**: DA3 rounds 16:9 frames to 504×280 (aspect 1.800 vs 1.778); the aspect
  check now allows 4%.
* **Watchdog blind spot** (found while this ran): RSS missed GPU and swap; see CHANGELOG (watchdog now uses
  phys_footprint).

**Next for video (not done):** register more of the walk (learned features such as DISK+LightGlue inside COLMAP, or a
looser sequential matcher), and allow bridges longer than 80 frames when the gap is a doorway turn.

## E22: Can COLMAP register more of a handheld walk? (2026-10-03) · negative result

`bench/local/e22_video_registration.sh` on the taped study video (`study_walk1.MOV`, 272 frames). Knobs: sequential
matching overlap and the registration threshold (`recon.video_abs_pose_min_inliers`; below 30 also lowers the inlier
ratio 0.25 → 0.15 and min matches 15 → 10).

| Overlap | Min inliers | Registered | Pieces | Per-keyframe scale spread | Lens (fov) | Study result |
|---|---|---|---|---|---|---|
| 10 | **30 (default)** | 175 | 95, 51, 24, 18, 12 (5 joined) | 20.4% | 58° | −4.0% / −6.9%, area −10.8% |
| 20 | 30 | 175 | identical | 20.4% | 58° | identical |
| 10 | 20 | 213 | 207, 22, 22, 2 | 47.8% | 91° | −16.1% / −7.2%, area −22.1%, ceiling −22.8 cm |
| 10 / 20 | 15 | 215 | 215, 10, 5 | 40.6% | 63° | no room: outline could not close |

* More overlap changes nothing: the breaks are not missing pairs but frames with too few good matches (blur in turns).
* A lower threshold registers more frames (79% vs 64%) **but registers them wrongly**: the per-keyframe scale spread
  doubles and the self-calibrated focal length drifts (58° → 91° at 20), so walls get worse or the room disappears.
  Fewer, correct poses beat more, wrong ones.
* Kept the default (30); the knob stays configurable. The remaining route for 2.5 is better features (DISK+LightGlue
  matches into COLMAP) or a protocol fix (slower turns), not a looser threshold.

## E24: A local damage detector (OWLv2), no API key (2026-10-03)

User question: why does damage need an Anthropic key? It doesn't have to. `roomscan/damage/detect_local.py` runs
google/owlv2-base-patch16-ensemble (Apache-2.0, open-vocabulary detection) offline and is now the default
(`damage.backend: owlv2`); Claude stays optional (`vlm`). Same output contract, so projection, rules and scope are shared.

**What can be measured without staged damage: false positives on clean rooms.** The 17 house_b photos (6 rooms) have
no visible damage (each detection was checked by eye).

| Version | Detections on 17 clean photos | What they were |
|---|---|---|
| v1: damage queries only ("a cracked ceiling", …), threshold 0.25 | whole-ceiling boxes in the 2 study photos tested | a query naming a surface matches the clean surface |
| v2: defect-only queries + 13 negative queries (wall, ceiling, window, door, shadow, …) | 4 | tile border ×2 ("crack"), door-frame gap ("hole"), skirting edge ("crack") |
| v3: + negatives for tile border, tiles, door frame, skirting, tap | 2 (0.26, 0.28, both "crack") | thin straight lines |
| **v3 + crack threshold 0.30 (shipped)** | **0** | |

* Negative queries are the main fix: a box whose best label is something undamaged is dropped, and it suppresses
  overlapping damage boxes in NMS.
* The crack threshold (0.30) was chosen on the same 17 photos it is reported on: **in-sample**, so 0/17 is optimistic.
* **Recall is not measured** (no staged-damage captures). The detector may miss faint stains; nothing here says how often.
* Cost: ~6.5 s per photo on the M1 CPU, ~0.9 GB; house_b full run 93 s with DA3 replayed. Outputs are cached in
  `cache/vlm/owlv2_*.json` (committed) like the Claude responses.
