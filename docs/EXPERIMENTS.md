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
