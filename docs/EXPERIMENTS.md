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

## E7 (planned): photo tier without EXIF metadata

**Question:** what does the photo tier deliver when the focal length is missing (photos sent through a chat app,
screenshots, edited exports)? The first study capture arrived like this: 6 of 9 JPGs at 1125×2200 with no EXIF.
**Why it matters:** E1 found the EXIF focal is the main source of metric scale (predicted focal was off by up to +24%,
scale error up to +18%). Today `check_capture.py` blocks such photos instead of degrading.
**Plan:** strip EXIF from each `study_*` variant (`bench/datasets/study_photo_variants.yaml`) and from the proxy sets,
run with the predicted focal, and compare wall error and interval width against the same set with EXIF. Decide
between (a) refusing, (b) running with predicted focal and a much wider interval, (c) a device-model focal lookup.
The interval must widen enough to stay calibrated; confident garbage on thin input caps the total score.

## E8: First real Camera-app photos: the study (2026-10-02)

8 variants of one 13-photo turn (`bench/datasets/study_photo_variants.yaml`), iPhone 14 Plus, scored against tape
(study ≈ 3.40 × 3.12 m incl. cove side, 10.6 m², ceiling 2.74 m; wall order not yet recorded, so long/short sides are compared).
Code at `5e8be9f`, `lite` profile, M1.

| Variant | n | Long wall | Short wall | Area | Truth in area interval |
|---|---|---|---|---|---|
| uw05_portrait (0.5×) | 4 | −11% | −18% | −27% | no |
| uwcrop (1.54 mm / 26 mm-eq) | 2 | −13% | −19% | −29% | no |
| w1x_landscape | 3 | +13% | +12% | +27% | yes |
| w1x (mixed orientation) | 6 | +46% | +11% | +61% | no |
| landscape_mixed (3 intrinsics) | 6 | +27% | +34% | +70% | no |
| portrait_mixed (0.5× + 1×) | 7 | +48% | −83% | +97% | no (6 walls) |
| w1x_portrait | 3 | −54% | −79% | −91% | no |
| uw05 (0.5×, mixed orientation) | 5 | crash | | | |

* **Crash:** a room folder mixing portrait and landscape fails in scale chaining: DA3 center-crops each window to its
  smallest image (378×378 vs 504×378), then the shared-view depth maps are compared with mismatched shapes
  (`operands could not be broadcast together (3,378,378) (3,504,378)`). Whether it fires depends on window grouping
  (`w1x` mixes orientations and ran). Walk-in risk: users will mix orientations.
* **Overconfidence:** truth outside the area interval in 6 of 7 runs. Intervals are not calibrated on real photos.
* **No ceiling height** on any photo run.
* 0.5× sets are the most consistent (−11 to −19% on walls), the same under-measurement as E6. 1× sets swing ±50%+.
* None of these sets follows the protocol (8 landscape 0.5×); a compliant study capture is still needed.

## E9: Multi-room photo set: study + bedroom + living (2026-10-02)

`data/raw/photos/house_a_portrait` (study 4 × 0.5×; bedroom 6 and living 7, mixed 0.5×/1×, all portrait), code at
`308b218`, `lite`, M1, 221 s. `house_a` (bedroom includes one landscape photo) crashes with the E8 orientation error,
so one sideways photo loses the whole property.

| Room | Output | Tape |
|---|---|---|
| study | 3.04 × 2.56 m, 7.8 m² | ≈ 3.40 × 3.12 m, 10.6 m² (−11% / −18%; identical to the single-room E8 run) |
| bedroom | 3.58 × 2.58 m, 9.3 m² | not measured |
| living | 10-wall outline incl. 0.01 m and 0.32 m walls, 38.8 m² | not measured |

* **Photo-tier stitch gate fails outright:** no doorway matching, so rooms are laid side by side and adjacency is empty
  (truth: study, bedroom and kitchen each open onto living). Openings, ceiling heights and damage are also absent.
* Living's outline has sliver walls: the outline needs a minimum wall length / simplification step.
