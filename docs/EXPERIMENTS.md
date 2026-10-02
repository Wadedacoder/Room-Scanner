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
