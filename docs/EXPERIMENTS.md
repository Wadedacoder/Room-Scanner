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
