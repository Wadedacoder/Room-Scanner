# Device matrix

Which tier runs on which iPhone, and what accuracy each tier has **measured** so far. Every accuracy figure below
comes from `docs/EXPERIMENTS.md`; where nothing has been measured, the cell says so.

## Capture devices

| Tier | Capture app | iPhone 15 / 15 Plus | iPhone 15 Pro / Pro Max (and 12 Pro or newer Pro) | Devices we tested on |
|---|---|---|---|---|
| Photos | Camera app, 0.5× | ✅ | ✅ | iPhone 14 Plus (house_b, study) |
| Video | Camera app, 0.5× | ✅ | ✅ | iPhone 14 Plus (study_walk1, house_b_walk) |
| LiDAR | Stray Scanner (free) | ❌ no LiDAR sensor | ✅ | a Pro iPhone (the three supplied captures, model not recorded) |

* The brief's walk-in uses an iPhone 15 or newer. Our own captures are from an **iPhone 14 Plus**: same 0.5× ultra-wide
  (13 mm equivalent) and 1× wide (26 mm) camera layout as the iPhone 15, but we have not run an iPhone 15 capture yet.
* LiDAR results come only from the three captures we were given; we have no Pro iPhone of our own, so there is no LiDAR
  capture of a taped room and no LiDAR accuracy number against tape yet.

## Measured accuracy (against tape unless noted)

| Tier | Walls | Floor area | Ceiling | Interval holds truth | Source |
|---|---|---|---|---|---|
| Photos, protocol capture (0.5× landscape, 4 photos of the study) | −1.6% / +4.5% | +0.7% | −2.0 cm | area ✅, ceiling ✅ | E11, E12, `bench/gates.py` |
| Photos, off-protocol (portrait / mixed lenses, 2–7 photos) | −90% … +54% (area) | | | 2 of 6 | E8 |
| Video, off-protocol (1× lens, fast turns, 34 s study walk) | −4.0% / −6.9% | −10.8% | not observed | area ✅ | E10, fix loop (`fixloop/`) |
| LiDAR | not measured against tape | not measured | not measured | | no taped LiDAR capture |
| LiDAR, internal consistency | revisit misalignment 19.7 → 3.5 cm with drift correction | | | | E7 |

Gates for reference: photo walls ±8%, video ±3%, ceiling ≤ 1.5 cm, openings ≤ 2 cm.

## Laptop running the pipeline

| Profile | Machine | Tested |
|---|---|---|
| `lite` (default on 8 GB) | M1 MacBook Air, 8 GB | ✅ every result above; GPU memory capped at 3.2 GB after a freeze (0.3.1) |
| `cuda-16gb` | NVIDIA 16 GB (Kaggle T4) | model experiments E1–E3 only, not the full pipeline |
| `mac-16gb`, `mac-32gb`, `cuda-24gb` | | not tested; settings extrapolated from E1 memory measurements |

Typical runtimes on the 8 GB M1: photos 10–13 s per room (47.7 s for 6 rooms; 4 s per room from checkpoints);
34 s video 306 s; 37 s LiDAR walk 20 s, 215 s walk ~100 s (+ ~110 s drift correction).
