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
| Photos, protocol capture (0.5× landscape, 4 photos of the study) | −1.6% / +4.5% | +0.7% | −2.0 cm | walls ✅ area ✅ ceiling ✅ | E11, E12, E26 |
| Photos, off-protocol (1× / portrait / mixed lenses / 2–3 photos; 9 sets of the study) | short side −20.5% … +15.5% (one −79%) | −31% … +27% | 2 sets observed | short 7/9, long 8/9, area 8/9 (widened ±15% term) | E26 |
| Video, off-protocol (1× lens, fast turns, 34 s study walk) | −4.0% / −6.9% | −10.8% | not observed | walls ✅ area ✅ (measured scale term, E21) | E10, fix loop, E21 |
| Video, whole house (0.5×, 212 s walk, 6 rooms) | no tape | 3 merged rooms instead of 6 | not observed | wide intervals, not usable | E21 |
| LiDAR | not measured against tape | not measured | not measured | | no taped LiDAR capture |
| LiDAR, internal consistency | revisit misalignment 19.7 → 3.5 cm with drift correction | | | | E7 |

Gates for reference: photo walls ±8%, video ±3%, ceiling ≤ 1.5 cm, openings ≤ 2 cm.

## Laptop running the pipeline

| Profile | Machine | Tested |
|---|---|---|
| `lite` (default on 8 GB) | M1 MacBook Air, 8 GB | ✅ every result above; GPU capped (3.2 GB, 4.5 GB for heavy steps), watchdog at 5.5 GB phys_footprint |
| `cuda-16gb` | NVIDIA 16 GB (Kaggle T4) | model experiments E1–E3 only, not the full pipeline |
| `mac-16gb`, `mac-32gb`, `cuda-24gb` | | not tested; settings extrapolated from E1 memory measurements |

Typical runtimes on the 8 GB M1 (live, no checkpoints; memory = phys_footprint incl. GPU): photos 15–28 s per room,
6-room house 109 s at 4.2 GB peak (≈ 8 s replayed from checkpoints); 34 s study video ~6 min at 3.2 GB; 212 s
whole-house video ~65 min (COLMAP 50 min); LiDAR 37 s walk ~30 s, 215 s walk ~3.5–5 min with drift correction at
≤ 0.4 GB.
