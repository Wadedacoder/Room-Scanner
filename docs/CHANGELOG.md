# Changelog

Every change to the pipeline, why it was made and how it was measured. Versions are git tags (`git checkout v0.2.0`
reproduces that version); every `plan.json` records the version that produced it in `capture.pipeline_version`.
Experiment numbers (E1, E2, …) refer to `docs/EXPERIMENTS.md`.

Format per entry: **Change** · **Why** (the evidence that prompted it) · **How it was checked** (the measurement).

## [Unreleased]

### Added
- **Door / window detection** (`roomscan/openings/detect.py`): openings are wall stretches the camera saw through
  (points beyond the wall line, ray crossing the gap); door if the see-through reaches the floor, window above a sill.
  Openings land in `plan.json` and are drawn in the SVG. Checked: LiDAR `c00a170fe1` finds the corridor–bathroom door
  from both rooms at the same place (0.60 / 0.65 m); a synthetic 0.9 m door test passes. Photo rooms: none found yet
  (E15, in progress).
- **Cross-room photo matching** experiment (E14): DISK + LightGlue finds the real kitchen↔living link (89 matches) and
  bathroom↔living (47); SIFT can't (≤ 44, noise level). `kornia` added for it.
- `scripts/healthcheck.sh` (tests, website, memory, stuck jobs, disk, git, offline weights), run every 30 min by a
  session watch; `docs/BACKLOG.md`: everything still required, prioritised.
- Video tracking settings `recon.video_fps`, `recon.video_matcher`, `recon.video_seq_overlap`.

### Changed
- **Video tracking is now repeatable:** COLMAP with a fixed seed and single-threaded extraction, matching and mapping.
  Before, the same video gave 92 vs 65 tracked frames and −3.4% vs −33.5% area; now two runs are identical (100 frames,
  area +2.3%). Best-of-4-seeds mapping is in the code but NOT yet verified (runs were interrupted).
- Models load from the local copy first (`load_pretrained`); a Hugging Face disconnect mid-run killed a photo job
  although every weight was cached.

### Fixed
- COLMAP child cleans SQLite's -wal/-shm files left by an interrupted run ("No registered database factory").

### Added
- `recon.photo_outline` option (`walls` default | `box`): the box rule (outermost long wall lines) was tested in E13
  and rejected (study area +7.7%, bedroom collapsed); kept only for future experiments.
- `bench/local/e13_walls.py`: top-down diagnosis of photo-tier rooms (wall points, cameras, outline, wall support).

### Measured (E13)
- Bedroom repeatability across two photo sets: 2.87 × 3.24 m vs 2.96 × 3.07 m (~5%). First real mixed
  portrait/landscape folder (house_a bedroom) runs, confirming the 0.5.1 letterbox change on real data.

## [0.5.2] - 2026-10-02 · ceiling fix: floor = lowest flat level, not the densest · tag `v0.5.2`

### Fixed
- **Ceilings measured from furniture.** The floor was the densest flat level below the camera; in the study that was
  the desk top and in the bedroom the bed (camera "0.85 m above the floor"). Now the lowest flat level with ≥ 40% of
  the densest level's support (`find_levels(floor_rule="lowest")`).
  - Why (E12): study ceiling 2.10 m vs 2.743 m tape, interval excluding the truth (confident garbage).
  - Checked (E12): study ceiling 2.723 m (−2.0 cm, interval holds); walls improve too (−1.6% / +1.6%, area 0.0%);
    kitchen, living, bathroom unchanged; LiDAR `c7d28f72c6` identical.

### Added
- Floor plausibility guard in the shared back-end: camera must be 1.0–1.95 m above the floor, else no ceiling is
  reported and a warning says why (the cluttered store room).

## [0.5.1] - 2026-10-02 · model memory release fixed; first protocol photos scored · tag `v0.5.1`

### Fixed
- One model at a time actually works now. 0.3.1 deleted only the helper's own reference; the caller and the windowing
  closure kept the geometry model alive while the metric model loaded, so `house_a_portrait` ran out of the 3.2 GB GPU
  cap. Checked: 6-room `house_b` runs at a 3.0 GB peak.

### Added (shipped in this tag without being listed at the time; noted afterwards)
- Mixed portrait/landscape photos in one room folder: minority-orientation views are scaled into a canvas of the
  majority shape (kept upright), intrinsics adjusted, padding masked out of the depth (`letterbox_to_common_shape`).
  This edit landed in the working tree from a command the user had rejected, and went into the 0.5.1 commit.
  Unit-tested (principal point lands on the image centre); NOT yet run on a real mixed set (`house_a`). No effect
  on E11: every house_b photo is landscape.

### Measured (E11)
- First protocol-style photos (landscape 0.5×): study area +2.1%, sides +2.9% / +6.5% (inside the ±8% gate).
- Known issue: study ceiling 2.10 m vs 2.743 m with an interval that excludes the truth (confident garbage).
- Known issue: rooms from separate folders are not connected into one plan.

## [0.5.0] - 2026-10-02 · video tier v1 · tag `v0.5.0`

### Added
- **Video tier end to end** (`roomscan run walk.MOV`): upright frames via ffmpeg (HDR squeezed to SDR), COLMAP
  sequential SfM in a child process with the focal length refined (videos don't record it), metric scale from
  DA3Metric depth at COLMAP's sparse points, dense cloud from keyframe metric depth, shared back-end.
  - Reports tracking coverage, the estimated lens (from the refined focal) and HDR as warnings.
  - Checked (E10): study video → 9.97 m² (−3.4%) but short side −12.6%, long side +8.9%; only 34% of frames tracked.
- The local website accepts videos now (same command underneath).

### Known issues
- Only COLMAP's largest piece is used; fast turns split the walk (E4/E5). Joining pieces is next.

## [0.4.1] - 2026-10-02 · capture protocol fixes from the capture audit · tag `v0.4.1`

### Changed
- `docs/CAPTURE_PROTOCOL.md`: HDR video off; one floor per capture; tap 0.5 and confirm it before recording video;
  LiDAR pace made concrete (one step every two seconds, a quarter turn in at least 2 s) plus a ceiling tilt per room.
  - Why (E9): every capture we hold deviates. LiDAR walks are ~1.4× too fast with turns up to 133°/s, two never look
    at the ceiling; both iPhone videos are HDR (untested in the pipeline) and likely 1×, which the file can't show.
  - Checked: n/a until the next protocol capture; the audit table in E9 is the baseline to compare against.

## [0.4.0] - 2026-10-02 · local test website · tag `v0.4.0`

### Added
- **`roomscan serve`: a local website for testing captures** (`roomscan/web/`). Upload a photo folder, video, Stray
  Scanner folder or zip (pick or drag and drop); see the floor plan, each room's area, ceiling and wall lengths with
  90% intervals, the pipeline's warnings and the run log; type tape values (pre-filled from the ground-truth sheet)
  to get errors against the brief's per-tier gate and whether each interval holds the tape value.
  - Why: the user asked to test captures themselves; the same page can serve as the walk-in demo screen.
  - Safety: one job at a time, each in its own process under `scripts/run_guarded.sh` with the profile's GPU cap
    (two concurrent photo runs would freeze an 8 GB Mac). Runs from the checkout's code, not a stale installed copy.
  - History: every run is kept with its profile and pipeline version (`runs/web/jobs.json`, survives restarts).
  - Checked: study photos uploaded through the API → photo tier ran (peak 1.8 GB), plan + SVG served; a video
    upload fails with "pipeline for tier 'video' not implemented yet"; 5 tests for upload detection and the API.
- `web` extra (fastapi, uvicorn, python-multipart).

### Added
- Study tape readings assigned to walls (`bench/ground_truth/home_tape.yaml`): door wall 123 in, whiteboard wall
  130 in, cupboard wall 123 in, brown wall ~138 in (approximate), ceiling 108 in. Opposite pairs 312.4/312.4 cm and
  330.2/~350.5 cm. Single readings; cove size, door widths and second readings still to record.

## [0.3.1] - 2026-10-02 · memory safety after the laptop crash · tag `v0.3.1`

### Fixed
- **The photo pipeline could freeze an 8 GB Mac.** The first real-photo benchmark (E8) held 4.46 GB of Apple-GPU
  memory; that memory is shared with macOS and invisible in process memory, free RAM fell to 1% and the laptop needed
  a forced restart.
  - How: `runtime.gpu_memory_fraction` caps MPS memory (lite: 0.6 of the 5.33 GB recommended working set = 3.2 GB;
    over the cap torch raises out-of-memory instead of starving the OS). Models are loaded one at a time and freed;
    the metric model runs one image per call (it is monocular, so results are identical); lite windows 4 → 3 views,
    because 4 views went over 3.2 GB on MPS.
  - Checked: same run, peak 2.43 GB GPU and < 1.9 GB process memory, 26 s instead of 154 s (no swapping).
- Note: the cap is relative to the GPU working set, not RAM (0.4 meant 2.13 GB, not 3.2 GB). Documented in the profile.

### Added
- `scripts/run_guarded.sh`: kills a run whose process memory passes a limit; used for every benchmark loop.
- `bench/score_study.py`: order-free tape scoring of the real study capture (E8).

### Known issues (E8, real photos)
- Photo-tier areas −90% … +54% across the study variants; only 2 of 6 intervals contain the truth.
- Portrait and landscape photos in one room folder crash windowed inference (out of memory, or shape mismatch).

## [0.3.0] - 2026-10-02 · drift correction · tag `v0.3.0` (code in 2ff3029, tag on the changelog commit after it)

### Added
- **LiDAR drift correction, on by default** (`roomscan/stitch/drift.py`). The walk is cut into ~4 s fragments; where
  it revisits a place, top-down wall images of the two fragments are matched (yaw ±3°, shift by phase correlation,
  accepted only if wall overlap improves and the shift is under 30 cm); a linear pose graph solves yaw + translation.
  - Why: the brief makes "poses used as-is" an automatic fail on the drift row.
  - Checked (E7): long walk `c7d28f72c6`, revisit misalignment 19.7 → 3.5 cm, wall sharpness 1.255 → 1.292.
    Short walk: no confident revisit, plan unchanged by design.
- `drift.method=off` for the required on/off ablation (`bench/local/e7_drift_ablation.py`).

### Changed
- Matching on 4 cm wall images with a coarse-then-fine yaw search: drift step 12 min → ~110 s on a 215 s walk.

### Fixed
- `--set drift.method=off` failed: YAML reads a bare `off` as boolean false. Normalised in the config loader; test added.

### Removed
- First drift attempt (Open3D point-to-plane ICP on fragment clouds). It diverged (metre-scale shifts, 40–90°
  rotations on revisits) and the pose graph returned no correction; never shipped.

### Known issues
- One room on the long walk changes 4.77 → 3.60 m² with correction on. Which is right needs ground truth.

## [0.2.0] - 2026-10-02 · photo tier · tag `v0.2.0` (commit 4d820e0)

### Added
- **Photo tier end to end** (`roomscan run <folder of room folders>`): EXIF 35 mm-equivalent focal → intrinsics,
  DA3-Base depth + poses in windows, metric scale from DA3Metric with the EXIF focal, gravity from the cameras' image
  axes refined by a floor-plane fit.
  - Why: the brief's floor is "any picture in, results out".
  - Checked (E6): runs on the proxy sets; rooms 43–69% too small (see Changed).
- **Shared geometry back-end** (`roomscan/pipeline/backend.py`) for all tiers, with a per-tier error model
  (LiDAR 1 cm; photos 3 cm + 5% scale; video 2 cm + 4%). The scale terms come from E1–E2 (−0.7% … −9.3%).
- Capture checker (`scripts/check_capture.py`) and the tape ground-truth sheet.

### Changed
- Photo rooms are outlined from the observed walls, not from ray carving.
  - Why (E6): with perfect depth and true camera positions, carving from 8 views still recovered only 28% of the
    living room; carving needs thousands of viewpoints.
  - Checked (E6): perfect-data ceiling went from 2.05 to 6.18 m² (living, LiDAR 7.41) and 3.13 to 5.25 m² (bathroom,
    LiDAR 5.62).

### Known issues
- Learned camera positions on the proxy frames are too wrong for the wall outline (non-overlapping 1× frames).
- Open-ended corridors leak into neighbouring rooms (no doorway detection). Rooms are not stitched.

## [0.1.0] - 2026-10-02 · LiDAR tier · tag `v0.1.0` (commit 15ca00b)

### Added
- Repository, output schema, plan, compliance matrix, capture protocol, hardware profiles with layered config.
- Stray Scanner loader. Poses verified as OpenCV camera-to-world (the ARKit convention put the floor at camera height).
- **LiDAR tier**: fused cloud; per-room ceilings, reported only when actually observed; interior from free-space
  carving; watershed room split at door width; walls re-fitted to the room's own face; `plan.json` + `plan.svg`.
  - Checked: room split matches the hand annotation on `c00a170fe1` (3 rooms); 5 rooms on `c7d28f72c6`.
- Experiments E1–E5 (model choice, scale, memory, video poses); the protocol was rewritten to an overlapping 0.5× ring.

### Fixed during development (each would have corrupted results)
- HEVC seeking returned the wrong frame (pixel diff up to 38): switched to sequential decoding.
- Rotated photos were scored against unrotated camera poses: poses rotated with the image (checked to 1 mm).
- Regular installs shipped without profiles or schema: both now packaged in the wheel.
- pycolmap and torch abort in one process (two OpenMP runtimes): COLMAP runs in a subprocess.
