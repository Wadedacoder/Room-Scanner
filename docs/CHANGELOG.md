# Changelog

Every change to the pipeline, why it was made and how it was measured. Versions are git tags (`git checkout v0.2.0`
reproduces that version); every `plan.json` records the version that produced it in `capture.pipeline_version`.
Experiment numbers (E1, E2, …) refer to `docs/EXPERIMENTS.md`.

Format per entry: **Change** · **Why** (the evidence that prompted it) · **How it was checked** (the measurement).

## [Unreleased]

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
