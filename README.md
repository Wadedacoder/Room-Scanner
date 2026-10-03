# Room-Scanner

Phone capture (photos, video or LiDAR) in, then a dimensioned, stitched floor plan out, with damage regions, scope line items and a calibrated interval on every measurement.

* **Submission summary: [docs/SUBMISSION.md](docs/SUBMISSION.md)** (PDF: `docs/Room-Scanner_submission.pdf`)
* Plan: [docs/PLAN.md](docs/PLAN.md)
* Capture protocol: [docs/CAPTURE_PROTOCOL.md](docs/CAPTURE_PROTOCOL.md)
* Walk-in runbook: [docs/WALKIN.md](docs/WALKIN.md)
* Benchmark: [docs/BENCHMARK.md](docs/BENCHMARK.md) · Technical report: [docs/TECH_REPORT.md](docs/TECH_REPORT.md)
* Compliance matrix: [docs/COMPLIANCE.md](docs/COMPLIANCE.md)
* Output schema: [schema/plan.schema.json](schema/plan.schema.json)

## Quick start (macOS / Linux)

```bash
./scripts/setup.sh                                  # uv + Python 3.11 venv + package; prints detected hardware
./scripts/fetch_data.sh                             # raw captures -> data/raw/
.venv/bin/roomscan run <capture> -o out/            # one command per capture
.venv/bin/pytest -q
```

Tier is auto-detected: a Stray Scanner folder means LiDAR, a `.MOV`/`.mp4` file means video, and a folder of per-room photo folders means photos. A `.zip` of any of these works too (extracted under the output folder). Off-protocol captures (loose photos, more than 8 per room, re-saved photos without EXIF) run with warnings instead of failing.

## Test it yourself (local website)

```bash
EXTRAS=geo,vlm,web,dev ./scripts/setup.sh && ./scripts/setup_models.sh   # once
.venv/bin/roomscan serve                                              # then open http://localhost:8765
```

Drop a capture on the page: a folder with one sub-folder of photos per room, a video, or a Stray Scanner folder or
zip. The page shows the plan, every measurement with its 90% interval, the pipeline's warnings, and a comparison
with tape values you type in (pre-filled from `bench/ground_truth/home_tape.yaml` when the room name matches).
Runs go one at a time through the memory watchdog; history and outputs are kept in `runs/web/`.

## Hardware profiles

The pipeline picks a profile for the machine it runs on. Stronger machines get larger models, more views per pass and finer voxels.

| Profile | Picked when | Photo/video model | Views/pass | LiDAR stride / voxel |
|---|---|---|---|---|
| `lite` | anything else (e.g. M1 8 GB, CPU-only) | DA3-Base + DA3Metric-Large, GPU memory capped at 3.2 GB | 3 | 4 / 2 cm |
| `mac-16gb` | Apple silicon ≥ 16 GB | DA3-Large-1.1 | 6 | 2 / 1 cm |
| `mac-32gb` | Apple silicon ≥ 32 GB | DA3-Large-1.1 | 8 | 1 / 1 cm |
| `cuda-16gb` | NVIDIA 12–23 GB (Kaggle T4/P100) | DA3-Giant-1.1 (fp16) | 4 | 1 / 1 cm |
| `cuda-24gb` | NVIDIA ≥ 24 GB | DA3-Nested-Giant-Large (bf16) | 6 | 1 / 1 cm |

```bash
roomscan hw                                   # detected hardware and the profile 'auto' picks
roomscan config -p mac-16gb                   # print a fully resolved config
roomscan run <capture> -p cuda-16gb -c my.yaml -s recon.max_views=24
```

Configs resolve in layers: `configs/default.yaml` < `configs/profiles/<p>.yaml` < `--config file` < `--set key=value`. A key that isn't in `default.yaml` is rejected, so a typo fails instead of being ignored. Every run writes `config.resolved.yaml`, including the hardware and a config digest, next to its output, so each reported number can be traced to the exact profile and models that produced it.

Licences: DA3-Giant and Nested-Giant are CC BY-NC 4.0; all models in `lite` and `mac-16gb` are Apache-2.0 (see `roomscan/models.py`).
