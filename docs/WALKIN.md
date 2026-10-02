# Walk-in test: runbook

Everything needed on the day, in order. Tested on the M1 MacBook Air 8 GB (`lite` profile), pipeline 0.8.x.

## Before (once, ~6 min, needs internet)
```bash
git clone https://github.com/Wadedacoder/Room-Scanner.git && cd Room-Scanner
EXTRAS=geo,vlm,web,dev ./scripts/setup.sh      # ~25 s
./scripts/setup_models.sh                       # ~5 min: torch, DA3, COLMAP, DISK/LightGlue, OWLv2 weights
.venv/bin/pytest -q                             # 48 tests, ~1 min
```
After this, runs need **no internet and no API key** (models load offline; damage uses the local OWLv2 detector).
Clean-machine timing: README → first plans in ≈ 10 min (docs/BENCHMARK.md §5b).

## Capture
Follow `docs/CAPTURE_PROTOCOL.md` literally. The single most important rules: **0.5× lens, landscape, 8 overlapping
photos per room from one spot, one photo through every doorway.** Off-protocol captures still run, but with wider
intervals and warnings (E26).

## Run (one command per capture)
```bash
.venv/bin/roomscan run <capture> -o out/
```
| Capture | Looks like | Live runtime on the M1 | Peak memory |
|---|---|---|---|
| Photos | folder of room folders (or loose photos = one room, or a .zip) | ~15–30 s per room; 6-room house 109 s | ≤ 4.4 GB |
| Video | one `.MOV` / `.mp4` | ~5–6 min for a 1-min walk; long walks (3+ min) can take 30–60 min | ≤ 3.4 GB |
| LiDAR | Stray Scanner folder or its .zip | ~30 s (short) to ~5 min (3.5-min walk with drift correction) | ≤ 1.3 GB |

Or open the website: `.venv/bin/roomscan serve` → http://localhost:8765, drop the capture, read the result.
For long jobs use the memory watchdog: `LIMIT_MB=5500 scripts/run_guarded.sh .venv/bin/roomscan run <capture> -o out/`.

## Read the output (`out/<capture>/`)
* `plan.svg`: rooms, wall lengths, doors/windows, damage (red), counts in the header.
* `plan.json` (schema `schema/plan.schema.json`): every number has `lo`/`hi` (90% interval).
* **Read the warnings first** (printed at the end of the run, also in plan.json): they say when a capture is
  off-protocol, a room could not be measured, rooms could not be linked, or a ceiling was not observed.
* `config.resolved.yaml`: exact settings, models and hardware for this run.

## If something goes wrong
| Symptom | Cause | What to do |
|---|---|---|
| "no room could be measured" | too little of the walls seen | re-capture following the protocol (8-photo ring / slow walk) |
| rooms "drawn to the side, not linked" | no photo sees through a doorway into the next room | add the doorway photo (protocol step 6) |
| "off-protocol photos (…)" | 1× lens, portrait, or < 4 photos | fine to keep; the interval is wider. Re-capture for accuracy |
| "no focal length in EXIF" | photos were re-saved / sent through a messenger | AirDrop or cable the originals |
| "camera tracking covered only N%" (video) | fast turns | walk and turn slower; or use photos |
| run killed by the watchdog | memory over the limit | close other apps; `-p lite`; photos instead of a long video |
| "COLMAP step failed: …" | the message names the cause | re-run once (seen once in the sweep, did not reproduce) |
| a result looks wrong | | rerun with `ROOMSCAN_NO_CACHE=1` to rule out a stale checkpoint |

## What to expect (measured, docs/BENCHMARK.md)
* Photos (protocol capture of the taped study): walls −1.6% / +4.5%, area +0.7%, ceiling −2.0 cm; intervals hold.
* Video: walls ±4–7% on a single room; a whole-house walk is not reliable (E21).
* LiDAR: drift corrected (19.7 → 3.5 cm on revisits); accuracy unmeasured (no taped LiDAR capture).
* Damage: 0 false detections on 17 clean photos (in-sample); recall unmeasured.
