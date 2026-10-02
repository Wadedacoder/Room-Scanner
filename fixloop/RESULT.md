# Fix-loop result

Declaration: `fixloop/DECLARATION.md` (committed in `5baa29e` before any code change; before-state tag `fixloop-before`).
After-state tag: `fixloop-after`. Readable diff of the fix:

```bash
git diff fixloop-before fixloop-after -- roomscan/recon/bridge.py roomscan/recon/sfm.py roomscan/pipeline/video.py configs/default.yaml
```

Regenerate (caches off so every model runs live):

```bash
git checkout fixloop-before && ROOMSCAN_NO_CACHE=1 roomscan run data/raw/video/study_walk1.MOV -o out/before
git checkout fixloop-after  && ROOMSCAN_NO_CACHE=1 roomscan run data/raw/video/study_walk1.MOV -o out/after
python bench/score_study.py out/before && python bench/score_study.py out/after
```
On the after-state, `-s recon.video_join_pieces=false` reproduces the before behaviour on the same code.

## Numbers (study_walk1.MOV, tape: short 3.124 m, long 3.302–3.505 m, area 10.3–10.95 m²)

| | Before | After v1 | **After (v2, shipped)** | Predicted | Prediction |
|---|---|---|---|---|---|
| frames used | 95 / 272 (35%) | 105 / 272 (39%) | **175 / 272 (64%)** | ≥ 70% | missed (64%) |
| pieces joined | 1 of 5 | 4 of 5 (largest left out) | **5 of 5** | all | met |
| short side | 2.005 m, −35.8% | 2.570 m, −17.7% | **3.000 m, −4.0%** | within ±10% | met |
| long side | 3.371 m, 0.0% | 3.730 m, +6.8% | 3.168 m, −4.0% | n/a | n/a |
| floor area | 6.76 m², −34.5%, interval misses | 9.59 m², −7.1% | **9.48 m², −8.1%, interval holds** | within ±12% | met |
| ±3% wall gate | fail | fail | **fail** | fail | as predicted |
| runtime | 214 s | 316 s | 306 s | | |

**After v1** was my first implementation of the declared fix. It joined pieces in time order and bridged only when the
next piece started after the previous one ended. COLMAP's pieces interleave in time (the 95-frame piece starts at frame
69, before an earlier piece ends), so the largest piece was the one left out. v2 anchors on the largest piece and
bridges each other piece at its closest frames in time. Both runs are kept (`fixloop/after_v1`, `fixloop/after`).

## Was the root cause right?
Yes. Using only the largest piece left 65% of the video out of the room. Joining the pieces moved the unseen short side
from −35.8% to −4.0% and the area from −34.5% to −8.1%, with the remaining error now on both sides (−4.0% / −4.0%)
instead of one.

## Why it falls short of the ±3% gate
* 36% of frames are still unused: COLMAP never registered them (not in any piece of ≥ 8 frames).
* The bridges are DA3 chains through fast turns. E5 measured ~10° median rotation error on bridged frames; the chain
  scale factors here range 1.18–7.53, a sign that the bridge has to absorb large scale differences between pieces.
* Metric scale comes from DA3Metric (E1–E2: −1% … −9% scale error on its own), now estimated per piece.
* The capture is off-protocol (1× lens, fast turns, E9): the protocol's 0.5× lens and slow turns target exactly these.
