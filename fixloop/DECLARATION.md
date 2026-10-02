# Fix-loop declaration (written and committed BEFORE the fix)

Date: 2026-10-02 23:45 IST. Code state of the before run: git tag `fixloop-before`.

## 1. Worst-performing gate, with the failing number
**Video tier, wall length ±3% (study room, tape ground truth).**
`roomscan run data/raw/video/study_walk1.MOV` (default settings, caches off, `ROOMSCAN_NO_CACHE=1`):

| Quantity | Pipeline | Tape | Error | Gate |
|---|---|---|---|---|
| short side | 2.005 m | 3.124 m | **−35.8%** | ±3%: fail |
| long side | 3.371 m | 3.302–3.505 m | 0.0% | pass |
| floor area | 6.76 m² | 10.3–10.95 m² | **−34.5%**, 90% interval excludes truth | fail |

Frames tracked by COLMAP: **95 of 272 (35%)**, in 5 pieces: 95, 51, 24, 18, 12.
Regenerate: `git checkout fixloop-before && ROOMSCAN_NO_CACHE=1 roomscan run data/raw/video/study_walk1.MOV -o out/`.

## 2. Root-cause hypothesis and evidence
**The video tier builds the room from COLMAP's largest piece only; fast turns split the walk into pieces, so most of
the room is never in the cloud and the outline closes early on the unseen side.**

Evidence:
* Only 35% of frames are in the piece used; the other 65% (four pieces) are discarded.
* The error is one-sided: the long side is right (seen) and the short side is 36% short (largely unseen).
* E10: a run that happened to track 190 frames in one piece gave 3.47 m × 3.72 m, both sides close to tape.
* E4/E5 (proxy video with ARKit truth): pieces break at fast turns (65–131° in about a second); inside a piece COLMAP is
  accurate to ~1° and a few cm.

## 3. The fix and the predicted number
**Fix:** join COLMAP's pieces instead of keeping only the largest one: chain DA3 through each gap between consecutive
pieces (frame by frame through the turn, as E5 did on the proxy video, 2 → 328 of 480 frames), give each piece a
similarity transform into the first piece's frame, and build the room from keyframes of every joined piece.

**Predicted after the fix (same video, same settings):**
* frames used: from 35% to **≥ 70%**;
* short-side error: from −35.8% to **within ±10%**;
* area error: from −34.5% to **within ±12%**, with the 90% interval containing the tape value.

I expect meaningful movement but **not** a pass of the ±3% gate: E5 left ~10° median rotation error on the bridged
frames, which is several centimetres of wall position at 2–3 m.
