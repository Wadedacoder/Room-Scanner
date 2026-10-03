# Room-Scanner: submission summary

Phone capture (photos, video or LiDAR) in → a dimensioned floor plan out (`plan.json` to a schema + `plan.svg`), with
walls, ceiling height, floor area, doors/windows, room adjacency, damage regions, concealed-damage flags and scope
line items, and a 90% interval on every number. One command per capture: `roomscan run <capture> -o out/`.
Built in 48 h on an 8 GB M1 MacBook Air; captures from an iPhone 14 Plus (no LiDAR on hand). Pipeline v0.9.0.

## What works (measured)
| | Result | Evidence |
|---|---|---|
| Photo tier, one room, protocol capture | walls −1.6% / +4.5%, area +0.7%, ceiling −2.0 cm vs tape | BENCHMARK §1 |
| Interval calibration | 90% intervals hold in 8–9 of 10 captures of the taped room | E26, BENCHMARK §2b |
| Repeatability | 3 live runs identical to 4 decimals (photos and video) | BENCHMARK §2 |
| Fix loop | declared worst gate (video short side −35.8%), fixed to −4.0%, predictions recorded | `fixloop/` |
| LiDAR | drift on revisits 19.7 → 3.5 cm; rooms split with adjacency | E7, E19 |
| Robustness | all 25 captures on disk complete; off-protocol captures run with warnings; clean-machine install ≈ 10 min | BENCHMARK §5b–5c |
| Damage | local detector (OWLv2, no API key); 0 false detections on 17 clean photos (in-sample) | E24 |

## What does not work (and why)
* **Stitching rooms into one plan from photos** is limited by evidence: only rooms that some photo sees *into* can be
  placed. Our house photos rarely look between rooms, so 3 of 6 rooms link with the default matcher (5 of 7 with
  MASt3R on a GPU, unverified layout). Unlinked rooms are measured and drawn apart, clearly marked, never guessed.
  The capture protocol's look-through photo per doorway is the fix (E14, E28–E32).
* **Video across several rooms** is not reliable: COLMAP splits a handheld walk into many small pieces whose joins
  disagree, so walls double and rooms merge (E21–E23, E34). Single-room video: walls −4% / −7% (fails the ±3% gate).
* **Not done**: head-to-head vs magicplan and LiDAR accuracy (no Pro iPhone); damage recall (no staged damage);
  only one room has tape, so all accuracy claims rest on it.

## Where to look
* `README.md`: setup and quick start · `docs/WALKIN.md`: runbook
* `docs/TECH_REPORT.md`: method, uncertainty, failure modes · `docs/BENCHMARK.md`: all numbers
* `docs/EXPERIMENTS.md`: E1–E34, every experiment incl. negative results · `docs/CHANGELOG.md`: versioned history
* `docs/COMPLIANCE.md`: requirement → file · `fixloop/`: the fix-loop bundle · `docs/CAPTURE_PROTOCOL.md`
