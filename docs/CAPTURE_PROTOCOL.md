# Capture protocol (one page) — DRAFT

Follow this literally. Pick the tier you want; each section is self-contained.

## Before any tier
* Turn on every light. Open interior doors fully. Cover or note any large mirrors.
* Hold the phone **upright (portrait)**, at chest height, **at 1× zoom**. Do not use 0.5× or 2×.
* Walk slowly — about one step per second.

## Tier 1 — Photos (any iPhone 15 or newer)
**Install:** nothing. Use the built-in Camera app, Photo mode, 1×.
For **each room**, take **6 photos** (minimum 4, maximum 8):
1. Stand in a corner, aim at the opposite corner so that floor *and* ceiling are visible. Repeat from the other 3 corners (4 photos).
2. **For every doorway/opening out of the room:** stand just inside the room, 1 m from the doorway, and take one photo of the doorway with the whole door frame visible.
**Hand-off:** AirDrop the photos to the laptop. Put them in one folder per room: `capture/kitchen/`, `capture/hall/` …
Run: `roomscan run capture/ -o out/`

## Tier 2 — Video (any iPhone 15 or newer)
**Install:** nothing. Camera app → Video, 1×, 1080p 30 fps (Settings → Camera → Record Video).
Start in the entry room. Walk along the walls of each room, phone tilted slightly down so the floor-wall line stays in view; turn slowly at corners; once per room, sweep up to the ceiling. Pass through each doorway once in each direction. **End where you started.** About 30–45 s per room, under 5 min total.
**Hand-off:** AirDrop the `.MOV`. Run: `roomscan run walk.MOV -o out/`

## Tier 3 — LiDAR (iPhone Pro 12 or newer)
**Install:** "Stray Scanner" (free, App Store).
Press record, then walk the same path as Tier 2, but at **half the speed**. Keep 1–3 m from the walls. **End where you started.**
**Hand-off:** in Stray Scanner, tap the recording → Share → AirDrop. Unzip it. Run: `roomscan run <folder> -o out/`

## Avoid
Running, fast spins, pointing at a bright window for long, people walking through the frame, and pointing the phone straight down at the floor for long stretches.
