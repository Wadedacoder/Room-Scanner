# Capture protocol (one page)

Follow it literally. Pick one tier; each section stands alone. Total time: about 2 minutes per room.

## Before any tier (all tiers)
1. Turn on every light and open the curtains. Open interior doors **fully**.
2. Settings → Camera → **Lens Correction: ON** (the default). Settings → Camera → Record Video → **HDR Video: OFF**.
   Leave everything else at its default.
3. Don't zoom with two fingers. Use only the **0.5×** or **1×** button where a step says to.
4. Keep people and pets out of the room while you capture.
5. **One floor per capture.** Don't go up or down stairs while capturing; capture each floor separately.

## Tier 1: Photos (any iPhone 15 or newer, no install)
For **each room**, in the Camera app, Photo mode, **0.5×**, phone held **sideways (landscape)**:
1. Stand in the middle of the room. If the room is too small (bathroom, closet), stand in its doorway instead.
2. Face a corner. Take a photo with the floor-wall line and the ceiling-wall line both visible.
3. **Turn clockwise by about half a screen** (until what was in the middle of the screen is at the left edge). Take the next photo.
4. Repeat until you are back where you started: **8 photos**. Every photo must share about half of its view with the one before it.
5. Every doorway out of the room must appear **whole** (both sides of the frame and the floor) in at least one photo.
6. **Doorway photo (one per doorway, counts toward the 8):** stand about 1 m back from each doorway, inside this room,
   and take one photo straight *through* it, so the next room fills the middle of the picture. This is what lets the
   rooms be joined into one floor plan.

Don't skip around the room or take photos from different spots. One spot, one full turn.

**Hand-off:** make one folder per room (`kitchen/`, `hall/`, …) and put that room's 8 photos in it.
* Mac: select the photos in Photos → AirDrop.
* Windows/Linux: connect a cable → copy from `DCIM`, or Photos app → Share → Save to Files → iCloud Drive / Google Drive.
HEIC and JPEG are both fine.

Run: `roomscan run capture/ -o out/`, where `capture/` contains the room folders.

## Tier 2: Video (any iPhone 15 or newer, no install)
Camera app → Video, phone held **upright (portrait)**, default resolution. Tap **0.5** before you press record and
check it still shows **0.5×** (the video file does not record which lens was used, so we cannot check it afterwards).
1. Start recording in the doorway of the first room.
2. Walk slowly along the walls (one step per second), phone tilted slightly down so the floor-wall line stays in view.
3. In each room, stop in the middle once and turn slowly through a full circle, tilting up to the ceiling for a moment.
4. Walk through each doorway once in each direction. **Finish where you started.** About 30–45 s per room.

**Hand-off:** AirDrop, or cable/Files as above, to get the `.MOV`. Run: `roomscan run walk.MOV -o out/`

## Tier 3: LiDAR (iPhone 12 Pro or newer Pro, install "Stray Scanner", free)
1. Open Stray Scanner, press record, then walk the same path as Tier 2 but at **half the speed: about one step
   every two seconds**, and turn slowly (a quarter turn should take at least 2 seconds). Stay 1–3 m from the walls.
   In each room, tilt up to the ceiling for a few seconds.
2. **Finish where you started**, then stop recording.

**Hand-off:** in Stray Scanner, open the recording → Share → AirDrop or Save to Files. You get one folder.
Run: `roomscan run <folder> -o out/`

## Avoid (all tiers)
Running or fast turns · pointing at a bright window for long · walking with your back to the room · covering the
camera with a finger at 0.5× (the ultra-wide lens sees your fingers). Mirrors and glass are fine; just don't film
**only** a mirror.
