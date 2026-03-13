# Day 3 — Crash Testing DimOS: Spatial Memory

## Video Script (~80 seconds)

### [0s–10s] INTRO

"Day 3 of crash testing DimOS. Today I dug into Spatial Memory — the feature that lets a robot remember what it saw and where it saw it. I wanted to visualize it, and it took some real debugging to get there."

### [10s–25s] PROBLEM 1: VISUALIZATION GAP

"DimOS has a built-in Rerun visualization system — the RerunBridgeModule — that auto-renders any live module publishing on LCM topics. But it's designed for live robot operation with the full Dask cluster running. Spatial Memory doesn't implement the `to_rerun()` hook yet, and we're replaying offline data, not running a live robot. So I built a standalone Rerun demo that wires directly into the recorded Go2 dataset — camera feed on the left, 3D trajectory map on the right."

### [25s–40s] PROBLEM 2: FAKE TRAJECTORY

"Next issue — the robot's path on the 3D map was a hardcoded sine wave. It looked nothing like the actual video. The robot walks through an office but the map shows a smooth wave — completely fake. Fix: I pulled in real odometry data from a recorded Unitree Go2 dataset — 877 position readings over 50 seconds. Now the trajectory on the map matches exactly what the camera sees."

### [40s–55s] PROBLEM 3: CLIP BLOCKING PLAYBACK

"Then CLIP killed the whole thing. Each frame takes half a second to embed on CPU, so the video would freeze every frame while CLIP processed. The fix was splitting the demo into two phases — first, play the full walk at real-time so you actually see the robot's journey. Then, go back and compute all the CLIP embeddings separately. No more freezing."

### [55s–70s] PROBLEM 4: METADATA MISMATCH

"One more — when I tried querying by location, nothing came back. Turns out Spatial Memory stores positions as `pos_x` and `pos_y` in the metadata, but the query function searches for `x` and `y`. Key mismatch. Text queries still worked though, and those are the real power — ask 'where's the door' and it points to the right spot on the map."

### [70s–80s] WRAP UP

"Four bugs, four fixes. No visualization hook — built a standalone demo. Fake trajectory — used real odometry. CLIP blocking — split into phases. Metadata mismatch — found and documented it. That's Day 3."

---

## Running the Demo

```bash
CI=1 uv run python examples/spatial_memory_demo.py
```

This launches a Rerun viewer with three phases:

1. **Playback** — Real-time replay of the Go2 office walk (camera + trajectory)
2. **CLIP Processing** — Builds spatial memory by embedding ~53 sampled frames
3. **Text Queries** — Searches like "door or hallway" highlight matching locations on the 3D map
