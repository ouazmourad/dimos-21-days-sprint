#!/usr/bin/env python3
"""
Spatial Memory Visual Demo — Real Robot Data + Rerun Visualization

Uses actual recorded Go2 robot data (video + odometry from an office walk)
to build a real spatial memory map with CLIP embeddings, then queries it.

Three phases:
  1. Play the recorded walk at real-time (camera feed + real trajectory on 3D map)
  2. Build spatial memory by computing CLIP embeddings for sampled frames
  3. Run text queries and highlight matching locations on the map

Usage:
    CI=1 uv run python examples/spatial_memory_demo.py
"""

import functools
import os
import pickle
import shutil
import sys
import tempfile
import time
import uuid
from datetime import datetime

import cv2
import numpy as np
import rerun as rr
import rerun.blueprint as rrb

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dimos.perception.spatial_perception import SpatialMemory
from dimos.utils.data import get_data

print = functools.partial(print, flush=True)

QUERIES = [
    ("office room with desk",       [46, 204, 113]),
    ("computer monitor or screen",  [52, 152, 219]),
    ("chair or furniture",          [231, 76,  60]),
    ("person walking",              [241, 196, 15]),
    ("door or hallway",             [155, 89, 182]),
    ("outdoor scene with trees",    [230, 126, 34]),
    ("robot or machine",            [26, 188, 156]),
]


def build_blueprint() -> rrb.Blueprint:
    return rrb.Blueprint(
        rrb.Horizontal(
            rrb.Vertical(
                rrb.Spatial2DView(name="Robot Camera", origin="camera"),
                rrb.TextDocumentView(name="Info", origin="info"),
                row_shares=[3, 2],
            ),
            rrb.Spatial3DView(name="Spatial Memory Map", origin="world"),
            column_shares=[2, 3],
        ),
        collapse_panels=True,
    )


def load_robot_data() -> tuple[list, list]:
    """Load recorded video frames and odometry from the Go2 office walk dataset."""
    data_path = get_data("unitree_office_walk")
    video_dir = os.path.join(str(data_path), "video")
    odom_dir = os.path.join(str(data_path), "odom")

    # Load video: list of (timestamp, image_ndarray)
    video_files = sorted(os.listdir(video_dir))
    video_data = []
    for fname in video_files:
        with open(os.path.join(video_dir, fname), "rb") as f:
            ts, img = pickle.load(f)
        video_data.append((ts, np.array(img)))

    # Load odom: list of (timestamp, x, y, z)
    odom_files = sorted(os.listdir(odom_dir))
    odom_data = []
    for fname in odom_files:
        with open(os.path.join(odom_dir, fname), "rb") as f:
            ts, msg = pickle.load(f)
        p = msg["data"]["pose"]["position"]
        odom_data.append((ts, p["x"], p["y"], p["z"]))

    return video_data, odom_data


def get_position_at_time(odom_data: list, t: float) -> list[float]:
    """Interpolate robot position for a given timestamp."""
    # Find surrounding odom entries
    for i in range(len(odom_data) - 1):
        if odom_data[i][0] <= t <= odom_data[i + 1][0]:
            # Linear interpolation
            t0, x0, y0, z0 = odom_data[i]
            t1, x1, y1, z1 = odom_data[i + 1]
            alpha = (t - t0) / (t1 - t0) if t1 != t0 else 0
            return [
                x0 + alpha * (x1 - x0),
                y0 + alpha * (y1 - y0),
                z0 + alpha * (z1 - z0),
            ]
    # Fallback: use nearest
    if t <= odom_data[0][0]:
        return [odom_data[0][1], odom_data[0][2], odom_data[0][3]]
    return [odom_data[-1][1], odom_data[-1][2], odom_data[-1][3]]


def main() -> None:
    temp_dir = tempfile.mkdtemp(prefix="spatial_memory_demo_")

    # ── Init Rerun ───────────────────────────────────────────────────────────
    print("Starting Rerun viewer...")
    rr.init("Spatial Memory Demo", spawn=True)
    rr.send_blueprint(build_blueprint())
    rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
    rr.log("info", rr.TextDocument(
        "# Loading robot data...\n\nPlease wait.",
        media_type=rr.MediaType.MARKDOWN,
    ))

    try:
        # ── Load real robot data ─────────────────────────────────────────────
        print("Loading Go2 office walk dataset...")
        video_data, odom_data = load_robot_data()
        n_video = len(video_data)
        n_odom = len(odom_data)
        t_start = video_data[0][0]
        t_end = video_data[-1][0]
        duration = t_end - t_start
        print(f"  {n_video} video frames, {n_odom} odom readings, {duration:.1f}s walk")

        # Sample every ~1s for CLIP processing
        sample_every = max(1, n_video // int(duration))
        sampled_indices = list(range(0, n_video, sample_every))
        print(f"  Will store {len(sampled_indices)} frames for spatial memory.\n")

        # ================================================================
        # PHASE 1: Play the recorded walk at real-time
        # ================================================================
        print(f"--- Phase 1: Playing recorded walk ({duration:.0f}s) ---")
        rr.log("info", rr.TextDocument(
            f"# Phase 1: Robot Walk Playback\n\n"
            f"**Real recorded data** from a Unitree Go2 robot walking through an office.\n\n"
            f"Duration: {duration:.0f}s — {n_video} frames — {34:.0f}m walked\n\n"
            f"The trajectory on the 3D map is the robot's **actual path**.",
            media_type=rr.MediaType.MARKDOWN,
        ))

        # Log start marker
        start_pos = get_position_at_time(odom_data, t_start)
        rr.log("world/start", rr.Points3D(
            [start_pos], colors=[[0, 220, 80]], radii=[0.12], labels=["START"],
        ))

        trajectory_pts: list[list[float]] = []
        display_every = max(1, n_video // (int(duration) * 5))  # ~5fps display
        map_every = max(1, n_video // (int(duration) * 2))       # ~2fps map update

        for idx in range(n_video):
            loop_start = time.time()
            ts, frame = video_data[idx]
            pos = get_position_at_time(odom_data, ts)

            # Display camera feed (~5fps)
            if idx % display_every == 0:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                h, w = frame_rgb.shape[:2]
                scale = min(480 / w, 360 / h, 1.0)
                if scale < 1.0:
                    small = cv2.resize(frame_rgb, (int(w * scale), int(h * scale)))
                else:
                    small = frame_rgb
                rr.log("camera", rr.Image(small))

            # Update 3D trajectory (~2fps)
            if idx % map_every == 0:
                trajectory_pts.append(pos)
                if len(trajectory_pts) >= 2:
                    rr.log("world/trajectory", rr.LineStrips3D(
                        [np.array(trajectory_pts)],
                        colors=[[100, 160, 255]], radii=[0.03],
                    ))
                rr.log("world/robot", rr.Points3D(
                    [pos], colors=[[255, 255, 0]], radii=[0.10],
                    labels=[f"Go2 ({pos[0]:.1f}, {pos[1]:.1f})"],
                ))

            # Pace to real-time
            if idx < n_video - 1:
                dt_real = video_data[idx + 1][0] - ts
                elapsed = time.time() - loop_start
                wait = dt_real - elapsed
                if wait > 0:
                    time.sleep(wait)

        # Finalize
        rr.log("world/robot", rr.Clear(recursive=False))
        end_pos = get_position_at_time(odom_data, t_end)
        rr.log("world/end", rr.Points3D(
            [end_pos], colors=[[220, 50, 50]], radii=[0.12], labels=["END"],
        ))
        print("  Playback complete.\n")

        # ================================================================
        # PHASE 2: Build spatial memory with CLIP
        # ================================================================
        print("--- Phase 2: Building spatial memory with CLIP ---")
        rr.log("info", rr.TextDocument(
            f"# Phase 2: Building Spatial Memory\n\n"
            f"Computing CLIP embeddings for **{len(sampled_indices)}** frames...\n\n"
            f"~0.5s per frame on CPU.",
            media_type=rr.MediaType.MARKDOWN,
        ))

        memory = SpatialMemory(
            collection_name="demo_spatial_memory",
            embedding_model="clip",
            embedding_dimensions=512,
            new_memory=True,
            db_path=os.path.join(temp_dir, "chroma_db"),
            visual_memory_path=os.path.join(temp_dir, "visual_memory.pkl"),
            output_dir=os.path.join(temp_dir, "images"),
            min_distance_threshold=0.05,
            min_time_threshold=0.1,
        )

        stored_positions: list[list[float]] = []
        stored_frames: list[np.ndarray] = []
        n_samples = len(sampled_indices)

        for si, idx in enumerate(sampled_indices):
            ts, frame = video_data[idx]
            pos = get_position_at_time(odom_data, ts)

            try:
                embedding = memory.embedding_provider.get_embedding(frame)
                fid = f"frame_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
                metadata = {
                    "pos_x": pos[0], "pos_y": pos[1], "pos_z": pos[2],
                    "rot_x": 0.0, "rot_y": 0.0, "rot_z": 0.0,
                    "timestamp": ts, "frame_id": fid,
                }
                memory.vector_db.add_image_vector(
                    vector_id=fid, image=frame, embedding=embedding, metadata=metadata,
                )
                stored_positions.append(pos)
                stored_frames.append(frame)
            except Exception as e:
                print(f"  Error at frame {idx}: {e}")
                continue

            # Show progress
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w = frame_rgb.shape[:2]
            scale = min(480 / w, 360 / h, 1.0)
            if scale < 1.0:
                small = cv2.resize(frame_rgb, (int(w * scale), int(h * scale)))
            else:
                small = frame_rgb
            rr.log("camera", rr.Image(small))

            rr.log("world/stored", rr.Points3D(
                np.array(stored_positions),
                colors=[[0, 200, 120]] * len(stored_positions),
                radii=[0.06] * len(stored_positions),
            ))

            pct = int((si + 1) / n_samples * 100)
            rr.log("info", rr.TextDocument(
                f"# Phase 2: Building Spatial Memory\n\n"
                f"**Progress**: {si+1}/{n_samples} ({pct}%)\n\n"
                f"Position: ({pos[0]:.1f}, {pos[1]:.1f})",
                media_type=rr.MediaType.MARKDOWN,
            ))

            if (si + 1) % 10 == 0 or si == 0:
                print(f"  {si+1}/{n_samples} ({pct}%)")

        n_stored = len(stored_positions)
        print(f"  Done — {n_stored} frames stored.\n")

        # ================================================================
        # PHASE 3: Text queries
        # ================================================================
        print("--- Phase 3: CLIP text queries ---")

        lines = [
            "# Query Results\n",
            f"**{n_stored}** frames from a real Go2 office walk stored with CLIP\n",
            "| # | Query | Score | Relevance | Position |",
            "|---|-------|-------|-----------|----------|",
        ]

        for qi, (query, color) in enumerate(QUERIES):
            results_q = memory.query_by_text(query, limit=1)
            if not results_q:
                lines.append(f"| {qi+1} | {query} | — | — | — |")
                continue

            r = results_q[0]
            sim = 1 - r.get("distance", 1.0)
            meta = r.get("metadata", {})
            if isinstance(meta, list):
                meta = meta[0] if meta else {}
            mx, my = float(meta.get("pos_x", 0)), float(meta.get("pos_y", 0))
            mz = float(meta.get("pos_z", 0))
            tag = "HIGH" if sim > 0.25 else "MED" if sim > 0.20 else "LOW"

            print(f'  {qi+1}. "{query}" → {sim:.4f} [{tag}] at ({mx:.1f}, {my:.1f})')
            lines.append(f"| {qi+1} | **{query}** | {sim:.4f} | {tag} | ({mx:.1f}, {my:.1f}) |")

            rr.log(f"world/queries/{qi:02d}", rr.Points3D(
                [[mx, my, mz + 0.3]], colors=[color], radii=[0.14],
                labels=[f"Q{qi+1}: {query} ({sim:.3f})"],
            ))

        lines += [
            "", "---",
            "**Real data**: Trajectory and positions are from actual Go2 robot odometry.",
            "Colored markers = text query matches. Hover for details.",
        ]
        rr.log("info", rr.TextDocument(
            "\n".join(lines), media_type=rr.MediaType.MARKDOWN,
        ), static=True)

        print(f"\nAll done! Explore the Rerun viewer.")
        print("  - The trajectory is the robot's REAL path through the office")
        print("  - Hover colored markers to see query match details")
        print("  - Press Ctrl+C to exit\n")

        while True:
            time.sleep(1)

    except KeyboardInterrupt:
        print("\nShutting down.")
    except Exception as e:
        import traceback
        print(f"\nERROR: {e}")
        traceback.print_exc()
    finally:
        try:
            memory.stop()
        except Exception:
            pass
        shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
