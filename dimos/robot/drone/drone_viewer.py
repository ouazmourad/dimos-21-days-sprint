#!/usr/bin/env python3
"""Standalone drone video viewer with keyboard teleop overlay.

Loads recorded drone footage and displays it with WASD-style
velocity controls overlaid using Pygame. No DimOS modules needed.

Usage:
    python dimos/robot/drone/drone_viewer.py
"""

import glob
import os
import pickle
import time

import numpy as np
import pygame

# Use X11 driver for compatibility
os.environ.setdefault("SDL_VIDEODRIVER", "x11")


def load_frames(data_dir: str = "data/drone/video") -> list[tuple[float, np.ndarray]]:
    """Load all replay frames sorted by index."""
    files = sorted(glob.glob(f"{data_dir}/*.pickle"))
    frames = []
    for path in files:
        with open(path, "rb") as f:
            ts, img = pickle.load(f)
            frames.append((ts, img.data))  # img.data is (H, W, 3) numpy array
    return frames


def main() -> None:
    print("Loading drone replay frames...")
    frames = load_frames()
    if not frames:
        print("No replay data found in data/drone/video/")
        print('Run: python -c "from dimos.utils.data import get_data; get_data(\'drone\')"')
        return

    src_h, src_w = frames[0][1].shape[:2]
    print(f"Loaded {len(frames)} frames ({src_w}x{src_h})")

    pygame.init()
    WIN_W, WIN_H = 960, 600  # extra height for HUD bar
    screen = pygame.display.set_mode((WIN_W, WIN_H))
    pygame.display.set_caption("Drone Replay + Teleop")
    clock = pygame.time.Clock()
    font = pygame.font.Font(None, 24)
    font_big = pygame.font.Font(None, 30)

    vel = {"x": 0.0, "y": 0.0, "z": 0.0, "yaw": 0.0}
    frame_idx = 0
    paused = False
    fps = 30.0
    target_fps = 30
    keys_held: set[int] = set()

    print("\nControls (hold keys for velocity, release to stop):")
    print("  W/S: Forward/Back    A/D: Yaw    Q/E: Strafe    R/F: Altitude")
    print("  Shift: Boost 2x      Ctrl: Slow 0.5x")
    print("  Space: E-Stop        P: Pause/Resume      ESC: Quit\n")

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                keys_held.add(event.key)
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_p:
                    paused = not paused
                elif event.key == pygame.K_SPACE:
                    keys_held.clear()
                    vel["x"] = vel["y"] = vel["z"] = vel["yaw"] = 0.0
                    print("E-STOP")
            elif event.type == pygame.KEYUP:
                keys_held.discard(event.key)

        # Compute velocity from held keys
        vel["x"] = vel["y"] = vel["z"] = vel["yaw"] = 0.0
        if pygame.K_w in keys_held:
            vel["x"] += 0.5
        if pygame.K_s in keys_held:
            vel["x"] -= 0.5
        if pygame.K_q in keys_held:
            vel["y"] += 0.5
        if pygame.K_e in keys_held:
            vel["y"] -= 0.5
        if pygame.K_r in keys_held:
            vel["z"] += 0.3
        if pygame.K_f in keys_held:
            vel["z"] -= 0.3
        if pygame.K_a in keys_held:
            vel["yaw"] += 0.5
        if pygame.K_d in keys_held:
            vel["yaw"] -= 0.5

        # Speed modifiers
        boost = pygame.K_LSHIFT in keys_held or pygame.K_RSHIFT in keys_held
        slow = pygame.K_LCTRL in keys_held or pygame.K_RCTRL in keys_held
        if boost:
            multiplier = 2.0
        elif slow:
            multiplier = 0.5
        else:
            multiplier = 1.0
        vel["x"] *= multiplier
        vel["y"] *= multiplier
        vel["z"] *= multiplier
        vel["yaw"] *= multiplier

        # Get frame and blit to screen
        _, rgb = frames[frame_idx % len(frames)]
        # Pygame expects (W, H, 3) surface from (H, W, 3) array
        surf = pygame.surfarray.make_surface(np.transpose(rgb, (1, 0, 2)))
        surf = pygame.transform.scale(surf, (WIN_W, WIN_H - 60))
        screen.blit(surf, (0, 0))

        # Draw HUD bar
        hud_y = WIN_H - 60
        pygame.draw.rect(screen, (15, 15, 25), (0, hud_y, WIN_W, 60))
        pygame.draw.line(screen, (0, 200, 200), (0, hud_y), (WIN_W, hud_y))

        # Velocity readout
        labels = [
            f"FWD: {vel['x']:+.1f}",
            f"LAT: {vel['y']:+.1f}",
            f"ALT: {vel['z']:+.1f}",
            f"YAW: {vel['yaw']:+.1f}",
        ]
        for i, text in enumerate(labels):
            s = font.render(text, True, (0, 255, 255))
            screen.blit(s, (15 + i * 140, hud_y + 8))

        # Mode indicator
        moving = any(vel[k] != 0 for k in ("x", "y", "z", "yaw"))
        if paused:
            mode, mode_color = "PAUSED", (255, 200, 0)
        elif moving:
            mode, mode_color = "MOVING", (255, 50, 50)
        else:
            mode, mode_color = "IDLE", (50, 255, 50)

        s = font_big.render(mode, True, mode_color)
        screen.blit(s, (WIN_W - 90, hud_y + 5))

        # FPS + frame counter
        s = font.render(f"{fps:.0f} FPS  #{frame_idx % len(frames)}", True, (150, 150, 150))
        screen.blit(s, (WIN_W - 140, hud_y + 35))

        # Controls help
        s = font.render("WASD:move  QE:strafe  RF:alt  Shift:boost  Space:stop  ESC:quit", True, (100, 100, 120))
        screen.blit(s, (15, hud_y + 35))

        # Boost/Slow indicator
        if boost:
            s = font_big.render("[BOOST 2x]", True, (0, 200, 255))
            screen.blit(s, (15, 8))
        elif slow:
            s = font_big.render("[SLOW 0.5x]", True, (255, 200, 0))
            screen.blit(s, (15, 8))

        # Title
        title_y = 35 if (boost or slow) else 8
        s = font.render("DRONE REPLAY + TELEOP", True, (0, 255, 255))
        screen.blit(s, (15, title_y))

        pygame.display.flip()

        # FPS tracking
        dt = clock.tick(target_fps) / 1000.0
        if dt > 0:
            fps = 0.9 * fps + 0.1 * (1.0 / dt)

        # Advance frame
        if not paused:
            frame_idx += 1

    pygame.quit()
    print("Viewer closed.")


if __name__ == "__main__":
    main()
