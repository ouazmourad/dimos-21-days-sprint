#!/usr/bin/env python3
"""Standalone keyboard teleop for ArduPilot SITL drone.

Connects directly to SITL via MAVProxy UDP and sends velocity commands.
Shows drone telemetry (GPS, altitude, heading, mode) in a Pygame window.

Usage:
    python dimos/robot/drone/drone_keyboard_sitl.py

Prerequisites:
    - ArduPilot SITL running in Docker
    - MAVProxy forwarding to UDP 14551
"""

import math
import os
import time

os.environ.setdefault("SDL_VIDEODRIVER", "x11")

import pygame
from pymavlink import mavutil

# Connection
MAV_URI = "udp:127.0.0.1:14551"

# Window
WIN_W, WIN_H = 560, 500


def main() -> None:
    # ── MAVLink connection ──────────────────────────────────────────
    print(f"Connecting to {MAV_URI}...")
    mav = mavutil.mavlink_connection(MAV_URI)
    mav.wait_heartbeat()
    print(f"Connected — system {mav.target_system}, component {mav.target_component}")

    # Read initial mode
    mode = "?"
    armed = False

    def update_state() -> None:
        nonlocal mode, armed
        msg = mav.recv_match(type="HEARTBEAT", blocking=False)
        if msg:
            mode = mavutil.mode_string_v10(msg)
            armed = bool(msg.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED)

    # State
    lat, lon, alt, heading = 0.0, 0.0, 0.0, 0.0
    vx, vy, vz = 0.0, 0.0, 0.0
    groundspeed = 0.0

    def update_gps() -> None:
        nonlocal lat, lon, alt, heading, groundspeed
        msg = mav.recv_match(type="GLOBAL_POSITION_INT", blocking=False)
        if msg:
            lat = msg.lat / 1e7
            lon = msg.lon / 1e7
            alt = msg.relative_alt / 1000.0
            heading = msg.hdg / 100.0
        msg = mav.recv_match(type="VFR_HUD", blocking=False)
        if msg:
            groundspeed = msg.groundspeed

    # ── Pygame ──────────────────────────────────────────────────────
    pygame.init()
    screen = pygame.display.set_mode((WIN_W, WIN_H), pygame.SWSURFACE)
    pygame.display.set_caption("Drone SITL Keyboard Control")
    clock = pygame.time.Clock()
    font = pygame.font.Font(None, 24)
    font_big = pygame.font.Font(None, 30)
    font_title = pygame.font.Font(None, 34)

    keys_held: set[int] = set()
    BASE_SPEED = 2.0  # m/s
    BASE_ALT_SPEED = 1.0  # m/s
    BASE_YAW_RATE = 0.5  # rad/s

    print("\n── Drone SITL Keyboard Control ──")
    print("  W/S: Forward/Back    A/D: Yaw")
    print("  Q/E: Strafe L/R      R/F: Alt Up/Down")
    print("  Shift: Boost 2x      Ctrl: Slow 0.5x")
    print("  Space: E-Stop        T: Takeoff 10m")
    print("  L: Land              G: GUIDED mode")
    print("  1: Arm               2: Disarm")
    print("  ESC: Quit\n")

    running = True
    last_cmd_time = 0.0
    status_messages: list[tuple[float, str, tuple[int, int, int]]] = []

    def add_status(text: str, color: tuple[int, int, int] = (0, 255, 255)) -> None:
        status_messages.append((time.time(), text, color))
        print(f">> {text}")

    while running:
        # ── Events ──────────────────────────────────────────────────
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                keys_held.add(event.key)

                if event.key == pygame.K_ESCAPE:
                    running = False

                elif event.key == pygame.K_SPACE:
                    # E-Stop: zero velocity
                    keys_held.clear()
                    send_velocity(mav, 0, 0, 0, 0)
                    add_status("E-STOP! All velocities zeroed", (255, 50, 50))

                elif event.key == pygame.K_t:
                    # Takeoff
                    add_status("Takeoff to 10m...", (0, 255, 100))
                    mav.set_mode("GUIDED")
                    time.sleep(0.5)
                    mav.mav.command_long_send(
                        mav.target_system, mav.target_component,
                        mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
                        0, 1, 0, 0, 0, 0, 0, 0,
                    )
                    time.sleep(1)
                    mav.mav.command_long_send(
                        mav.target_system, mav.target_component,
                        mavutil.mavlink.MAV_CMD_NAV_TAKEOFF,
                        0, 0, 0, 0, 0, 0, 0, 10,
                    )
                    add_status("Takeoff command sent", (0, 255, 100))

                elif event.key == pygame.K_l:
                    # Land
                    add_status("Landing...", (255, 200, 0))
                    mav.set_mode("LAND")

                elif event.key == pygame.K_g:
                    # GUIDED mode
                    mav.set_mode("GUIDED")
                    add_status("GUIDED mode set", (0, 200, 255))

                elif event.key == pygame.K_1:
                    # Arm
                    mav.mav.command_long_send(
                        mav.target_system, mav.target_component,
                        mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
                        0, 1, 0, 0, 0, 0, 0, 0,
                    )
                    add_status("ARM command sent", (0, 255, 100))

                elif event.key == pygame.K_2:
                    # Disarm
                    mav.mav.command_long_send(
                        mav.target_system, mav.target_component,
                        mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
                        0, 0, 0, 0, 0, 0, 0, 0,
                    )
                    add_status("DISARM command sent", (255, 200, 0))

            elif event.type == pygame.KEYUP:
                keys_held.discard(event.key)

        # ── Compute velocity from keys ──────────────────────────────
        vx = vy = vz = yaw_rate = 0.0
        if pygame.K_w in keys_held:
            vx += BASE_SPEED
        if pygame.K_s in keys_held:
            vx -= BASE_SPEED
        if pygame.K_q in keys_held:
            vy -= BASE_SPEED
        if pygame.K_e in keys_held:
            vy += BASE_SPEED
        if pygame.K_r in keys_held:
            vz -= BASE_ALT_SPEED  # NED: negative Z = up
        if pygame.K_f in keys_held:
            vz += BASE_ALT_SPEED  # NED: positive Z = down
        if pygame.K_a in keys_held:
            yaw_rate -= BASE_YAW_RATE
        if pygame.K_d in keys_held:
            yaw_rate += BASE_YAW_RATE

        # Speed modifiers
        boost = pygame.K_LSHIFT in keys_held or pygame.K_RSHIFT in keys_held
        slow = pygame.K_LCTRL in keys_held or pygame.K_RCTRL in keys_held
        if boost:
            mult = 2.0
        elif slow:
            mult = 0.5
        else:
            mult = 1.0
        vx *= mult
        vy *= mult
        vz *= mult
        yaw_rate *= mult

        # Send velocity at ~10Hz
        now = time.time()
        if now - last_cmd_time > 0.1:
            if armed and mode == "GUIDED":
                send_velocity(mav, vx, vy, vz, yaw_rate)
            last_cmd_time = now

        # ── Update telemetry ────────────────────────────────────────
        # Drain MAVLink messages
        for _ in range(20):
            update_state()
            update_gps()

        # ── Render ──────────────────────────────────────────────────
        screen.fill((20, 20, 30))
        y = 15

        # Title
        s = font_title.render("Drone SITL — Keyboard Control", True, (0, 255, 255))
        screen.blit(s, (15, y))
        y += 40

        # Mode + Armed status
        arm_text = "ARMED" if armed else "DISARMED"
        arm_color = (0, 255, 100) if armed else (255, 80, 80)
        s = font_big.render(f"Mode: {mode}", True, (255, 255, 255))
        screen.blit(s, (15, y))
        s = font_big.render(arm_text, True, arm_color)
        screen.blit(s, (WIN_W - 130, y))
        y += 35

        # GPS
        s = font.render(f"LAT: {lat:.6f}   LON: {lon:.6f}", True, (200, 200, 220))
        screen.blit(s, (15, y))
        y += 25
        s = font.render(f"ALT: {alt:.1f}m   HDG: {heading:.0f}°   GS: {groundspeed:.1f}m/s", True, (200, 200, 220))
        screen.blit(s, (15, y))
        y += 30

        # Separator
        pygame.draw.line(screen, (60, 60, 80), (15, y), (WIN_W - 15, y))
        y += 10

        # Velocity readout
        moving = vx != 0 or vy != 0 or vz != 0 or yaw_rate != 0
        labels = [
            f"FWD: {vx:+.1f}",
            f"LAT: {vy:+.1f}",
            f"ALT: {-vz:+.1f}",  # Show positive=up
            f"YAW: {yaw_rate:+.2f}",
        ]
        for i, text in enumerate(labels):
            c = (0, 255, 255) if text.split(":")[1].strip() != "+0.0" and text.split(":")[1].strip() != "+0.00" else (100, 100, 120)
            s = font.render(text, True, c)
            screen.blit(s, (15 + i * 135, y))
        y += 30

        # Speed modifier
        if boost:
            s = font_big.render("[BOOST 2x]", True, (0, 200, 255))
            screen.blit(s, (15, y))
        elif slow:
            s = font_big.render("[SLOW 0.5x]", True, (255, 200, 0))
            screen.blit(s, (15, y))

        # Moving indicator
        indicator_color = (255, 50, 50) if moving else (50, 255, 50)
        indicator_label = "MOVING" if moving else "HOVER"
        pygame.draw.circle(screen, indicator_color, (WIN_W - 45, y + 12), 10)
        s = font.render(indicator_label, True, indicator_color)
        screen.blit(s, (WIN_W - 120, y + 3))
        y += 35

        # Separator
        pygame.draw.line(screen, (60, 60, 80), (15, y), (WIN_W - 15, y))
        y += 10

        # Controls help
        help_lines = [
            "W/S: Forward/Back     Q/E: Strafe L/R",
            "A/D: Yaw Left/Right   R/F: Altitude Up/Down",
            "Shift: Boost (2x)     Ctrl: Slow (0.5x)",
            "Space: E-Stop         T: Takeoff   L: Land",
            "G: GUIDED mode        1: Arm   2: Disarm",
        ]
        for line in help_lines:
            s = font.render(line, True, (100, 100, 120))
            screen.blit(s, (15, y))
            y += 22

        y += 10

        # Status messages (last 3, fade after 5s)
        now = time.time()
        status_messages[:] = [(t, m, c) for t, m, c in status_messages if now - t < 8]
        for t, msg, color in status_messages[-3:]:
            alpha = max(0, min(255, int(255 * (1 - (now - t) / 8))))
            faded = tuple(max(0, int(c * alpha / 255)) for c in color)
            s = font.render(f">> {msg}", True, faded)
            screen.blit(s, (15, y))
            y += 22

        pygame.display.flip()
        clock.tick(30)

    # ── Cleanup ─────────────────────────────────────────────────────
    # Zero velocity before exit
    send_velocity(mav, 0, 0, 0, 0)
    mav.close()
    pygame.quit()
    print("Keyboard control closed.")


def send_velocity(mav: mavutil.mavlink_connection, vx: float, vy: float, vz: float, yaw_rate: float) -> None:
    """Send SET_POSITION_TARGET_LOCAL_NED with velocity in NED frame."""
    mav.mav.set_position_target_local_ned_send(
        0,  # time_boot_ms
        mav.target_system,
        mav.target_component,
        mavutil.mavlink.MAV_FRAME_BODY_NED,  # body frame so vx=forward
        0b0000_0100_1100_0111,  # type_mask: enable vx, vy, vz, yaw_rate
        0, 0, 0,  # position (ignored)
        vx, vy, vz,  # velocity m/s
        0, 0, 0,  # acceleration (ignored)
        0,  # yaw (ignored)
        yaw_rate,  # yaw_rate rad/s
    )


if __name__ == "__main__":
    main()
