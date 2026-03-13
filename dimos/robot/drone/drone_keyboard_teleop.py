#!/usr/bin/env python3
# Copyright 2025-2026 Dimensional Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import threading

import pygame

from dimos.core.core import rpc
from dimos.core.module import Module
from dimos.core.stream import Out
from dimos.msgs.geometry_msgs import Twist, Vector3

# Force X11 driver to avoid OpenGL threading issues
os.environ["SDL_VIDEODRIVER"] = "x11"


class DroneKeyboardTeleop(Module):
    """Pygame-based keyboard teleop for drone 3D velocity control.

    Publishes Twist messages with full 3D linear velocity + yaw.

    Controls:
        W / S       Forward / Backward
        A / D       Yaw left / right
        Q / E       Strafe left / right
        R / F       Altitude up / down
        Shift       Boost (2x speed)
        Ctrl        Slow (0.5x speed)
        Space       Emergency stop (zero all velocities)
        ESC         Quit
    """

    cmd_vel: Out[Twist]

    _stop_event: threading.Event
    _keys_held: set[int] | None = None
    _thread: threading.Thread | None = None
    _screen: pygame.Surface | None = None
    _clock: pygame.time.Clock | None = None
    _font: pygame.font.Font | None = None

    def __init__(self) -> None:
        super().__init__()
        self._stop_event = threading.Event()

    @rpc
    def start(self) -> None:
        super().start()
        self._keys_held = set()
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._pygame_loop, daemon=True)
        self._thread.start()

    @rpc
    def stop(self) -> None:
        stop_twist = Twist()
        stop_twist.linear = Vector3(0, 0, 0)
        stop_twist.angular = Vector3(0, 0, 0)
        self.cmd_vel.publish(stop_twist)
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(2)
        super().stop()

    def _pygame_loop(self) -> None:
        if self._keys_held is None:
            raise RuntimeError("_keys_held not initialized")

        pygame.init()
        self._screen = pygame.display.set_mode((520, 440), pygame.SWSURFACE)
        pygame.display.set_caption("Drone Keyboard Teleop")
        self._clock = pygame.time.Clock()
        self._font = pygame.font.Font(None, 24)

        while not self._stop_event.is_set():
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self._stop_event.set()
                elif event.type == pygame.KEYDOWN:
                    self._keys_held.add(event.key)
                    if event.key == pygame.K_SPACE:
                        self._keys_held.clear()
                        stop_twist = Twist()
                        stop_twist.linear = Vector3(0, 0, 0)
                        stop_twist.angular = Vector3(0, 0, 0)
                        self.cmd_vel.publish(stop_twist)
                        print("EMERGENCY STOP!")
                    elif event.key == pygame.K_ESCAPE:
                        self._stop_event.set()
                elif event.type == pygame.KEYUP:
                    self._keys_held.discard(event.key)

            twist = Twist()
            twist.linear = Vector3(0, 0, 0)
            twist.angular = Vector3(0, 0, 0)

            # Forward / Backward (W/S)
            if pygame.K_w in self._keys_held:
                twist.linear.x = 0.5
            if pygame.K_s in self._keys_held:
                twist.linear.x = -0.5

            # Strafe left / right (Q/E)
            if pygame.K_q in self._keys_held:
                twist.linear.y = 0.5
            if pygame.K_e in self._keys_held:
                twist.linear.y = -0.5

            # Altitude up / down (R/F)
            if pygame.K_r in self._keys_held:
                twist.linear.z = 0.3
            if pygame.K_f in self._keys_held:
                twist.linear.z = -0.3

            # Yaw left / right (A/D)
            if pygame.K_a in self._keys_held:
                twist.angular.z = 0.5
            if pygame.K_d in self._keys_held:
                twist.angular.z = -0.5

            # Speed modifiers
            speed_multiplier = 1.0
            if pygame.K_LSHIFT in self._keys_held or pygame.K_RSHIFT in self._keys_held:
                speed_multiplier = 2.0
            elif pygame.K_LCTRL in self._keys_held or pygame.K_RCTRL in self._keys_held:
                speed_multiplier = 0.5

            twist.linear.x *= speed_multiplier
            twist.linear.y *= speed_multiplier
            twist.linear.z *= speed_multiplier
            twist.angular.z *= speed_multiplier

            self.cmd_vel.publish(twist)
            self._update_display(twist)

            if self._clock is None:
                raise RuntimeError("_clock not initialized")
            self._clock.tick(50)

        pygame.quit()

    def _update_display(self, twist: Twist) -> None:
        if self._screen is None or self._font is None or self._keys_held is None:
            raise RuntimeError("Not initialized correctly")

        self._screen.fill((20, 20, 30))

        y = 15

        # Title + speed modifier
        speed_tag = ""
        if pygame.K_LSHIFT in self._keys_held or pygame.K_RSHIFT in self._keys_held:
            speed_tag = " [BOOST 2x]"
        elif pygame.K_LCTRL in self._keys_held or pygame.K_RCTRL in self._keys_held:
            speed_tag = " [SLOW 0.5x]"

        title = self._font.render("Drone Keyboard Teleop" + speed_tag, True, (0, 255, 255))
        self._screen.blit(title, (15, y))
        y += 35

        # Velocity readout
        lines = [
            f"Forward/Back (W/S):  {twist.linear.x:+.2f} m/s",
            f"Strafe L/R   (Q/E):  {twist.linear.y:+.2f} m/s",
            f"Altitude     (R/F):  {twist.linear.z:+.2f} m/s",
            f"Yaw          (A/D):  {twist.angular.z:+.2f} rad/s",
        ]
        for line in lines:
            surf = self._font.render(line, True, (255, 255, 255))
            self._screen.blit(surf, (15, y))
            y += 28

        y += 10

        # Active keys
        active = ", ".join(
            pygame.key.name(k).upper() for k in sorted(self._keys_held) if k < 256
        )
        surf = self._font.render(f"Keys: {active}", True, (180, 180, 180))
        self._screen.blit(surf, (15, y))
        y += 35

        # Status indicator
        moving = (
            twist.linear.x != 0
            or twist.linear.y != 0
            or twist.linear.z != 0
            or twist.angular.z != 0
        )
        color = (255, 50, 50) if moving else (50, 255, 50)
        label = "MOVING" if moving else "IDLE"
        pygame.draw.circle(self._screen, color, (470, 30), 15)
        surf = self._font.render(label, True, color)
        self._screen.blit(surf, (420, 50))

        # Help section
        y += 5
        pygame.draw.line(self._screen, (60, 60, 80), (15, y), (505, y))
        y += 10
        help_lines = [
            "W/S: Forward/Back    Q/E: Strafe",
            "A/D: Yaw Left/Right  R/F: Alt Up/Down",
            "Shift: Boost (2x)    Ctrl: Slow (0.5x)",
            "Space: E-Stop        ESC: Quit",
            "",
            "Use web UI or agent for: takeoff / land / arm",
        ]
        for line in help_lines:
            surf = self._font.render(line, True, (120, 120, 140))
            self._screen.blit(surf, (15, y))
            y += 22

        pygame.display.flip()


drone_keyboard_teleop = DroneKeyboardTeleop.blueprint

__all__ = ["DroneKeyboardTeleop", "drone_keyboard_teleop"]
