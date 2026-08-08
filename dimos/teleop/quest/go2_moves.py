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

"""Quest teleop for the Go2: thumbstick driving + controller buttons.

`Go2TeleopModule` gives thumbstick driving and the robot camera in the headset.
This adds the two things a one-handed operator needs:

  * HOLD-TO-TURN buttons — hold X to rotate right, Y to rotate left. Turning is a
    velocity blended into cmd_vel while held, not a canned animation, so it stops
    the instant you release.
  * ONE-SHOT sport moves — a button press fires a Unitree move (wave, stand up, …).
    Edge-triggered (fires on press, not every frame while held) and rate limited,
    since moves take seconds to play out and queuing them makes the robot lurch.

Default layout — everything on the LEFT controller, so it stays usable when the
right controller is dead/unpaired (common on second-hand headsets):

    Left stick            drive (forward/back + strafe)
    X            (hold)   rotate RIGHT
    Y            (hold)   rotate LEFT
    Grip (palm)  (press)  Hello — wave
    Trigger      (press)  RecoveryStand — STAND BACK UP (after StandDown or a fall)
    Stick press  (press)  Dance1

The right stick still yaws when a right controller is present. Everything is
configurable: `button_moves`, `turn_buttons`, `button_turn_speed`, speeds, deadzone.

Run:
    dimos run teleop-quest-go2-moves
then open https://<this-pc-ip>:8443/teleop in the Quest browser, accept the
self-signed certificate, and tap Connect.
"""

from __future__ import annotations

import time
from typing import Any

from unitree_webrtc_connect.constants import RTC_TOPIC, SPORT_CMD

from dimos.core.core import rpc
from dimos.msgs.geometry_msgs.Twist import Twist
from dimos.msgs.geometry_msgs.Vector3 import Vector3
from dimos.robot.unitree.go2.connection_spec import GO2ConnectionSpec
from dimos.teleop.quest.quest_extensions import Go2TeleopConfig, Go2TeleopModule
from dimos.teleop.quest.quest_teleop_module import Hand, QuestTeleopModule
from dimos.utils.logging_config import setup_logger

logger = setup_logger()

# button id -> one-shot Unitree sport move.
_DEFAULT_BUTTON_MOVES: dict[str, str] = {
    "left_grip": "Hello",  # palm button: wave
    "left_trigger": "RecoveryStand",  # stand back up
    "left_thumbstick_press": "Dance1",
    # Right controller (simply never fires when it isn't connected).
    "right_primary": "Dance1",
    "right_secondary": "Hello",
}

# button id -> turn direction, applied as a velocity WHILE HELD.
_DEFAULT_TURN_BUTTONS: dict[str, str] = {
    "left_primary": "right",  # X
    "left_secondary": "left",  # Y
}

# button id -> (hand, QuestControllerState attribute)
_BUTTON_ATTRS: dict[str, tuple[str, str]] = {
    "left_primary": ("left", "primary"),  # X
    "left_secondary": ("left", "secondary"),  # Y
    "left_touchpad": ("left", "touchpad"),
    "left_thumbstick_press": ("left", "thumbstick_press"),
    "right_primary": ("right", "primary"),  # A
    "right_secondary": ("right", "secondary"),  # B
    "right_touchpad": ("right", "touchpad"),
    "right_thumbstick_press": ("right", "thumbstick_press"),
    # Analog axes, thresholded by analog_press_threshold.
    "left_trigger": ("left", "trigger"),
    "left_grip": ("left", "grip"),
    "right_trigger": ("right", "trigger"),
    "right_grip": ("right", "grip"),
}


class Go2MovesTeleopConfig(Go2TeleopConfig):
    """Configuration for Go2MovesTeleopModule."""

    button_moves: dict[str, str] = dict(_DEFAULT_BUTTON_MOVES)
    turn_buttons: dict[str, str] = dict(_DEFAULT_TURN_BUTTONS)
    # Yaw rate applied while a turn button is held (rad/s).
    button_turn_speed: float = 0.8
    # Ignore further one-shot moves this long after firing one.
    move_cooldown_s: float = 2.5
    # Analog trigger/grip count as pressed above this value.
    analog_press_threshold: float = 0.7
    # Left stick X steers instead of strafing (handy with only one controller).
    left_stick_steers: bool = False


class Go2MovesTeleopModule(Go2TeleopModule):
    """Go2 Quest teleop: stick driving + hold-to-turn buttons + one-shot sport moves."""

    config: Go2MovesTeleopConfig
    _connection: GO2ConnectionSpec

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._button_prev: dict[str, bool] = {}
        self._last_move_at: float = 0.0

    @rpc
    def start(self) -> None:
        super().start()
        moves = ", ".join(f"{b}->{m}" for b, m in self.config.button_moves.items())
        turns = ", ".join(f"{b}->turn {d}" for b, d in self.config.turn_buttons.items())
        logger.info(f"Go2 Quest teleop: moves ({moves}); turn buttons ({turns})")

    # ---- helpers ------------------------------------------------------------
    def _pressed(self, states: dict[str, Any], button_id: str) -> bool:
        attr = _BUTTON_ATTRS.get(button_id)
        if attr is None:
            return False
        hand_key, field = attr
        state = states.get(hand_key)
        raw = getattr(state, field, False) if state is not None else False
        if isinstance(raw, bool):
            return raw
        if isinstance(raw, (int, float)):
            return raw > self.config.analog_press_threshold
        return bool(raw)

    def _fire_move(self, command: str) -> None:
        api_id = SPORT_CMD.get(command)
        if api_id is None:
            logger.warning(f"Quest teleop: unknown sport command {command!r} — ignoring")
            return
        try:
            self._connection.publish_request(RTC_TOPIC["SPORT_MOD"], {"api_id": api_id})
            logger.info(f"Quest teleop: fired sport move {command}")
        except Exception as e:  # noqa: BLE001
            logger.warning(f"Quest teleop: sport move {command} failed: {e}")

    # ---- driving ------------------------------------------------------------
    def _publish_drive(self, states: dict[str, Any]) -> None:
        """Blend stick input and held turn buttons into a single cmd_vel Twist."""
        left = states.get("left")
        right = states.get("right")
        twist = Twist()
        twist.linear = Vector3(0.0, 0.0, 0.0)
        twist.angular = Vector3(0.0, 0.0, 0.0)

        if left is not None:
            twist.linear.x = -self._deadzone(left.thumbstick.y) * self.config.linear_speed
            if self.config.left_stick_steers:
                twist.angular.z = -self._deadzone(left.thumbstick.x) * self.config.angular_speed
            else:
                twist.linear.y = -self._deadzone(left.thumbstick.x) * self.config.linear_speed
        if right is not None and not self.config.left_stick_steers:
            twist.angular.z = -self._deadzone(right.thumbstick.x) * self.config.angular_speed

        # Hold-to-turn buttons add yaw on top of any stick yaw. Negative = turn right,
        # matching the stick convention in Go2TeleopModule.
        turn = 0.0
        for button_id, direction in self.config.turn_buttons.items():
            if self._pressed(states, button_id):
                turn += (
                    -self.config.button_turn_speed
                    if str(direction).lower().startswith("r")
                    else self.config.button_turn_speed
                )
        if turn:
            limit = self.config.button_turn_speed
            twist.angular.z = max(-limit, min(limit, twist.angular.z + turn))

        self.cmd_vel.publish(twist)

    # ---- one-shot moves -----------------------------------------------------
    def _handle_move_buttons(self, states: dict[str, Any]) -> None:
        now = time.monotonic()
        for button_id, command in self.config.button_moves.items():
            pressed = self._pressed(states, button_id)
            was_pressed = self._button_prev.get(button_id, False)
            self._button_prev[button_id] = pressed
            if pressed and not was_pressed:  # rising edge only
                if now - self._last_move_at < self.config.move_cooldown_s:
                    logger.info(f"Quest teleop: {command} ignored (cooldown)")
                    continue
                self._last_move_at = now
                self._fire_move(command)

    def _on_joy_bytes(self, data: bytes) -> None:
        # Grandparent updates controller state. We deliberately skip
        # Go2TeleopModule's own publish so driving is computed once here, with the
        # hold-to-turn buttons folded into the same Twist.
        QuestTeleopModule._on_joy_bytes(self, data)
        with self._lock:  # RLock — safe to re-acquire
            states = {
                "left": self._controllers.get(Hand.LEFT),
                "right": self._controllers.get(Hand.RIGHT),
            }
        self._publish_drive(states)
        self._handle_move_buttons(states)


# --- blueprint ----------------------------------------------------------------
# Mirrors the shipped `teleop_quest_go2` wiring (cmd_vel over LCM, camera over shared
# memory). cmd_vel reaches the legs via the sport `Move` command, and GO2Connection's
# LiDAR collision guard still vetoes driving straight into obstacles.
from dimos.constants import DEFAULT_CAPACITY_COLOR_IMAGE  # noqa: E402
from dimos.core.coordination.blueprints import autoconnect  # noqa: E402
from dimos.core.transport import LCMTransport, pSHMTransport  # noqa: E402
from dimos.msgs.sensor_msgs.Image import Image  # noqa: E402
from dimos.robot.unitree.go2.connection import GO2Connection  # noqa: E402

teleop_quest_go2_moves = (
    autoconnect(
        Go2MovesTeleopModule.blueprint(),
        GO2Connection.blueprint(),
    )
    .transports(
        {
            ("cmd_vel", Twist): LCMTransport("/cmd_vel", Twist),
            ("color_image", Image): pSHMTransport(
                "color_image", default_capacity=DEFAULT_CAPACITY_COLOR_IMAGE
            ),
        }
    )
    .global_config(robot_model="unitree_go2")
)

__all__ = ["Go2MovesTeleopConfig", "Go2MovesTeleopModule", "teleop_quest_go2_moves"]
