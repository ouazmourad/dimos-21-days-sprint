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

from __future__ import annotations

import datetime
import difflib
import math
import time

from unitree_webrtc_connect.constants import RTC_TOPIC, SPORT_CMD

from dimos.agents.annotation import skill
from dimos.core.core import rpc
from dimos.core.module import Module
from dimos.msgs.geometry_msgs.PoseStamped import PoseStamped
from dimos.msgs.geometry_msgs.Quaternion import Quaternion
from dimos.msgs.geometry_msgs.Vector3 import Vector3
from dimos.navigation.base import NavigationState
from dimos.navigation.navigation_spec import NavigationInterfaceSpec
from dimos.robot.unitree.go2.connection_spec import GO2ConnectionSpec
from dimos.utils.logging_config import setup_logger

logger = setup_logger()


UNITREE_WEBRTC_CONTROLS: list[tuple[str, int, str]] = [
    # ("Damp", 1001, "Lowers the robot to the ground fully."),
    (
        "BalanceStand",
        1002,
        "Activates a mode that maintains the robot in a balanced standing position.",
    ),
    (
        "StandUp",
        1004,
        "Commands the robot to transition from a sitting or prone position to a standing posture.",
    ),
    (
        "StandDown",
        1005,
        "Instructs the robot to move from a standing position to a sitting or prone posture.",
    ),
    (
        "RecoveryStand",
        1006,
        "Recovers the robot to a state from which it can take more commands. Useful to run after multiple dynamic commands like front flips, Must run after skills like sit and jump and standup.",
    ),
    ("Sit", 1009, "Commands the robot to sit down from a standing or moving stance."),
    (
        "RiseSit",
        1010,
        "Commands the robot to rise back to a standing position from a sitting posture.",
    ),
    (
        "SwitchGait",
        1011,
        "Switches the robot's walking pattern or style dynamically, suitable for different terrains or speeds.",
    ),
    ("Trigger", 1012, "Triggers a specific action or custom routine programmed into the robot."),
    (
        "BodyHeight",
        1013,
        "Adjusts the height of the robot's body from the ground, useful for navigating various obstacles.",
    ),
    (
        "FootRaiseHeight",
        1014,
        "Controls how high the robot lifts its feet during movement, which can be adjusted for different surfaces.",
    ),
    (
        "SpeedLevel",
        1015,
        "Sets or adjusts the speed at which the robot moves, with various levels available for different operational needs.",
    ),
    (
        "Hello",
        1016,
        "Performs a greeting action, which could involve a wave or other friendly gesture.",
    ),
    ("Stretch", 1017, "Engages the robot in a stretching routine."),
    (
        "TrajectoryFollow",
        1018,
        "Directs the robot to follow a predefined trajectory, which could involve complex paths or maneuvers.",
    ),
    (
        "ContinuousGait",
        1019,
        "Enables a mode for continuous walking or running, ideal for long-distance travel.",
    ),
    ("Content", 1020, "To display or trigger when the robot is happy."),
    ("Wallow", 1021, "The robot falls onto its back and rolls around."),
    (
        "Dance1",
        1022,
        "Performs a predefined dance routine 1, programmed for entertainment or demonstration.",
    ),
    ("Dance2", 1023, "Performs another variant of a predefined dance routine 2."),
    ("GetBodyHeight", 1024, "Retrieves the current height of the robot's body from the ground."),
    (
        "GetFootRaiseHeight",
        1025,
        "Retrieves the current height at which the robot's feet are being raised during movement.",
    ),
    (
        "GetSpeedLevel",
        1026,
        "Retrieves the current speed level setting of the robot.",
    ),
    (
        "SwitchJoystick",
        1027,
        "Switches the robot's control mode to respond to joystick input for manual operation.",
    ),
    (
        "Pose",
        1028,
        "Commands the robot to assume a specific pose or posture as predefined in its programming.",
    ),
    ("Scrape", 1029, "The robot performs a scraping motion."),
    (
        "FrontFlip",
        1030,
        "Commands the robot to perform a front flip, showcasing its agility and dynamic movement capabilities.",
    ),
    (
        "FrontJump",
        1031,
        "Instructs the robot to jump forward, demonstrating its explosive movement capabilities.",
    ),
    (
        "FrontPounce",
        1032,
        "Commands the robot to perform a pouncing motion forward.",
    ),
    (
        "WiggleHips",
        1033,
        "The robot performs a hip wiggling motion, often used for entertainment or demonstration purposes.",
    ),
    (
        "GetState",
        1034,
        "Retrieves the current operational state of the robot, including its mode, position, and status.",
    ),
    (
        "EconomicGait",
        1035,
        "Engages a more energy-efficient walking or running mode to conserve battery life.",
    ),
    ("FingerHeart", 1036, "Performs a finger heart gesture while on its hind legs."),
    (
        "Handstand",
        1301,
        "Commands the robot to perform a handstand, demonstrating balance and control.",
    ),
    (
        "CrossStep",
        1302,
        "Commands the robot to perform cross-step movements.",
    ),
    (
        "OnesidedStep",
        1303,
        "Commands the robot to perform one-sided step movements.",
    ),
    ("Bound", 1304, "Commands the robot to perform bounding movements."),
    ("MoonWalk", 1305, "Commands the robot to perform a moonwalk motion."),
    ("LeftFlip", 1042, "Executes a flip towards the left side."),
    ("RightFlip", 1043, "Performs a flip towards the right side."),
    ("Backflip", 1044, "Executes a backflip, a complex and dynamic maneuver."),
]


_UNITREE_COMMANDS = {
    name: (id_, description)
    for name, id_, description in UNITREE_WEBRTC_CONTROLS
    if name not in ["Reverse", "Spin"]
}


class UnitreeSkillContainer(Module):
    """Container for Unitree Go2 robot skills using the new framework."""

    # Optional: only `relative_move` uses it (nav-map path planning, which the Go2
    # Air can't do anyway). Lightweight blueprints omit the navigation stack; the
    # direct `move` skill works without it. Full blueprints still provide it.
    _navigation: NavigationInterfaceSpec | None = None
    _connection: GO2ConnectionSpec

    @rpc
    def start(self) -> None:
        super().start()
        # Initialize TF early so it can start receiving transforms.
        _ = self.tf

    @rpc
    def stop(self) -> None:
        super().stop()

    @skill
    def relative_move(self, forward: float = 0.0, left: float = 0.0, degrees: float = 0.0) -> str:
        """Move the robot relative to its current position.

        The `degrees` arguments refers to the rotation the robot should be at the end, relative to its current rotation.

        Example calls:

            # Move to a point that's 2 meters forward and 1 to the right.
            relative_move(forward=2, left=-1, degrees=0)

            # Move back 1 meter, while still facing the same direction.
            relative_move(forward=-1, left=0, degrees=0)

            # Rotate 90 degrees to the right (in place)
            relative_move(forward=0, left=0, degrees=-90)

            # Move 3 meters left, and face that direction
            relative_move(forward=0, left=3, degrees=90)
        """
        if self._navigation is None:
            return (
                "relative_move needs the navigation stack, which isn't loaded in this "
                "blueprint. Use move(forward=..., left=..., turn_degrees=...) for direct "
                "movement instead."
            )

        forward, left, degrees = float(forward), float(left), float(degrees)

        tf = self.tf.get("world", "base_link")
        if tf is None:
            return "Failed to get the position of the robot."

        # TODO: Improve this. This is not a nice way to do it. I should
        # subscribe to arrival/cancellation events instead.

        self._navigation.set_goal(self._generate_new_goal(tf.to_pose(), forward, left, degrees))

        time.sleep(1.0)

        start_time = time.monotonic()
        timeout = 100.0
        while self._navigation.get_state() == NavigationState.FOLLOWING_PATH:
            if time.monotonic() - start_time > timeout:
                return "Navigation timed out"
            time.sleep(0.1)

        time.sleep(1.0)

        if not self._navigation.is_goal_reached():
            return "Navigation was cancelled or failed"
        else:
            return "Navigation goal reached"

    def _generate_new_goal(
        self, current_pose: PoseStamped, forward: float, left: float, degrees: float
    ) -> PoseStamped:
        local_offset = Vector3(forward, left, 0)
        global_offset = current_pose.orientation.rotate_vector(local_offset)
        goal_position = current_pose.position + global_offset

        current_euler = current_pose.orientation.to_euler()
        goal_yaw = current_euler.yaw + math.radians(degrees)
        goal_euler = Vector3(current_euler.roll, current_euler.pitch, goal_yaw)
        goal_orientation = Quaternion.from_euler(goal_euler)

        return PoseStamped(position=goal_position, orientation=goal_orientation)

    @skill
    def move(
        self,
        forward: float = 0.0,
        left: float = 0.0,
        turn_degrees: float = 0.0,
        seconds: float = 1.5,
    ) -> str:
        """Move the robot directly — no map or path planning needed. Use this (NOT
        relative_move) for simple movement, especially on a Go2 Air where navigation fails.

        Both turns and straight forward/back moves are CLOSED-LOOP on odometry: a turn
        watches the heading and stops at the exact angle (turn_degrees=90 -> ~90°), and a
        forward/back move watches position and stops after ~(forward * seconds) meters.

        Args:
            forward: forward speed in m/s (negative = backward). ~0.3-0.6 is a gentle walk.
            left: sideways speed in m/s (negative = right).
            turn_degrees: degrees to turn in place (positive = left / counter-clockwise).
                For a pure turn leave forward/left at 0; `seconds` is ignored (it turns
                until the measured angle is reached).
            seconds: how long a straight/strafing move lasts (the robot auto-stops after).

        Examples:
            move(forward=0.4, seconds=3)   # walk forward ~3 seconds
            move(forward=-0.3, seconds=2)  # back up
            move(turn_degrees=90)          # turn exactly ~90 degrees left in place
            move(turn_degrees=-360)        # spin a full circle to the right
        """
        forward, left, turn_degrees = float(forward), float(left), float(turn_degrees)
        seconds = min(max(0.1, float(seconds)), 20.0)
        MAX_LIN, MAX_ANG = 0.6, 1.5  # m/s, rad/s
        SEND_HZ = 10.0  # steady Move-command rate (publish_request is fire-and-forget)
        dt = 1.0 / SEND_HZ
        SPORT = RTC_TOPIC["SPORT_MOD"]
        stop = {"api_id": SPORT_CMD["StopMove"]}

        def _send_move(x: float, y: float, z: float) -> None:
            # Sport Move velocity command (api_id 1008): x=fwd m/s, y=left m/s, z=yaw rad/s.
            # publish_request is now FIRE-AND-FORGET (it no longer blocks on the robot's
            # ack), so re-sending it at SEND_HZ keeps the velocity stream steady — a
            # blocking send per loop iteration stalled the stream and the dog moved in
            # start-stop bursts (a 360° spin crawling ~60° every few seconds; forward
            # under-travelling). NOT the rt/wirelesscontroller joystick (its flood fails
            # to move the Air and kills the data channel).
            self._connection.publish_request(
                SPORT, {"api_id": SPORT_CMD["Move"], "parameter": {"x": x, "y": y, "z": z}}
            )

        def _yaw() -> float | None:
            tf = self.tf.get("world", "base_link")
            return tf.to_pose().orientation.to_euler().yaw if tf is not None else None

        def _xy() -> tuple[float, float] | None:
            tf = self.tf.get("world", "base_link")
            if tf is None:
                return None
            p = tf.to_pose().position
            return (p.x, p.y)

        try:
            # A turn is CLOSED-LOOP: spin while watching the odometry heading and stop when
            # the MEASURED angle hits the target. Open-loop velocity*time is hopelessly
            # inaccurate on the Go2. Verified closed-loop: 90°->~91°.
            if turn_degrees != 0.0 and forward == 0.0 and left == 0.0:
                target = math.radians(turn_degrees)
                speed = min(MAX_ANG, 1.2)  # rad/s spin rate
                lead = math.radians(10.0) * speed  # stop early; scales with spin speed
                z = math.copysign(speed, turn_degrees)
                prev = _yaw()
                if prev is None:
                    return "Couldn't read the robot's heading (no odometry)."
                acc = 0.0
                # Generous cap: time to spin even at half the commanded rate, plus slack —
                # so a full 360° completes instead of timing out partway.
                deadline = time.monotonic() + abs(target) / (speed * 0.5) + 10.0
                while abs(acc) < max(0.0, abs(target) - lead) and time.monotonic() < deadline:
                    _send_move(0.0, 0.0, z)
                    time.sleep(dt)
                    cur = _yaw()
                    if cur is not None:
                        d = cur - prev
                        while d > math.pi:
                            d -= 2 * math.pi
                        while d < -math.pi:
                            d += 2 * math.pi
                        acc += d
                        prev = cur
                self._connection.publish_request(SPORT, stop)
                return f"Turned {math.degrees(acc):.0f}° (requested {turn_degrees:.0f}°)."

            forward = max(-MAX_LIN, min(MAX_LIN, forward))
            left = max(-MAX_LIN, min(MAX_LIN, left))

            # Pure straight move (no strafe, no turn): CLOSED-LOOP on odometry DISTANCE so
            # "~1 m" really travels ~1 m, independent of how well the Air realises the
            # commanded velocity. Target distance = |forward| * seconds (the intent).
            if forward != 0.0 and left == 0.0 and turn_degrees == 0.0:
                target_d = abs(forward) * seconds
                lead_d = 0.10  # stop ~10 cm early to cancel stop-latency creep
                start = _xy()
                if start is None:
                    return "Couldn't read the robot's position (no odometry)."
                x0, y0 = start
                travelled = 0.0
                deadline = time.monotonic() + target_d / max(0.1, abs(forward) * 0.5) + 8.0
                while travelled < max(0.0, target_d - lead_d) and time.monotonic() < deadline:
                    _send_move(forward, 0.0, 0.0)
                    time.sleep(dt)
                    cur = _xy()
                    if cur is not None:
                        travelled = math.hypot(cur[0] - x0, cur[1] - y0)
                self._connection.publish_request(SPORT, stop)
                return f"Moved {travelled:.2f} m forward (target {target_d:.2f} m)."

            # Strafing / combined move: open-loop velocity for the requested duration.
            wz = max(-MAX_ANG, min(MAX_ANG, math.radians(turn_degrees) / seconds))
            end = time.monotonic() + seconds
            while time.monotonic() < end:
                _send_move(forward, left, wz)
                time.sleep(dt)
            self._connection.publish_request(SPORT, stop)
        except Exception as e:  # noqa: BLE001
            logger.error(f"move failed: {e}")
            return "Failed to send the movement command (the connection may be down)."
        return f"Moved forward={forward} m/s, left={left} m/s over {seconds:.1f}s."

    @skill
    def wait(self, seconds: float) -> str:
        """Wait for a specified amount of time.

        Args:
            seconds: Seconds to wait
        """
        time.sleep(seconds)
        return f"Wait completed with length={seconds}s"

    @skill
    def current_time(self) -> str:
        """Provides current time."""
        return str(datetime.datetime.now())

    @skill
    def execute_sport_command(self, command_name: str) -> str:
        if command_name not in _UNITREE_COMMANDS:
            suggestions = difflib.get_close_matches(
                command_name, _UNITREE_COMMANDS.keys(), n=3, cutoff=0.6
            )
            return f"There's no '{command_name}' command. Did you mean: {suggestions}"

        id_, _ = _UNITREE_COMMANDS[command_name]

        try:
            self._connection.publish_request(RTC_TOPIC["SPORT_MOD"], {"api_id": id_})
            return f"'{command_name}' command executed successfully."
        except Exception as e:
            logger.error(f"Failed to execute {command_name}: {e}")
            return "Failed to execute the command."


_commands = "\n".join(
    [f'- "{name}": {description}' for name, (_, description) in _UNITREE_COMMANDS.items()]
)

UnitreeSkillContainer.execute_sport_command.__doc__ = f"""Execute a Unitree sport command.

Example usage:

    execute_sport_command("FrontPounce")

Here are all the command names and what they do.

{_commands}
"""
