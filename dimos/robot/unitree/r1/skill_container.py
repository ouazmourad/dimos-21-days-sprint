# Copyright 2026 Dimensional Inc.
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

"""
Unitree R1 skill container for the new agents framework.
Dynamically generates skills for R1 humanoid robot including arm controls and movement modes.
"""

import difflib

from dimos.agents.annotation import skill
from dimos.core.core import rpc
from dimos.core.module import Module
from dimos.msgs.geometry_msgs.Twist import Twist
from dimos.msgs.geometry_msgs.Vector3 import Vector3
from dimos.robot.unitree.r1.connection_spec import R1ConnectionSpec
from dimos.robot.unitree.r1.effectors.high_level.speak_proxy import R1SpeakProxy
from dimos.utils.logging_config import setup_logger

logger = setup_logger()

# Arm actions: api_id 7106 ("execute action") on the "arm" service, driven by
# PC1's r1_arm_client (G1ArmActionClient). Ids/names below were read live from the
# R1 via GetActionList (api 7107) on 2026-06-17 — authoritative for this robot.
# Note: the arm server may require the robot to be in a locomotion mode for the
# arms to actually move; execute_arm_command surfaces a non-zero return code.
R1_ARM_CONTROLS = [
    ("ReleaseArm", 99, "Relax both arms back to rest (use to end/release any held gesture)."),
    ("BlowKissBothHands", 11, "Blow a kiss with both hands."),
    ("BlowKissLeftHand", 12, "Blow a kiss with the left hand."),
    ("BlowKissRightHand", 13, "Blow a kiss with the right hand."),
    ("BothHandsUp", 15, "Raise both hands up in the air."),
    ("Clamp", 17, "Bring both hands together (clap/clamp)."),
    ("HighFive", 18, "Give a high five with the right hand."),
    ("Hug", 19, "Open both arms for a hug."),
    ("Refuse", 22, "Make a refusal / 'no' gesture."),
    ("RightHandUp", 23, "Raise the right hand up."),
    ("UltramanRay", 24, "Strike the Ultraman ray pose."),
    ("WaveUnderHead", 25, "Wave with the hand around chest level."),
    ("WaveAboveHead", 26, "Wave with the hand raised above the head."),
    ("ShakeHand", 27, "Extend the right hand forward for a handshake."),
    ("BoxLeftHandWin", 28, "Boxing victory pose, left hand."),
    ("BoxRightHandWin", 29, "Boxing victory pose, right hand."),
    ("BoxBothHandWin", 30, "Boxing victory pose, both hands."),
    ("ExtendRightArmForward", 31, "Extend the right arm straight forward."),
    ("RightHandOnHeart", 33, "Place the right hand over the heart."),
    ("BothHandsUpDeviateRight", 34, "Raise both hands up, leaning to the right."),
    ("Emphasize", 35, "Make an emphasizing hand gesture."),
    ("ForwardPush", 36, "Push both hands forward."),
]

# R1 Movement Modes - api_id 7101 on topic "rt/api/sport/request". NOTE: these are
# still the G1 placeholder ids; the R1's real loco FSM ids differ (StandUp=4,
# Start=811), so 500/501/801 may not apply. Unverified — confirm before relying.
R1_MODE_CONTROLS = [
    ("WalkMode", 500, "Switch to normal walking mode."),
    ("WalkControlWaist", 501, "Switch to walking mode with waist control."),
    ("RunMode", 801, "Switch to running mode."),
]

_ARM_COMMANDS: dict[str, tuple[int, str]] = {
    name: (id_, description) for name, id_, description in R1_ARM_CONTROLS
}

_MODE_COMMANDS: dict[str, tuple[int, str]] = {
    name: (id_, description) for name, id_, description in R1_MODE_CONTROLS
}


class UnitreeR1SkillContainer(Module):
    _connection: R1ConnectionSpec

    @rpc
    def start(self) -> None:
        super().start()

    @rpc
    def stop(self) -> None:
        super().stop()

    @skill
    def move(self, x: float, y: float = 0.0, yaw: float = 0.0, duration: float = 0.0) -> str:
        """Move the robot using direct velocity commands. Determine duration required based on user distance instructions.

        Example call:
            args = { "x": 0.5, "y": 0.0, "yaw": 0.0, "duration": 2.0 }
            move(**args)

        Args:
            x: Forward velocity (m/s)
            y: Left/right velocity (m/s)
            yaw: Rotational velocity (rad/s)
            duration: How long to move (seconds)
        """

        twist = Twist(linear=Vector3(x, y, 0), angular=Vector3(0, 0, yaw))
        self._connection.move(twist, duration=duration)
        return f"Started moving with velocity=({x}, {y}, {yaw}) for {duration} seconds"

    @skill
    def execute_arm_command(self, command_name: str) -> str:
        # Arm ids verified live via r1_arm_client --list (api 7106, "arm" service).
        return self._execute_r1_command(_ARM_COMMANDS, 7106, "rt/api/arm/request", command_name)

    @skill
    def execute_mode_command(self, command_name: str) -> str:
        # TODO(R1): confirm R1 mode command ids/topic/api_id — currently mirrors G1.
        return self._execute_r1_command(_MODE_COMMANDS, 7101, "rt/api/sport/request", command_name)

    def _execute_r1_command(
        self,
        command_dict: dict[str, tuple[int, str]],
        api_id: int,
        topic: str,
        command_name: str,
    ) -> str:
        if command_name not in command_dict:
            suggestions = difflib.get_close_matches(
                command_name, command_dict.keys(), n=3, cutoff=0.6
            )
            return f"There's no '{command_name}' command. Did you mean: {suggestions}"

        id_, _ = command_dict[command_name]

        try:
            result = self._connection.publish_request(
                topic, {"api_id": api_id, "parameter": {"data": id_}}
            )
            code = result.get("code", 0) if isinstance(result, dict) else 0
            if code != 0:
                return (
                    f"'{command_name}' was sent but the robot returned error code {code}. "
                    f"Arm actions may require the robot to be in a walk/locomotion mode first."
                )
            return f"'{command_name}' command executed successfully."
        except Exception as e:
            logger.error(f"Failed to execute {command_name}: {e}")
            return "Failed to execute the command."


_arm_commands = "\n".join(
    [f'- "{name}": {description}' for name, (_, description) in _ARM_COMMANDS.items()]
)

UnitreeR1SkillContainer.execute_arm_command.__doc__ = f"""Execute a Unitree R1 arm command.

Example usage:

    execute_arm_command("ArmHeart")

Here are all the command names and what they do.

{_arm_commands}
"""

_mode_commands = "\n".join(
    [f'- "{name}": {description}' for name, (_, description) in _MODE_COMMANDS.items()]
)

UnitreeR1SkillContainer.execute_mode_command.__doc__ = f"""Execute a Unitree R1 mode command.

Example usage:

    execute_mode_command("RunMode")

Here are all the command names and what they do.

{_mode_commands}
"""


class R1SpeakSkill(Module):
    """Agent skill: speak through the R1's own onboard speaker (robot-side TTS).

    Replaces the generic ``SpeakSkill`` (which renders OpenAI TTS on the *laptop*
    speakers) for the R1: here the robot itself synthesizes and plays the speech
    through its built-in speaker via PC1's ``r1_audio_client`` (see
    :class:`R1SpeakProxy`). Same ``speak`` tool name/contract the agent already
    expects, so no prompt changes are needed.
    """

    _proxy: R1SpeakProxy | None = None

    @rpc
    def start(self) -> None:
        super().start()
        # Robot speech is auxiliary: if PC1 is unreachable, log and keep the rest
        # of the agent running rather than crashing the whole suite at startup.
        try:
            self._proxy = R1SpeakProxy()
            self._proxy.start()
        except Exception as e:
            logger.warning(f"R1 robot speech unavailable (proxy failed to start): {e}")
            self._proxy = None

    @rpc
    def stop(self) -> None:
        if self._proxy is not None:
            self._proxy.stop()
            self._proxy = None
        super().stop()

    @skill
    def speak(self, text: str) -> str:
        """Speak text out loud through the robot's own speaker.

        USE THIS TOOL AS OFTEN AS NEEDED. People can't see your text, but they can
        hear what you speak. Be concise — speaking takes time, so get to the point.

        Example usage:

            speak("Hello, I am the R1 humanoid.")
        """
        if self._proxy is None:
            return "Error: robot speech not initialized"
        try:
            self._proxy.speak(text)
            return f"Spoke: {text}"
        except Exception as e:
            logger.error(f"R1 speak failed: {e}")
            return f"Error speaking: {e}"
