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

"""R1 loco control proxied through PC1's native unitree_sdk2 client.

The R1 main board's DDS types are newer than the pip ``unitree-sdk2py-dimos``,
so the laptop cannot match endpoints (CycloneDDS XTypes type-consistency) and
the native ``dds`` backend never reaches the robot. PC1 — the robot's onboard
Jetson (192.168.123.164) — ships the firmware-matching ``unitree_sdk2`` with a
prebuilt ``r1_loco_client``. This proxy opens one SSH session to PC1 and invokes
that client per command, so DimOS on the laptop drives the real R1.

``paramiko`` is imported lazily inside :meth:`start` so this module imports
without it (mirrors the lazy SDK imports in :mod:`dds_sdk`).

Latency note: each command spawns a fresh ``r1_loco_client`` on PC1 (DDS init +
send + exit, ~1-2 s), which is fine for discrete commands (stand/damp/timed
move) but not for high-rate continuous teleop.
"""

import os
import time
from typing import Any

from dimos.msgs.geometry_msgs.Twist import Twist
from dimos.utils.logging_config import setup_logger

logger = setup_logger()

# Loco client on PC1 and the api ids it accepts via publish_request (matches the
# R1's r1_loco_api.hpp: SET_FSM_ID=7101, SET_VELOCITY=7105).
_CLIENT = "~/unitree_sdk2/build/bin/r1_loco_client"
_API_SET_FSM_ID = 7101
_API_SET_VELOCITY = 7105


class R1LocoProxy:
    """Drives the R1 by invoking PC1's native ``r1_loco_client`` over SSH."""

    def __init__(
        self,
        host: str = "192.168.123.164",
        user: str = "unitree",
        password: str | None = None,
        interface: str = "eth10",
    ) -> None:
        self.host = host
        self.user = user
        # Default to the env var, then the Unitree factory default. Prefer
        # setting R1_PC1_PASS rather than relying on the default.
        self.password = password or os.environ.get("R1_PC1_PASS", "123")
        self.interface = interface
        self._client: Any = None  # paramiko.SSHClient

    def start(self) -> None:
        try:
            import paramiko
        except ImportError as e:
            raise ImportError("R1 'pc1' backend needs paramiko: pip install paramiko") from e

        self._client = paramiko.SSHClient()
        self._client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        # The R1 ethernet link can flap (coupler joint), so retry briefly: a
        # transient blip at startup shouldn't crash the whole run.
        last_err: Exception | None = None
        for attempt in range(5):
            try:
                self._client.connect(
                    self.host, username=self.user, password=self.password, timeout=10
                )
                break
            except Exception as e:
                last_err = e
                logger.warning(f"PC1 SSH connect {attempt + 1}/5 failed ({e}); retrying in 2s ...")
                time.sleep(2.0)
        else:
            raise RuntimeError(
                f"Could not SSH to PC1 {self.user}@{self.host}:22 after 5 tries. Is the ethernet "
                f"link up (`cat /sys/class/net/enp2s0/carrier` should be 1) and PC1 booted? "
                f"Last error: {last_err}"
            )
        logger.info(
            f"R1 loco proxy connected: {self.user}@{self.host} -> r1_loco_client (iface {self.interface})"
        )
        logger.info(f"R1 current FSM state: {self.get_state()}")

    def _run(self, flag: str, timeout: float = 30.0) -> str:
        assert self._client is not None, "R1LocoProxy not started"
        cmd = f"{_CLIENT} --network_interface={self.interface} {flag}"
        logger.debug(f"PC1$ {cmd}")
        _stdin, stdout, stderr = self._client.exec_command(cmd, timeout=timeout)
        out = stdout.read().decode(errors="replace")
        err = stderr.read().decode(errors="replace")
        if err.strip():
            logger.warning(f"r1_loco_client stderr: {err.strip()}")
        return out

    def move(self, twist: Twist, duration: float = 0.0) -> bool:
        vx, vy, vyaw = twist.linear.x, twist.linear.y, twist.angular.z
        dur = duration if duration and duration > 0 else 1.0
        self._run(f'--set_velocity="{vx} {vy} {vyaw} {dur}"')
        return True

    def stand_up(self) -> bool:
        self._run("--stand_up")
        return True

    def start_locomotion(self) -> bool:
        # R1 Start (FSM id 811): enter active balance/locomotion mode. Velocity
        # commands are ignored in StandUp (4) — the robot must be in Start to
        # walk or turn.
        self._run("--start")
        return True

    def lie_down(self) -> bool:
        # R1 has no squat/sit; Damp (FSM id 1) is the safe relaxed state.
        self._run("--damp")
        return True

    def damp(self) -> bool:
        self._run("--damp")
        return True

    def get_state(self) -> str:
        out = self._run("--get_fsm_id", timeout=15)
        for line in out.splitlines():
            if "fsm_id:" in line:
                return line.split("fsm_id:")[-1].strip()
        return "unknown"

    def publish_request(self, topic: str, data: dict[str, Any]) -> dict[Any, Any]:
        api_id = data.get("api_id")
        param = data.get("parameter", {}) or {}
        if api_id == _API_SET_FSM_ID:
            self._run(f"--set_fsm_id={int(param.get('data', 0))}")
            return {"code": 0}
        if api_id == _API_SET_VELOCITY:
            v = param.get("velocity", [0.0, 0.0, 0.0])
            d = param.get("duration", 1.0)
            self._run(f'--set_velocity="{v[0]} {v[1]} {v[2]} {d}"')
            return {"code": 0}
        logger.warning(f"R1LocoProxy: unsupported api_id {api_id}")
        return {"code": -1, "error": "unsupported_api"}

    def stop(self) -> None:
        try:
            if self._client is not None:
                self._run("--stop_move", timeout=15)
        except Exception as e:
            logger.error(f"R1LocoProxy stop error: {e}")
        finally:
            if self._client is not None:
                self._client.close()
                self._client = None


__all__ = ["R1LocoProxy"]
