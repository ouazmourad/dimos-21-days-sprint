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

"""R1 onboard-speaker TTS, proxied through PC1's native unitree_sdk2.

The R1 synthesizes and plays speech on its own built-in speaker via the DDS
``voice`` service (api ``ROBOT_API_ID_AUDIO_TTS`` = 1001). The laptop cannot
reach that service directly (the same CycloneDDS XTypes type mismatch that
blocks the native loco/camera paths), so this proxy SSHes to PC1 — the robot's
onboard Jetson — and invokes a small ``r1_audio_client`` that calls
``unitree::robot::g1::AudioClient::TtsMaker``. The g1 AudioClient targets the
service name ``voice`` shared across ``unitree_hg`` humanoids, so it drives the
R1's speaker; there is no R1-specific audio client in the SDK.

Mirrors :class:`R1LocoProxy` (one SSH session, one short-lived client process
per command) and :class:`R1PC1Camera` (auto-provisions + builds the helper on
PC1 from the embedded source if the binary is missing). ``paramiko`` is imported
lazily inside :meth:`start`.

Latency note: each ``speak`` spawns a fresh ``r1_audio_client`` (DDS init + send
+ exit, ~1-2 s). Playback itself is server-side on the robot and continues after
the client exits, so this is fire-and-forget — fine for the agent's speak skill.
"""

import os
import shlex
import time
from typing import Any

from dimos.utils.logging_config import setup_logger

logger = setup_logger()

_BIN = "~/unitree_sdk2/build/bin/r1_audio_client"
_SRC = "~/unitree_sdk2/example/r1/high_level/r1_audio_client.cpp"

# Source for the PC1 helper, kept here so a fresh robot can be re-provisioned
# (see _ensure_client). Must stay in sync with the binary built on PC1.
_CLIENT_SRC = r'''// r1_audio_client - minimal CLI to drive the R1's onboard voice service.
// Mirrors r1_loco_client_example.cpp's --key=value parsing; uses the g1
// AudioClient (DDS service "voice", shared across unitree_hg humanoids) so the
// robot does its own TTS through its built-in speaker.
//
//   r1_audio_client --network_interface=eth10 --set_volume=85 --tts="hello"
//   r1_audio_client --network_interface=eth10 --tts="hi" --speaker=1
//   r1_audio_client --network_interface=eth10 --get_volume
#include <iostream>
#include <map>
#include <string>

#include <unitree/common/time/time_tool.hpp>
#include <unitree/robot/g1/audio/g1_audio_client.hpp>

int main(int argc, char const *argv[]) {
  std::map<std::string, std::string> args = {{"network_interface", "lo"}};
  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg.substr(0, 2) == "--") {
      size_t pos = arg.find("=");
      std::string key, value;
      if (pos != std::string::npos) {
        key = arg.substr(2, pos - 2);
        value = arg.substr(pos + 1);
        if (!value.empty() && value.front() == '"' && value.back() == '"') {
          value = value.substr(1, value.length() - 2);
        }
      } else {
        key = arg.substr(2);
        value = "";
      }
      args[key] = value;
    }
  }

  unitree::robot::ChannelFactory::Instance()->Init(0, args["network_interface"]);
  unitree::robot::g1::AudioClient client;
  client.Init();
  client.SetTimeout(10.0f);

  int32_t speaker_id = args.count("speaker") ? std::stoi(args["speaker"]) : 1;
  int32_t ret = 0;

  if (args.count("get_volume")) {
    uint8_t v = 0;
    ret = client.GetVolume(v);
    std::cout << "volume: " << std::to_string(v) << " (ret " << ret << ")" << std::endl;
  }
  if (args.count("set_volume")) {
    ret = client.SetVolume(static_cast<uint8_t>(std::stoi(args["set_volume"])));
    std::cout << "set_volume ret: " << ret << std::endl;
  }
  if (args.count("tts")) {
    ret = client.TtsMaker(args["tts"], speaker_id);
    std::cout << "tts ret: " << ret << " (speaker " << speaker_id << ")" << std::endl;
    unitree::common::Sleep(1);  // let the RPC land before we exit; playback is server-side
  }
  return ret;
}
'''


class R1SpeakProxy:
    """Speaks through the R1's onboard speaker via PC1's native ``r1_audio_client``."""

    def __init__(
        self,
        host: str = "192.168.123.164",
        user: str = "unitree",
        password: str | None = None,
        interface: str = "eth10",
        volume: int = 85,
        speaker_id: int = 1,
    ) -> None:
        self.host = host
        self.user = user
        self.password = password or os.environ.get("R1_PC1_PASS", "123")
        self.interface = interface
        self.volume = volume
        self.speaker_id = speaker_id  # g1 AudioClient: 0 = Chinese, 1 = English
        self._client: Any = None  # paramiko.SSHClient

    def start(self) -> None:
        try:
            import paramiko
        except ImportError as e:
            raise ImportError("R1 speech needs paramiko: pip install paramiko") from e

        self._client = paramiko.SSHClient()
        self._client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
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
        self._ensure_client()
        logger.info(
            f"R1 speak proxy connected: {self.user}@{self.host} -> r1_audio_client (iface {self.interface})"
        )
        try:
            self._run(f"--set_volume={int(self.volume)}")
        except Exception as e:
            logger.warning(f"R1 set_volume failed: {e}")

    def _exec(self, cmd: str, timeout: float = 60.0) -> tuple[str, str]:
        assert self._client is not None, "R1SpeakProxy not started"
        _stdin, stdout, stderr = self._client.exec_command(cmd, timeout=timeout)
        return stdout.read().decode(errors="replace"), stderr.read().decode(errors="replace")

    def _ensure_client(self) -> None:
        """Build r1_audio_client on PC1 from the embedded source if it's missing."""
        out, _ = self._exec(f"test -x {_BIN} && echo OK || echo MISSING")
        if "OK" in out:
            return
        logger.info("r1_audio_client missing on PC1 — provisioning + building ...")
        import base64

        b64 = base64.b64encode(_CLIENT_SRC.encode()).decode()
        self._exec(f"echo {b64} | base64 -d > {_SRC}")
        # Register the CMake target once (idempotent), then build just that target.
        self._exec(
            "grep -q r1_audio_client ~/unitree_sdk2/example/r1/CMakeLists.txt || "
            "printf '\\nadd_executable(r1_audio_client high_level/r1_audio_client.cpp)\\n"
            "target_link_libraries(r1_audio_client unitree_sdk2)\\n' "
            ">> ~/unitree_sdk2/example/r1/CMakeLists.txt"
        )
        out, err = self._exec(
            "cd ~/unitree_sdk2/build && cmake .. >/dev/null 2>&1 && "
            "make r1_audio_client 2>&1 | tail -3",
            timeout=240,
        )
        logger.info(f"r1_audio_client build: {out.strip()} {err.strip()}")
        ok, _ = self._exec(f"test -x {_BIN} && echo OK || echo MISSING")
        if "OK" not in ok:
            raise RuntimeError("Failed to build r1_audio_client on PC1; see logs above")

    def _run(self, flag: str, timeout: float = 30.0) -> str:
        cmd = f"{_BIN} --network_interface={self.interface} {flag}"
        logger.debug(f"PC1$ {cmd}")
        out, err = self._exec(cmd, timeout=timeout)
        if err.strip():
            logger.warning(f"r1_audio_client stderr: {err.strip()}")
        return out

    def speak(self, text: str, speaker_id: int | None = None) -> bool:
        """Speak ``text`` on the robot's speaker (server-side TTS, fire-and-forget)."""
        text = text.strip()
        if not text:
            return False
        sid = self.speaker_id if speaker_id is None else speaker_id
        # shlex.quote makes the text one safe argv token; the CLI takes everything
        # after the first '=' as the value, so apostrophes/'='/etc. pass through.
        self._run(f"--tts={shlex.quote(text)} --speaker={int(sid)}")
        return True

    def set_volume(self, volume: int) -> bool:
        self._run(f"--set_volume={int(volume)}")
        return True

    def stop(self) -> None:
        if self._client is not None:
            self._client.close()
            self._client = None


__all__ = ["R1SpeakProxy"]
