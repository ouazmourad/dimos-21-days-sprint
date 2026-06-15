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

"""R1 front camera via PC1's native video client, bridged over SSH.

The R1 publishes its camera on ``rt/frontvideostream`` (``unitree_go``
``Go2FrontVideoData_``, JPEG-encoded). The laptop's pip SDK can't read the R1's
DDS (type mismatch), but PC1's firmware-matching ``unitree_sdk2`` can. This
``CameraHardware`` SSHes to PC1, runs a tiny streaming client (``r1_video_stream``,
auto-compiled from the bundled source below if the binary is missing), reads
length-prefixed JPEG frames and emits them as DimOS ``Image`` messages
(~17 fps, 1280x720). ``paramiko``/``cv2`` are imported lazily.
"""

from functools import cache
import os
import struct
import threading
import time
from typing import Any

from pydantic import Field
from reactivex import create
from reactivex.observable import Observable

from dimos.hardware.sensors.camera.spec import CameraConfig, CameraHardware
from dimos.msgs.sensor_msgs.CameraInfo import CameraInfo
from dimos.msgs.sensor_msgs.Image import Image, ImageFormat
from dimos.utils.logging_config import setup_logger
from dimos.utils.reactive import backpressure

logger = setup_logger()

# C++ streamer: reads rt/frontvideostream via the firmware-matching SDK on PC1
# and writes [uint32 big-endian length][jpeg] frames to stdout. Auto-compiled on
# PC1 if the binary is absent (e.g. after a reseat of the OS image).
_STREAM_CPP = r"""#include <unitree/robot/go2/video/video_client.hpp>
#include <arpa/inet.h>
#include <unistd.h>
#include <cstdint>
#include <cstdio>
#include <vector>
int main() {
    unitree::robot::ChannelFactory::Instance()->Init(0);
    unitree::robot::go2::VideoClient vc; vc.SetTimeout(1.0f); vc.Init();
    fprintf(stderr, "[r1_video_stream] started\n");
    std::vector<uint8_t> buf;
    while (true) {
        if (vc.GetImageSample(buf) == 0 && !buf.empty()) {
            uint32_t n = htonl((uint32_t)buf.size());
            fwrite(&n, 4, 1, stdout); fwrite(buf.data(), 1, buf.size(), stdout); fflush(stdout);
        }
        usleep(30000);
    }
}
"""


def _compile_cmd(sdk: str, src: str, out: str) -> str:
    a = f"{sdk}/lib/aarch64/libunitree_sdk2.a"
    tp = f"{sdk}/thirdparty"
    return (
        f"g++ -std=c++17 -O2 {src} -o {out} "
        f"-I{sdk}/include -I{tp}/include/ddscxx -I{tp}/include/ddsc -I{tp}/include "
        f"-Wl,--start-group {a} {tp}/lib/aarch64/libddsc.so {tp}/lib/aarch64/libddscxx.so "
        f"-Wl,--end-group -pthread -Wl,-rpath,{tp}/lib/aarch64; echo EXIT=$?"
    )


class R1PC1CameraConfig(CameraConfig):
    frame_id_prefix: str | None = None
    # Tamed defaults for the live dashboard: full-rate 720p is ~60 MB/s over LCM,
    # which floods Rerun/the browser and can OOM-freeze the host. Frames are
    # throttled to `fps` and downscaled to width x height. Raise once stable.
    width: int = 640
    height: int = 360
    fps: float = 5.0
    camera_info: CameraInfo = Field(default_factory=CameraInfo)
    # PC1 SSH proxy + paths (password falls back to R1_PC1_PASS env, then "123").
    host: str = "192.168.123.164"
    user: str = "unitree"
    password: str | None = None
    sdk_path: str = "/home/unitree/unitree_sdk2"
    stream_bin: str = "/home/unitree/r1_video_stream"
    stream_src: str = "/home/unitree/r1_video_stream.cpp"


class R1PC1Camera(CameraHardware):
    config: R1PC1CameraConfig

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._client: Any = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._observer: Any = None

    @property
    def camera_info(self) -> CameraInfo:
        return self.config.camera_info

    @cache
    def image_stream(self) -> Observable[Image]:
        def subscribe(observer: Any, _scheduler: Any = None) -> Any:
            self._observer = observer
            try:
                self.start()
            except Exception as e:
                observer.on_error(e)
                return None

            def dispose() -> None:
                self._observer = None
                self.stop()

            return dispose

        return backpressure(create(subscribe))

    def _password(self) -> str:
        return self.config.password or os.environ.get("R1_PC1_PASS", "123")

    def _connect(self) -> Any:
        import paramiko

        client = paramiko.SSHClient()
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        client.connect(
            self.config.host, username=self.config.user, password=self._password(), timeout=10
        )
        return client

    def _ensure_binary(self, client: Any) -> None:
        cfg = self.config
        _i, o, _e = client.exec_command(f"test -x {cfg.stream_bin} && echo OK", timeout=10)
        if o.read().decode().strip() == "OK":
            return
        logger.info("R1 PC1 camera streamer missing; compiling on PC1 ...")
        sftp = client.open_sftp()
        with sftp.open(cfg.stream_src, "w") as f:
            f.write(_STREAM_CPP)
        sftp.close()
        _i, o, e = client.exec_command(
            _compile_cmd(cfg.sdk_path, cfg.stream_src, cfg.stream_bin), timeout=180
        )
        out = o.read().decode(errors="replace") + e.read().decode(errors="replace")
        if "EXIT=0" not in out:
            raise RuntimeError(f"Failed to compile r1_video_stream on PC1:\n{out[-800:]}")
        logger.info("R1 PC1 camera streamer compiled")

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._client = self._connect()
        self._ensure_binary(self._client)
        self._stop.clear()
        self._thread = threading.Thread(target=self._reader, daemon=True)
        self._thread.start()
        logger.info(f"R1 PC1 camera streaming from {self.config.user}@{self.config.host}")

    def _reader(self) -> None:
        import cv2
        import numpy as np

        _i, stdout, _e = self._client.exec_command(self.config.stream_bin, timeout=None)
        chan = stdout.channel
        chan.settimeout(5.0)
        buf = b""
        min_interval = 1.0 / self.config.fps if self.config.fps > 0 else 0.0
        last_emit = 0.0
        while not self._stop.is_set():
            try:
                chunk = chan.recv(65536)
            except Exception:
                continue
            if not chunk:
                break
            buf += chunk
            while len(buf) >= 4:
                ln = struct.unpack(">I", buf[:4])[0]
                if ln <= 0 or ln > 5_000_000:
                    buf = b""
                    break
                if len(buf) < 4 + ln:
                    break
                jpg, buf = buf[4 : 4 + ln], buf[4 + ln :]
                # Throttle to config.fps *before* the expensive decode, so full-rate
                # 720p doesn't flood LCM/Rerun/the browser and OOM the host.
                now = time.time()
                if min_interval > 0 and (now - last_emit) < min_interval:
                    continue
                if self._observer is None or self._stop.is_set():
                    continue
                img = cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR)
                if img is None:
                    continue
                last_emit = now
                if img.shape[1] != self.config.width or img.shape[0] != self.config.height:
                    img = cv2.resize(img, (self.config.width, self.config.height))
                rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                self._observer.on_next(
                    Image.from_numpy(
                        rgb, format=ImageFormat.RGB, frame_id=self._frame_id(), ts=time.time()
                    )
                )

    def _frame_id(self) -> str:
        if self.config.frame_id_prefix:
            return f"{self.config.frame_id_prefix}/camera_optical"
        return "camera_optical"

    def stop(self) -> None:
        self._stop.set()
        if self._client is not None:
            try:
                self._client.exec_command("pkill -f r1_video_stream")
            except Exception:
                pass
            try:
                self._client.close()
            except Exception:
                pass
            self._client = None
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)


__all__ = ["R1PC1Camera", "R1PC1CameraConfig"]
