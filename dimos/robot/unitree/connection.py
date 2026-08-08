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

import asyncio
from dataclasses import dataclass
import functools
import os
import threading
import time
from typing import Any, TypeAlias, TypeVar

import numpy as np
from numpy.typing import NDArray
from reactivex import operators as ops
from reactivex.observable import Observable
from reactivex.subject import Subject
from unitree_webrtc_connect.constants import (
    RTC_TOPIC,
    SPORT_CMD,
    VUI_COLOR,
)
from unitree_webrtc_connect.webrtc_driver import (
    UnitreeWebRTCConnection as LegionConnection,
    WebRTCConnectionMethod,
)

from dimos.constants import DEFAULT_THREAD_JOIN_TIMEOUT
from dimos.core.resource import Resource
from dimos.msgs.geometry_msgs.Pose import Pose
from dimos.msgs.geometry_msgs.Transform import Transform
from dimos.msgs.geometry_msgs.Twist import Twist
from dimos.msgs.sensor_msgs.Image import Image, ImageFormat
from dimos.msgs.sensor_msgs.PointCloud2 import PointCloud2
from dimos.robot.unitree.type.lidar import (
    RawLidarMsg,
    pointcloud2_from_webrtc_lidar,
)
from dimos.robot.unitree.type.lowstate import LowStateMsg
from dimos.robot.unitree.type.odometry import Odometry
from dimos.types.timestamped import Timestamped
from dimos.utils.decorators.decorators import simple_mcache
from dimos.utils.reactive import backpressure, callback_to_observable

VideoMessage: TypeAlias = NDArray[np.uint8]  # Shape: (height, width, 3)


_T = TypeVar("_T", bound=Timestamped)


def time_is_now(x: _T) -> _T:
    x.ts = time.time()
    return x


@dataclass
class SerializableVideoFrame:
    """Pickleable wrapper for av.VideoFrame with all metadata"""

    data: np.ndarray
    pts: int | None = None
    time: float | None = None
    dts: int | None = None
    width: int | None = None
    height: int | None = None
    format: str | None = None

    @classmethod
    def from_av_frame(cls, frame):  # type: ignore[no-untyped-def]
        return cls(
            data=frame.to_ndarray(format="rgb24"),
            pts=frame.pts,
            time=frame.time,
            dts=frame.dts,
            width=frame.width,
            height=frame.height,
            format=frame.format.name if hasattr(frame, "format") and frame.format else None,
        )

    def to_ndarray(self, format=None):  # type: ignore[no-untyped-def]
        return self.data


class UnitreeWebRTCConnection(Resource):
    _SPORT_API_ID_RAGEMODE: int = 2059

    # --- auto-reconnect tuning ---
    # The Go2 Air's single WebRTC data channel drops fairly often (idle timeouts,
    # network blips, brief contention). A watchdog running in the connection's own
    # event loop detects this and transparently rebuilds the channel.
    _WATCHDOG_INTERVAL_S: float = 2.0  # how often the watchdog checks health
    _DATA_TIMEOUT_S: float = 10.0  # no data on any subscribed stream this long => stale => reconnect
    _RECONNECT_BACKOFF_S: float = 3.0  # wait before retrying a failed reconnect

    def __init__(self, ip: str, mode: str = "ai") -> None:
        self.ip = ip
        self.mode = mode
        self.stop_timer: threading.Timer | None = None
        self.cmd_vel_timeout = 0.2
        # --- auto-reconnect state ---
        # topic -> wrapped callback, re-applied to the new data channel on reconnect
        self._active_subs: dict[str, Any] = {}
        # monotonic time of the last message on ANY subscribed stream (freshness signal)
        self._last_data_ts: float | None = None
        self._stop_watchdog = False
        # data2=3 firmware (Go2 >= 1.1.15) requires a per-device AES-128 key for
        # the LAN handshake; supply via GO2_AES_KEY (tools/fetch_go2_aes_key.py).
        aes_128_key = os.environ.get("GO2_AES_KEY") or None
        # Connection method: LocalSTA (dog on your LAN, reached by ROBOT_IP — the
        # default) or LocalAP (laptop joined the dog's OWN hotspot; the connector
        # forces ip=192.168.12.1 and ignores ROBOT_IP). Override via env:
        # GO2_WEBRTC_METHOD=LocalAP (e.g. in .env). AES key applies to both.
        method = getattr(
            WebRTCConnectionMethod, os.environ.get("GO2_WEBRTC_METHOD", "LocalSTA")
        )
        self.conn = LegionConnection(method, ip=self.ip, aes_128_key=aes_128_key)
        self.connect()

    def connect(self) -> None:
        self.loop = asyncio.new_event_loop()
        self.task = None
        self.connected_event = asyncio.Event()
        self.connection_ready = threading.Event()

        async def async_connect() -> None:
            await self.conn.connect()
            await self._post_connect_setup()

            self.connected_event.set()
            self.connection_ready.set()

            # Auto-reconnect watchdog: detect a dropped/stalled channel and rebuild
            # it transparently (re-apply motion mode + re-subscribe active streams)
            # so downstream consumers (camera/lidar/odom) never see the gap.
            while not self._stop_watchdog:
                await asyncio.sleep(self._WATCHDOG_INTERVAL_S)
                if self._stop_watchdog:
                    break
                try:
                    if self._should_reconnect():
                        await self._do_reconnect()
                except Exception as e:  # noqa: BLE001 — watchdog must never die
                    print(f"Go2 auto-reconnect watchdog error: {e}")

        def start_background_loop() -> None:
            asyncio.set_event_loop(self.loop)
            self.task = self.loop.create_task(async_connect())
            self.loop.run_forever()

        self.loop = asyncio.new_event_loop()
        self.thread = threading.Thread(target=start_background_loop, daemon=True)
        self.thread.start()
        self.connection_ready.wait()

    async def _post_connect_setup(self) -> None:
        """(Re-)apply data-channel configuration after every (re)connect."""
        await self.conn.datachannel.disableTrafficSaving(True)
        self.conn.datachannel.set_decoder(decoder_type="native")
        await self.conn.datachannel.pub_sub.publish_request_new(
            RTC_TOPIC["MOTION_SWITCHER"], {"api_id": 1002, "parameter": {"name": self.mode}}
        )
        # Treat a fresh connect as fresh data so the freshness watchdog doesn't
        # immediately fire before the first message arrives.
        self._last_data_ts = time.monotonic()

    def _should_reconnect(self) -> bool:
        """True when the channel looks dead.

        Two independent signals: the connector reports it's no longer connected
        (peer state went to "closed"), OR — if any stream is subscribed — no data
        has arrived for _DATA_TIMEOUT_S (catches silent stalls where the connector
        still believes it is connected but odom/video have frozen).
        """
        if not getattr(self.conn, "isConnected", True):
            return True
        if self._active_subs and self._last_data_ts is not None:
            if time.monotonic() - self._last_data_ts > self._DATA_TIMEOUT_S:
                return True
        return False

    async def _do_reconnect(self) -> None:
        """Rebuild the peer + data channel, re-apply config, re-subscribe streams."""
        print("Go2 WebRTC channel lost — attempting auto-reconnect…")
        try:
            await self.conn.reconnect()
            await self._post_connect_setup()
            # Re-register every active subscription on the NEW data channel; the
            # connector recreates self.datachannel (and its pub_sub) on reconnect,
            # so the old registrations are gone.
            for topic, cb in list(self._active_subs.items()):
                self.conn.datachannel.pub_sub.subscribe(topic, cb)
            self._last_data_ts = time.monotonic()
            print(f"Go2 WebRTC auto-reconnect succeeded ({len(self._active_subs)} streams restored)")
        except Exception as e:  # noqa: BLE001
            print(f"Go2 WebRTC auto-reconnect failed: {e}; retrying shortly")
            await asyncio.sleep(self._RECONNECT_BACKOFF_S)

    def start(self) -> None:
        pass

    def stop(self) -> None:
        # Tell the auto-reconnect watchdog to exit so it doesn't fight the teardown.
        self._stop_watchdog = True

        # Cancel timer
        if self.stop_timer:
            self.stop_timer.cancel()
            self.stop_timer = None

        if self.task:
            self.task.cancel()

        async def async_disconnect() -> None:
            try:
                # Send stop command directly since we're already in the event loop.
                self.conn.datachannel.pub_sub.publish_without_callback(
                    RTC_TOPIC["WIRELESS_CONTROLLER"],
                    data={"lx": 0, "ly": 0, "rx": 0, "ry": 0},
                )
                await self.conn.disconnect()
            except Exception:
                pass

        if self.loop.is_running():
            asyncio.run_coroutine_threadsafe(async_disconnect(), self.loop)

            self.loop.call_soon_threadsafe(self.loop.stop)

        if self.thread.is_alive():
            self.thread.join(timeout=DEFAULT_THREAD_JOIN_TIMEOUT)

    def move(self, twist: Twist, duration: float = 0.0) -> bool:
        """Send movement command to the robot using Twist commands.

        Args:
            twist: Twist message with linear and angular velocities
            duration: How long to move (seconds). If 0, command is continuous

        Returns:
            bool: True if command was sent successfully
        """
        x, y, yaw = twist.linear.x, twist.linear.y, twist.angular.z

        # Drive with the SPORT `Move` command (api_id 1008), NOT the
        # rt/wirelesscontroller joystick topic.
        #
        # The joystick path (previously used here) does NOT move a Go2 Air: verified
        # against odometry, the robot barely translates, and flooding it also knocks
        # over the WebRTC data channel. The visible symptom when the navigation stack
        # drives through this method is a robot that twitches/rotates in place and
        # never follows its planned path — A* plans fine, but the velocities never
        # reach the legs. Sport Move is odometry-verified (0.3 m/s x 1.5 s -> 0.35 m;
        # a commanded 360 deg spin measured 349 deg).
        #
        # Sport Move takes body-frame velocities directly: x forward m/s, y left m/s,
        # z yaw rad/s — the same convention as Twist, so no sign flips.
        def send_move() -> None:
            self.publish_request(
                RTC_TOPIC["SPORT_MOD"],
                {"api_id": SPORT_CMD["Move"], "parameter": {"x": x, "y": y, "z": yaw}},
            )

        # Cancel existing timer and start a new one
        if self.stop_timer:
            self.stop_timer.cancel()

        # Auto-stop shortly after the last command so a dropped cmd_vel stream doesn't
        # leave the robot walking.
        self.stop_timer = threading.Timer(self.cmd_vel_timeout, self.stop_movement)
        self.stop_timer.daemon = True
        self.stop_timer.start()

        try:
            if duration > 0:
                # Re-send at ~10 Hz so the robot keeps walking for the whole duration
                # (the sport controller expects a repeating velocity command). 10 Hz,
                # not the old 100 Hz, which was enough to destabilise the channel.
                end = time.monotonic() + duration
                while time.monotonic() < end:
                    send_move()
                    time.sleep(0.1)
                self.stop_movement()
            else:
                # One velocity update; the caller (e.g. nav cmd_vel) keeps streaming.
                send_move()
            return True
        except Exception as e:
            print(f"Failed to send movement command: {e}")
            return False

    # Generic conversion of unitree subscription to Subject (used for all subs)
    def unitree_sub_stream(self, topic_name: str):  # type: ignore[no-untyped-def]
        def subscribe_in_thread(cb) -> None:  # type: ignore[no-untyped-def]
            # Wrap the callback so every received message refreshes the freshness
            # timestamp (drives the stall-detection watchdog) AND so we keep a
            # handle to re-subscribe this exact callback after an auto-reconnect.
            def wrapped(msg) -> None:  # type: ignore[no-untyped-def]
                self._last_data_ts = time.monotonic()
                cb(msg)

            self._active_subs[topic_name] = wrapped

            # Run the subscription in the background thread that has the event loop
            def run_subscription() -> None:
                self.conn.datachannel.pub_sub.subscribe(topic_name, wrapped)

            # Use call_soon_threadsafe to run in the background thread
            self.loop.call_soon_threadsafe(run_subscription)

        def unsubscribe_in_thread(cb) -> None:  # type: ignore[no-untyped-def]
            self._active_subs.pop(topic_name, None)

            # Run the unsubscription in the background thread that has the event loop
            def run_unsubscription() -> None:
                self.conn.datachannel.pub_sub.unsubscribe(topic_name)

            # Use call_soon_threadsafe to run in the background thread
            self.loop.call_soon_threadsafe(run_unsubscription)

        return callback_to_observable(
            start=subscribe_in_thread,
            stop=unsubscribe_in_thread,
        )

    # Generic sync API call (we jump into the client thread)
    def publish_request(self, topic: str, data: dict[Any, Any]) -> Any:
        """Send a sport REQUEST. FIRE-AND-FORGET: schedule it on the connection
        loop but do NOT block on the robot's ack.

        The connector's `publish()` does `return await future`, resolved only when
        the robot replies to that request id. Awaiting it inside a high-rate
        velocity loop stalls the command stream between sends, so the dog moves in
        start-stop bursts (a 360° spin crawling ~60° every few seconds; forward
        moves under-travelling). The command is still delivered — `channel.send`
        runs inside the coroutine before that await — we just don't wait for the
        reply. The ack future resolves/GCs on its own; we retrieve any error so
        asyncio doesn't log it as unretrieved.
        """
        future = asyncio.run_coroutine_threadsafe(
            self.conn.datachannel.pub_sub.publish_request_new(topic, data), self.loop
        )
        future.add_done_callback(lambda f: f.cancelled() or f.exception())
        return {"status": "sent"}

    @simple_mcache
    def raw_lidar_stream(self) -> Observable[RawLidarMsg]:
        return backpressure(self.unitree_sub_stream(RTC_TOPIC["ULIDAR_ARRAY"]))

    @simple_mcache
    def raw_odom_stream(self) -> Observable[Pose]:
        return backpressure(self.unitree_sub_stream(RTC_TOPIC["ROBOTODOM"]))

    @simple_mcache
    def lidar_stream(self) -> Observable[PointCloud2]:
        return backpressure(
            self.raw_lidar_stream().pipe(
                ops.map(pointcloud2_from_webrtc_lidar),
                ops.map(time_is_now),
                # repair_stale_ts(),
            )
        )

    @simple_mcache
    def tf_stream(self) -> Observable[Transform]:
        base_link = functools.partial(Transform.from_pose, "base_link")
        return backpressure(self.odom_stream().pipe(ops.map(base_link)))

    @simple_mcache
    def odom_stream(self) -> Observable[Pose]:
        return backpressure(
            self.raw_odom_stream().pipe(
                ops.map(
                    Odometry.from_msg,
                ),
                ops.map(time_is_now),
            )
        )

    @simple_mcache
    def video_stream(self) -> Observable[Image]:
        return backpressure(
            self.raw_video_stream().pipe(
                ops.filter(lambda frame: frame is not None),
                ops.map(
                    lambda frame: Image.from_numpy(
                        # np.ascontiguousarray(frame.to_ndarray("rgb24")),
                        frame.to_ndarray(format="rgb24"),  # type: ignore[attr-defined]
                        format=ImageFormat.RGB,  # Frame is RGB24, not BGR
                        frame_id="camera_optical",
                    ),
                ),
                ops.map(time_is_now),
            )
        )

    @simple_mcache
    def lowstate_stream(self) -> Observable[LowStateMsg]:
        return backpressure(self.unitree_sub_stream(RTC_TOPIC["LOW_STATE"]))

    def standup(self) -> bool:
        return bool(self.publish_request(RTC_TOPIC["SPORT_MOD"], {"api_id": SPORT_CMD["StandUp"]}))

    def balance_stand(self) -> bool:
        """Activate BalanceStand mode — enables WIRELESS_CONTROLLER joystick commands."""
        return bool(
            self.publish_request(RTC_TOPIC["SPORT_MOD"], {"api_id": SPORT_CMD["BalanceStand"]})
        )

    def set_obstacle_avoidance(self, enabled: bool = True) -> None:
        self.publish_request(
            RTC_TOPIC["OBSTACLES_AVOID"],
            {"api_id": 1001, "parameter": {"enable": int(enabled)}},
        )

    def free_walk(self) -> bool:
        """Activate FreeWalk locomotion mode — enables walking and velocity commands."""
        return bool(self.publish_request(RTC_TOPIC["SPORT_MOD"], {"api_id": SPORT_CMD["FreeWalk"]}))

    def enable_rage_mode(self) -> bool:
        """Enable Rage Mode on the Go2 via WebRTC.
        Assumes the robot is already in BalanceStand.
        """
        rage_ok = bool(
            self.publish_request(
                RTC_TOPIC["SPORT_MOD"],
                {"api_id": self._SPORT_API_ID_RAGEMODE, "parameter": {"data": True}},
            )
        )
        time.sleep(2.0)

        joystick_ok = bool(
            self.publish_request(
                RTC_TOPIC["SPORT_MOD"],
                {
                    "api_id": SPORT_CMD["SwitchJoystick"],
                    "parameter": {"data": True},
                },
            )
        )
        return rage_ok and joystick_ok

    def liedown(self) -> bool:
        return bool(
            self.publish_request(RTC_TOPIC["SPORT_MOD"], {"api_id": SPORT_CMD["StandDown"]})
        )

    async def handstand(self):  # type: ignore[no-untyped-def]
        return self.publish_request(
            RTC_TOPIC["SPORT_MOD"],
            {"api_id": SPORT_CMD["Standup"], "parameter": {"data": True}},
        )

    def color(self, color: VUI_COLOR = VUI_COLOR.RED, colortime: int = 60) -> bool:
        return self.publish_request(  # type: ignore[no-any-return]
            RTC_TOPIC["VUI"],
            {
                "api_id": 1001,
                "parameter": {
                    "color": color,
                    "time": colortime,
                },
            },
        )

    @simple_mcache
    def raw_video_stream(self) -> Observable[VideoMessage]:
        subject: Subject[VideoMessage] = Subject()
        stop_event = threading.Event()

        from aiortc import MediaStreamTrack

        async def accept_track(track: MediaStreamTrack) -> None:
            while True:
                if stop_event.is_set():
                    return
                frame = await track.recv()
                serializable_frame = SerializableVideoFrame.from_av_frame(frame)  # type: ignore[no-untyped-call]
                subject.on_next(serializable_frame)

        self.conn.video.add_track_callback(accept_track)

        # Run the video channel switching in the background thread
        def switch_video_channel() -> None:
            self.conn.video.switchVideoChannel(True)

        self.loop.call_soon_threadsafe(switch_video_channel)

        def stop() -> None:
            stop_event.set()  # Signal the loop to stop
            self.conn.video.track_callbacks.remove(accept_track)

            # Run the video channel switching off in the background thread
            def switch_video_channel_off() -> None:
                self.conn.video.switchVideoChannel(False)

            self.loop.call_soon_threadsafe(switch_video_channel_off)

        return subject.pipe(ops.finally_action(stop))

    def get_video_stream(self, fps: int = 30) -> Observable[Image]:
        """Get the video stream from the robot's camera.

        Implements the AbstractRobot interface method.

        Args:
            fps: Frames per second. This parameter is included for API compatibility,
                 but doesn't affect the actual frame rate which is determined by the camera.

        Returns:
            Observable: An observable stream of video frames or None if video is not available.
        """
        return self.video_stream()

    def stop_movement(self) -> None:
        """Stop the robot and cancel the auto-stop timer.

        This previously only cancelled the timer, so a robot walking under a sport
        `Move` velocity kept going when the command stream stopped. Now it also sends
        StopMove, which is what actually halts the legs.
        """
        if self.stop_timer:
            self.stop_timer.cancel()
            self.stop_timer = None
        try:
            self.publish_request(
                RTC_TOPIC["SPORT_MOD"], {"api_id": SPORT_CMD["StopMove"]}
            )
        except Exception:  # noqa: BLE001
            pass

    def disconnect(self) -> None:
        """Disconnect from the robot and clean up resources."""
        # Cancel timer
        if self.stop_timer:
            self.stop_timer.cancel()
            self.stop_timer = None

        if hasattr(self, "task") and self.task:
            self.task.cancel()
        if hasattr(self, "conn"):

            async def async_disconnect() -> None:
                try:
                    await self.conn.disconnect()
                except:
                    pass

            if hasattr(self, "loop") and self.loop.is_running():
                asyncio.run_coroutine_threadsafe(async_disconnect(), self.loop)

        if hasattr(self, "loop") and self.loop.is_running():
            self.loop.call_soon_threadsafe(self.loop.stop)

        if hasattr(self, "thread") and self.thread.is_alive():
            self.thread.join(timeout=DEFAULT_THREAD_JOIN_TIMEOUT)
