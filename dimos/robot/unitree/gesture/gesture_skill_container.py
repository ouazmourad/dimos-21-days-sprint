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

"""Go2 hand-gesture control as a DimOS skill (deployable over WebRTC).

Ported from keaganchs/go2_gesture_recognition. The vision + gesture logic lives in
`gesture_engine`; this container is the DimOS-native wrapper: it takes frames from
DimOS's own camera stream (`color_image`) — NOT a second WebRTC connection — and
sends motion through the shared GO2 sport connection (`Move`/`StopMove`/emotes),
exactly like FootballSkillContainer.

Skill:
  * follow_gestures(seconds) -- run the vision->velocity loop for a bounded time.
      PEACE (right) enables, FLAT palm (right) disables; POINT at the floor walks
      there, POINT sideways turns in place, palm-down wave reverses; a LEFT-hand
      peace held 3 s arms emotes, then LEFT 1-4 fingers = Hello/Stretch/WiggleHips/
      Dance1 and a RIGHT peace = FingerHeart.

SAFETY: the robot walks toward wherever a person points and cannot see walls or
furniture. Run only with clear floor space. Any send failure / timeout halts it.
"""

from __future__ import annotations

import threading
import time
from typing import Any

import cv2

from unitree_webrtc_connect.constants import RTC_TOPIC, SPORT_CMD

from dimos.agents.annotation import skill
from dimos.core.core import rpc
from dimos.core.module import Module
from dimos.core.stream import In
from dimos.msgs.sensor_msgs.Image import Image, ImageFormat
from dimos.robot.unitree.go2.connection_spec import GO2ConnectionSpec
from dimos.utils.logging_config import setup_logger

logger = setup_logger()

_CMD_RATE = 10.0        # Hz — steady Move/StopMove stream (the Air's data channel
# chokes on high-rate floods; 10 Hz is the source app's proven rate).
_CMD_FRESHNESS = 0.5    # s — watchdog: if the newest vision command is older than
# this the robot is halted, so a stalled vision loop can't leave it driving blind.


class GestureSkillContainer(Module):
    """Gesture-driven "go where I point" control for the Go2 (velocity interface)."""

    color_image: In[Image]
    _connection: GO2ConnectionSpec

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._frame_lock = threading.Lock()
        self._latest_bgr = None          # newest camera frame, converted to BGR
        self._frame_ver = 0
        self._unsub = None
        self._engine: Any = None         # gesture_engine.GestureEngine (lazy)
        self._engine_lock = threading.Lock()
        self._active = False             # a follow_gestures loop is running
        self._session_thread: threading.Thread | None = None
        # Live dashboard (MJPEG over HTTP) — the headless replacement for the source
        # app's local OpenCV window. Started on first follow_gestures, served for the
        # life of the module.
        self._viz_lock = threading.Lock()
        self._viz_jpeg: bytes | None = None
        self._httpd: Any = None
        self._viz_port: int | None = None
        self._viz_thread: threading.Thread | None = None

    @rpc
    def start(self) -> None:
        super().start()

        def on_frame(msg: Image) -> None:
            data = msg.data
            if data is None or getattr(data, "ndim", 0) < 3:
                return
            bgr = cv2.cvtColor(data, cv2.COLOR_RGB2BGR) if msg.format == ImageFormat.RGB else data
            with self._frame_lock:
                self._latest_bgr = bgr
                self._frame_ver += 1

        self._unsub = self.color_image.subscribe(on_frame)
        # Bring the dashboard up at boot so the live camera preview is viewable
        # immediately (no mediapipe here — safe to start early).
        self._ensure_viz_server()

    @rpc
    def stop(self) -> None:
        self._active = False
        if self._unsub is not None:
            try:
                self._unsub()
            except Exception:
                pass
            self._unsub = None
        if self._engine is not None:
            try:
                self._engine.stop()
            except Exception:
                pass
            self._engine = None
        if self._httpd is not None:
            try:
                self._httpd.shutdown()
                self._httpd.server_close()
            except Exception:
                pass
            self._httpd = None
        super().stop()

    # ---- helpers ------------------------------------------------------------
    def _frame_getter(self):
        with self._frame_lock:
            if self._latest_bgr is None:
                return None, self._frame_ver
            return self._latest_bgr.copy(), self._frame_ver

    def _ensure_engine(self):
        if self._engine is not None:
            return
        with self._engine_lock:
            if self._engine is not None:
                return
            # Import (and thereby load mediapipe) LAZILY, here at first use — NOT at
            # module top level. mediapipe bundles its own abseil/protobuf/TFLite
            # native libs; if they load BEFORE dimos's native stack the process
            # segfaults on import. Deferring to skill-call time guarantees dimos is
            # fully loaded first (verified: dimos-first = OK, mediapipe-first = SIGSEGV).
            from dimos.robot.unitree.gesture.gesture_engine import GestureEngine

            self._engine = GestureEngine(self._frame_getter)
            self._engine.start()
            logger.info("Gesture: MediaPipe engine started")

    def _ensure_viz_server(self) -> int | None:
        """Start (once) an MJPEG HTTP server that streams the annotated dashboard —
        the headless stand-in for the source app's cv2 window. View it in a browser
        at http://localhost:<port>. Port from $GO2_GESTURE_VIZ_PORT (default 8600)."""
        if self._httpd is not None:
            return self._viz_port
        import http.server
        import os
        import socketserver

        port = int(os.environ.get("GO2_GESTURE_VIZ_PORT", "8600"))
        container = self

        class _Handler(http.server.BaseHTTPRequestHandler):
            def log_message(self, *args):  # keep the dimos log clean
                pass

            def do_GET(self):  # noqa: N802
                if self.path in ("/", "/index.html"):
                    body = (
                        b"<!doctype html><meta charset=utf-8>"
                        b"<title>Go2 Gesture Dashboard</title>"
                        b"<body style='margin:0;background:#111;text-align:center'>"
                        b"<img src='/stream' style='max-width:100%;height:auto'>"
                    )
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                    return
                if self.path == "/stream":
                    self.send_response(200)
                    self.send_header(
                        "Content-Type", "multipart/x-mixed-replace; boundary=frame"
                    )
                    self.end_headers()
                    try:
                        while container._httpd is not None:
                            with container._viz_lock:
                                jpg = container._viz_jpeg
                            if jpg is None:
                                time.sleep(0.05)
                                continue
                            self.wfile.write(
                                b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                                + str(len(jpg)).encode()
                                + b"\r\n\r\n"
                            )
                            self.wfile.write(jpg)
                            self.wfile.write(b"\r\n")
                            time.sleep(1.0 / 15.0)
                    except (BrokenPipeError, ConnectionResetError, OSError):
                        pass
                    return
                self.send_error(404)

        class _Server(socketserver.ThreadingTCPServer):
            allow_reuse_address = True  # survive a quick blueprint restart on the same port
            daemon_threads = True

        try:
            httpd = _Server(("0.0.0.0", port), _Handler)
        except OSError as e:
            logger.warning(f"Gesture: dashboard server couldn't bind :{port} ({e})")
            return None
        threading.Thread(target=httpd.serve_forever, daemon=True).start()
        self._httpd = httpd
        self._viz_port = port
        # Always-on preview: keep the stream showing the LIVE camera even when no
        # session is running, so it never sits on a stale frozen frame. _run_session
        # takes over rendering (full overlay) while active; this fills the gaps.
        self._viz_thread = threading.Thread(target=self._viz_loop, daemon=True)
        self._viz_thread.start()
        logger.info(f"Gesture: dashboard live at http://localhost:{port}")
        return port

    def _viz_loop(self) -> None:
        """Render a live camera preview to the MJPEG buffer whenever no control
        session is active (during a session, _run_session renders the full overlay)."""
        while self._httpd is not None:
            if not self._active:
                with self._frame_lock:
                    frame = None if self._latest_bgr is None else self._latest_bgr.copy()
                if frame is not None:
                    cv2.putText(frame, "NO ACTIVE SESSION - run follow_gestures",
                                (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 4, cv2.LINE_AA)
                    cv2.putText(frame, "NO ACTIVE SESSION - run follow_gestures",
                                (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 220, 255), 2, cv2.LINE_AA)
                    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                    if ok:
                        with self._viz_lock:
                            self._viz_jpeg = buf.tobytes()
            time.sleep(1.0 / 15.0)

    def _move(self, x: float, y: float, z: float) -> None:
        # Tolerate transient send failures (the GO2Connection @rpc intermittently
        # hits a stub mid-session) — skip the bad command and keep going.
        try:
            self._connection.publish_request(
                RTC_TOPIC["SPORT_MOD"],
                {"api_id": SPORT_CMD["Move"], "parameter": {"x": x, "y": y, "z": z}},
            )
        except Exception:  # noqa: BLE001
            pass

    def _stop_move(self) -> None:
        try:
            self._connection.publish_request(RTC_TOPIC["SPORT_MOD"], {"api_id": SPORT_CMD["StopMove"]})
        except Exception:  # noqa: BLE001
            pass

    def _emote(self, name: str) -> None:
        cmd = SPORT_CMD.get(name)
        if cmd is None:
            return
        try:
            self._connection.publish_request(RTC_TOPIC["SPORT_MOD"], {"api_id": cmd})
        except Exception:  # noqa: BLE001
            pass

    # ---- behaviour ----------------------------------------------------------
    @skill
    def follow_gestures(self, seconds: float = 60.0, hand: str = "both") -> str:
        """Start watching the person through the camera and moving on their hand
        gestures. This is a BACKGROUND skill: it starts a detector and returns
        immediately (it does NOT block for the whole session). Call it ONCE, then
        stay silent — the person drives me with their hands until the time runs out
        or `stop_gestures` is called. Do NOT call move/execute_sport_command while
        it runs.

        Gestures: PEACE sign (right hand) ENABLES; POINT at a floor spot -> I walk
        there; point sideways -> I turn that way; palm-down back-wave -> I step
        backwards; FLAT palm (right) DISABLES. Hold a LEFT-hand peace sign 3 s to arm
        emotes, then LEFT 1-4 fingers -> Hello / Stretch / WiggleHips / Dance1;
        right-hand peace -> FingerHeart.

        SAFETY: I walk toward where you point and cannot see walls or furniture —
        run this only with clear floor space around me.

        Args:
            seconds: how long to accept gestures for before auto-stopping (5-600).
            hand: which hand's POINT drives me — 'right', 'left' or 'both'.
        """
        seconds = min(max(5.0, float(seconds)), 600.0)
        hand = hand if hand in ("right", "left", "both") else "both"

        if self._active:
            return "Gesture control is already running — call stop_gestures to end it first."
        if self._latest_bgr is None:
            return "No camera frames yet — is the robot connected and streaming?"
        try:
            self._ensure_engine()
        except FileNotFoundError as e:
            return f"Gesture models not found: {e}"
        except Exception as e:  # noqa: BLE001
            logger.error(f"gesture engine init failed: {e}")
            return "Couldn't start the gesture engine (is mediapipe installed?)."

        viz_port = self._ensure_viz_server()

        # Run the vision->velocity loop on a BACKGROUND thread and return now, so the
        # MCP call doesn't block for the whole session (a blocking call times out
        # client- and server-side at 30/120 s). stop_gestures / the timeout end it.
        self._active = True
        self._session_thread = threading.Thread(
            target=self._run_session, args=(seconds, hand), daemon=True
        )
        self._session_thread.start()
        dash = f" Live dashboard: http://localhost:{viz_port}" if viz_port else ""
        return (
            f"Gesture control ON — I'll watch your hands for {seconds:.0f}s. Show a "
            f"PEACE sign with your right hand to wake me, then point where I should go. "
            f"Flat palm stops me; call stop_gestures (or say stop) to end early.{dash}"
        )

    @skill
    def stop_gestures(self) -> str:
        """Stop the gesture-control session started by follow_gestures (halts the
        robot). Use this to end early instead of waiting for the timeout."""
        if not self._active:
            return "Gesture control isn't running."
        self._active = False
        t = self._session_thread
        if t is not None:
            t.join(timeout=5)
        self._stop_move()
        return "Gesture control OFF — I'm back under voice control."

    def _run_session(self, seconds: float, hand: str) -> None:
        """Background vision->velocity control loop for one follow_gestures session.

        A sender thread drains the latest command at a steady 10 Hz with a freshness
        watchdog (so a slow inference frame can't leave the robot coasting); this
        loop consumes engine results and runs the gesture state machine. On exit it
        halts the robot and releases the MediaPipe engine (frees CPU until next time).
        """
        from dimos.robot.unitree.gesture.gesture_engine import FloorProjector, Tracker
        from dimos.robot.unitree.gesture.gesture_viz import draw_overlay

        tracker = Tracker(point_hand=hand)
        cmd_lock = threading.Lock()
        cmd = {"mode": "idle", "v": (0.0, 0.0, 0.0), "stamp": 0.0, "emote": None}

        def _sender():
            stop_sent = True
            while self._active:
                t0 = time.monotonic()
                with cmd_lock:
                    mode, v, stamp = cmd["mode"], cmd["v"], cmd["stamp"]
                    emote = cmd["emote"]
                    cmd["emote"] = None
                fresh = time.monotonic() - stamp < _CMD_FRESHNESS
                if emote:
                    self._emote(emote)
                if mode == "move" and fresh:
                    stop_sent = False
                    self._move(v[0], v[1], v[2])
                elif (mode == "stop" or (mode == "move" and not fresh)) and not stop_sent:
                    stop_sent = True
                    self._stop_move()
                time.sleep(max(0.0, 1.0 / _CMD_RATE - (time.monotonic() - t0)))
            self._stop_move()

        sender = threading.Thread(target=_sender, daemon=True)
        sender.start()

        projector = None
        last_ver = -1
        end = time.monotonic() + seconds
        try:
            while self._active and time.monotonic() < end:
                out, ver = self._engine.latest()
                if out is None or ver == last_ver:
                    time.sleep(0.005)
                    continue
                last_ver = ver
                frame, hands = out["frame"], out["hands"]
                if projector is None:
                    h, w = frame.shape[:2]
                    projector = FloorProjector(w, h)
                now = time.monotonic()
                v, mode, debug = tracker.update(hands, projector, now)
                with cmd_lock:
                    if mode in ("move", "stop"):
                        cmd["mode"], cmd["v"], cmd["stamp"] = mode, v, now
                    elif mode == "emote":
                        cmd["emote"] = debug["emote"]
                    elif mode == "idle" and cmd["mode"] == "move":
                        cmd["mode"], cmd["stamp"] = "stop", now

                # Render the dashboard frame for the MJPEG stream (best-effort).
                if self._httpd is not None:
                    try:
                        canvas = draw_overlay(
                            frame.copy(), hands, tracker, debug, v,
                            self._engine.fps, "connected", True, out["vision"],
                        )
                        ok, buf = cv2.imencode(".jpg", canvas, [cv2.IMWRITE_JPEG_QUALITY, 80])
                        if ok:
                            with self._viz_lock:
                                self._viz_jpeg = buf.tobytes()
                    except Exception:  # noqa: BLE001
                        pass
        except Exception as e:  # noqa: BLE001
            logger.error(f"follow_gestures session failed: {e}")
        finally:
            self._active = False
            with cmd_lock:
                cmd["mode"], cmd["stamp"] = "stop", time.monotonic()
            sender.join(timeout=2)
            self._stop_move()
            if self._engine is not None:
                try:
                    self._engine.stop()
                except Exception:
                    pass
                self._engine = None
            logger.info("Gesture: session ended")

    @skill
    def dump_gesture_frame(self) -> str:
        """Debug: save the current camera frame (and detected hands, if the engine
        is running) to /tmp/go2_gesture_debug so recognition can be inspected."""
        import os

        with self._frame_lock:
            frame = None if self._latest_bgr is None else self._latest_bgr.copy()
        if frame is None:
            return "No camera frame to dump."
        out_dir = "/tmp/go2_gesture_debug"
        os.makedirs(out_dir, exist_ok=True)
        cv2.imwrite(f"{out_dir}/frame.jpg", frame)
        hands_txt = "engine not running (call follow_gestures first)"
        if self._engine is not None:
            out, _ = self._engine.latest()
            if out is not None:
                hands_txt = ", ".join(
                    f"{h['handedness']}[{h['model'][0]}]" for h in out["hands"]
                ) or "no hands"
        return f"Saved frame to {out_dir}. Hands: {hands_txt}."
