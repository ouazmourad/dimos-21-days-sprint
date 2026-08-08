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

"""Go2 Air "play football" behaviour (deployable over WebRTC, no low-level control).

The Air only exposes high-level *sport* velocity commands, so this treats walking
as a black box and runs a vision -> velocity control loop on top:

  detect the ball -> turn toward it (yaw) and drive into it (forward) -> push/dribble.

Detection uses YOLO (COCO "sports ball", class 32). An earlier shape-only detector
(HoughCircles) chased ANY round thing on the floor — chairs, vacuum drums — because
it can't tell a ball from furniture. YOLO knows what a ball is, so it ignores
furniture. Runs on CPU by default (~5 Hz, no GPU/VRAM use -> no risk of the rerun
GPU freeze); flip _DEVICE to "cuda" for a faster loop.

Skills:
  * find_ball()        -- perception only, never moves. Confirms the ball is seen.
  * dump_frame()       -- debug: save the current frame + the detected box.
  * play_football(...) -- the chase-and-push control loop (bounded by `seconds`).
"""

from __future__ import annotations

import math
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

_SPORTS_BALL_CLASS = 32  # COCO class id for "sports ball"
_CONF = 0.25  # YOLO confidence threshold for the ball
_YOLO_MODEL = "yolo11s.pt"  # "n" (nano) never proposed "sports ball" AT ALL (down to
# conf=0.02) for this ball's bold star pattern from some angles -- live-verified on a
# saved frame it completely missed. "s" (small, ~19MB) correctly scored it 0.389 on
# the SAME frame via the exact production (class-filtered) call. Still tiny/safe on
# the 4GB GPU (yolo11n itself only used ~0.5GB).
_DEVICE = "cuda"  # CPU was only ~5Hz -> too slow a control loop -> lost the ball
# during turns. CUDA -> ~20Hz+. (The earlier GPU freeze was the rerun pointcloud,
# not YOLO compute.)

# Real camera calibration (dimos/robot/unitree/go2/front_camera_720.yaml), model
# "equidistant" (fisheye): for this model r_px = f*theta, so the angle of a pixel
# off the optical axis is EXACTLY (pixel_offset / focal_length) radians -- no
# small-angle approximation needed. This makes bearing-to-target exact, not a
# tuned gain.
_CAM_FX = 797.4756164864929
_CAM_FY = 796.4872112769983
_CAM_CX = 643.5352167821186
# Distance is estimated via similar triangles from the ball's apparent size, which
# needs an assumed real diameter. 0.22m = a standard size-5 football (matches the
# ball in this robot's camera feed). If the real ball is a different size, tell me
# its diameter and this constant is the only thing to change -- bearing accuracy is
# unaffected either way (it only depends on the calibrated camera, not ball size).
_ASSUMED_BALL_DIAMETER_M = 0.22


class FootballSkillContainer(Module):
    """Vision-driven 'play football' behaviour for the Go2 (velocity interface)."""

    color_image: In[Image]
    _connection: GO2ConnectionSpec

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._frame_lock = threading.Lock()
        self._latest_frame = None
        self._latest_is_rgb: bool = True
        self._unsub = None
        self._yolo: Any = None
        self._model_lock = threading.Lock()
        # Shared "where did we last see it" hint, updated by EVERY successful
        # detection (any skill). Lets a fresh approach_ball call -- or a blind
        # recovery scan after a few missed frames -- search toward the side the
        # ball was actually last on, instead of guessing/always-one-fixed-direction
        # (live-confirmed bug: a fixed-direction scan overshot a ball that only
        # needed a small correction, pushing it out to the frame edge and losing it
        # entirely on the very first approach_ball attempt).
        self._last_ball_bearing_rad: float | None = None
        self._last_ball_ts: float = 0.0

    @rpc
    def start(self) -> None:
        super().start()
        _ = self.tf

        def on_frame(msg: Image) -> None:
            with self._frame_lock:
                self._latest_frame = msg.data
                self._latest_is_rgb = msg.format == ImageFormat.RGB

        self._unsub = self.color_image.subscribe(on_frame)

    @rpc
    def stop(self) -> None:
        if self._unsub is not None:
            try:
                self._unsub()
            except Exception:
                pass
            self._unsub = None
        super().stop()

    # ---- perception (YOLO sports-ball) -------------------------------------
    def _ensure_model(self) -> None:
        if self._yolo is not None:
            return
        with self._model_lock:
            if self._yolo is not None:
                return
            from ultralytics import YOLO

            from dimos.utils.data import get_data

            self._yolo = YOLO(get_data("models_yolo") / _YOLO_MODEL, task="detect")
            logger.info(f"Football: loaded YOLO ball detector on {_DEVICE}")

    def _detect_ball(self) -> dict[str, float] | None:
        """Return the largest detected sports ball as {cx, cy, r, w, h}, or None."""
        with self._frame_lock:
            frame = None if self._latest_frame is None else self._latest_frame.copy()
            is_rgb = self._latest_is_rgb
        if frame is None or frame.ndim < 3:
            return None
        self._ensure_model()
        h, w = frame.shape[:2]
        bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR) if is_rgb else frame
        res = self._yolo.predict(
            source=bgr, conf=_CONF, classes=[_SPORTS_BALL_CLASS], device=_DEVICE, verbose=False
        )[0]
        best: tuple[float, float, float, float] | None = None  # (area, cx, cy, r)
        for box in res.boxes:
            x1, y1, x2, y2 = (float(v) for v in box.xyxy[0].tolist())
            area = (x2 - x1) * (y2 - y1)
            if best is None or area > best[0]:
                best = (area, (x1 + x2) / 2, (y1 + y2) / 2, max(x2 - x1, y2 - y1) / 2)
        if best is None:
            return None
        ball = {"cx": best[1], "cy": best[2], "r": best[3], "w": float(w), "h": float(h)}
        self._last_ball_bearing_rad = (ball["cx"] - _CAM_CX) / _CAM_FX
        self._last_ball_ts = time.monotonic()
        return ball

    def _estimate_ball_position(self, ball: dict[str, float]) -> tuple[float, float]:
        """Estimate (distance_m, bearing_rad) to the ball from a detection box.

        bearing_rad: EXACT given the calibrated camera (equidistant/fisheye model:
        theta = pixel_offset_from_principal_point / focal_length). Positive = ball
        is to the right of the optical axis.
        distance_m: similar-triangles from focal length and _ASSUMED_BALL_DIAMETER_M.
        Scales linearly with that assumption if the real ball differs in size, but
        that only affects absolute distance -- since we re-detect and recompute
        after every move, a consistent scale error still converges correctly.
        """
        diameter_px = max(1.0, 2.0 * ball["r"])
        distance_m = (_ASSUMED_BALL_DIAMETER_M * _CAM_FX) / diameter_px
        bearing_rad = (ball["cx"] - _CAM_CX) / _CAM_FX
        return distance_m, bearing_rad

    @skill
    def find_ball(self) -> str:
        """Report whether a ball is currently in view, with an estimated distance and
        bearing. Does NOT move the robot.

        Detection is YOLO (COCO 'sports ball'), so it won't confuse furniture for a
        ball. Distance assumes a 0.22m (standard size-5) ball -- tell me the real
        diameter if it looks off. Use this to confirm the ball is seen before
        approach_ball / play_football.
        """
        if self._latest_frame is None:
            return "No camera frames yet (is the robot connected and streaming?)."
        ball = self._detect_ball()
        if ball is None:
            return "No ball in view."
        distance_m, bearing_rad = self._estimate_ball_position(ball)
        deg = math.degrees(bearing_rad)
        side = "straight ahead" if abs(deg) < 3 else (f"{abs(deg):.0f}° right" if deg > 0 else f"{abs(deg):.0f}° left")
        return f"Found a ball: ~{distance_m:.2f}m away, {side} (assumes a 0.22m ball)."

    @skill
    def dump_frame(self) -> str:
        """Debug: save the current camera frame + the detected ball box to
        /tmp/go2_football_debug, so detection can be inspected from a real image.
        """
        import os

        with self._frame_lock:
            frame = None if self._latest_frame is None else self._latest_frame.copy()
            is_rgb = self._latest_is_rgb
        if frame is None:
            return "No camera frame to dump."
        out_dir = "/tmp/go2_football_debug"
        os.makedirs(out_dir, exist_ok=True)
        bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR) if is_rgb else frame.copy()
        cv2.imwrite(f"{out_dir}/frame.jpg", bgr)
        ball = self._detect_ball()
        ann = bgr.copy()
        if ball is not None:
            c = (int(ball["cx"]), int(ball["cy"]))
            cv2.circle(ann, c, int(ball["r"]), (0, 255, 0), 3)
            cv2.circle(ann, c, 3, (0, 255, 0), -1)
        cv2.imwrite(f"{out_dir}/detected.jpg", ann)
        return (
            f"Saved frame + detection to {out_dir}. Detector says: "
            f"{('ball at %d,%d r=%d' % (ball['cx'], ball['cy'], ball['r'])) if ball else 'no ball'}."
        )

    # ---- behaviour ----------------------------------------------------------
    @skill
    def play_football(self, seconds: float = 30.0) -> str:
        """Chase the ball and push it (dribble) using the camera.

        Turns toward the ball and drives into it. Pure velocity control over the
        Go2's sport interface — it pushes/nudges the ball, it does not kick. Runs
        until `seconds` elapse. Call find_ball first to confirm the ball is seen.

        Args:
            seconds: how long to play (2-120).
        """
        seconds = min(max(2.0, float(seconds)), 120.0)
        SPORT = RTC_TOPIC["SPORT_MOD"]
        MAX_LIN, MAX_ANG = 0.35, 0.7  # m/s, rad/s — conservative for safety
        K_YAW = 1.3  # how hard to turn toward the ball (flip sign if it turns away)
        ALIGN = 0.30  # |offset| below which we drive forward at full speed
        SEARCH_YAW = 0.5  # spin speed when hunting for a lost ball
        CLOSE_R = 85.0  # ball radius (px) beyond which we're in contact/push range
        PUSH_HOLD = 1.2  # after losing a close ball, keep driving forward this long
        COAST_HOLD = 0.5  # on a brief detection flicker, coast on the last command this long
        dt = 0.05  # control loop period; actual rate is gated by detector latency

        def _move(x: float, y: float, z: float) -> None:
            # Tolerate transient send failures (the GO2Connection @rpc intermittently
            # hits a stub mid-session) — skip the bad command and keep playing.
            try:
                self._connection.publish_request(
                    SPORT, {"api_id": SPORT_CMD["Move"], "parameter": {"x": x, "y": y, "z": z}}
                )
            except Exception:  # noqa: BLE001
                pass

        def _stop() -> None:
            try:
                self._connection.publish_request(SPORT, {"api_id": SPORT_CMD["StopMove"]})
            except Exception:  # noqa: BLE001
                pass

        if self._latest_frame is None:
            return "No camera frames yet — is the robot connected and the camera streaming?"

        end = time.monotonic() + seconds
        ticks = seen = 0
        last_seen = 0.0
        last_offset = 0.0
        last_cmd = (0.0, 0.0)
        pushing_until = 0.0
        try:
            while time.monotonic() < end:
                now = time.monotonic()
                ticks += 1
                ball = self._detect_ball()
                if ball is not None:
                    seen += 1
                    last_seen = now
                    offset = (ball["cx"] - ball["w"] / 2) / (ball["w"] / 2)  # -1..1
                    last_offset = offset
                    yaw = max(-MAX_ANG, min(MAX_ANG, -K_YAW * offset))
                    if ball["r"] >= CLOSE_R and abs(offset) < ALIGN * 1.5:
                        fwd = MAX_LIN
                        pushing_until = now + PUSH_HOLD
                    else:
                        fwd = MAX_LIN if abs(offset) < ALIGN else 0.18
                    last_cmd = (fwd, yaw)
                    _move(fwd, 0.0, yaw)
                elif now < pushing_until:
                    _move(MAX_LIN, 0.0, 0.0)  # ball under chin -> push through
                elif now - last_seen < COAST_HOLD:
                    _move(last_cmd[0], 0.0, last_cmd[1])  # coast through a brief flicker
                else:
                    _move(0.0, 0.0, SEARCH_YAW if last_offset <= 0 else -SEARCH_YAW)
                time.sleep(dt)
            _stop()
        except Exception as e:  # noqa: BLE001
            _stop()
            logger.error(f"play_football failed: {e}")
            return "Football stopped (the connection may be down)."

        pct = 100.0 * seen / max(1, ticks)
        return (
            f"Played football for {seconds:.0f}s. Saw the ball in {seen}/{ticks} "
            f"frames (~{pct:.0f}% of the time)."
        )

    # ---- resilient low-level sends (shared) --------------------------------
    def _drive(self, x: float, y: float, z: float) -> None:
        """Send a sport Move velocity (x=fwd, y=left, z=yaw). Tolerates transient
        send failures so a single routing blip doesn't abort a behaviour."""
        try:
            self._connection.publish_request(
                RTC_TOPIC["SPORT_MOD"],
                {"api_id": SPORT_CMD["Move"], "parameter": {"x": x, "y": y, "z": z}},
            )
        except Exception:  # noqa: BLE001
            pass

    def _halt(self) -> None:
        try:
            self._connection.publish_request(RTC_TOPIC["SPORT_MOD"], {"api_id": SPORT_CMD["StopMove"]})
        except Exception:  # noqa: BLE001
            pass

    def _turn_by_rad(self, target_rad: float, max_seconds: float = 6.0) -> float:
        """Closed-loop turn by a signed angle (rad; +=left/CCW, matching the Move
        API's z convention) using odometry yaw feedback -- same proven approach as
        the verified unitree_skill_container.move() turn (349 of a requested 360).
        Returns the actual measured turn achieved (rad).
        """
        if abs(target_rad) < 1e-3:
            return 0.0
        SPEED = 0.6  # rad/s
        LEAD = math.radians(4.0)  # stop a bit early to cancel stop-latency overshoot
        z = math.copysign(SPEED, target_rad)
        tf0 = self.tf.get("world", "base_link")
        if tf0 is None:
            return 0.0
        prev = tf0.to_pose().orientation.to_euler().yaw
        acc = 0.0
        deadline = time.monotonic() + min(max_seconds, abs(target_rad) / (SPEED * 0.4) + 3.0)
        while abs(acc) < max(0.0, abs(target_rad) - LEAD) and time.monotonic() < deadline:
            self._drive(0.0, 0.0, z)
            time.sleep(0.05)
            tf = self.tf.get("world", "base_link")
            if tf is not None:
                cur = tf.to_pose().orientation.to_euler().yaw
                d = cur - prev
                while d > math.pi:
                    d -= 2 * math.pi
                while d < -math.pi:
                    d += 2 * math.pi
                acc += d
                prev = cur
        self._halt()
        return acc

    def _walk_by_m(self, target_m: float, max_seconds: float = 6.0) -> float:
        """Closed-loop forward walk by target_m (>=0) using odometry position
        feedback. Returns the actual measured distance traveled (m)."""
        if target_m <= 0.02:
            return 0.0
        SPEED = 0.28  # m/s -- gentle; this is fine positioning, not a long walk
        LEAD = 0.05  # stop 5cm early to cancel stop-latency creep
        tf0 = self.tf.get("world", "base_link")
        if tf0 is None:
            return 0.0
        p0 = tf0.to_pose().position
        x0, y0 = p0.x, p0.y
        traveled = 0.0
        deadline = time.monotonic() + min(max_seconds, target_m / (SPEED * 0.4) + 3.0)
        while traveled < max(0.0, target_m - LEAD) and time.monotonic() < deadline:
            self._drive(SPEED, 0.0, 0.0)
            time.sleep(0.05)
            tf = self.tf.get("world", "base_link")
            if tf is not None:
                p = tf.to_pose().position
                traveled = math.hypot(p.x - x0, p.y - y0)
        self._halt()
        return traveled

    @skill
    def approach_ball(self, timeout: float = 45.0) -> str:
        """Walk up to the ball using a real position estimate, and arrive with it at
        the robot's feet, ready to kick.

        Each cycle: stop, detect the ball, compute an exact bearing (from the
        calibrated camera) and an estimated distance (from its apparent size),
        then turn or step accordingly -- closed-loop on odometry, not open-loop
        timing. Once close and centered, it takes one final CALCULATED step (using
        the last real distance estimate) and explicitly CHECKS AGAIN: the ball is
        expected to vanish from the camera at this range (it's directly under the
        lens, not actually gone), so "not visible" only counts as arrival if the
        geometry says it should be right there -- otherwise it keeps trying.

        Args:
            timeout: max seconds to spend getting there (2-90).
        """
        timeout = min(max(2.0, float(timeout)), 90.0)
        ARRIVE_DISTANCE = 0.32  # enter the final closing phase within this range (m)
        BEARING_DEADBAND = math.radians(6.0)
        MAX_STEP_M = 0.28  # SMALL steps -- re-checking often is how we discover the
        # real vanish-from-view distance empirically, instead of assuming one and
        # overshooting it blind (a single 0.5m step from 1.18m away lost the ball
        # well before the old 0.32m assumption -- the true FOV cutoff is farther out
        # than that guess, and varies with camera angle/ball size, so don't hardcode it).
        MAX_BLIND_STEP_M = 0.5  # safety cap on the final calculated step
        STANDOFF_M = 0.14  # stop this far from the ball's NEAR EDGE, not its center
        SETTLE = 0.35  # pause after stopping so the detected frame is sharp
        MAX_SCAN_TRIES = 5
        MAX_ITERS = 40
        ball_radius_m = _ASSUMED_BALL_DIAMETER_M / 2.0

        if self._latest_frame is None:
            return "No camera frames yet — is the camera streaming?"

        def _detect_debounced(tries: int = 3, gap: float = 0.15) -> dict[str, float] | None:
            # YOLO detection flickers frame-to-frame (live-observed: dump_frame found
            # the ball on one call immediately after find_ball reported none, same
            # detector, split seconds apart). A single missed frame shouldn't mean
            # "lost" -- retry a couple of quick times before trusting a None.
            for i in range(tries):
                ball = self._detect_ball()
                if ball is not None:
                    return ball
                if i < tries - 1:
                    time.sleep(gap)
            return None

        def _final_close(last_distance_m: float) -> str:
            # One more real look first -- maybe still visible, or the estimate was off.
            self._halt()
            time.sleep(SETTLE)
            ball = _detect_debounced()
            if ball is not None:
                d, b = self._estimate_ball_position(ball)
                if d > ARRIVE_DISTANCE or abs(b) > BEARING_DEADBAND:
                    return ""  # not actually ready yet -- caller keeps servoing
                last_distance_m = d
            remaining = max(0.0, min(last_distance_m - ball_radius_m - STANDOFF_M, MAX_BLIND_STEP_M))
            moved = self._walk_by_m(remaining) if remaining > 0.02 else 0.0
            # Check again (possibly more than once) rather than assuming.
            still_visible = None
            for _ in range(2):
                time.sleep(0.3)
                recheck = self._detect_ball()
                if recheck is None:
                    still_visible = False
                    break
                still_visible = True
                d2, _b2 = self._estimate_ball_position(recheck)
                extra = max(0.0, min(d2 - ball_radius_m - STANDOFF_M, 0.2))
                if extra > 0.02:
                    self._walk_by_m(extra)
            if still_visible is False or still_visible is None:
                return (
                    f"Arrived — moved the final calculated {moved:.2f}m; the ball is no "
                    f"longer visible now, which is expected this close (it's directly "
                    f"under my camera, not gone). Ready to kick."
                )
            return (
                f"Got close (moved {moved:.2f}m total) but the ball is still visible — "
                f"treating this as arrived since I'm within the calculated contact range."
            )

        end = time.monotonic() + timeout
        last_known_distance: float | None = None  # from the most recent successful detection
        last_step_taken = 0.0  # forward distance just walked (0 if we just turned/scanned)
        scan_tries = 0
        iters = 0
        try:
            while time.monotonic() < end and iters < MAX_ITERS:
                iters += 1
                self._halt()
                time.sleep(SETTLE)
                ball = _detect_debounced()
                if ball is None:
                    if last_known_distance is not None and last_step_taken > 0.01:
                        # We just walked toward a ball that was tracked and centered a
                        # moment ago, and now it's gone -- almost certainly the camera's
                        # FOV cutoff (it's close, under the lens), not a real loss. Close
                        # the geometrically-remaining gap (measured distance minus the
                        # step we already took) and verify, rather than assuming.
                        remaining_before = max(0.0, last_known_distance - last_step_taken)
                        result = _final_close(remaining_before)
                        if result:
                            return result
                        last_known_distance = None
                        last_step_taken = 0.0
                        continue
                    scan_tries += 1
                    if scan_tries > MAX_SCAN_TRIES:
                        self._halt()
                        return "Lost the ball — it's not in view. Please put it back in front of me."
                    # Scan TOWARD wherever the ball was last actually seen (any recent
                    # detection, even from a different call), not a fixed direction --
                    # a directionless scan previously overshot a ball that only needed
                    # a small correction, sweeping it out to the frame edge and losing
                    # it entirely within the scan budget.
                    hint_fresh = time.monotonic() - self._last_ball_ts < 5.0
                    if hint_fresh and self._last_ball_bearing_rad is not None:
                        scan_z = -0.35 if self._last_ball_bearing_rad > 0 else 0.35
                    else:
                        scan_z = 0.35
                    self._drive(0.0, 0.0, scan_z)
                    time.sleep(0.2)
                    self._halt()
                    last_step_taken = 0.0
                    continue
                scan_tries = 0
                distance_m, bearing_rad = self._estimate_ball_position(ball)
                last_known_distance = distance_m
                last_step_taken = 0.0
                if distance_m <= ARRIVE_DISTANCE and abs(bearing_rad) <= BEARING_DEADBAND:
                    result = _final_close(distance_m)
                    if result:
                        return result
                    continue
                if abs(bearing_rad) > BEARING_DEADBAND:
                    # Turn toward it: bearing>0 (ball right) needs a right turn, which is
                    # NEGATIVE z on the Move API (verified: this sign correctly centered a
                    # ball that was "83% right" earlier this session).
                    self._turn_by_rad(-bearing_rad)
                else:
                    step = min(MAX_STEP_M, max(0.10, distance_m - ARRIVE_DISTANCE))
                    self._walk_by_m(step)
                    last_step_taken = step
            self._halt()
            return "Couldn't reach the ball in time (try placing it closer or in clearer view)."
        except Exception as e:  # noqa: BLE001
            self._halt()
            logger.error(f"approach_ball failed: {e}")
            return "Approach stopped (the connection may be down)."

    @skill
    def kick_ball(self, direction: str = "right", seconds: float = 0.7) -> str:
        """'Kick' the ball left or right. The Go2 Air has no low-level leg control, so
        this is a quick forward lunge angled to the side that drives THROUGH the ball
        and knocks it that way. Best called right after approach_ball.

        SAFETY: the robot cannot see walls/furniture — make sure there is ~1 m of
        clear space on the side it will kick toward (it travels ~0.4 m that way).

        Args:
            direction: 'right' or 'left'.
            seconds: lunge duration (0.3-3.0). Default is a short punt, not a charge.
        """
        seconds = min(max(0.3, float(seconds)), 3.0)
        right = not direction.lower().startswith("l")
        strafe = -0.50 if right else 0.50  # y<0 = strafe right
        end = time.monotonic() + seconds
        try:
            while time.monotonic() < end:
                self._drive(0.30, strafe, 0.0)  # forward + sideways = sweep the ball aside
                time.sleep(0.05)
            self._halt()
        except Exception as e:  # noqa: BLE001
            self._halt()
            logger.error(f"kick_ball failed: {e}")
            return "Kick stopped (the connection may be down)."
        return f"Kicked the ball to the {'right' if right else 'left'}."

    @skill
    def penalty_kick(self, direction: str = "right", timeout: float = 45.0) -> str:
        """One complete penalty attempt: walk up to the ball and immediately kick it
        in the given direction. Deterministic single call (approach + kick), so use
        this for penalty-shootout rounds instead of separate approach/kick calls.

        The kick only fires if the approach actually confirms arrival at the ball —
        if the ball is lost or unreachable, NO kick happens and the returned message
        says why (ask the referee to reset the ball, then retry).

        Args:
            direction: 'right' or 'left' — which way to send the ball.
            timeout: max seconds for the walk-up (2-90).
        """
        res = self.approach_ball(timeout=timeout)
        if "Arrived" not in res and "arrived" not in res:
            return f"Penalty NOT taken — {res}"
        kick = self.kick_ball(direction=direction)
        return f"Penalty taken: walked to the ball and kicked {direction}. ({kick})"

    @skill
    def dribble_circle(self, seconds: float = 40.0, turn: str = "right") -> str:
        """Herd the ball along a curving/circular path (velocity-level, the most the Air
        allows — NOT leg control). Repeats a robust STOP-LOOK-MOVE cycle: stop, find the
        ball (detect while stationary = reliable), line up behind it, nudge it forward,
        then add a small turn so the ball's path curves. Expect imprecise nudge-herding,
        re-acquiring the ball after each nudge (the forward camera can't see it underfoot).

        Args:
            seconds: total run time (5-120).
            turn: 'right' or 'left' — direction the circle curves.
        """
        seconds = min(max(5.0, float(seconds)), 120.0)
        right = not turn.lower().startswith("l")
        bias = -1.0 if right else 1.0  # yaw<0 = turn right
        CENTER = 0.22       # |offset| to count as lined up behind the ball
        NUDGE_R = 75.0      # ball radius (px): below = approach (far), at/above = close enough to nudge
        TURN_RATE = 0.6
        SETTLE = 0.35
        if self._latest_frame is None:
            return "No camera frames yet — is the camera streaming?"
        end = time.monotonic() + seconds
        misses = 0
        nudges = 0
        turn_sign = 1.0
        prev_offset: float | None = None
        try:
            while time.monotonic() < end:
                self._halt()
                time.sleep(SETTLE)
                ball = self._detect_ball()
                if ball is None:
                    misses += 1
                    if misses >= 5:
                        self._halt()
                        return f"Lost the ball after {nudges} nudges — reposition it in front of me."
                    self._drive(0.0, 0.0, TURN_RATE * bias)  # scan in the circle direction
                    time.sleep(0.3)
                    self._halt()
                    continue
                misses = 0
                offset = (ball["cx"] - ball["w"] / 2) / (ball["w"] / 2)
                if prev_offset is not None and abs(offset) > abs(prev_offset) + 0.06:
                    turn_sign *= -1.0  # last centering turn went the wrong way -> flip
                if abs(offset) > CENTER:
                    # Turn to line up behind the ball.
                    base = -1.0 if offset > 0 else 1.0
                    self._drive(0.0, 0.0, TURN_RATE * base * turn_sign)
                    time.sleep(min(0.6, 0.2 + 0.5 * abs(offset)))
                    self._halt()
                    prev_offset = offset
                elif ball["r"] < NUDGE_R:
                    # Lined up but far -> step forward to close in.
                    self._drive(0.30, 0.0, 0.0)
                    time.sleep(0.5)
                    self._halt()
                    prev_offset = None
                else:
                    # Lined up and close -> NUDGE the ball forward, then add the circle-turn
                    # so the next nudge strikes it at an angle and the path curves.
                    self._drive(0.35, 0.0, 0.0)
                    time.sleep(0.5)
                    self._halt()
                    nudges += 1
                    self._drive(0.0, 0.0, TURN_RATE * bias)
                    time.sleep(0.4)
                    self._halt()
                    prev_offset = None
            self._halt()
            return f"Dribbled the ball {nudges} nudges over {seconds:.0f}s, curving {'right' if right else 'left'}."
        except Exception as e:  # noqa: BLE001
            self._halt()
            logger.error(f"dribble_circle failed: {e}")
            return "Dribble stopped (the connection may be down)."
