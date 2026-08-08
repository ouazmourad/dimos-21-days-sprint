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

"""Autonomous "find my lost keys" patrol for the Go2 Air.

The dog patrols the floor, looks for keys with a bounding-box detector, walks up to
each candidate, then does a CLOSE INSPECTION — it backs off slightly and uses
`StandDown` to drop its body so the (forward-facing, high-mounted) camera can
actually see the floor right in front of its feet — re-checks, and only then
announces a find out loud.

Why a VLM detector and not COCO YOLO: COCO's 80 classes contain no "keys" class, so
plain YOLO11 cannot see keys at all. This uses the configured `detection_model`
(set `detection_model=openai` in .env -> gpt-4o) whose `query_detections()` returns
pixel bounding boxes for ARBITRARY text — a true open-vocabulary box detector, with
no local GPU/VRAM cost. A local open-vocab alternative (YOLOe, text-prompted) exists
in the repo (`dimos/perception/detection/detectors/yoloe.py`) and can be swapped in
here later; it needs its MobileCLIP text encoder on disk.

Movement is VELOCITY-level (sport `Move`), not nav goals: the Air has no usable
LiDAR costmap, so `relative_move`/`navigate_with_text` fail ("No path found").
Patrol uses a stop-look-move cycle (detect only while stationary => sharp frames).
OBSTACLES: the robot's onboard OBSTACLES_AVOID is requested but proved unreliable on
the Air (the dog walked into furniture), so every forward step is additionally gated on
our OWN LiDAR check — the world-frame point cloud is rotated into the robot frame and
the corridor ahead is measured; if anything is within _STOP_DIST_M the robot turns away
instead of pushing on. `check_clearance()` exposes that measurement.

Skills:
  * look_for_keys()   -- scan the current view only. Never moves. Use to test.
  * check_clearance() -- report LiDAR free space ahead. Never moves. Verifies obstacle gating.
  * inspect_below()   -- do just the StandDown close-inspection here. Never patrols.
  * find_my_keys()    -- start the autonomous patrol (background); speaks on find.
  * stop_searching()  -- stop the patrol.
  * search_status()   -- how the search is going.
"""

from __future__ import annotations

import math
import threading
import time
from typing import Any

import numpy as np

from unitree_webrtc_connect.constants import RTC_TOPIC, SPORT_CMD

from dimos.agents.annotation import skill
from dimos.agents.skills.speak_skill_spec import SpeakSkillSpec
from dimos.core.core import rpc
from dimos.core.module import Module
from dimos.core.stream import In, Out
from dimos.models.vl.create import create
from dimos.msgs.sensor_msgs.Image import Image
from dimos.msgs.sensor_msgs.PointCloud2 import PointCloud2
from dimos_lcm.std_msgs import Bool  # must match WavefrontFrontierExplorer's type
from dimos.robot.unitree.go2.connection_spec import GO2ConnectionSpec
from dimos.utils.logging_config import setup_logger

logger = setup_logger()

# What to ask the detector for. Kept broad (keys are small and easily confused with
# coins/screws/jewellery) — the StandDown close-inspection is what rejects the noise.
_KEYS_QUERY = "keys, a key ring, a bunch of keys, a keychain lying on the floor"

# Yes/no confirmation prompt used up close, after StandDown.
_CONFIRM_PROMPT = (
    "Look at the floor in this image. Are there KEYS (a key, key ring, keychain, or "
    "bunch of keys) clearly visible? Answer with exactly YES or NO on the first line, "
    "then one short sentence describing what you actually see."
)

_CLOSE_CY_FRAC = 0.72  # bbox center below this fraction of frame height == close to the feet
_CENTER_TOL = 0.20  # |x offset| within this == lined up with the candidate
_MAX_FRAME_AGE_S = 3.0  # older than this == the video feed is frozen/dead, refuse to use it

# --- obstacle gating (the robot's onboard OBSTACLES_AVOID is not trustworthy on the
# Air: it silently did nothing and the dog walked into furniture, so we gate forward
# motion on the LiDAR point cloud ourselves) ---
_STOP_DIST_M = 0.65  # refuse to step forward if something is closer than this ahead
_CORRIDOR_HALF_W_M = 0.32  # half-width of the corridor we care about (robot ~0.31 m wide)
_LOOKAHEAD_M = 1.60  # ignore points further ahead than this
_MIN_OBST_H_M = 0.08  # ignore points below this height (floor) ...
_MAX_OBST_H_M = 1.10  # ... and above this (ceiling / overhangs it walks under)
_MAX_CLOUD_AGE_S = 2.0  # older than this == lidar stale, treat clearance as unknown

# --- exploration bootstrap ---
# Frontier exploration needs a map that already contains reachable frontiers. From a
# parked spot (e.g. facing furniture) the explorer just logs "No frontier found ...
# Retrying in 2 seconds" forever and the robot never moves. When that happens we open
# up unknown space ourselves with a small lidar-gated manoeuvre, then re-arm
# exploration — the automated version of driving it a few metres by hand first.
_BOOTSTRAP_IDLE_S = 20.0  # no pose change for this long while exploring => bootstrap
_BOOTSTRAP_MOVE_EPS_M = 0.12  # translation that counts as "it moved"
_BOOTSTRAP_YAW_EPS_RAD = 0.25  # rotation that counts as "it moved" (~14 deg)


class KeyFinderSkillContainer(Module):
    """Autonomous key-finding patrol (detect -> approach -> stand-down verify -> alert)."""

    color_image: In[Image]
    lidar: In[PointCloud2]
    # Optional: present only in blueprints that include the navigation/exploration
    # stack (VoxelGridMapper -> CostMapper -> WavefrontFrontierExplorer -> A*).
    # Publishing here is exactly what the Command Center's "Start Exploration"
    # button does, and it covers a house far better than a hand-rolled sweep.
    explore_cmd: Out[Bool]
    stop_explore_cmd: Out[Bool]
    _connection: GO2ConnectionSpec
    _speak_skill: SpeakSkillSpec

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._frame_lock = threading.Lock()
        self._latest: Image | None = None
        self._latest_at: float = 0.0  # monotonic arrival time of self._latest
        self._frames_seen = 0
        self._unsub = None
        self._cloud_lock = threading.Lock()
        self._cloud: np.ndarray | None = None  # (N,3) world-frame points
        self._cloud_at: float = 0.0
        self._clouds_seen = 0
        self._blocked_stops = 0
        self._bootstraps = 0
        self._unsub_lidar = None
        self._vl_model: Any = None
        self._model_lock = threading.Lock()
        # search state
        self._thread: threading.Thread | None = None
        self._stop_flag = threading.Event()
        self._state = "idle"
        self._scans = 0
        self._candidates = 0
        self._rejected = 0
        self._found_note: str | None = None

    @rpc
    def start(self) -> None:
        super().start()
        _ = self.tf

        def on_frame(msg: Image) -> None:
            with self._frame_lock:
                self._latest = msg
                self._latest_at = time.monotonic()
                self._frames_seen += 1

        self._unsub = self.color_image.subscribe(on_frame)

        def on_cloud(msg: PointCloud2) -> None:
            try:
                pts = msg.points_f32()
            except Exception:  # noqa: BLE001
                return
            with self._cloud_lock:
                self._cloud = pts
                self._cloud_at = time.monotonic()
                self._clouds_seen += 1

        self._unsub_lidar = self.lidar.subscribe(on_cloud)

    @rpc
    def stop(self) -> None:
        self._stop_flag.set()
        if self._unsub is not None:
            try:
                self._unsub()
            except Exception:
                pass
            self._unsub = None
        if self._unsub_lidar is not None:
            try:
                self._unsub_lidar()
            except Exception:
                pass
            self._unsub_lidar = None
        if self._vl_model is not None:
            try:
                self._vl_model.stop()
            except Exception:
                pass
        super().stop()

    # ---- plumbing -----------------------------------------------------------
    def _frame(self) -> Image | None:
        """Latest camera frame, or None if the feed is stale.

        CRITICAL: the Go2's WebRTC video track can fail to attach (an `accept_track`
        error) or freeze mid-run. Without this check the module happily re-analyses one
        frozen frame forever — it "sees" a scene from minutes ago and reports nonsense
        (e.g. the same view after the robot has turned 360 degrees). Never trust an old
        frame; report the feed as dead instead.
        """
        with self._frame_lock:
            if self._latest is None:
                return None
            if time.monotonic() - self._latest_at > _MAX_FRAME_AGE_S:
                return None
            return self._latest

    def _feed_note(self) -> str:
        with self._frame_lock:
            age = time.monotonic() - self._latest_at if self._latest is not None else -1.0
            seen = self._frames_seen
        if seen == 0:
            return "no camera frames have EVER arrived (WebRTC video track never attached)"
        return f"camera feed is stale — last frame {age:.0f}s ago ({seen} frames total, then it froze)"

    def _forward_clearance(self) -> float | None:
        """Distance (m) to the nearest obstacle in the corridor straight ahead.

        Returns None when we genuinely don't know (no/stale lidar or no robot pose).
        The LiDAR cloud is published in the WORLD frame, so points are rotated into the
        robot's frame using the world->base_link transform before filtering.
        """
        with self._cloud_lock:
            pts = None if self._cloud is None else self._cloud
            age = time.monotonic() - self._cloud_at
        if pts is None or len(pts) == 0 or age > _MAX_CLOUD_AGE_S:
            return None
        tf = self.tf.get("world", "base_link")
        if tf is None:
            return None
        pose = tf.to_pose()
        x0, y0, z0 = pose.position.x, pose.position.y, pose.position.z
        yaw = pose.orientation.to_euler().yaw
        dx = pts[:, 0] - x0
        dy = pts[:, 1] - y0
        c, sn = math.cos(yaw), math.sin(yaw)
        fwd = dx * c + dy * sn  # along the robot's heading
        left = -dx * sn + dy * c
        height = pts[:, 2] - (z0 - 0.30)  # approx height above the floor under the robot
        mask = (
            (fwd > 0.12)
            & (fwd < _LOOKAHEAD_M)
            & (np.abs(left) < _CORRIDOR_HALF_W_M)
            & (height > _MIN_OBST_H_M)
            & (height < _MAX_OBST_H_M)
        )
        if not bool(mask.any()):
            return float(_LOOKAHEAD_M)  # corridor is clear as far as we look
        return float(np.min(fwd[mask]))

    def _path_blocked(self) -> bool:
        """True if something is too close ahead. Unknown clearance => not blocked (but warned)."""
        d = self._forward_clearance()
        if d is None:
            return False
        return d < _STOP_DIST_M

    def _exploration_available(self) -> bool:
        """True when this blueprint wires the frontier-exploration stack."""
        try:
            return self.explore_cmd.transport is not None
        except Exception:  # noqa: BLE001
            return False

    def _explore(self, on: bool) -> None:
        """Start/stop autonomous frontier exploration (A* replanning drives the robot)."""
        try:
            if on:
                self.explore_cmd.publish(Bool(data=True))
            else:
                self.stop_explore_cmd.publish(Bool(data=True))
            logger.info(f"KeyFinder: exploration -> {'START' if on else 'STOP'}")
        except Exception as e:  # noqa: BLE001
            logger.warning(f"KeyFinder: exploration command failed: {e}")

    def _pose(self) -> tuple[float, float, float] | None:
        """Robot (x, y, yaw) in the world frame, or None if tf isn't available yet."""
        tf = self.tf.get("world", "base_link")
        if tf is None:
            return None
        p = tf.to_pose()
        return (p.position.x, p.position.y, p.orientation.to_euler().yaw)

    def _bootstrap_map(self) -> None:
        """Open unknown space so the frontier explorer has a frontier to chase.

        Rotation is always safe (the collision guard only vetoes forward motion), so we
        turn first; then step forward if the lidar corridor is clear, otherwise reverse
        a little and turn again. Either way the LiDAR sweeps new territory and the
        costmap grows, which is what the explorer needs.
        """
        self._bootstraps += 1
        logger.info(
            f"KeyFinder: exploration produced no motion — bootstrapping the map "
            f"(#{self._bootstraps})"
        )
        self._turn(0.7, 1.1)  # ~45 deg
        if not self._step(0.28, 0.9):  # blocked ahead?
            self._step(-0.20, 0.6)  # reverse (never gated) to make room
            self._turn(0.7, 1.1)

    def _model(self) -> Any:
        if self._vl_model is None:
            with self._model_lock:
                if self._vl_model is None:
                    m = create(self.config.g.detection_model)
                    m.start()
                    self._vl_model = m
                    logger.info(f"KeyFinder: VLM ready ({self.config.g.detection_model})")
        return self._vl_model

    def _sport(self, api_id: int, parameter: dict[str, Any] | None = None) -> None:
        """Send a sport command, tolerating transient send failures."""
        payload: dict[str, Any] = {"api_id": api_id}
        if parameter is not None:
            payload["parameter"] = parameter
        try:
            self._connection.publish_request(RTC_TOPIC["SPORT_MOD"], payload)
        except Exception:  # noqa: BLE001
            pass

    def _drive(self, x: float, y: float, z: float) -> None:
        self._sport(SPORT_CMD["Move"], {"x": x, "y": y, "z": z})

    def _halt(self) -> None:
        self._sport(SPORT_CMD["StopMove"])

    def _obstacle_avoidance(self, enabled: bool) -> None:
        """Ask the robot for its onboard avoidance. Best-effort ONLY — on the Air this
        proved unreliable (the dog still walked into furniture), which is why forward
        motion is additionally gated on our own lidar clearance check. Failures are
        logged rather than swallowed so we can see whether the robot accepted it."""
        try:
            self._connection.publish_request(
                RTC_TOPIC["OBSTACLES_AVOID"],
                {"api_id": 1001, "parameter": {"enable": int(enabled)}},
            )
            logger.info(f"KeyFinder: onboard obstacle avoidance -> {enabled}")
        except Exception as e:  # noqa: BLE001
            logger.warning(f"KeyFinder: onboard obstacle avoidance request failed: {e}")

    # ---- perception ---------------------------------------------------------
    def _scan(self, image: Image) -> list[dict[str, float]]:
        """Return key candidates in `image` as [{cx,cy,x1,y1,x2,y2,w,h}] (biggest first)."""
        dets = self._model().query_detections(image, _KEYS_QUERY)
        h, w = image.data.shape[:2]
        out: list[dict[str, float]] = []
        for d in dets.detections:
            x1, y1, x2, y2 = (float(v) for v in d.bbox)
            out.append(
                {
                    "cx": (x1 + x2) / 2,
                    "cy": (y1 + y2) / 2,
                    "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                    "area": max(0.0, (x2 - x1)) * max(0.0, (y2 - y1)),
                    "w": float(w), "h": float(h),
                }
            )
        out.sort(key=lambda c: c["area"], reverse=True)
        return out

    def _confirm(self, image: Image) -> tuple[bool, str]:
        """Ask the VLM to confirm keys are visible. Returns (is_keys, description)."""
        answer = (self._model().query(image, _CONFIRM_PROMPT) or "").strip()
        first = answer.splitlines()[0].strip().upper() if answer else ""
        yes = first.startswith("YES") or first == "Y"
        return yes, answer.replace("\n", " ")[:300]

    # ---- skills -------------------------------------------------------------
    @skill
    def look_for_keys(self) -> str:
        """Scan what the camera sees RIGHT NOW for keys. Does NOT move the robot.

        Use this to check the detector before starting a patrol.
        """
        img = self._frame()
        if img is None:
            return f"Can't look — {self._feed_note()}. Restart the robot connection."
        try:
            cands = self._scan(img)
        except Exception as e:  # noqa: BLE001
            logger.error(f"look_for_keys failed: {e}")
            return "Couldn't run the key detector (check the vision model / API key)."
        if not cands:
            return "No key-like objects in view."
        c = cands[0]
        off = (c["cx"] - c["w"] / 2) / (c["w"] / 2)
        side = "centered" if abs(off) < 0.15 else ("right" if off > 0 else "left")
        near = "close to my feet" if c["cy"] / c["h"] > _CLOSE_CY_FRAC else "further ahead"
        return (
            f"{len(cands)} key-like candidate(s). Best: {abs(off) * 100:.0f}% to the {side}, "
            f"{near} (box {c['x2'] - c['x1']:.0f}x{c['y2'] - c['y1']:.0f}px). "
            "Not confirmed yet — inspect_below() verifies up close."
        )

    @skill
    def check_clearance(self) -> str:
        """Report how much free space the LiDAR sees straight ahead. Does NOT move.

        Use this to confirm obstacle sensing works before letting the robot patrol.
        """
        d = self._forward_clearance()
        with self._cloud_lock:
            clouds = self._clouds_seen
            age = time.monotonic() - self._cloud_at if self._cloud is not None else -1.0
            n = 0 if self._cloud is None else len(self._cloud)
        if d is None:
            if clouds == 0:
                return "No LiDAR data has EVER arrived — obstacle gating is BLIND (movement not protected)."
            if age > _MAX_CLOUD_AGE_S:
                return f"LiDAR is stale ({age:.1f}s old, {clouds} clouds seen) — clearance unknown."
            return "No robot pose (tf world->base_link) yet — clearance unknown."
        verdict = "BLOCKED" if d < _STOP_DIST_M else "clear"
        return (
            f"{verdict}: nearest obstacle ahead {d:.2f} m "
            f"(stop threshold {_STOP_DIST_M} m, corridor +/-{_CORRIDOR_HALF_W_M} m, "
            f"{n} lidar points, {age:.1f}s old)."
        )

    @skill
    def inspect_below(self) -> str:
        """Do the close-inspection: back off a little, StandDown so the camera can see the
        floor right in front of the feet, double-check for keys, then stand back up.

        The forward camera sits high and cannot see the ground a few centimetres ahead;
        dropping the body brings that patch into view.
        """
        img0 = self._frame()
        if img0 is None:
            return f"Can't inspect — {self._feed_note()}."
        try:
            # Back off slightly so ground near the feet enters the field of view.
            for _ in range(6):
                self._drive(-0.18, 0.0, 0.0)
                time.sleep(0.1)
            self._halt()
            time.sleep(0.3)
            # Drop the body to look at the floor closely.
            self._sport(SPORT_CMD["StandDown"])
            time.sleep(2.5)
            img = self._frame()
            verdict, note = (False, "no frame")
            if img is not None:
                verdict, note = self._confirm(img)
            # Back up to a normal stance.
            self._sport(SPORT_CMD["RecoveryStand"])
            time.sleep(1.5)
            self._sport(SPORT_CMD["BalanceStand"])
        except Exception as e:  # noqa: BLE001
            logger.error(f"inspect_below failed: {e}")
            try:
                self._sport(SPORT_CMD["RecoveryStand"])
            except Exception:
                pass
            return "Close inspection failed (the connection may be down)."
        return f"{'CONFIRMED: these are keys.' if verdict else 'Not keys.'} ({note})"

    @skill
    def find_my_keys(self, minutes: float = 5.0) -> str:
        """Patrol the floor autonomously looking for lost keys, and say so out loud when
        found. Runs in the background — poll with search_status(), halt with stop_searching().

        The robot walks in a stop-look-move cycle with its own obstacle avoidance on,
        scans the floor for key-like objects, walks up to any candidate, drops into
        StandDown to verify it up close, and announces a confirmed find.

        Args:
            minutes: how long to search before giving up (0.5-30).
        """
        if self._thread is not None and self._thread.is_alive():
            return "Already searching — use stop_searching() first."
        if self._frame() is None:
            return f"Can't start the search — {self._feed_note()}."
        minutes = min(max(0.5, float(minutes)), 30.0)
        self._stop_flag.clear()
        self._state = "searching"
        self._scans = self._candidates = self._rejected = 0
        self._found_note = None
        self._thread = threading.Thread(
            target=self._search_loop, args=(minutes * 60.0,), daemon=True
        )
        self._thread.start()
        return (
            f"Searching for your keys for up to {minutes:.0f} minute(s). I'll walk around, "
            "check the floor, and tell you out loud if I find them."
        )

    @skill
    def stop_searching(self) -> str:
        """Stop the key-finding patrol."""
        self._stop_flag.set()
        if self._thread is not None:
            self._thread.join(timeout=8.0)
        self._halt()
        self._state = "idle"
        return "Stopped searching."

    @skill
    def search_status(self) -> str:
        """Report how the key search is going (state, scans, candidates checked)."""
        alive = self._thread is not None and self._thread.is_alive()
        d = self._forward_clearance()
        with self._cloud_lock:
            clouds = self._clouds_seen
        clear = "lidar: NO DATA" if d is None else f"clear ahead: {d:.2f}m"
        base = (
            f"state={self._state}{' (running)' if alive else ''}, scans={self._scans}, "
            f"candidates={self._candidates}, rejected={self._rejected}, {clear} "
            f"(clouds={clouds}, blocked_stops={self._blocked_stops}, bootstraps={self._bootstraps})"
        )
        return base + (f", FOUND: {self._found_note}" if self._found_note else "")

    # ---- the autonomous search ---------------------------------------------
    def _search_loop(self, budget_s: float) -> None:
        end = time.monotonic() + budget_s
        self._obstacle_avoidance(True)
        legs = 0
        # Prefer the real exploration stack (frontier goals + A* replanning) when the
        # blueprint provides it — it covers a house properly. The manual step/turn
        # sweep below is only a fallback for lightweight blueprints without nav.
        exploring = self._exploration_available()
        if exploring:
            self._explore(True)
        last_pose = self._pose()
        last_move_at = time.monotonic()
        try:
            while not self._stop_flag.is_set() and time.monotonic() < end:
                # --- LOOK (stationary => sharp frame) ---
                self._halt()
                time.sleep(0.4)
                img = self._frame()
                if img is None:
                    time.sleep(0.5)
                    continue
                self._scans += 1
                try:
                    cands = self._scan(img)
                except Exception as e:  # noqa: BLE001
                    logger.error(f"key scan failed: {e}")
                    cands = []

                if cands:
                    self._candidates += 1
                    self._state = "checking a candidate"
                    if exploring:
                        self._explore(False)  # pause exploration; we drive now
                        time.sleep(0.4)
                    if self._approach(cands[0]):
                        verdict, note = self._verify_here()
                        if verdict:
                            self._state = "found"
                            self._found_note = note
                            self._announce("I found your keys! They are right here.")
                            return
                    self._rejected += 1
                    self._state = "searching"
                    if exploring:
                        self._explore(True)  # resume systematic coverage
                    else:
                        # Turn away from the false positive so we don't re-check it.
                        self._turn(0.8, 0.6)
                    continue

                # --- MOVE ---
                self._state = "exploring (frontier + A*)" if exploring else "searching"
                if exploring:
                    # The nav stack is driving; just keep scanning as it covers ground.
                    # But if it never actually moves us (the explorer can sit on
                    # "No frontier found" forever from a parked spot), open up the map
                    # ourselves and re-arm exploration.
                    pose = self._pose()
                    if pose is not None:
                        if last_pose is None:
                            last_pose, last_move_at = pose, time.monotonic()
                        else:
                            moved = math.hypot(
                                pose[0] - last_pose[0], pose[1] - last_pose[1]
                            )
                            dyaw = abs(
                                (pose[2] - last_pose[2] + math.pi) % (2 * math.pi) - math.pi
                            )
                            if moved > _BOOTSTRAP_MOVE_EPS_M or dyaw > _BOOTSTRAP_YAW_EPS_RAD:
                                last_pose, last_move_at = pose, time.monotonic()
                        if time.monotonic() - last_move_at > _BOOTSTRAP_IDLE_S:
                            self._state = "bootstrapping the map (no frontier yet)"
                            self._explore(False)
                            time.sleep(0.3)
                            self._bootstrap_map()
                            self._explore(True)
                            last_pose = self._pose()
                            last_move_at = time.monotonic()
                    time.sleep(1.5)
                    continue
                legs += 1
                if legs % 4 == 0:
                    self._turn(0.6, 0.9)  # sweep to a new heading
                elif not self._step(0.28, 0.8):
                    # Obstacle ahead (lidar) -> turn away instead of pushing into it.
                    self._state = "blocked ahead, turning away"
                    self._turn(0.7, 1.0)
        except Exception as e:  # noqa: BLE001
            logger.error(f"key search loop failed: {e}")
        finally:
            if exploring:
                self._explore(False)
            self._halt()
            self._obstacle_avoidance(False)
            if self._state != "found":
                self._state = "idle"
                if not self._stop_flag.is_set():
                    self._announce("I finished looking but I could not find your keys.")

    def _announce(self, text: str) -> None:
        try:
            self._speak_skill.speak(text, blocking=False)
        except Exception as e:  # noqa: BLE001
            logger.error(f"KeyFinder speak failed: {e}")

    def _step(self, v: float, seconds: float) -> bool:
        """Step forward, but ONLY while the lidar corridor ahead is clear.

        Returns False if we refused to move / stopped early because of an obstacle.
        The robot's own OBSTACLES_AVOID proved unreliable on the Air (it walked into
        furniture), so clearance is re-checked every control tick here.
        """
        if v > 0 and self._path_blocked():
            self._blocked_stops += 1
            self._halt()
            return False
        t_end = time.monotonic() + seconds
        moved = True
        while time.monotonic() < t_end and not self._stop_flag.is_set():
            if v > 0 and self._path_blocked():
                self._blocked_stops += 1
                moved = False
                break
            self._drive(v, 0.0, 0.0)
            time.sleep(0.1)
        self._halt()
        return moved

    def _turn(self, rate: float, seconds: float) -> None:
        t_end = time.monotonic() + seconds
        while time.monotonic() < t_end and not self._stop_flag.is_set():
            self._drive(0.0, 0.0, rate)
            time.sleep(0.1)
        self._halt()

    def _approach(self, cand: dict[str, float]) -> bool:
        """Walk up to a candidate using stop-look-move. True if we got close enough."""
        for _ in range(6):
            if self._stop_flag.is_set():
                return False
            off = (cand["cx"] - cand["w"] / 2) / (cand["w"] / 2)
            if cand["cy"] / cand["h"] > _CLOSE_CY_FRAC:
                return True  # close to the feet
            if abs(off) > _CENTER_TOL:
                self._turn(-0.55 if off > 0 else 0.55, min(0.6, 0.2 + 0.5 * abs(off)))
            else:
                if not self._step(0.25, 0.6):
                    return True  # blocked: can't close in, verify from where we are
            self._halt()
            time.sleep(0.4)
            img = self._frame()
            if img is None:
                return False
            try:
                cands = self._scan(img)
            except Exception:  # noqa: BLE001
                return False
            if not cands:
                return False  # lost it
            cand = cands[0]
        return True  # ran out of steps; verify anyway

    def _verify_here(self) -> tuple[bool, str]:
        """Close-inspect the spot in front of the robot (back off, StandDown, confirm)."""
        self._state = "inspecting up close"
        try:
            for _ in range(6):
                self._drive(-0.18, 0.0, 0.0)
                time.sleep(0.1)
            self._halt()
            time.sleep(0.3)
            self._sport(SPORT_CMD["StandDown"])
            time.sleep(2.5)
            img = self._frame()
            verdict, note = (False, "no frame")
            if img is not None:
                verdict, note = self._confirm(img)
            self._sport(SPORT_CMD["RecoveryStand"])
            time.sleep(1.5)
            self._sport(SPORT_CMD["BalanceStand"])
            time.sleep(0.5)
            return verdict, note
        except Exception as e:  # noqa: BLE001
            logger.error(f"close verify failed: {e}")
            try:
                self._sport(SPORT_CMD["RecoveryStand"])
            except Exception:
                pass
            return False, "inspection error"
