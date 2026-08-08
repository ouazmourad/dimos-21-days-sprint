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

"""MediaPipe hand-gesture engine + gesture->velocity state machine for the Go2.

Ported from keaganchs/go2_gesture_recognition (point_follow.py). The vision and
gesture logic here is UNCHANGED in behaviour; what was removed is everything that
belongs to the DimOS layer instead: the app's own WebRTC connection, the `SHARED`
global mailbox, argparse/main, and the OpenCV visualizer window. Frames now come in
via a getter callback (fed by the DimOS camera stream) and control comes out as
(vx, vy, vyaw) velocities + emote requests for the skill container to publish.

Pipeline (per newest frame, on a background worker thread):
  low-light enhance -> MediaPipe GestureRecognizer (2 hands) + PoseLandmarker in
  parallel -> handedness relabel from pose wrists -> at most one pose-guided crop
  recovery -> Tracker state machine -> (vx, vy, vyaw), mode, debug.

Gestures (enable/disable listen to the RIGHT hand only):
  PEACE (right)            -> enable
  FLAT palm (right)        -> disable
  POINT at floor           -> walk to the projected floor spot
  POINT horizontally L/R   -> turn in place toward that heading
  palm-down back-wave      -> walk slowly backwards
  LEFT hand 1-4 fingers    -> emotes (Hello/Stretch/WiggleHips/Dance1)
  RIGHT peace while enabled -> FingerHeart
(hold a LEFT-hand peace sign 3 s to arm emotes first).
"""

from __future__ import annotations

import collections
import math
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Model files (MediaPipe .task). Resolved from $GO2_GESTURE_MODEL_DIR, else the
# cloned go2_gesture_recognition repo at the DimOS repo root, else CWD.
# ---------------------------------------------------------------------------

_MODEL_CANDIDATES = [
    os.environ.get("GO2_GESTURE_MODEL_DIR"),
    # dimos/robot/unitree/gesture/gesture_engine.py -> parents[4] == repo root
    str(Path(__file__).resolve().parents[4] / "go2_gesture_recognition"),
    str(Path.cwd() / "go2_gesture_recognition"),
    str(Path.cwd()),
]
GESTURE_MODEL_NAME = "gesture_recognizer.task"
POSE_MODEL_NAME = "pose_landmarker_lite.task"


def find_model_dir() -> str | None:
    """First candidate directory that actually contains the gesture model."""
    for cand in _MODEL_CANDIDATES:
        if cand and (Path(cand) / GESTURE_MODEL_NAME).is_file():
            return cand
    return None


# ---------------------------------------------------------------------------
# Configuration (identical to the source app; tune here)
# ---------------------------------------------------------------------------

# Low-light enhancement fades in below DARK_LUMA_ON, full at DARK_LUMA_FULL.
DARK_LUMA_ON = 90
DARK_LUMA_FULL = 45
TARGET_LUMA = 110
GAMMA_MIN = 0.45

POSE_EVERY = 2  # run the pose model every Nth frame while both hands are tracked

# Pose-guided hand recovery
POSE_MIN_VIS = 0.5
CROP_MIN_PX = 64
CROP_UPSCALE = 320
RELABEL_MAX_DIST = 0.12
STICKY_MAX_DIST = 0.15

POINT_HANDS = {"right": ("Right",), "left": ("Left",), "both": ("Right", "Left")}

ENABLE_GESTURE = "Victory"      # peace sign
DISABLE_GESTURE = "Open_Palm"   # flat palm
GESTURE_MIN_SCORE = 0.5

# Camera model (Go2 front camera). f is derived from the horizontal FOV.
HFOV_DEG = 104.0
CAM_HEIGHT = 0.38     # camera height above floor while standing [m]
CAM_PITCH_DEG = 0.0
CAM_FWD_OFFSET = 0.30

SWAP_HANDEDNESS = False

STABLE_FRAMES = 6
TOGGLE_COOLDOWN = 2.0
POINT_GRACE = 0.1

# Backward "wave"
BACK_SPEED = 0.15
WAVE_WINDOW = 0.8
WAVE_MIN_SAMPLES = 6
WAVE_DEADBAND = 0.006
WAVE_AMP = 0.03
WAVE_MIN_REVERSALS = 2
PALM_HORIZ_MIN = 0.55

# Control
MAX_VX = 0.4
MAX_VYAW = 0.6
K_LIN = 0.6
K_YAW = 1.5
STOP_RADIUS = 0.35
TARGET_EMA = 0.4
MIN_DOWNWARD = 0.30
TURN_DEADBAND = 0.15

# Palm-relative finger states for POINT / PEACE
FINGER_EXT_RATIO = 2.0
FINGER_CURL_RATIO = 1.5
POINT_FORGIVE = 0.20
_POINT_BAND = POINT_FORGIVE * (FINGER_EXT_RATIO - FINGER_CURL_RATIO)
POINT_EXT_RATIO = FINGER_EXT_RATIO - _POINT_BAND
POINT_CURL_RATIO = FINGER_CURL_RATIO + _POINT_BAND

# Emotes (require ENABLED + a standing robot; debounced + cooldown)
LEFT_EMOTES = {1: "Hello", 2: "Stretch", 3: "WiggleHips", 4: "Dance1"}
RIGHT_PEACE_EMOTE = "FingerHeart"
EMOTE_COOLDOWN = 8.0
EMOTE_ENABLE_HOLD = 3.0

# Hand landmark indices
WRIST = 0
I_MCP, I_PIP, I_TIP = 5, 6, 8
M_MCP, M_PIP, M_TIP = 9, 10, 12
R_MCP, R_PIP, R_TIP = 13, 14, 16
P_MCP, P_PIP, P_TIP = 17, 18, 20

# Pose landmark indices (BlazePose)
P_L_WRIST, P_R_WRIST = 15, 16
P_L_PINKY, P_R_PINKY = 17, 18
P_L_INDEX, P_R_INDEX = 19, 20


def np3(lm):
    return np.array([lm.x, lm.y, lm.z], dtype=float)


# ---------------------------------------------------------------------------
# Geometry: finger ray -> floor target
# ---------------------------------------------------------------------------


class FloorProjector:
    """Projects the index-finger ray of a hand onto the floor plane.

    Camera frame: x right, y down, z forward. Robot frame: x forward, y left.
    """

    def __init__(self, width, height):
        self.f = (width / 2.0) / math.tan(math.radians(HFOV_DEG) / 2.0)
        self.cx, self.cy = width / 2.0, height / 2.0
        self.w, self.h = width, height
        p = math.radians(CAM_PITCH_DEG)
        self.sin_p, self.cos_p = math.sin(p), math.cos(p)

    def to_level(self, v):
        x, y, z = v
        return np.array([
            x,
            y * self.cos_p + z * self.sin_p,
            -y * self.sin_p + z * self.cos_p,
        ])

    def pixel(self, lm):
        return np.array([lm.x * self.w, lm.y * self.h])

    def project_cam(self, p):
        if p[2] <= 0.05:
            return None
        return (
            int(self.f * p[0] / p[2] + self.cx),
            int(self.f * p[1] / p[2] + self.cy),
        )

    def hand_depth(self, img_lms, world_lms):
        ests = []
        for a, b in ((WRIST, M_MCP), (I_MCP, P_MCP)):
            px = np.linalg.norm(self.pixel(img_lms[a]) - self.pixel(img_lms[b]))
            wa, wb = np3(world_lms[a]), np3(world_lms[b])
            plane = math.hypot(wb[0] - wa[0], wb[1] - wa[1])
            if px > 4 and plane > 0.01:
                ests.append(self.f * plane / px)
        return float(np.median(ests)) if ests else None

    def analyze_pointing(self, img_lms, world_lms):
        """None, or {"type": "floor", "target": (x_fwd, y_left), "ray": ...}
        / {"type": "turn", "azimuth": rad, "ray": ...}."""
        d = np3(world_lms[I_TIP]) - np3(world_lms[I_MCP])
        n = np.linalg.norm(d)
        if n < 1e-6:
            return None
        d = d / n
        d_l = self.to_level(d)
        azimuth = math.atan2(-d_l[0], d_l[2])
        tip_px = tuple(self.pixel(img_lms[I_TIP]).astype(int))

        origin = None
        Z = self.hand_depth(img_lms, world_lms)
        if Z is not None and 0.3 < Z < 6.0:
            u, v = self.pixel(img_lms[I_MCP])
            origin = np.array(
                [(u - self.cx) / self.f * Z, (v - self.cy) / self.f * Z, Z])

        if d_l[1] > MIN_DOWNWARD:  # pointing at the floor
            if origin is None:
                return None
            o_l = self.to_level(origin)
            t = (CAM_HEIGHT - o_l[1]) / d_l[1]
            if t <= 0:
                return None
            floor_cam = origin + t * d
            floor_lvl = o_l + t * d_l
            return {
                "type": "floor",
                "target": (floor_lvl[2] + CAM_FWD_OFFSET, -floor_lvl[0]),
                "ray": (tip_px, self.project_cam(floor_cam)),
            }

        if origin is not None:
            end_px = self.project_cam(origin + 1.0 * d)
        else:
            v2 = self.pixel(img_lms[I_TIP]) - self.pixel(img_lms[I_MCP])
            nv = np.linalg.norm(v2)
            end_px = (tuple((self.pixel(img_lms[I_TIP]) + v2 / nv * 120)
                            .astype(int)) if nv > 1 else None)
        return {"type": "turn", "azimuth": azimuth, "ray": (tip_px, end_px)}


# ---------------------------------------------------------------------------
# Low-light enhancement
# ---------------------------------------------------------------------------


class LowLightEnhancer:
    """Adaptive gamma + CLAHE on the luma channel, faded by darkness."""

    def __init__(self):
        self.clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        self.gamma = 1.0
        self.strength = 0.0
        self.mean_luma = 0.0

    @property
    def active(self):
        return self.strength > 0.02

    def __call__(self, bgr):
        ycc = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
        luma = ycc[:, :, 0]
        self.mean_luma = float(luma.mean())

        target_s = (DARK_LUMA_ON - self.mean_luma) / (DARK_LUMA_ON - DARK_LUMA_FULL)
        target_s = max(0.0, min(1.0, target_s))
        self.strength = 0.85 * self.strength + 0.15 * target_s

        if not self.active:
            self.gamma = 1.0
            return bgr

        m = max(self.mean_luma, 1.0) / 255.0
        g_full = math.log(TARGET_LUMA / 255.0) / math.log(m)
        g_full = max(GAMMA_MIN, min(1.0, g_full))
        self.gamma = 1.0 + self.strength * (g_full - 1.0)

        lut = ((np.arange(256) / 255.0) ** self.gamma * 255).astype(np.uint8)
        lifted = cv2.LUT(luma, lut)
        ycc[:, :, 0] = cv2.addWeighted(
            self.clahe.apply(lifted), self.strength,
            lifted, 1.0 - self.strength, 0)
        return cv2.cvtColor(ycc, cv2.COLOR_YCrCb2BGR)


# ---------------------------------------------------------------------------
# Pose-guided hand localization
# ---------------------------------------------------------------------------


class SimpleLm:
    __slots__ = ("x", "y", "z")

    def __init__(self, x, y, z):
        self.x, self.y, self.z = x, y, z


class PoseHelper:
    """Runs the person/pose model and uses its wrists to (a) fix handedness
    labels on detected hands and (b) recover hands the full-frame recognizer
    missed, by re-running recognition on upscaled wrist crops."""

    def __init__(self, crop_recognizer, pose_landmarker):
        self.pose = pose_landmarker
        self.crop_rec = crop_recognizer

    def detect(self, mp_img, ts_ms):
        res = self.pose.detect_for_video(mp_img, ts_ms)
        return res.pose_landmarks[0] if res.pose_landmarks else None

    @staticmethod
    def wrists_px(pose, w, h):
        out = {}
        for side, idx in (("Left", P_L_WRIST), ("Right", P_R_WRIST)):
            lm = pose[idx]
            if getattr(lm, "visibility", 1.0) >= POSE_MIN_VIS:
                out[side] = (lm.x * w, lm.y * h)
        return out

    def relabel(self, hands, pose, w, h):
        wrists = self.wrists_px(pose, w, h)
        if not wrists:
            return
        for hand in hands:
            hx, hy = hand["img"][WRIST].x * w, hand["img"][WRIST].y * h
            side, d = min(
                ((s, math.hypot(px - hx, py - hy)) for s, (px, py) in wrists.items()),
                key=lambda t: t[1],
            )
            if d < RELABEL_MAX_DIST * w:
                hand["handedness"] = side
        if len(hands) == 2 and hands[0]["handedness"] == hands[1]["handedness"]:
            s = hands[0]["handedness"]
            other = "Left" if s == "Right" else "Right"
            px, py = wrists.get(s, (0, 0))
            d0 = math.hypot(hands[0]["img"][WRIST].x * w - px,
                            hands[0]["img"][WRIST].y * h - py)
            d1 = math.hypot(hands[1]["img"][WRIST].x * w - px,
                            hands[1]["img"][WRIST].y * h - py)
            hands[1 if d1 > d0 else 0]["handedness"] = other

    @staticmethod
    def hand_roi(pose, side, w, h):
        wr, ix, pk = ((P_L_WRIST, P_L_INDEX, P_L_PINKY) if side == "Left"
                      else (P_R_WRIST, P_R_INDEX, P_R_PINKY))
        if getattr(pose[wr], "visibility", 1.0) < POSE_MIN_VIS:
            return None
        wx, wy = pose[wr].x * w, pose[wr].y * h
        fx = (pose[ix].x + pose[pk].x) / 2 * w
        fy = (pose[ix].y + pose[pk].y) / 2 * h
        cx, cy = wx + 1.3 * (fx - wx), wy + 1.3 * (fy - wy)
        span = math.hypot(fx - wx, fy - wy)
        half = max(2.2 * span, CROP_MIN_PX)
        x0, y0 = int(max(0, cx - half)), int(max(0, cy - half))
        x1, y1 = int(min(w, cx + half)), int(min(h, cy + half))
        if x1 - x0 < CROP_MIN_PX or y1 - y0 < CROP_MIN_PX:
            return None
        return x0, y0, x1, y1

    def recover(self, frame, pose, side, w, h, mp):
        roi = self.hand_roi(pose, side, w, h)
        if roi is None:
            return None
        x0, y0, x1, y1 = roi
        crop = frame[y0:y1, x0:x1]
        if max(crop.shape[:2]) < CROP_UPSCALE:
            s = CROP_UPSCALE / max(crop.shape[:2])
            crop = cv2.resize(crop, None, fx=s, fy=s, interpolation=cv2.INTER_CUBIC)
        res = self.crop_rec.recognize(
            mp.Image(image_format=mp.ImageFormat.SRGB,
                     data=cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)))
        if not res.hand_landmarks:
            return None
        img_lms = [SimpleLm((x0 + lm.x * (x1 - x0)) / w,
                            (y0 + lm.y * (y1 - y0)) / h, lm.z)
                   for lm in res.hand_landmarks[0]]
        gname, gscore = "None", 0.0
        if res.gestures and res.gestures[0]:
            cat = res.gestures[0][0]
            gname, gscore = cat.category_name, cat.score
            if gscore < GESTURE_MIN_SCORE:
                gname = "None"
        return {
            "img": img_lms,
            "world": res.hand_world_landmarks[0],
            "handedness": side,
            "model": (gname, gscore),
            "roi": roi,
        }


def stabilize_labels(hands, prev, w, h):
    for hand in hands:
        hx, hy = hand["img"][WRIST].x * w, hand["img"][WRIST].y * h
        best, bd = None, 1e9
        for p in prev:
            d = math.hypot(p["px"][0] - hx, p["px"][1] - hy)
            if d < bd:
                best, bd = p, d
        if best is not None and bd < STICKY_MAX_DIST * w:
            hand["handedness"] = best["label"]
    if len(hands) == 2 and hands[0]["handedness"] == hands[1]["handedness"]:
        a, b = sorted(hands, key=lambda hd: hd["img"][WRIST].x)
        a["handedness"], b["handedness"] = "Right", "Left"


# ---------------------------------------------------------------------------
# Gesture classification (palm-relative, rotation-invariant)
# ---------------------------------------------------------------------------


def finger_extended(w, mcp, pip, tip):
    wrist = np3(w[WRIST])
    d_tip = np.linalg.norm(np3(w[tip]) - wrist)
    d_pip = np.linalg.norm(np3(w[pip]) - wrist)
    seg = np.linalg.norm(np3(w[pip]) - np3(w[mcp])) + np.linalg.norm(np3(w[tip]) - np3(w[pip]))
    straight = np.linalg.norm(np3(w[tip]) - np3(w[mcp])) / max(seg, 1e-6)
    return d_tip > d_pip and straight > 0.8


PALM_REFS = (WRIST, I_MCP, M_MCP, R_MCP, P_MCP)


def palm_geometry(w):
    pts = np.array([np3(w[i]) for i in PALM_REFS])
    center = pts.mean(axis=0)
    radius = float(np.linalg.norm(pts - center, axis=1).mean())
    return center, max(radius, 1e-6)


def finger_state(w, tip, center, radius, ext=FINGER_EXT_RATIO, curl=FINGER_CURL_RATIO):
    r = np.linalg.norm(np3(w[tip]) - center) / radius
    if r >= ext:
        return 1
    if r <= curl:
        return -1
    return 0


def hand_pose(hand):
    w = hand["world"]
    center, radius = palm_geometry(w)
    p_idx = finger_state(w, I_TIP, center, radius, POINT_EXT_RATIO, POINT_CURL_RATIO)
    p_mid = finger_state(w, M_TIP, center, radius, POINT_EXT_RATIO, POINT_CURL_RATIO)
    p_rng = finger_state(w, R_TIP, center, radius, POINT_EXT_RATIO, POINT_CURL_RATIO)
    if p_idx == 1 and p_mid == -1 and p_rng == -1:
        return "POINT"
    idx = finger_state(w, I_TIP, center, radius)
    mid = finger_state(w, M_TIP, center, radius)
    rng = finger_state(w, R_TIP, center, radius)
    pky = finger_state(w, P_TIP, center, radius)
    if idx == 1 and mid == 1 and rng == -1 and pky == -1:
        return "PEACE"
    return None


def is_pointing(hand):
    return hand["model"][0] == "Pointing_Up" or hand_pose(hand) == "POINT"


def is_peace(hand):
    return hand["model"][0] == "Victory" or hand_pose(hand) == "PEACE"


def _hand_px(hand):
    lm = hand["img"][I_MCP]
    return (lm.x, lm.y)


def _hand_dist2(hand, px):
    x, y = _hand_px(hand)
    return (x - px[0]) ** 2 + (y - px[1]) ** 2


def palm_normal(w):
    o = np3(w[WRIST])
    n = np.cross(np3(w[I_MCP]) - o, np3(w[P_MCP]) - o)
    return n / max(float(np.linalg.norm(n)), 1e-6)


def is_palm_horizontal(hand):
    return abs(palm_normal(hand["world"])[1]) >= PALM_HORIZ_MIN


def open_fingers(hand):
    w = hand["world"]
    center, radius = palm_geometry(w)
    return sum(finger_state(w, t, center, radius) == 1
               for t in (I_TIP, M_TIP, R_TIP, P_TIP))


def wave_signal(hand):
    img = hand["img"]
    tips = float(np.mean([img[t].y for t in (I_TIP, M_TIP, R_TIP, P_TIP)]))
    return tips - img[WRIST].y


class WaveState:
    """Detects a palm-down up/down flick from a per-hand history of the
    vertical fingertip signal — a real wave, not a static open palm."""

    def __init__(self):
        self.hist = collections.deque()

    def update(self, now, hand):
        gate = (hand is not None and is_palm_horizontal(hand)
                and open_fingers(hand) >= 2)
        if gate:
            self.hist.append((now, wave_signal(hand)))
        while self.hist and now - self.hist[0][0] > WAVE_WINDOW:
            self.hist.popleft()
        if not gate or len(self.hist) < WAVE_MIN_SAMPLES:
            return False
        sigs = [s for _, s in self.hist]
        if max(sigs) - min(sigs) < WAVE_AMP:
            return False
        reversals, dirn, prev = 0, 0, sigs[0]
        for s in sigs[1:]:
            d = s - prev
            if abs(d) < WAVE_DEADBAND:
                continue
            nd = 1 if d > 0 else -1
            if dirn and nd != dirn:
                reversals += 1
            dirn, prev = nd, s
        return reversals >= WAVE_MIN_REVERSALS


FINGERS = ((I_MCP, I_PIP, I_TIP), (M_MCP, M_PIP, M_TIP),
           (R_MCP, R_PIP, R_TIP), (P_MCP, P_PIP, P_TIP))


def count_fingers_up(hand):
    w, img = hand["world"], hand["img"]
    n = 0
    for mcp, pip, tip in FINGERS:
        if finger_extended(w, mcp, pip, tip) and img[tip].y < img[WRIST].y:
            n += 1
    return n


# ---------------------------------------------------------------------------
# State machine + controller
# ---------------------------------------------------------------------------


class Tracker:
    def __init__(self, point_hand="both"):
        self.state = "DISABLED"        # DISABLED | ENABLED
        self.motion = "STAY"           # STAY | MOVE (only meaningful in ENABLED)
        self.counts = {"enable": 0, "disable": 0, "point": 0, "emote": 0, "peace": 0}
        self.point_hand = point_hand   # "right" | "left" | "both"
        self.pointer_px = None
        self.wave = WaveState()
        self.last_toggle = 0.0
        self.last_point_seen = -1e9
        self.target = None
        self.turn_azimuth = None
        self.emote_n = 0
        self.last_emote = -1e9
        self.last_emote_name = "-"
        self.emotes_enabled = False
        self.left_peace_since = None
        self.status_msg = "show PEACE sign to enable"

    def _reset_counts(self):
        self.counts = dict.fromkeys(self.counts, 0)

    def _bump(self, key, cond):
        self.counts[key] = self.counts[key] + 1 if cond else 0
        return self.counts[key] >= STABLE_FRAMES

    def update(self, hands, projector, now):
        """hands: [{'img','world','handedness','model'}...].
        Returns (vx, vy, vyaw), mode, debug."""
        debug = {"ray": None, "target": None, "gesture": "-", "azimuth": None,
                 "fingers": 0, "emote": None, "emote_hold": 0.0}
        right = next((h for h in hands if h["handedness"] == "Right"), None)
        if right is not None:
            debug["gesture"] = (f"{right['model'][0]}/{hand_pose(right) or '-'}"
                                f" {right['model'][1]:.2f}")

        right_gesture = right["model"][0] if right is not None else "None"
        right_peace = right is not None and is_peace(right)
        can_toggle = now - self.last_toggle > TOGGLE_COOLDOWN
        backwave = self.wave.update(now, right)

        if self.state == "DISABLED":
            if self._bump("enable", right_peace and can_toggle):
                self.state, self.motion = "ENABLED", "STAY"
                self.target = None
                self.last_toggle = now
                self._reset_counts()
                self.status_msg = "enabled: point where the robot should go"
            return (0, 0, 0), "idle", debug

        # ENABLED
        if backwave:
            self.motion = "STAY"
            self.target = self.turn_azimuth = self.pointer_px = None
            self.counts["disable"] = self.counts["point"] = 0
            self.status_msg = "back-wave -> reversing"
            debug["gesture"] += " BACKWAVE"
            return (round(-BACK_SPEED, 3), 0.0, 0.0), "move", debug

        if self._bump("disable", right_gesture == DISABLE_GESTURE and can_toggle):
            self.state, self.motion = "DISABLED", "STAY"
            self.target = None
            self.last_toggle = now
            self._reset_counts()
            self.status_msg = "flat palm -> DISABLED (peace sign enables)"
            return (0, 0, 0), "stop", debug

        left = next((h for h in hands if h["handedness"] == "Left"), None)
        debug["fingers"] = count_fingers_up(left) if left is not None else 0

        debug["emote_hold"] = 0.0
        if left is not None and is_peace(left):
            if self.left_peace_since is None:
                self.left_peace_since = now
            held = now - self.left_peace_since
            if not self.emotes_enabled:
                debug["emote_hold"] = min(1.0, held / EMOTE_ENABLE_HOLD)
                if held >= EMOTE_ENABLE_HOLD:
                    self.emotes_enabled = True
                    self.last_emote = now
                    self.status_msg = "emotes enabled"
        else:
            self.left_peace_since = None

        if (self.emotes_enabled and self.motion == "STAY"
                and now - self.last_emote > EMOTE_COOLDOWN and can_toggle):
            n = debug["fingers"] if debug["fingers"] in LEFT_EMOTES else 0
            if n != self.emote_n:
                self.emote_n, self.counts["emote"] = n, 0
            emote = None
            if n and self._bump("emote", True):
                emote = LEFT_EMOTES[n]
            elif self._bump("peace", right_peace):
                emote = RIGHT_PEACE_EMOTE
            if emote:
                self.last_emote = now
                self.last_emote_name = emote
                self.emote_n = 0
                self._reset_counts()
                self.status_msg = f"emote: {emote}"
                debug["emote"] = emote
                return (0, 0, 0), "emote", debug

        allowed = POINT_HANDS[self.point_hand]
        pointers = [h for h in hands
                    if h["handedness"] in allowed and is_pointing(h)]
        pointer = None
        if pointers:
            if self.pointer_px is not None:
                pointer = min(pointers, key=lambda h: _hand_dist2(h, self.pointer_px))
            else:
                pointer = next((h for h in pointers
                                if h["handedness"] == "Right"), pointers[0])
            self.pointer_px = _hand_px(pointer)

        res = None
        if pointer is not None:
            debug["gesture"] += f" POINT({pointer['handedness'][0]})"
            res = projector.analyze_pointing(pointer["img"], pointer["world"])

        if res is not None:
            self.last_point_seen = now
            debug["ray"] = res["ray"]
            if res["type"] == "floor":
                self.turn_azimuth = None
                x, y = res["target"]
                if self.target is None:
                    self.target = (x, y)
                else:
                    ax = TARGET_EMA
                    self.target = (ax * x + (1 - ax) * self.target[0],
                                   ax * y + (1 - ax) * self.target[1])
            else:
                self.target = None
                self.turn_azimuth = res["azimuth"]
                debug["azimuth"] = res["azimuth"]
            if self._bump("point", True):
                self.motion = "MOVE"
                self.status_msg = ("turning to pointed heading"
                                   if self.turn_azimuth is not None
                                   else "walking to pointed target")
        elif now - self.last_point_seen > POINT_GRACE:
            self.counts["point"] = 0
            self.pointer_px = None
            if self.motion == "MOVE":
                self.motion = "STAY"
                self.target = None
                self.turn_azimuth = None
                self.status_msg = "not pointing -> stay"
                return (0, 0, 0), "stop", debug

        if self.motion == "MOVE":
            if self.turn_azimuth is not None:
                az = self.turn_azimuth
                debug["azimuth"] = az
                if abs(az) < TURN_DEADBAND:
                    self.motion = "STAY"
                    self.status_msg = "aligned with pointed heading"
                    return (0, 0, 0), "stop", debug
                vyaw = max(-MAX_VYAW, min(MAX_VYAW, K_YAW * az))
                return (0.0, 0.0, round(vyaw, 3)), "move", debug
            if self.target is not None:
                debug["target"] = self.target
                vx, vy, vyaw = self._control(self.target)
                if (vx, vy, vyaw) == (0.0, 0.0, 0.0):
                    self.motion = "STAY"
                    self.status_msg = "arrived at target"
                    return (0, 0, 0), "stop", debug
                return (vx, vy, vyaw), "move", debug

        return (0, 0, 0), "idle", debug

    @staticmethod
    def _control(target):
        x, y = target
        dist = math.hypot(x, y)
        if dist < STOP_RADIUS:
            return (0.0, 0.0, 0.0)
        bearing = math.atan2(y, x)
        vyaw = max(-MAX_VYAW, min(MAX_VYAW, K_YAW * bearing))
        vx = 0.0
        if abs(bearing) < 1.0:
            vx = max(0.0, min(MAX_VX, K_LIN * (dist - STOP_RADIUS)))
        return (round(vx, 3), 0.0, round(vyaw, 3))


# ---------------------------------------------------------------------------
# GestureEngine: owns the MediaPipe models + a background worker that turns the
# newest camera frame into (hands, vision). Replaces the source app's
# VisionWorker (which pulled from the SHARED global); here frames come from a
# getter callback the DimOS container supplies.
# ---------------------------------------------------------------------------


class GestureEngine:
    """MediaPipe gesture/pose inference on a background worker thread.

    frame_getter() -> (frame_bgr | None, version:int). The worker always jumps to
    the newest frame (never queues), runs gesture + pose inference concurrently on
    a 2-thread pool (MediaPipe releases the GIL), does handedness relabel + one
    crop recovery, and publishes the latest result via latest().
    """

    def __init__(self, frame_getter, model_dir: str | None = None):
        import mediapipe as mp  # lazy: only needed when gesture control actually runs
        from mediapipe.tasks import python as mp_tasks
        from mediapipe.tasks.python import vision as mp_vision

        self._mp = mp
        self._frame_getter = frame_getter
        model_dir = model_dir or find_model_dir()
        if model_dir is None:
            raise FileNotFoundError(
                f"Could not find '{GESTURE_MODEL_NAME}'. Set $GO2_GESTURE_MODEL_DIR "
                f"to the go2_gesture_recognition checkout (searched: {_MODEL_CANDIDATES})."
            )
        gesture_path = str(Path(model_dir) / GESTURE_MODEL_NAME)
        pose_path = str(Path(model_dir) / POSE_MODEL_NAME)

        self._recognizer = mp_vision.GestureRecognizer.create_from_options(
            mp_vision.GestureRecognizerOptions(
                base_options=mp_tasks.BaseOptions(model_asset_path=gesture_path),
                running_mode=mp_vision.RunningMode.VIDEO,
                num_hands=2,
                min_hand_detection_confidence=0.4,
                min_hand_presence_confidence=0.4,
                min_tracking_confidence=0.4,
            )
        )
        crop_recognizer = mp_vision.GestureRecognizer.create_from_options(
            mp_vision.GestureRecognizerOptions(
                base_options=mp_tasks.BaseOptions(model_asset_path=gesture_path),
                running_mode=mp_vision.RunningMode.IMAGE,
                num_hands=1,
                min_hand_detection_confidence=0.4,
            )
        )
        pose_landmarker = mp_vision.PoseLandmarker.create_from_options(
            mp_vision.PoseLandmarkerOptions(
                base_options=mp_tasks.BaseOptions(model_asset_path=pose_path),
                running_mode=mp_vision.RunningMode.VIDEO,
                num_poses=1,
            )
        )
        self._ph = PoseHelper(crop_recognizer, pose_landmarker)
        self.enhancer = LowLightEnhancer()
        self._pool = ThreadPoolExecutor(max_workers=2)

        self._lock = threading.Lock()
        self._out = None
        self._out_ver = 0
        self.fps = 0.0
        self._running = False
        self._thread: threading.Thread | None = None

    def start(self):
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2)
        try:
            self._recognizer.close()
        except Exception:
            pass

    def latest(self):
        with self._lock:
            return self._out, self._out_ver

    def _run(self):
        mp = self._mp
        last_ver, ts_ms, t_prev = -1, 0, time.monotonic()
        frame_idx = 0
        pose, pose_age = None, 999
        recover_side = "Left"
        prev_hands = []

        while self._running:
            frame, ver = self._frame_getter()
            if frame is None or ver == last_ver:
                time.sleep(0.002)
                continue
            last_ver = ver
            frame = self.enhancer(frame.copy())
            h, w = frame.shape[:2]
            frame_idx += 1

            ts_ms = max(ts_ms + 1, int(time.monotonic() * 1000))
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB,
                              data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            hands_were_missing = self._out is None or len(self._out["hands"]) < 2
            want_pose = hands_were_missing or frame_idx % POSE_EVERY == 0
            f_rec = self._pool.submit(self._recognizer.recognize_for_video, mp_img, ts_ms)
            f_pose = (self._pool.submit(self._ph.detect, mp_img, ts_ms)
                      if want_pose else None)
            result = f_rec.result()
            if f_pose is not None:
                pose, pose_age = f_pose.result(), 0
            else:
                pose_age += 1
            if pose_age > 3 * POSE_EVERY:
                pose = None

            hands = []
            for i, img_lms in enumerate(result.hand_landmarks):
                label = result.handedness[i][0].category_name
                if SWAP_HANDEDNESS:
                    label = "Right" if label == "Left" else "Left"
                gname, gscore = "None", 0.0
                if result.gestures and result.gestures[i]:
                    cat = result.gestures[i][0]
                    gname, gscore = cat.category_name, cat.score
                    if gscore < GESTURE_MIN_SCORE:
                        gname = "None"
                hands.append({
                    "img": img_lms,
                    "world": result.hand_world_landmarks[i],
                    "handedness": label,
                    "model": (gname, gscore),
                })

            stabilize_labels(hands, prev_hands, w, h)
            recovered = []
            if pose is not None:
                if pose_age == 0:
                    self._ph.relabel(hands, pose, w, h)
                present = {hd["handedness"] for hd in hands}
                missing = [s for s in ("Left", "Right") if s not in present]
                if missing:
                    if len(missing) == 2:
                        recover_side = "Right" if recover_side == "Left" else "Left"
                        missing = [recover_side]
                    hd = self._ph.recover(frame, pose, missing[0], w, h, mp)
                    if hd is not None:
                        hands.append(hd)
                        recovered.append(missing[0])

            prev_hands = [{"px": (hd["img"][WRIST].x * w, hd["img"][WRIST].y * h),
                           "label": hd["handedness"]} for hd in hands]

            now = time.monotonic()
            dt = now - t_prev
            t_prev = now
            if dt > 0:
                self.fps = 0.9 * self.fps + 0.1 * (1.0 / dt)

            vision = {"pose": pose, "recovered": recovered,
                      "luma": self.enhancer.mean_luma, "gamma": self.enhancer.gamma,
                      "strength": self.enhancer.strength, "enhanced": self.enhancer.active}
            with self._lock:
                self._out = {"frame": frame, "hands": hands, "vision": vision}
                self._out_ver += 1
