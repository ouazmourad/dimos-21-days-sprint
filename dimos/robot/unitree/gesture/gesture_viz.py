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

"""The gesture dashboard overlay (ported from point_follow.py draw_overlay).

Renders skeletons, gesture labels, the pointing ray + floor cross, a state panel,
velocity bars and a top-down 5 m map onto a frame. Pure drawing — no I/O. The
skill container encodes the result to JPEG and streams it over HTTP (see
gesture_skill_container._ensure_viz_server) so it can be viewed headlessly in a
browser instead of the original app's local OpenCV window.
"""

from __future__ import annotations

import math
import time

import cv2
import numpy as np

from dimos.robot.unitree.gesture.gesture_engine import (
    EMOTE_COOLDOWN,
    MAX_VX,
    MAX_VYAW,
    STABLE_FRAMES,
    WRIST,
)

PANEL_W = 300

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
]

POSE_CONNECTIONS = [
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
    (11, 23), (12, 24), (23, 24),
]


def draw_hand(img, lms, w, h, color):
    pts = [(int(l.x * w), int(l.y * h)) for l in lms]
    for a, b in HAND_CONNECTIONS:
        cv2.line(img, pts[a], pts[b], color, 2)
    for p in pts:
        cv2.circle(img, p, 3, (255, 255, 255), -1)
    return pts


def draw_overlay(img, hands, tracker, debug, cmd, fps, conn_status, move_enabled, vision):
    """Return the annotated frame with the state panel hstacked on the right."""
    h, w = img.shape[:2]

    if vision["pose"] is not None:
        pose = vision["pose"]
        for a, b in POSE_CONNECTIONS:
            pa = (int(pose[a].x * w), int(pose[a].y * h))
            pb = (int(pose[b].x * w), int(pose[b].y * h))
            cv2.line(img, pa, pb, (90, 90, 90), 2)
    for hand in hands:
        if "roi" in hand:
            x0, y0, x1, y1 = hand["roi"]
            cv2.rectangle(img, (x0, y0), (x1, y1), (255, 200, 0), 1)
            cv2.putText(img, "recovered", (x0 + 3, y0 + 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 200, 0), 1)

    for hand in hands:
        is_right = hand["handedness"] == "Right"
        tracked = is_right and tracker.state == "ENABLED"
        color = (0, 220, 0) if tracked else (0, 160, 255) if is_right else (200, 120, 0)
        pts = draw_hand(img, hand["img"], w, h, color)
        label = f"{hand['handedness']} [{hand['model'][0]}]"
        cv2.putText(img, label, (pts[WRIST][0] - 20, pts[WRIST][1] + 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    if debug["ray"] is not None:
        tip_px, floor_px = debug["ray"]
        if floor_px is not None:
            cv2.arrowedLine(img, tip_px, floor_px, (0, 255, 255), 2, tipLength=0.03)
            cv2.circle(img, floor_px, 10, (0, 0, 255), 2)
            cv2.drawMarker(img, floor_px, (0, 0, 255), cv2.MARKER_CROSS, 20, 2)

    # ---- side panel ----
    panel = np.full((h, PANEL_W, 3), 30, np.uint8)

    def put(text, y, color=(220, 220, 220), scale=0.5):
        cv2.putText(panel, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)

    state_color = (0, 220, 0) if tracker.state == "ENABLED" else (0, 160, 255)
    put(f"STATE: {tracker.state}", 30, state_color, 0.7)
    put(f"motion: {tracker.motion if tracker.state == 'ENABLED' else '-'}"
        f"   point hand: {tracker.point_hand}", 55)
    put(tracker.status_msg, 78, (180, 180, 255))
    put(f"conn: {conn_status}", 100, (160, 220, 160))
    put(f"move enabled: {move_enabled}", 120, (160, 220, 160) if move_enabled else (0, 0, 255))
    put(f"vision fps: {fps:.1f}", 140)
    lum = f"luma: {vision['luma']:.0f}"
    if vision["enhanced"]:
        lum += f"  LOW LIGHT {vision['strength'] * 100:.0f}% (gamma {vision['gamma']:.2f})"
    put(lum, 160, (0, 255, 255) if vision["enhanced"] else (180, 180, 180))
    pose_txt = "pose: ok" if vision["pose"] is not None else "pose: -"
    if vision["recovered"]:
        pose_txt += "  recovered: " + "+".join(vision["recovered"])
    put(pose_txt, 178, (255, 200, 0) if vision["recovered"] else (180, 180, 180))
    put(f"hands: {len(hands)}  right: {debug['gesture']}", 196)
    key = "enable" if tracker.state == "DISABLED" else "disable"
    n = min(tracker.counts[key], STABLE_FRAMES)
    put(f"{key}: {'#' * n}{'.' * (STABLE_FRAMES - n)}", 214)

    if debug["target"] is not None:
        x, y = debug["target"]
        put(f"target: x={x:+.2f}m  y={y:+.2f}m", 232, (0, 255, 255))
        put(f"dist={math.hypot(x, y):.2f}m  brg={math.degrees(math.atan2(y, x)):+.0f}deg",
            250, (0, 255, 255))
    elif debug["azimuth"] is not None:
        put(f"target: TURN {math.degrees(debug['azimuth']):+.0f}deg", 232, (255, 0, 255))
        put("(horizontal point -> rotate)", 250, (255, 0, 255))
    else:
        put("target: -", 232)

    if not tracker.emotes_enabled:
        if debug["emote_hold"] > 0:
            bars = int(debug["emote_hold"] * 10)
            put(f"emotes: arming {'#' * bars}{'.' * (10 - bars)}", 266, (0, 255, 255))
        else:
            put("emotes: OFF (hold L peace 3s)", 266, (120, 120, 120))
    else:
        cd = max(0.0, EMOTE_COOLDOWN - (time.monotonic() - tracker.last_emote))
        put(f"L fingers: {debug['fingers']}  emote: {tracker.last_emote_name}"
            + (f" (cd {cd:.0f}s)" if cd > 0 else ""), 266,
            (0, 255, 255) if debug["emote"] else (180, 180, 180))

    vx, vy, vyaw = cmd
    put(f"cmd vx={vx:+.2f}  vyaw={vyaw:+.2f}", 284, (0, 220, 0))
    cv2.rectangle(panel, (10, 292), (10 + PANEL_W - 20, 304), (60, 60, 60), 1)
    cv2.rectangle(panel, (PANEL_W // 2, 292),
                  (PANEL_W // 2 + int((PANEL_W - 20) / 2 * vx / MAX_VX), 304), (0, 220, 0), -1)
    cv2.rectangle(panel, (10, 308), (10 + PANEL_W - 20, 320), (60, 60, 60), 1)
    cv2.rectangle(panel, (PANEL_W // 2, 308),
                  (PANEL_W // 2 + int((PANEL_W - 20) / 2 * vyaw / MAX_VYAW), 320), (0, 160, 255), -1)

    # ---- top-down map (robot at bottom centre, x fwd = up) ----
    map_top = 332
    map_size = min(PANEL_W - 20, max(80, h - map_top - 80))
    scale = map_size / 5.0  # 5 m view
    cv2.rectangle(panel, (10, map_top), (10 + map_size, map_top + map_size), (60, 60, 60), 1)
    rob = (10 + map_size // 2, map_top + map_size - 15)
    for r in (1, 2):
        cv2.circle(panel, rob, int(r * scale), (55, 55, 55), 1)
    cv2.drawMarker(panel, rob, (255, 255, 255), cv2.MARKER_TRIANGLE_UP, 14, 2)
    if debug["target"] is not None:
        x, y = debug["target"]
        tx = int(rob[0] - y * scale)
        ty = int(rob[1] - x * scale)
        if 10 < tx < 10 + map_size and map_top < ty < map_top + map_size:
            cv2.line(panel, rob, (tx, ty), (0, 255, 255), 1)
            cv2.circle(panel, (tx, ty), 6, (0, 0, 255), -1)
    elif debug["azimuth"] is not None:
        az = debug["azimuth"]
        end = (int(rob[0] - math.sin(az) * 1.5 * scale),
               int(rob[1] - math.cos(az) * 1.5 * scale))
        cv2.arrowedLine(panel, rob, end, (255, 0, 255), 2, tipLength=0.2)
    put("top view (5m)", map_top + map_size + 20, (150, 150, 150))
    put("PEACE=enable PALM=disable POINT=go/turn", h - 30, (150, 150, 150), 0.42)
    put("L peace 3s=arm emotes  L 1-4=emote  R peace=heart", h - 12, (150, 150, 150), 0.42)

    return np.hstack([img, panel])
