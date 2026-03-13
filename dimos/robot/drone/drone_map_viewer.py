#!/usr/bin/env python3
"""Drone map viewer — shows simulated drone on a Google Maps satellite image.

Connects to DimOS WebSocket vis module (port 7779) for live GPS telemetry
and renders the drone position on a Google Maps static tile.

Usage:
    python dimos/robot/drone/drone_map_viewer.py
"""

import io
import math
import os
import time
import urllib.request

import numpy as np
import pygame

# Config
MAP_CENTER_LAT = 37.780967
MAP_CENTER_LON = -122.406883
MAP_ZOOM = 18
MAP_W, MAP_H = 640, 640
WIN_W, WIN_H = 700, 780
API_KEY = os.environ.get("GOOGLE_MAPS_API_KEY", "")

# SF Office GPS locations from system prompt
WAYPOINTS = {
    "Office (454 Natoma)": (37.780967, -122.406883),
    "6th & Natoma": (37.780199, -122.407708),
    "5th & Mission": (37.782598, -122.406494),
    "6th & Mission": (37.781007, -122.408684),
}


def fetch_map_tile() -> pygame.Surface | None:
    """Fetch Google Maps static satellite tile."""
    if not API_KEY:
        print("No GOOGLE_MAPS_API_KEY set — using plain grid background")
        return None

    url = (
        f"https://maps.googleapis.com/maps/api/staticmap?"
        f"center={MAP_CENTER_LAT},{MAP_CENTER_LON}"
        f"&zoom={MAP_ZOOM}&size={MAP_W}x{MAP_H}"
        f"&maptype=satellite&key={API_KEY}"
    )
    print(f"Fetching map tile...")
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "DimOS-Drone/1.0"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = resp.read()
        return pygame.image.load(io.BytesIO(data))
    except Exception as e:
        print(f"Failed to fetch map: {e}")
        return None


def gps_to_pixel(lat: float, lon: float) -> tuple[int, int]:
    """Convert GPS coords to pixel position on the map tile."""
    # Mercator projection math for Google Maps static tiles
    scale = 2 ** MAP_ZOOM
    world_x = (lon + 180) / 360 * 256 * scale
    sin_lat = math.sin(lat * math.pi / 180)
    world_y = (0.5 - math.log((1 + sin_lat) / (1 - sin_lat)) / (4 * math.pi)) * 256 * scale

    center_x = (MAP_CENTER_LON + 180) / 360 * 256 * scale
    sin_center = math.sin(MAP_CENTER_LAT * math.pi / 180)
    center_y = (0.5 - math.log((1 + sin_center) / (1 - sin_center)) / (4 * math.pi)) * 256 * scale

    px = int(MAP_W / 2 + (world_x - center_x))
    py = int(MAP_H / 2 + (world_y - center_y))
    return px, py


def draw_drone(screen: pygame.Surface, x: int, y: int, heading: float, color: tuple) -> None:
    """Draw a drone icon (triangle pointing in heading direction)."""
    size = 12
    angle = math.radians(heading)
    # Triangle points
    nose = (x + size * math.sin(angle), y - size * math.cos(angle))
    left = (x + size * 0.6 * math.sin(angle + 2.5), y - size * 0.6 * math.cos(angle + 2.5))
    right = (x + size * 0.6 * math.sin(angle - 2.5), y - size * 0.6 * math.cos(angle - 2.5))
    pygame.draw.polygon(screen, color, [nose, left, right])
    pygame.draw.polygon(screen, (255, 255, 255), [nose, left, right], 2)
    # Pulsing circle
    pulse = int(8 + 4 * math.sin(time.time() * 3))
    s = pygame.Surface((pulse * 2, pulse * 2), pygame.SRCALPHA)
    pygame.draw.circle(s, (*color, 80), (pulse, pulse), pulse)
    screen.blit(s, (x - pulse, y - pulse))


def main() -> None:
    # Try to connect to DimOS websocket for live telemetry
    sio = None
    drone_state = {
        "lat": MAP_CENTER_LAT, "lon": MAP_CENTER_LON,
        "alt": 0.0, "heading": 270.0, "connected": False,
        "mode": "UNKNOWN", "armed": False,
    }

    try:
        import socketio as sio_lib
        sio = sio_lib.Client()

        @sio.on("gps_location")
        def on_gps(data):
            if isinstance(data, dict):
                drone_state["lat"] = data.get("lat", drone_state["lat"])
                drone_state["lon"] = data.get("lon", drone_state["lon"])
                drone_state["connected"] = True

        @sio.on("odom")
        def on_odom(data):
            if isinstance(data, dict):
                pos = data.get("position", {})
                drone_state["alt"] = pos.get("z", drone_state["alt"])

        @sio.on("full_state")
        def on_full_state(data):
            if isinstance(data, dict):
                gps = data.get("gps_location", {})
                if gps:
                    drone_state["lat"] = gps.get("lat", drone_state["lat"])
                    drone_state["lon"] = gps.get("lon", drone_state["lon"])
                    drone_state["connected"] = True

        print("Connecting to DimOS WebSocket at localhost:7779...")
        sio.connect("http://localhost:7779", transports=["websocket"])
        print("Connected to DimOS!")
        drone_state["connected"] = True
    except Exception as e:
        print(f"WebSocket connection failed: {e}")
        print("Running in standalone mode — drone stays at home position")

    pygame.init()
    os.environ.setdefault("SDL_VIDEODRIVER", "x11")
    screen = pygame.display.set_mode((WIN_W, WIN_H))
    pygame.display.set_caption("DimOS Drone — Google Maps View")
    clock = pygame.time.Clock()
    font = pygame.font.Font(None, 22)
    font_big = pygame.font.Font(None, 28)
    font_title = pygame.font.Font(None, 32)

    map_tile = fetch_map_tile()
    map_offset_x = (WIN_W - MAP_W) // 2
    map_offset_y = 5

    print("\nDrone Map Viewer running. Waiting for telemetry...")
    print("Use the web UI at http://localhost:5555 to command the drone.\n")

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                running = False

        screen.fill((20, 20, 30))

        # Draw map
        if map_tile:
            screen.blit(map_tile, (map_offset_x, map_offset_y))
        else:
            # Grid background
            pygame.draw.rect(screen, (30, 40, 30), (map_offset_x, map_offset_y, MAP_W, MAP_H))
            for i in range(0, MAP_W, 40):
                pygame.draw.line(screen, (40, 55, 40),
                                 (map_offset_x + i, map_offset_y),
                                 (map_offset_x + i, map_offset_y + MAP_H))
            for j in range(0, MAP_H, 40):
                pygame.draw.line(screen, (40, 55, 40),
                                 (map_offset_x, map_offset_y + j),
                                 (map_offset_x + MAP_W, map_offset_y + j))

        # Draw waypoints
        for name, (wlat, wlon) in WAYPOINTS.items():
            wx, wy = gps_to_pixel(wlat, wlon)
            wx += map_offset_x
            wy += map_offset_y
            if 0 <= wx < WIN_W and 0 <= wy < WIN_H:
                pygame.draw.circle(screen, (255, 200, 0), (wx, wy), 5)
                pygame.draw.circle(screen, (255, 255, 255), (wx, wy), 5, 1)
                s = font.render(name, True, (255, 255, 200))
                screen.blit(s, (wx + 8, wy - 8))

        # Draw drone
        dx, dy = gps_to_pixel(drone_state["lat"], drone_state["lon"])
        dx += map_offset_x
        dy += map_offset_y
        color = (0, 255, 100) if drone_state["connected"] else (255, 80, 80)
        draw_drone(screen, dx, dy, drone_state["heading"], color)

        # HUD bar at bottom
        hud_y = MAP_H + map_offset_y + 10
        pygame.draw.rect(screen, (15, 15, 25), (0, hud_y, WIN_W, WIN_H - hud_y))
        pygame.draw.line(screen, (0, 200, 200), (0, hud_y), (WIN_W, hud_y))

        # Title
        s = font_title.render("DimOS Drone — ArduPilot SITL", True, (0, 255, 255))
        screen.blit(s, (15, hud_y + 8))

        # Connection status
        status = "CONNECTED" if drone_state["connected"] else "DISCONNECTED"
        status_color = (0, 255, 100) if drone_state["connected"] else (255, 80, 80)
        s = font_big.render(status, True, status_color)
        screen.blit(s, (WIN_W - 160, hud_y + 8))

        # Telemetry
        y = hud_y + 40
        lines = [
            f"LAT: {drone_state['lat']:.6f}    LON: {drone_state['lon']:.6f}    ALT: {drone_state['alt']:.1f}m",
            f"Heading: {drone_state['heading']:.0f}°",
            "",
            "Control via http://localhost:5555:",
            '  "arm and takeoff to 10 meters"',
            '  "fly to 6th and mission intersection"',
            '  "land"',
        ]
        for line in lines:
            s = font.render(line, True, (180, 180, 200))
            screen.blit(s, (15, y))
            y += 22

        pygame.display.flip()
        clock.tick(30)

    if sio and sio.connected:
        sio.disconnect()
    pygame.quit()
    print("Map viewer closed.")


if __name__ == "__main__":
    main()
