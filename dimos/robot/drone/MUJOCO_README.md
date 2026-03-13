# MuJoCo Drone Simulation

Physics-simulated Skydio X2 quadrotor integrated with DimOS. Opens a MuJoCo 3D viewer with a city scene and connects to the Nightwatch agent — type natural language commands at `localhost:5555` and watch the drone fly in real-time.

## Quick Start

```bash
.venv/bin/python dimos/robot/drone/mujoco_sim.py
```

This opens two things:
- **MuJoCo 3D viewer** — the drone hovering in a city scene
- **Nightwatch agent** at http://localhost:5555 — type commands like "fly forward", "go up 2 meters", "draw DimOS in the sky"

## Requirements

- Python 3.12 (project virtualenv)
- `mujoco >= 3.5.0`
- `mujoco_playground` or `mujoco_menagerie` (for the Skydio X2 mesh assets)
- An OpenAI API key set as `OPENAI_API_KEY` (for the Nightwatch agent)

All dependencies are in the project's `.venv`.

## Terminal Keyboard Controls

While the sim is running, the terminal accepts keyboard input:

| Key | Action |
|-----|--------|
| `w` / `s` | Forward / Backward (2 m/s) |
| `q` / `e` | Strafe left / right (2 m/s) |
| `r` / `f` | Altitude up / down (1 m/s) |
| `a` / `d` | Yaw left / right |
| `t` | Takeoff |
| `l` | Land |
| `x` | Stop (hover) |
| `ESC` | Quit |

## Architecture

```
┌─────────────────────┐     UDP 19876      ┌──────────────────────┐
│   DimOS Agent        │ ──────────────────>│  MuJoCo Sim (main)   │
│   (Nightwatch)       │                    │                      │
│                      │                    │  - Physics engine     │
│  MuJocoSkillProxy    │                    │  - 3D viewer          │
│  WebsocketVisModule  │                    │  - PD flight ctrl     │
│  Agent + web_input   │                    │  - DimOSBridge        │
│                      │                    │                      │
│  http://localhost:5555│                    │  MuJoCo window        │
└─────────────────────┘                    └──────────────────────┘
```

The simulation runs in the **main thread** (MuJoCo viewer requires it). The DimOS agent runs in a **background thread**. Commands flow from the agent through `MuJocoSkillProxy` → UDP → `DimOSBridge` → `MuJocoDroneController`.

## Files

| File | Purpose |
|------|---------|
| `mujoco_sim.py` | Standalone runner — 3D viewer + DimOS agent |
| `mujoco_drone.py` | `MuJocoDroneController` (PD flight controller) + `MuJocoDroneModule` (DimOS Module) |
| `mujoco_skill_proxy.py` | Lightweight DimOS Module that sends skill commands via UDP |
| `city_scene.xml` | MuJoCo scene — city with buildings, cars, roads, and the Skydio X2 |
| `assets/` | Drone mesh and texture (copied from mujoco_menagerie) |

## Agent Skills

The Nightwatch agent has these skills available:

- `move(x, y, z, duration)` — velocity command (m/s)
- `move_with_yaw(vx, vy, vz, yaw_rate, duration)` — velocity + yaw rotation
- `takeoff(altitude)` / `land()`
- `arm()` / `disarm()`
- `fly_to(lat, lon, alt)` — GPS waypoint
- `draw_dimos()` — choreographed flight tracing "DimOS" in the air
- `execute_path(moves)` — semicolon-separated velocity segments

## Flight Controller

The `MuJocoDroneController` is a PD attitude controller:

1. **Velocity → tilt angle**: horizontal velocity error mapped to target pitch/roll (clamped to ±23°)
2. **Attitude PD**: proportional on angle error, derivative on angular rate
3. **Yaw heading lock**: holds current heading when no yaw command is active
4. **Yaw feedforward**: compensates for the parasitic yaw torque that roll commands create (due to the X-config motor layout where same-side motors share gear signs)
5. **Motor mixing**: X-configuration mapping to 4 motor thrusts, clipped to [0, 13] N

Pitch and roll are extracted from the **body-frame gravity vector** (not Euler angles), making the controller stable at any yaw orientation.

## City Scene

The drone spawns at `(0, -3, 3)` facing along the road (+Y direction, 90° yaw). The city includes:

- Main road along Y with cross streets
- Buildings: glass skyscraper, office towers, brick apartments, shops, parking garage
- 7 parked cars (sedans, SUVs, taxi)
- Park with trees
- Street lights

All city geometry is visual-only (`contype="0"`) — the drone flies through buildings without collision. Only the ground floor has collision enabled for landing.

## Using as a DimOS Module

`MuJocoDroneModule` is a drop-in replacement for `DroneConnectionModule` with the same stream interface:

```python
from dimos.robot.drone.mujoco_drone import MuJocoDroneModule

# In a blueprint
bp = autoconnect(
    MuJocoDroneModule.blueprint(headless=True),
    WebsocketVisModule.blueprint(),
    agent(system_prompt=DRONE_SYSTEM_PROMPT, model="gpt-4o"),
    web_input(),
)
bp.build().loop()
```

Streams: `odom`, `gps_location`, `video`, `status`, `telemetry` (outputs); `movecmd`, `movecmd_twist`, `gps_goal` (inputs).
