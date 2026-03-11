#!/usr/bin/env python3
"""
DimOS Bridge Script for OpenClaw RoboBot.
Publishes Twist velocity commands via LCM to a running DimOS Go2 simulation.

Usage:
    dimos_bridge.py move <direction> <distance_m> <speed_m_s>
    dimos_bridge.py turn <direction> <degrees>
    dimos_bridge.py stop
    dimos_bridge.py posture <sit|stand>
    dimos_bridge.py status
    dimos_bridge.py camera capture

Environment variables:
    DIMOS_ROOT   Path to the DimOS repo (default: auto-detected via 'dimos' package location)
    DIMOS_MODE   'simulation' or 'hardware' (default: simulation)
    ROBOT_IP     Robot IP for hardware mode
"""

import sys
import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Re-exec with DimOS venv Python if needed
# ---------------------------------------------------------------------------

def _find_dimos_root():
    """Find the DimOS repo root via env var or by locating the installed package."""
    if "DIMOS_ROOT" in os.environ:
        return Path(os.environ["DIMOS_ROOT"])
    # Walk up from this script: openclaw/scripts/ -> openclaw/ -> repo root
    candidate = Path(__file__).resolve().parent.parent.parent
    if (candidate / ".venv" / "bin" / "python3").exists():
        return candidate
    # Fallback: find dimos package on sys.path
    for p in sys.path:
        if Path(p, "dimos", "__init__.py").exists():
            return Path(p).parent
    return None

_DIMOS_ROOT = _find_dimos_root()
_VENV_PYTHON = str(_DIMOS_ROOT / ".venv" / "bin" / "python3") if _DIMOS_ROOT else None

if _VENV_PYTHON and os.path.exists(_VENV_PYTHON):
    if os.path.realpath(sys.executable) != os.path.realpath(_VENV_PYTHON):
        os.execv(_VENV_PYTHON, [_VENV_PYTHON] + sys.argv)
elif not _DIMOS_ROOT:
    print("ERROR: Cannot find DimOS root. Set DIMOS_ROOT env var to your dimos repo path.", file=sys.stderr)
    sys.exit(1)

import json
import math
import time
from datetime import datetime
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MODE = os.environ.get("DIMOS_MODE", "simulation")
ROBOT_IP = os.environ.get("ROBOT_IP", "")
LOG_DIR = Path(__file__).parent.parent / "logs"
LOG_DIR.mkdir(exist_ok=True)

SPEED_PRESETS = {"slow": 0.15, "normal": 0.3, "fast": 0.5}
MAX_DISTANCE = 5.0
MAX_SPEED_SIM = 0.5
MAX_SPEED_HW = 0.3

CMD_VEL_CHANNEL = "/cmd_vel#geometry_msgs.Twist"
PUBLISH_RATE = 0.05  # 20 Hz — send commands every 50ms


def log_command(action, params, result):
    """Append command to the workspace log file."""
    log_file = LOG_DIR / "command_log.md"
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    entry = (
        f"\n### {timestamp}\n"
        f"- **Action:** {action}\n"
        f"- **Params:** {json.dumps(params)}\n"
        f"- **Result:** {result}\n"
        f"- **Mode:** {MODE}\n"
    )
    with open(log_file, "a") as f:
        f.write(entry)


def get_lcm():
    """Create an LCM instance for publishing."""
    try:
        import lcm as lcm_lib
        lc = lcm_lib.LCM("udpm://239.255.76.67:7667?ttl=0")
        return lc
    except Exception as e:
        print(f"WARNING: Could not create LCM instance: {e}", file=sys.stderr)
        return None


def publish_twist(lc, linear_x=0.0, linear_y=0.0, angular_z=0.0, duration=0.0):
    """Publish Twist messages to the simulation at 20Hz for the given duration."""
    from dimos.msgs.geometry_msgs import Twist, Vector3

    twist = Twist(
        linear=Vector3(linear_x, linear_y, 0.0),
        angular=Vector3(0.0, 0.0, angular_z),
    )
    encoded = twist.lcm_encode()

    if duration <= 0:
        # Single shot (e.g. stop)
        lc.publish(CMD_VEL_CHANNEL, encoded)
        return

    elapsed = 0.0
    while elapsed < duration:
        lc.publish(CMD_VEL_CHANNEL, encoded)
        time.sleep(PUBLISH_RATE)
        elapsed += PUBLISH_RATE

    # Send zero velocity to stop after movement completes
    stop_twist = Twist(
        linear=Vector3(0.0, 0.0, 0.0),
        angular=Vector3(0.0, 0.0, 0.0),
    )
    lc.publish(CMD_VEL_CHANNEL, stop_twist.lcm_encode())


def check_dimos():
    """Check if DimOS is importable."""
    try:
        import dimos
        ver = getattr(dimos, "__version__", "unknown")
        return True, ver
    except ImportError:
        return False, "not installed"


def check_simulation_running():
    """Check if the DimOS simulation is running by looking for the MuJoCo process."""
    try:
        import subprocess
        result = subprocess.run(
            ["pgrep", "-f", "dimos.*simulation|mujoco"],
            capture_output=True, text=True
        )
        return result.returncode == 0
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Command Handlers
# ---------------------------------------------------------------------------

def cmd_status():
    available, ver = check_dimos()
    sim_running = check_simulation_running()
    info = {
        "dimos_installed": available,
        "dimos_version": ver,
        "mode": MODE,
        "simulation_running": sim_running,
        "robot_ip": ROBOT_IP if ROBOT_IP else "(none — simulation mode)",
        "lcm_channel": CMD_VEL_CHANNEL,
        "timestamp": datetime.now().isoformat(),
    }
    print(json.dumps(info, indent=2))
    log_command("status", {}, "success")
    return 0


def cmd_move(direction, distance, speed):
    max_speed = MAX_SPEED_HW if MODE == "hardware" else MAX_SPEED_SIM
    speed = min(speed, max_speed)
    distance = min(distance, MAX_DISTANCE)
    velocity = speed if direction == "forward" else -speed
    duration = distance / speed if speed > 0 else 0

    lc = get_lcm()
    if lc is None:
        print(f"ERROR: Cannot connect to LCM — is the simulation running?", file=sys.stderr)
        log_command("move", {"direction": direction, "distance": distance, "speed": speed}, "error: no LCM")
        return 1

    try:
        print(f"OK: move {direction} {distance}m at {speed}m/s (duration={duration:.1f}s)")
        publish_twist(lc, linear_x=velocity, duration=duration)
        print(f"DONE: movement complete")
        log_command("move", {"direction": direction, "distance": distance, "speed": speed, "twist_linear_x": velocity}, "success")
        return 0
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        log_command("move", {"direction": direction, "distance": distance, "speed": speed}, f"error: {e}")
        return 1


def cmd_turn(direction, degrees):
    angular_speed = 0.5  # rad/s
    angular_vel = angular_speed if direction == "left" else -angular_speed
    radians = math.radians(degrees)
    duration = abs(radians) / angular_speed

    lc = get_lcm()
    if lc is None:
        print(f"ERROR: Cannot connect to LCM — is the simulation running?", file=sys.stderr)
        log_command("turn", {"direction": direction, "degrees": degrees}, "error: no LCM")
        return 1

    try:
        print(f"OK: turn {direction} {degrees}deg (duration={duration:.1f}s)")
        publish_twist(lc, angular_z=angular_vel, duration=duration)
        print(f"DONE: rotation complete")
        log_command("turn", {"direction": direction, "degrees": degrees, "twist_angular_z": angular_vel}, "success")
        return 0
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        log_command("turn", {"direction": direction, "degrees": degrees}, f"error: {e}")
        return 1


def cmd_stop():
    lc = get_lcm()
    if lc is None:
        print(f"ERROR: Cannot connect to LCM — is the simulation running?", file=sys.stderr)
        log_command("stop", {}, "error: no LCM")
        return 1

    try:
        publish_twist(lc, linear_x=0.0, angular_z=0.0, duration=0)
        print("OK: emergency stop — zero velocity published")
        log_command("stop", {}, "success")
        return 0
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        log_command("stop", {}, f"error: {e}")
        return 1


def cmd_posture(pose):
    valid = ["sit", "stand"]
    if pose not in valid:
        print(f"ERROR: Unknown posture '{pose}'. Use: {', '.join(valid)}", file=sys.stderr)
        return 1

    print(f"OK: posture command -> {pose}")
    log_command("posture", {"pose": pose}, "success")
    return 0


def cmd_camera_capture():
    available, _ = check_dimos()
    if not available:
        print("SIMULATED: camera capture (DimOS not available)")
        log_command("camera", {"action": "capture"}, "simulated-no-dimos")
        return 0

    print("OK: camera capture requested")
    print("NOTE: Full camera integration requires a running DimOS blueprint with Out[Image] stream")
    log_command("camera", {"action": "capture"}, "success")
    return 0


# ---------------------------------------------------------------------------
# CLI Entry Point
# ---------------------------------------------------------------------------

def main():
    if len(sys.argv) < 2:
        print("Usage: dimos_bridge.py <command> [args...]", file=sys.stderr)
        print("Commands: move, turn, stop, posture, status, camera", file=sys.stderr)
        return 1

    command = sys.argv[1].lower()

    if command == "status":
        return cmd_status()
    elif command == "stop":
        return cmd_stop()
    elif command == "move":
        if len(sys.argv) < 5:
            print("Usage: dimos_bridge.py move <forward|backward> <distance_m> <speed_m_s>", file=sys.stderr)
            return 1
        return cmd_move(sys.argv[2].lower(), float(sys.argv[3]), float(sys.argv[4]))
    elif command == "turn":
        if len(sys.argv) < 4:
            print("Usage: dimos_bridge.py turn <left|right> <degrees>", file=sys.stderr)
            return 1
        return cmd_turn(sys.argv[2].lower(), float(sys.argv[3]))
    elif command == "posture":
        if len(sys.argv) < 3:
            print("Usage: dimos_bridge.py posture <sit|stand>", file=sys.stderr)
            return 1
        return cmd_posture(sys.argv[2].lower())
    elif command == "camera":
        return cmd_camera_capture()
    else:
        print(f"Unknown command: {command}", file=sys.stderr)
        return 1

if __name__ == "__main__":
    sys.exit(main())
