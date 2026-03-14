#!/usr/bin/env python3
"""
DimOS Night Watch Bridge Script for OpenClaw RoboBot.
Non-blocking, daemon-accelerated command dispatch.

Architecture:
    CLI invocation → Unix socket → persistent daemon (pre-loaded imports) → LCM → MuJoCo
    Fallback: if daemon isn't running, executes directly (slower, ~560ms import).

Usage:
    dimos_bridge.py move <direction> <distance_m> <speed_m_s>
    dimos_bridge.py turn <direction> <degrees>
    dimos_bridge.py stop
    dimos_bridge.py posture <sit|stand>
    dimos_bridge.py status
    dimos_bridge.py camera capture
    dimos_bridge.py patrol start|stop
    dimos_bridge.py ask <question>
    dimos_bridge.py alert check|history [N]
    dimos_bridge.py latency [N]
    dimos_bridge.py daemon                   # Start persistent daemon
"""

import sys
import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Re-exec with DimOS venv Python if needed
# ---------------------------------------------------------------------------

def _find_dimos_root():
    if "DIMOS_ROOT" in os.environ:
        return Path(os.environ["DIMOS_ROOT"])
    candidate = Path(__file__).resolve().parent.parent.parent
    if (candidate / ".venv" / "bin" / "python3").exists():
        return candidate
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
    print("ERROR: Cannot find DimOS root. Set DIMOS_ROOT env var.", file=sys.stderr)
    sys.exit(1)

import json
import math
import socket
import struct
import threading
import time
from datetime import datetime

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MODE = os.environ.get("DIMOS_MODE", "simulation")
ROBOT_IP = os.environ.get("ROBOT_IP", "")
LOG_DIR = Path(__file__).parent.parent / "logs"
LOG_DIR.mkdir(exist_ok=True)

MAX_DISTANCE = 5.0
MAX_SPEED_SIM = 1.0   # Raised from 0.5 — Go2 can handle ~1.5 m/s
MAX_SPEED_HW = 0.3

CMD_VEL_CHANNEL = "/cmd_vel#geometry_msgs.Twist"
HUMAN_INPUT_CHANNEL = "/human_input#builtins.str"

PERSISTENT_MEMORY_DIR = Path.home() / ".local" / "state" / "dimos" / "temporal_memory"
PERSISTENT_JSONL = PERSISTENT_MEMORY_DIR / "temporal_memory.jsonl"
LATENCY_LOG = LOG_DIR / "latency.jsonl"

TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_USER_ID = os.environ.get("TELEGRAM_USER_ID", "")

# Daemon socket path
_SOCK_PATH = Path("/tmp/dimos_bridge.sock")

# ---------------------------------------------------------------------------
# Auto-stop state
# ---------------------------------------------------------------------------
_stop_timer_lock = threading.Lock()
_stop_timer: threading.Timer | None = None


def log_command(action, params, result):
    log_file = LOG_DIR / "command_log.md"
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    entry = f"\n### {ts}\n- **Action:** {action}\n- **Params:** {json.dumps(params)}\n- **Result:** {result}\n- **Mode:** {MODE}\n"
    with open(log_file, "a") as f:
        f.write(entry)


def log_latency(action, params, t_start, t_published, t_returned, via="direct"):
    record = {
        "action": action,
        "params": params,
        "via": via,
        "t_start": t_start,
        "t_published": t_published,
        "t_returned": t_returned,
        "latency_publish_ms": round((t_published - t_start) * 1000, 1),
        "latency_total_ms": round((t_returned - t_start) * 1000, 1),
        "timestamp": datetime.now().isoformat(),
    }
    try:
        with open(LATENCY_LOG, "a") as f:
            f.write(json.dumps(record) + "\n")
    except Exception:
        pass


# ---------------------------------------------------------------------------
# LCM helpers (only imported when actually needed)
# ---------------------------------------------------------------------------

_lcm_instance = None
_twist_module = None


def _ensure_lcm():
    """Lazy-init LCM connection and Twist import (cached for daemon lifetime)."""
    global _lcm_instance, _twist_module
    if _lcm_instance is None:
        import lcm as lcm_lib
        _lcm_instance = lcm_lib.LCM("udpm://239.255.76.67:7667?ttl=0")
    if _twist_module is None:
        from dimos.msgs.geometry_msgs import Twist, Vector3
        _twist_module = (Twist, Vector3)
    return _lcm_instance, _twist_module


def _publish_twist_once(linear_x=0.0, linear_y=0.0, angular_z=0.0):
    lc, (Twist, Vector3) = _ensure_lcm()
    twist = Twist(linear=Vector3(linear_x, linear_y, 0.0), angular=Vector3(0.0, 0.0, angular_z))
    lc.publish(CMD_VEL_CHANNEL, twist.lcm_encode())


def _schedule_auto_stop(duration):
    global _stop_timer
    def _do_stop():
        try:
            _publish_twist_once(0.0, 0.0, 0.0)
        except Exception:
            pass
    with _stop_timer_lock:
        if _stop_timer is not None:
            _stop_timer.cancel()
        _stop_timer = threading.Timer(duration, _do_stop)
        _stop_timer.daemon = True
        _stop_timer.start()


def _cancel_auto_stop():
    global _stop_timer
    with _stop_timer_lock:
        if _stop_timer is not None:
            _stop_timer.cancel()
            _stop_timer = None


def publish_human_input(message):
    try:
        from dimos.core.transport import pLCMTransport
        transport = pLCMTransport("/human_input")
        transport.start()
        transport.lcm.publish("/human_input", message)
        time.sleep(0.1)
        transport.stop()
        return True
    except Exception as e:
        print(f"ERROR publishing to agent: {e}", file=sys.stderr)
        return False


def check_simulation_running():
    try:
        import subprocess
        result = subprocess.run(["pgrep", "-f", "dimos.*simulation|mujoco"], capture_output=True, text=True)
        return result.returncode == 0
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Command handlers (return (exit_code, stdout_text))
# ---------------------------------------------------------------------------

def cmd_status():
    try:
        import dimos
        ver = getattr(dimos, "__version__", "unknown")
        available = True
    except ImportError:
        ver = "not installed"
        available = False
    info = {
        "dimos_installed": available,
        "dimos_version": ver,
        "mode": MODE,
        "simulation_running": check_simulation_running(),
        "robot_ip": ROBOT_IP or "(none — simulation mode)",
        "lcm_channel": CMD_VEL_CHANNEL,
        "temporal_memory_exists": PERSISTENT_JSONL.exists(),
        "daemon_running": _SOCK_PATH.exists(),
        "max_speed": MAX_SPEED_SIM,
        "timestamp": datetime.now().isoformat(),
    }
    text = json.dumps(info, indent=2)
    log_command("status", {}, "success")
    return 0, text


def cmd_move(direction, distance, speed):
    t_start = time.monotonic()
    max_speed = MAX_SPEED_HW if MODE == "hardware" else MAX_SPEED_SIM
    speed = min(speed, max_speed)
    distance = min(distance, MAX_DISTANCE)
    velocity = speed if direction == "forward" else -speed
    duration = distance / speed if speed > 0 else 0

    try:
        _publish_twist_once(linear_x=velocity)
        t_pub = time.monotonic()
        if duration > 0:
            _schedule_auto_stop(duration)
        t_ret = time.monotonic()
        text = f"OK: move {direction} {distance}m at {speed}m/s (auto-stop in {duration:.1f}s)"
        log_command("move", {"direction": direction, "distance": distance, "speed": speed}, "success")
        log_latency("move", {"direction": direction, "distance": distance, "speed": speed}, t_start, t_pub, t_ret)
        return 0, text
    except Exception as e:
        log_command("move", {"direction": direction, "distance": distance, "speed": speed}, f"error: {e}")
        return 1, f"ERROR: {e}"


def cmd_turn(direction, degrees):
    t_start = time.monotonic()
    angular_speed = 0.5
    angular_vel = angular_speed if direction == "left" else -angular_speed
    radians = math.radians(degrees)
    duration = abs(radians) / angular_speed

    try:
        _publish_twist_once(angular_z=angular_vel)
        t_pub = time.monotonic()
        if duration > 0:
            _schedule_auto_stop(duration)
        t_ret = time.monotonic()
        text = f"OK: turn {direction} {degrees}deg (auto-stop in {duration:.1f}s)"
        log_command("turn", {"direction": direction, "degrees": degrees}, "success")
        log_latency("turn", {"direction": direction, "degrees": degrees}, t_start, t_pub, t_ret)
        return 0, text
    except Exception as e:
        log_command("turn", {"direction": direction, "degrees": degrees}, f"error: {e}")
        return 1, f"ERROR: {e}"


def cmd_stop():
    t_start = time.monotonic()
    _cancel_auto_stop()
    try:
        _publish_twist_once(0.0, 0.0, 0.0)
        t_pub = time.monotonic()
        text = "OK: emergency stop — zero velocity published"
        log_command("stop", {}, "success")
        log_latency("stop", {}, t_start, t_pub, time.monotonic())
        return 0, text
    except Exception as e:
        log_command("stop", {}, f"error: {e}")
        return 1, f"ERROR: {e}"


def cmd_posture(pose):
    valid = ["sit", "stand"]
    if pose not in valid:
        return 1, f"ERROR: Unknown posture '{pose}'. Use: {', '.join(valid)}"
    log_command("posture", {"pose": pose}, "success")
    return 0, f"OK: posture command -> {pose}"


def cmd_camera_capture():
    log_command("camera", {"action": "capture"}, "success")
    return 0, "OK: camera capture requested"


# ---------------------------------------------------------------------------
# Person walking simulation
# ---------------------------------------------------------------------------
_person_walk_thread: threading.Thread | None = None
_person_walk_stop = threading.Event()


def cmd_person(action):
    """Start or stop a simulated person walking around the scene."""
    global _person_walk_thread

    if action == "start":
        if _person_walk_thread and _person_walk_thread.is_alive():
            return 0, "OK: person is already walking"

        _person_walk_stop.clear()

        def _walk_loop():
            from dimos.simulation.mujoco.person_on_track import PersonTrackPublisher
            # Simple back-and-forth along open corridor (avoids furniture)
            track = [
                (1.0, 0.0),
                (-1.0, 0.0),
            ]
            pub = PersonTrackPublisher(track)
            while not _person_walk_stop.is_set():
                pub.tick()
                time.sleep(1 / 60)
            pub.stop()

        _person_walk_thread = threading.Thread(target=_walk_loop, daemon=True)
        _person_walk_thread.start()
        log_command("person", {"action": "start"}, "success")
        return 0, "OK: person started walking a patrol route"

    elif action == "stop":
        _person_walk_stop.set()
        _person_walk_thread = None
        log_command("person", {"action": "stop"}, "success")
        return 0, "OK: person stopped"

    return 1, f"ERROR: Unknown person action '{action}'. Use: start, stop"


def cmd_patrol(action):
    if action == "start":
        msg = ("Begin your Night Watch patrol now. "
               "Start exploration and continuously look out for: "
               "fallen furniture, unknown people, fallen objects on floor, "
               "open doors, anything out of place, spills or messes. "
               "Announce all findings via speak. Tag anomaly locations.")
        if publish_human_input(msg):
            log_command("patrol", {"action": "start"}, "success")
            return 0, "OK: Patrol start command sent to Night Watch agent"
        log_command("patrol", {"action": "start"}, "error: failed to publish")
        return 1, "ERROR: Failed to send patrol start"
    elif action == "stop":
        msg = ("Stop patrol immediately. Call stop_following if following anyone, "
               "call end_exploration, and speak 'Night Watch standing down.' "
               "Wait for further instructions.")
        if publish_human_input(msg):
            log_command("patrol", {"action": "stop"}, "success")
            return 0, "OK: Patrol stop command sent to Night Watch agent"
        log_command("patrol", {"action": "stop"}, "error: failed to publish")
        return 1, "ERROR: Failed to send patrol stop"
    return 1, f"ERROR: Unknown patrol action '{action}'. Use: start, stop"


def cmd_ask(question):
    if publish_human_input(question):
        log_command("ask", {"question": question}, "success")
        return 0, f"OK: Question sent to Night Watch agent: {question}"
    log_command("ask", {"question": question}, "error: failed to publish")
    return 1, "ERROR: Failed to send question"


def cmd_alert(action, count=50):
    if action not in ("check", "history"):
        return 1, f"ERROR: Unknown alert action '{action}'. Use: check, history"
    if not PERSISTENT_JSONL.exists():
        return 0, "No temporal memory data yet. Is Night Watch running?"
    events = []
    try:
        with open(PERSISTENT_JSONL, "r") as f:
            lines = f.readlines()
        for line in lines[-count:]:
            line = line.strip()
            if line:
                try:
                    events.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    except Exception as e:
        return 1, f"ERROR reading temporal memory: {e}"
    if not events:
        return 0, "No events recorded yet."
    parts = [f"Last {len(events)} temporal memory events:"]
    for ev in events:
        ts = ev.get("timestamp", "unknown")
        etype = ev.get("type", "unknown")
        if "caption" in ev:
            parts.append(f"  [{ts}] {etype}: {ev['caption'][:120]}")
        elif "entities" in ev:
            enames = [e.get("descriptor", e.get("id", "?")) for e in ev["entities"]]
            parts.append(f"  [{ts}] {etype}: entities={enames}")
        else:
            parts.append(f"  [{ts}] {etype}: {json.dumps(ev)[:120]}")
    log_command("alert", {"action": action, "events_found": len(events)}, "success")
    return 0, "\n".join(parts)


def cmd_latency(count=20):
    if not LATENCY_LOG.exists():
        return 0, "No latency data yet. Run some move/turn/stop commands first."
    records = []
    try:
        with open(LATENCY_LOG, "r") as f:
            lines = f.readlines()
        for line in lines[-count:]:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    except Exception as e:
        return 1, f"ERROR reading latency log: {e}"
    if not records:
        return 0, "No latency records yet."
    parts = [f"Last {len(records)} bridge latency records:",
             f"  {'Action':<8} {'Via':<8} {'Pub ms':>7} {'Total ms':>9}  Params",
             f"  {'-'*8} {'-'*8} {'-'*7} {'-'*9}  {'-'*30}"]
    for r in records:
        a = r.get("action", "?")
        v = r.get("via", "?")
        p = r.get("latency_publish_ms", "?")
        t = r.get("latency_total_ms", "?")
        params = json.dumps(r.get("params", {}))[:40]
        parts.append(f"  {a:<8} {v:<8} {p:>7} {t:>9}  {params}")
    pub_times = [r["latency_publish_ms"] for r in records if "latency_publish_ms" in r]
    if pub_times:
        parts.append(f"\n  Avg publish: {sum(pub_times)/len(pub_times):.1f}ms  Max: {max(pub_times):.1f}ms")
    return 0, "\n".join(parts)


# ---------------------------------------------------------------------------
# Command dispatcher
# ---------------------------------------------------------------------------

def dispatch(argv):
    """Parse argv and run the appropriate command. Returns (exit_code, output_text)."""
    if len(argv) < 2:
        return 1, "Usage: dimos_bridge.py <command> [args...]\nCommands: move, turn, stop, posture, status, camera, patrol, ask, alert, latency, daemon"

    cmd = argv[1].lower()

    if cmd == "status":
        return cmd_status()
    elif cmd == "stop":
        return cmd_stop()
    elif cmd == "move":
        if len(argv) < 5:
            return 1, "Usage: dimos_bridge.py move <forward|backward> <distance_m> <speed_m_s>"
        return cmd_move(argv[2].lower(), float(argv[3]), float(argv[4]))
    elif cmd == "turn":
        if len(argv) < 4:
            return 1, "Usage: dimos_bridge.py turn <left|right> <degrees>"
        return cmd_turn(argv[2].lower(), float(argv[3]))
    elif cmd == "posture":
        if len(argv) < 3:
            return 1, "Usage: dimos_bridge.py posture <sit|stand>"
        return cmd_posture(argv[2].lower())
    elif cmd == "camera":
        return cmd_camera_capture()
    elif cmd == "patrol":
        if len(argv) < 3:
            return 1, "Usage: dimos_bridge.py patrol <start|stop>"
        return cmd_patrol(argv[2].lower())
    elif cmd == "ask":
        if len(argv) < 3:
            return 1, "Usage: dimos_bridge.py ask <question>"
        return cmd_ask(" ".join(argv[2:]))
    elif cmd == "alert":
        if len(argv) < 3:
            return cmd_alert("check")
        action = argv[2].lower()
        count = int(argv[3]) if len(argv) > 3 else 50
        return cmd_alert(action, count)
    elif cmd == "latency":
        count = int(argv[2]) if len(argv) > 2 else 20
        return cmd_latency(count)
    elif cmd == "person":
        if len(argv) < 3:
            return 1, "Usage: dimos_bridge.py person <start|stop>"
        return cmd_person(argv[2].lower())
    else:
        return 1, f"Unknown command: {cmd}"


# ---------------------------------------------------------------------------
# Daemon: persistent process that pre-loads all imports and listens on a
# Unix socket. Each client sends a JSON-encoded argv, gets back a JSON
# response with exit_code and output. Eliminates ~560ms import cost.
# ---------------------------------------------------------------------------

def _daemon_serve():
    """Run the bridge daemon (foreground, blocks until killed)."""
    # Pre-load heavy imports now so commands are instant
    print("Daemon: pre-loading imports...", flush=True)
    t0 = time.monotonic()
    _ensure_lcm()
    elapsed = time.monotonic() - t0
    print(f"Daemon: ready in {elapsed*1000:.0f}ms (LCM + geometry_msgs cached)", flush=True)

    # Clean up stale socket
    if _SOCK_PATH.exists():
        _SOCK_PATH.unlink()

    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server.bind(str(_SOCK_PATH))
    server.listen(4)
    print(f"Daemon: listening on {_SOCK_PATH}", flush=True)

    try:
        while True:
            conn, _ = server.accept()
            try:
                raw = b""
                while True:
                    chunk = conn.recv(4096)
                    if not chunk:
                        break
                    raw += chunk
                    # Protocol: 4-byte length prefix + JSON payload
                    if len(raw) >= 4:
                        msg_len = struct.unpack("!I", raw[:4])[0]
                        if len(raw) >= 4 + msg_len:
                            break

                if len(raw) < 4:
                    continue
                msg_len = struct.unpack("!I", raw[:4])[0]
                payload = raw[4:4 + msg_len]
                argv = json.loads(payload)

                t_start = time.monotonic()
                code, text = dispatch(argv)
                t_done = time.monotonic()

                resp = json.dumps({"code": code, "text": text,
                                   "daemon_dispatch_ms": round((t_done - t_start) * 1000, 2)})
                resp_bytes = resp.encode()
                conn.sendall(struct.pack("!I", len(resp_bytes)) + resp_bytes)
            except Exception as e:
                try:
                    err = json.dumps({"code": 1, "text": f"Daemon error: {e}", "daemon_dispatch_ms": -1})
                    err_bytes = err.encode()
                    conn.sendall(struct.pack("!I", len(err_bytes)) + err_bytes)
                except Exception:
                    pass
            finally:
                conn.close()
    except KeyboardInterrupt:
        print("\nDaemon: shutting down")
    finally:
        server.close()
        if _SOCK_PATH.exists():
            _SOCK_PATH.unlink()


def _client_send(argv):
    """Send a command to the daemon and return (exit_code, output_text)."""
    t_client_start = time.monotonic()
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    sock.settimeout(5.0)
    sock.connect(str(_SOCK_PATH))

    payload = json.dumps(argv).encode()
    sock.sendall(struct.pack("!I", len(payload)) + payload)
    sock.shutdown(socket.SHUT_WR)  # Signal end of request

    raw = b""
    while True:
        chunk = sock.recv(4096)
        if not chunk:
            break
        raw += chunk
        if len(raw) >= 4:
            msg_len = struct.unpack("!I", raw[:4])[0]
            if len(raw) >= 4 + msg_len:
                break
    sock.close()

    msg_len = struct.unpack("!I", raw[:4])[0]
    resp = json.loads(raw[4:4 + msg_len])
    t_client_end = time.monotonic()

    daemon_ms = resp.get("daemon_dispatch_ms", -1)
    client_ms = round((t_client_end - t_client_start) * 1000, 1)

    # Annotate output with timing
    text = resp["text"]
    if resp["code"] == 0 and daemon_ms >= 0:
        text += f" [{daemon_ms}ms daemon, {client_ms}ms total]"

    return resp["code"], text


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    if len(sys.argv) >= 2 and sys.argv[1].lower() == "daemon":
        _daemon_serve()
        return 0

    # Try daemon first (fast path: ~2-5ms total)
    if _SOCK_PATH.exists():
        try:
            code, text = _client_send(sys.argv)
            print(text)
            return code
        except (ConnectionRefusedError, FileNotFoundError, OSError):
            # Daemon not actually running, stale socket — fall through
            pass

    # Direct execution fallback (~560ms import cost)
    code, text = dispatch(sys.argv)
    print(text)
    return code


if __name__ == "__main__":
    sys.exit(main())
