#!/usr/bin/env python3
"""Control the Unitree R1 from the laptop by proxying to PC1's native loco client.

Why this exists
---------------
The R1's main board (192.168.123.161) speaks ``unitree_hg``/CycloneDDS, but the
laptop's pip ``unitree-sdk2py-dimos`` (1.0.3) is OLDER than the R1 firmware, so
CycloneDDS XTypes type-consistency rejects every endpoint match and commands
never reach the robot. PC1 — the onboard Jetson at 192.168.123.164 — ships the
robot's *matching* ``unitree_sdk2`` with a prebuilt ``r1_loco_client``. This
tool SSHes into PC1 and runs that client, so you can drive the R1 from the
laptop until DimOS has a firmware-matching SDK of its own.

Safety
------
``start``, ``stand_up``, ``move`` and ``set_velocity`` cause real motion on a
humanoid that can fall. Keep ~2 m clear and the e-stop/remote in hand.

Usage
-----
    python tools/r1_control.py get_fsm_id          # read current FSM state (no motion)
    python tools/r1_control.py get_fsm_mode
    python tools/r1_control.py damp                # FSM 1  (compliant hold)
    python tools/r1_control.py stand_up            # FSM 4  (rise) -- MOTION
    python tools/r1_control.py start               # FSM 811 (ready/walk) -- MOTION
    python tools/r1_control.py zero_torque         # FSM 0  (limp)
    python tools/r1_control.py move 0.2 0 0        # vx vy omega, ~1 s  -- MOTION
    python tools/r1_control.py set_velocity 0.2 0 0 1.5
    python tools/r1_control.py stop                # stop_move
    python tools/r1_control.py set_fsm_id 4
    python tools/r1_control.py speed_mode 1

Env overrides: R1_PC1_HOST, R1_PC1_USER, R1_PC1_PASS, R1_PC1_IFACE.
"""

import os
import sys

HOST = os.environ.get("R1_PC1_HOST", "192.168.123.164")
USER = os.environ.get("R1_PC1_USER", "unitree")
PW = os.environ.get("R1_PC1_PASS", "123")
IFACE = os.environ.get("R1_PC1_IFACE", "eth10")
CLIENT = "~/unitree_sdk2/build/bin/r1_loco_client"

# verbs that map straight to a `--<verb>` flag with no value
SIMPLE = {"get_fsm_id", "get_fsm_mode", "damp", "start", "stand_up", "zero_torque", "stop_move"}


def build_flag(argv: list[str]) -> str:
    cmd, rest = argv[0], argv[1:]
    if cmd == "stop":
        cmd = "stop_move"
    if cmd in SIMPLE:
        return f"--{cmd}"
    if cmd == "move":
        vx, vy, om = (rest + ["0", "0", "0"])[:3]
        return f'--move="{vx} {vy} {om}"'
    if cmd == "set_velocity":
        if not 3 <= len(rest) <= 4:
            raise SystemExit("set_velocity needs: vx vy omega [duration]")
        return f'--set_velocity="{" ".join(rest)}"'
    if cmd == "set_fsm_id":
        return f"--set_fsm_id={int(rest[0])}"
    if cmd == "speed_mode":
        return f"--set_speed_mode={int(rest[0])}"
    raise SystemExit(f"unknown command: {cmd!r}\n\n{__doc__}")


def main() -> int:
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
        print(__doc__)
        return 2

    try:
        import paramiko
    except ImportError:
        raise SystemExit("paramiko required: pip install paramiko")

    flag = build_flag(sys.argv[1:])
    full = f"{CLIENT} --network_interface={IFACE} {flag}"

    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(HOST, username=USER, password=PW, timeout=10)
    print(f"[r1] {USER}@{HOST}$ {full}")
    _stdin, stdout, stderr = client.exec_command(full, timeout=30)
    out = stdout.read().decode(errors="replace")
    err = stderr.read().decode(errors="replace")
    rc = stdout.channel.recv_exit_status()
    client.close()

    print(out, end="" if out.endswith("\n") else "\n")
    if err.strip():
        print("[stderr]", err.strip())
    return rc


if __name__ == "__main__":
    sys.exit(main())
