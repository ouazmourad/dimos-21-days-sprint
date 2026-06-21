#!/usr/bin/env python3
# Copyright 2026 Dimensional Inc.
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

"""Diagnose why the Go2 WebRTC connection stalls after signaling.

Connects with the connector directly (verbose, bounded by a timeout) and prints
where it stalls — so we can tell a data-channel/permission problem apart from an
ICE/network one.

Close the Unitree app first (one WebRTC client at a time), and have GO2_AES_KEY
set (same key DimOS uses):

    export GO2_AES_KEY=<key>
    .venv/bin/python tools/go2_connect_diag.py            # uses ROBOT_IP or .123
    .venv/bin/python tools/go2_connect_diag.py 192.168.69.123
"""

import asyncio
import logging
import os
import sys

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")


def main() -> int:
    ip = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("ROBOT_IP", "192.168.69.123")
    key = os.environ.get("GO2_AES_KEY")
    if not key:
        print("Set GO2_AES_KEY first:  export GO2_AES_KEY=<key>")
        return 2

    print(f"Connecting to {ip}  (key {key[:6]}…, 45s timeout). App must be closed.\n")
    from unitree_webrtc_connect.webrtc_driver import (
        UnitreeWebRTCConnection,
        WebRTCConnectionMethod,
    )

    conn = UnitreeWebRTCConnection(WebRTCConnectionMethod.LocalSTA, ip=ip, aes_128_key=key)

    async def go() -> None:
        await asyncio.wait_for(conn.connect(), timeout=45)

    try:
        asyncio.run(go())
        print("\n=== RESULT: CONNECTED OK — the connection itself works. ===")
        return 0
    except asyncio.TimeoutError:
        print(
            "\n=== RESULT: TIMEOUT — signaling succeeded but the peer/data channel "
            "never opened.\n"
            "    If the logs show ICE never reaching 'connected' -> network "
            "(Wi-Fi client isolation, firewall).\n"
            "    If ICE connected but the channel didn't validate -> the "
            "STA-T remote-control permission or the key. ==="
        )
        return 1
    except Exception as e:  # noqa: BLE001
        print(f"\n=== RESULT: {type(e).__name__}: {e} ===")
        return 1


if __name__ == "__main__":
    sys.exit(main())
