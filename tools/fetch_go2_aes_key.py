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

"""Fetch your Go2's per-device AES-128 key from the Unitree cloud.

Newer Go2 firmware (>= 1.1.15, the "data2=3" LAN handshake) requires a
per-device AES-128 key before DimOS can connect over local WebRTC. This key
lives in your Unitree account (the same login as the mobile app).

Run it LOCALLY. Your password is entered via a hidden prompt and is sent only
to Unitree's cloud (md5-hashed, exactly as the app does) — it never goes
anywhere else. The printed AES key is what DimOS needs:

    python tools/fetch_go2_aes_key.py
    # then, in the shell you launch dimos from:
    export GO2_AES_KEY=<the key it prints>

Region defaults to "global"; pass --region china if your account is on the
China server.
"""

import argparse
import getpass
import sys


def main() -> int:
    ap = argparse.ArgumentParser(description="Fetch the Go2 per-device AES-128 key.")
    ap.add_argument("--region", default="global", help="Unitree account region (global|china)")
    ap.add_argument("--email", default=None, help="account email (omit to be prompted)")
    args = ap.parse_args()

    try:
        from unitree_webrtc_connect.unitree_cloud import UnitreeCloud
    except Exception as e:  # noqa: BLE001
        print(f"Could not import the Unitree cloud client: {e}", file=sys.stderr)
        print("Make sure connector 2.1.2 is installed (unitree-webrtc-connect).", file=sys.stderr)
        return 2

    email = args.email or input("Unitree account email: ").strip()
    password = getpass.getpass("Unitree account password (hidden, stays local): ")

    cloud = UnitreeCloud(region=args.region, device_type="Go2")
    cloud.login_email(email, password)
    devices = cloud.list_devices()

    if not devices:
        print("No robots are bound to this account.")
        return 1

    print(f"\nFound {len(devices)} device(s) on your account:\n")
    for d in devices:
        sn = getattr(d, "sn", "?")
        name = getattr(d, "name", "") or "?"
        key = getattr(d, "key", "") or ""
        print(f"  • {name}   sn={sn}")
        if key:
            print(f"      AES-128 key:  {key}")
        else:
            print("      AES-128 key:  (empty — this robot's firmware is < 1.1.15, "
                  "so it doesn't use data2=3; no key needed)")
    print("\nFor DimOS, in the shell you run `dimos` from:")
    print("  export GO2_AES_KEY=<the AES-128 key above>")
    return 0


if __name__ == "__main__":
    sys.exit(main())
