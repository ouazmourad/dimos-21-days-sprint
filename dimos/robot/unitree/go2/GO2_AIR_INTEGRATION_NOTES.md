# My Go2 Air — Integration Reference

Everything I learned getting my **personal Unitree Go2 Air** to talk to software,
distilled into the facts that are **framework-agnostic** — i.e. what I need to
remember to wire it into *any* stack (not just DimOS). The DimOS-specific glue is
noted only where it points at a reusable fact.

> ⚠️ The single biggest thing: a newer-firmware Go2 (mine) speaks a **different
> local WebRTC handshake** (`data2=3`) that needs a **per-device AES-128 key**, and
> it allows **only one WebRTC client at a time**. Get those two things right and
> everything else follows. Get them wrong and nothing connects.

---

## 1. The robot in one paragraph

A Go2 Air has **no Ethernet SDK2 / DDS path** for control — unlike the EDU/Pro
low-level SDK, the Air is driven entirely over **WebRTC**. One WebRTC peer
connection carries *everything*: movement commands, camera video, lidar, audio,
state. So "linking it to a framework" = "open one WebRTC data channel and speak its
JSON-RPC-ish topic protocol."

| Property | Value |
|---|---|
| Model | Unitree **Go2 Air** |
| Control transport | **WebRTC only** (no SDK2/DDS, no RTSP) |
| Front camera | **1280×720**, `equidistant` (fisheye) distortion |
| Lidar | **Unitree 4D LiDAR L1** (built in; data over `rt/utlidar/*`) |
| Speaker | Yes (`audio_hub` + `audio_player_service` services run) |
| Concurrent WebRTC clients | **1** (the app OR your code, never both) |

---

## 2. Network & connection

- **Connection method:** WebRTC **`LocalSTA`** (a.k.a. STA-L / LAN mode) to the
  robot's IP on your network.
- **Robot IP:** DHCP-assigned on your WiFi — mine came up as **`192.168.69.123`**,
  but *re-check every session*. (When you connect to the robot's **own hotspot**
  instead of your router, the gateway is the documented Go2 default — verify it on
  the unit; don't assume.)
- **Handshake signaling (new firmware, `data2=3`):** `con_notify` on **port 9991**.
- **Old handshake (`data2=2`):** was `/offer` on **port 8081** — **closed** on my
  firmware. If you see "connection refused 8081" / "RSA key format not supported" /
  "Could not get SDP from the peer", you're using an old connector against new
  firmware.

### Firmware / app prerequisites (do these in the Unitree app first)
- Firmware **≥ V1.1.6** → the robot exposes **Motion Services V2.0**.
- In the app, enable **STA-T remote control** *and* **Public Network Remote
  Connection** so the LAN WebRTC offer is accepted.
- Air/Pro must use **WebRTC**, not the local Ethernet SDK2 (that path doesn't exist
  on these models).

---

## 3. The per-device AES-128 key (required on `data2=3`)

Newer firmware (Go2 **≥ 1.1.15**) encrypts the LAN handshake with a key that is
**unique to your robot** and stored in **your Unitree cloud account** (same login
as the mobile app).

How to get it (run locally — password is md5-hashed and sent only to Unitree):

```python
from unitree_webrtc_connect.unitree_cloud import UnitreeCloud
cloud = UnitreeCloud(region="global", device_type="Go2")   # region "china" if applicable
cloud.login_email(email, password)
dev = cloud.list_devices()[0]
print(dev.key)        # <-- this is the AES-128 key
```

Then your connector receives it, e.g.:

```python
conn = UnitreeWebRTCConnection(WebRTCConnectionMethod.LocalSTA, ip=ROBOT_IP, aes_128_key=KEY)
```

> The key is per-robot and stable; fetch once and stash it (I export
> `GO2_AES_KEY=...`). Treat it like a credential — keep it local.

---

## 4. The connector library (what actually works)

| Need | Use |
|---|---|
| Connector | **`unitree-webrtc-connect==2.1.2`** (mainline). |
| HTTP/TLS impersonation dep | **`curl_cffi==0.15.0`** (the handshake mimics a browser). |
| ❌ Avoid | the **`leshy` fork 2.0.7** — only does `data2=2`, won't connect to new firmware. |

`pip install unitree-webrtc-connect==2.1.2 curl_cffi`

---

## 5. The one-connection rule (most common foot-gun)

The Go2 accepts **exactly one** WebRTC client. While the **Unitree app is
connected, your code gets `RobotBusyError`** (and vice-versa). There is **no second
channel** — you cannot "drive from the app while another program watches the
camera." Whoever holds the connection does *everything*.

→ **Before connecting from any framework: fully close the Unitree app.**

---

## 6. Data-channel protocol (topics)

Everything is a request published on a topic over the data channel. The useful ones:

| Purpose | Topic |
|---|---|
| Movement / sport commands | `rt/api/sport/request` (MCF sport api_ids since fw 1.1.7, same topic) |
| Sport state (feedback) | `rt/sportmodestate`, `rt/lf/sportmodestate` |
| Camera / video | `rt/api/videohub/request` (video is a **WebRTC track**, not RTSP) |
| Obstacle avoidance | `rt/api/obstacles_avoid/request` |
| Audio hub | `rt/api/audiohub/request` |
| Voice UI | `rt/api/vui/request` |
| Motion switcher | `rt/api/motion_switcher/request` |
| Lidar (4D LiDAR L1) | `rt/utlidar/switch`, `rt/utlidar/voxel_map`, `rt/utlidar/voxel_map_compressed`, `rt/utlidar/lidar_state` |
| Odometry | `rt/utlidar/robot_pose` |

Movement = publish `Twist`-style velocity via the sport API; the higher-level
helpers map to standup / balance_stand / liedown sport commands.

---

## 7. Camera

- **WebRTC video track only** — subscribe to the video stream after the peer
  connection is up. **RTSP (port 8554) is closed**; there is no separate stream.
- Resolution **1280×720**, fisheye (`equidistant`) — undistort if you need metric
  geometry.
- Because camera and control share the **one** connection, the camera is only
  available to whoever holds that connection (see §5).

---

## 8. Audio (speaker)

The Air **does** have a speaker. Control it via the **AudioHub** on
`rt/api/audiohub/request`. Two ways to make sound:

- **Library playback:** `upload_audio_file(path)` → `get_audio_list()` → find your
  clip → `play_by_uuid(uuid)`.
- **Megaphone (live):** `enter_megaphone()` → `upload_megaphone(path)` →
  `exit_megaphone()`.

AudioHub methods: `get_audio_list`, `play_by_uuid`, `pause`, `resume`,
`set_play_mode`, `get_play_mode`, `rename_record`, `delete_record`,
`upload_audio_file`, `enter_megaphone` / `exit_megaphone` / `upload_megaphone`.

**Gotchas I hit:**
- Audio must be **WAV at 44.1 kHz**. (I resample 24 kHz TTS → 44100 with the stdlib
  `audioop.ratecv`, no ffmpeg needed.)
- `get_audio_list()` returns **nested** `data["data"]` JSON with **UPPERCASE keys**:
  `UNIQUE_ID`, `CUSTOM_NAME`, `ADD_TIME`. Pick the newest by max `ADD_TIME`.
- **There is NO volume API.** Volume is **app-only**. My playback was nearly
  inaudible until I raised the volume **in the app first** — there is no
  programmatic way to do it.

---

## 9. Quick checklist to connect from a fresh framework

1. **Close the Unitree app** (one client only).
2. In-app once: firmware ≥ 1.1.6, enable **STA-T remote control** + **Public Network
   Remote Connection**.
3. `pip install unitree-webrtc-connect==2.1.2 curl_cffi`.
4. Fetch the **per-device AES-128 key** from your Unitree account (§3).
5. Find the robot's **LAN IP** (re-check; DHCP).
6. Connect: `WebRTCConnectionMethod.LocalSTA`, `ip=<robot ip>`, `aes_128_key=<key>`.
7. After the data channel is up: publish to `rt/api/sport/request` to move,
   subscribe to the WebRTC **video track** to see, use `rt/api/audiohub/request`
   to speak.
8. Remember: the Air **has** the 4D LiDAR L1 (`rt/utlidar/*`, enable via switch);
   **volume** is app-only; camera is **fisheye 720p**.

---

## 10. Symptom → cause cheat-sheet

| Symptom | Cause / fix |
|---|---|
| `RobotBusyError` | The Unitree app (or another client) holds the one connection — close it. |
| Refused on 8081 / "RSA key not supported" / "no SDP from peer" | Old `data2=2` connector vs new firmware — use connector 2.1.2 + `curl_cffi`, supply the AES key. |
| Connects but handshake fails on `data2=3` | Missing/wrong **AES-128 key**. |
| "Api not support" in the app when picking region | App/region/firmware mismatch — update the app; toggle the STA-T / public-network settings. |
| Camera never appears | You're expecting RTSP — there isn't one; subscribe to the WebRTC video track. |
| Robot too quiet / silent | No volume API — raise volume **in the app**; confirm `audio_hub` / `audio_player_service` are running. |
| Lidar topics empty | Enable the sensor via `rt/utlidar/switch` — the Air **does** have the 4D LiDAR L1. |

---

*These notes are reconstructed from getting my own Go2 Air online. IPs, firmware
versions, and the AES key are specific to my unit — verify them against the robot
each time. The protocol facts (WebRTC-only, `data2=3` + AES key, one-client limit,
the `rt/api/*` topics, fisheye 720p camera, app-only volume) are the portable part.*
