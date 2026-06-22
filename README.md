# Unitree R1 — the talking, promptable humanoid

DimOS integration for the **Unitree R1 EDU**: a Claude-powered agent you chat with
from your laptop that makes the robot **speak through its own onboard speaker**,
**move and turn**, **perform arm gestures**, and **see through its front camera**.

This is the setup guide written from how this was actually built and run on a real
R1 EDU, with the gotchas called out — because it is **not** plug-and-play, and
knowing *why* will save you hours.

> **Base install is the upstream source of truth.** Everything *R1-specific* below
> (the SSH-to-PC1 architecture, the IPs, `eth10`, the FSM modes, `paramiko`, the
> blueprint names) is unique to this branch and won't be in the upstream docs. For
> the canonical DimOS install, follow the repository's top-level `README.md`.

---

## 0. What you're actually building (read this first)

You run **DimOS** (the robot framework) on your **laptop**. The laptop is the
"brain": it runs a Claude-powered agent you chat with, and sends commands to the
robot.

The catch that shapes everything: the R1's Python SDK (`unitree_sdk2` pip, v1.0.3)
has a **DDS type mismatch** that stops a laptop from driving the robot directly. So
instead, DimOS **SSHes into the robot's own onboard computer (PC1, a Jetson)** and
runs tiny native C++ helper programs *there* — those talk to the robot correctly
over its internal network.

So:

- **Laptop = brain** (DimOS + Claude)
- **PC1 = hands & mouth** (native clients)

Speech, movement, arm gestures, and the camera all flow through that SSH bridge to
PC1. This means you need the **EDU** version — it gives you SSH access to PC1. A
non-EDU R1 won't work this way.

```
┌─────────────┐   chat / prompts    ┌──────────────────┐
│   Laptop    │  ─────────────────► │   You (operator) │
│             │                     └──────────────────┘
│  DimOS      │   SSH (paramiko)        internal LAN
│  + Claude   │  ───────────────►  ┌──────────────────────────────┐
│  agent      │   runs native      │ R1 robot (192.168.123.x)     │
└─────────────┘   helper clients   │  • PC1 Jetson  .164  (eth10) │
                                    │  • Mainboard   .161          │
                                    └──────────────────────────────┘
```

---

## 1. Network setup

The R1 has two addresses on its internal LAN (`192.168.123.x`):

| Component | Address |
|---|---|
| Mainboard / robot controller | `192.168.123.161` |
| PC1 (onboard Jetson computer) | `192.168.123.164` |

1. Power on the R1 and let it fully boot.
2. Connect your laptop to the robot's network (ethernet to the robot, or its WiFi).
   Your laptop should get a `192.168.123.x` address.
3. **Verify you can reach PC1** — this is the make-or-break step:
   ```bash
   ssh unitree@192.168.123.164
   # password: 123   (Unitree default on the isolated robot LAN)
   ```
   If this works, you're 80% of the way there. If it doesn't, nothing else will —
   fix this first (check the ethernet link, confirm PC1 booted).
4. From your SSH session, confirm PC1 can see the robot:
   ```bash
   ping 192.168.123.161
   ```

---

## 2. Prepare PC1 (the onboard Jetson)

While SSHed into PC1, make sure the native SDK is present and built:

```bash
# On PC1:
ls ~/unitree_sdk2/build/bin     # should exist

# If unitree_sdk2 isn't built yet:
cd ~/unitree_sdk2 && mkdir -p build && cd build && cmake .. && make -j
```

DimOS will **auto-build** the small helper clients (`r1_audio_client`, etc.) from
embedded source on first run — but only if `~/unitree_sdk2` itself is present and
compiled. The robot's internal network interface on PC1 is **`eth10`** (the helpers
are invoked with `--network_interface=eth10`).

---

## 3. Laptop: clone the code & install DimOS

```bash
git clone https://github.com/ouazmourad/dimos-21-days-sprint.git
cd dimos-21-days-sprint
git checkout unitree-r1-integration

# Set up DimOS per the repo's top-level README (Python 3.12 venv + install).
# Typically:
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e .

# R1-specific extra — the SSH bridge needs paramiko:
pip install paramiko
```

After install, the `dimos` CLI is available.

---

## 4. API key (the agent's brain)

The agent uses Claude, so set your own key:

```bash
export ANTHROPIC_API_KEY=sk-ant-...
```

The R1 speaks with its **own onboard speaker** via the native audio client, so you
do **not** need a separate TTS service for speech.

---

## 5. Put the robot in a safe state

- Clear space around it; keep the e-stop within reach.
- Bring it up to standing (damping → stand). The agent's `move` skill auto-enters
  the robot's "Start" locomotion mode (**FSM 811**) before driving.
- ⚠️ **Arm gestures only work when the robot is in the correct FSM mode** — if you
  trigger one off-locomotion you'll see error **`7404`**. Stand it up first.

---

## 6. Run it

Start with the **lean speech + chat** blueprint (no camera — simplest, matches
"it talks and I prompt it"):

```bash
dimos run unitree-r1-control
```

Once it's up, open the **chat UI at http://localhost:5555** and type prompts. Ask
it to introduce itself or describe an action — it replies **and speaks through the
robot's speaker**, and can move/turn/do arm gestures on command.

When that works, graduate to the **full agentic** blueprint, which adds the robot's
front camera (bridged through PC1) so it can see:

```bash
dimos --rerun-open web run unitree-r1-agentic
```

`--rerun-open web` serves the Rerun viewer in your browser; it **auto-opens at
http://localhost:9878** (camera feed + 3D world view). Use the web viewer here, not
the bundled native `dimos-viewer` binary — it's version-incompatible on this setup
and never starts.

---

## Available blueprints

| Blueprint | What it is | Use it for |
|---|---|---|
| **`unitree-r1-control`** | Lean: R1 connection (PC1 backend) + Claude MCP agent + chat/voice + loco/arm skills. **No camera/viz.** Chat UI at `:5555`. | The first thing to run — "it talks, moves, and I prompt it." |
| **`unitree-r1-agentic`** | Full stack: `unitree-r1-basic` + agentic skills, **adds the front camera** (via PC1), mapping, and the Rerun viewer (`:9878`). | When you want the robot to *see* as well as talk and move. |
| `unitree-r1-basic` | `unitree-r1-primitive-no-nav` + `R1Connection`. Building block. | Composition only — not a direct run target. |
| `unitree-r1-primitive-no-nav` | Minimal viz + camera + voxel/costmap/frontier modules; no connection of its own. Building block. | Composition only — not a direct run target. |

---

## 7. What works vs. what doesn't (honest status)

| Capability | Status |
|---|---|
| Robot speaks from its **own speaker** | ✅ Works (confirmed) |
| Chat-prompt the agent (Claude) | ✅ Works |
| Move / turn on command | ✅ Works (auto-enters Start mode, FSM 811) |
| Arm gestures | ✅ Works, but **FSM-gated** (stand up first, else err 7404) |
| Front camera into DimOS | ✅ Works (via PC1 bridge, `unitree-r1-agentic`) |
| **Dex3 hands** (e.g. "hold a box") | ❌ Not implemented yet |
| Direct laptop→robot control (no SSH) | ❌ Blocked by the SDK type mismatch — that's why we proxy through PC1 |

---

## 8. Troubleshooting

| Symptom | Fix |
|---|---|
| **SSH to PC1 fails** | Robot not fully booted, or the ethernet link is down. This blocks *everything* — fix first. |
| **Robot doesn't speak** | Check the speaker volume (the audio client sets it, default 85; you can raise it). Confirm the `voice`/audio services are running on the robot. |
| **Helper binary missing on PC1** | Confirm `~/unitree_sdk2` is built (Step 2); DimOS builds the small client from embedded source but needs the SDK there. |
| **Arm command does nothing / error 7404** | Robot isn't in the locomotion FSM mode — stand it up. |
| **Rerun viewer never opens / `--connect` error** | Use `--rerun-open web` (browser viewer at `:9878`); the native `dimos-viewer` binary is version-incompatible here. |
| **IPs / interface differ on your unit** | The `192.168.123.161` / `.164` and `eth10` values are the defaults that worked for us — verify them against your own robot. |

---

## How the R1-specific pieces fit together

| File | Role |
|---|---|
| `connection.py`, `connection_spec.py` | `R1Connection` — the "pc1" SSH-proxy backend that drives the real robot. |
| `effectors/high_level/dds_sdk.py`, `loco_proxy.py` | Loco FSM ids/api + arm api routing; `move()` auto-enters Start mode (FSM 811). |
| `effectors/high_level/speak_proxy.py` | Onboard-speaker TTS via a PC1 `r1_audio_client` (auto-provisioned/built from embedded source). |
| `sensors/pc1_camera.py` | `R1PC1Camera` — front camera bridged into DimOS through PC1 (throttled to 5 fps / 640×360). |
| `skill_container.py`, `system_prompt.py` | R1 agent skills (incl. real arm-gesture table + speak skill) and prompt. |
| `blueprints/` | The four blueprints in the table above. |
| `tools/r1_*.py` (repo root) | Standalone scripts: `r1_control.py`, `r1_camera_view.py`, `r1_dimos_demo.py`, `r1_video_demo.py`. |
