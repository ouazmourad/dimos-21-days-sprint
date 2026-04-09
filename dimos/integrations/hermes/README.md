# Hermes &harr; DIMOS Integration

Drive a DIMOS-controlled Unitree Go2 quadruped (in MuJoCo simulation) using
[Hermes Agent](https://github.com/nousresearch/hermes-agent) as the LLM brain.

## Architecture

```
+-------------------+         HTTP / JSON-RPC          +---------------------+
|                   |  POST http://localhost:9990/mcp  |                     |
|   HERMES AGENT    |--------------------------------> |  DIMOS MCP SERVER   |
|  (LLM + memory)   |                                  |  (FastAPI)          |
|                   | <--------------------------------|                     |
+-------------------+         tool results              +---------+-----------+
                                                                  |
                                                                  | RPC
                                                                  v
                                                       +---------------------+
                                                       |  DIMOS MODULES      |
                                                       |  - go2_connection   |
                                                       |  - unitree_skills   |
                                                       |  - speak_skill      |
                                                       |  - navigation_skill |
                                                       |  - spatial_memory   |
                                                       +---------+-----------+
                                                                 |
                                                                 v
                                                        +-----------------+
                                                        |  MuJoCo (Go2)   |
                                                        +-----------------+
```

Hermes never knows about DIMOS internals — it sees a list of tools
(`move`, `speak`, `navigate_with_text`, etc.) and calls them like any
other MCP server. DIMOS handles the embodiment.

## Quick Start

### 0. (One-time) Install Hermes Agent

Hermes is a separate Python package — install it into the same venv DIMOS uses
so you can launch it from anywhere:

```bash
# Clone Hermes (skip if you already have it)
git clone https://github.com/nousresearch/hermes-agent /tmp/hermes-agent

# Install into the dimos venv
cd ~/Desktop/dimos-21days-sprint/dimos-21-days-sprint
.venv/bin/python -m pip install /tmp/hermes-agent

# If you hit `SyntaxError: invalid syntax` in asyncio/base_events.py,
# remove the rogue Python-3.4 asyncio package that some old installs leave
# behind:
.venv/bin/python -m pip uninstall -y asyncio

# Symlink the hermes binary so it's on PATH
mkdir -p ~/.local/bin && ln -sf "$(pwd)/.venv/bin/hermes" ~/.local/bin/hermes

# Verify
hermes --version           # should print "Hermes Agent v0.8.0 ..."
hermes mcp list            # should show "dimos" with status ✓ enabled
```

The first time you run Hermes, drop your Anthropic key into `~/.hermes/.env`:

```bash
mkdir -p ~/.hermes
echo 'ANTHROPIC_API_KEY=sk-ant-api03-...' >> ~/.hermes/.env
chmod 600 ~/.hermes/.env

# Copy the example config (or merge into your existing one)
cp dimos/integrations/hermes/hermes-config.example.yaml ~/.hermes/config.yaml
```

### 1. Terminal A — start DIMOS

```bash
cd ~/Desktop/dimos-21days-sprint/dimos-21-days-sprint
CI=1 .venv/bin/python -m dimos.integrations.hermes.run_dimos_mcp
```

The runner already forces `performance_tier=low` and `n_workers=1`
internally — no extra flags needed. If you launch via `dimos run` instead,
make sure to pass `--performance-tier low` so your laptop doesn't crash.

Wait until you see `MCP server is now accepting connections.` and a
MuJoCo viewer window appears with the Go2 standing in the office scene.

### 2. (Optional) Smoke-test the bridge without Hermes

In a third terminal:

```bash
cd ~/Desktop/dimos-21days-sprint/dimos-21-days-sprint
.venv/bin/python -m dimos.integrations.hermes.smoke_test
```

You should see `ALL CHECKS PASSED` — confirming the MCP endpoint is
live, tools are registered, and the speak skill works.

### 3. Terminal B — start Hermes

First add the MCP server to your Hermes config. Copy the example:

```bash
mkdir -p ~/.hermes
cp dimos/integrations/hermes/hermes-config.example.yaml ~/.hermes/config.yaml
# (or merge into your existing one — only the mcp_servers.dimos block matters)
```

Then launch Hermes:

```bash
hermes
```

Inside Hermes, verify the DIMOS tools were discovered:

```
/tools list
```

You should see entries like `dimos__move`, `dimos__speak`,
`dimos__navigate_with_text`, etc.

### 4. Drive the robot

Talk to Hermes naturally:

```
Walk forward for 2 seconds.
```
```
Wave to me.
```
```
Tell me what's around you, then navigate to the desk.
```

Hermes' LLM picks the appropriate tool, calls the DIMOS MCP server,
and the robot reacts in MuJoCo.

## Files

| File | Purpose |
|------|---------|
| `blueprint.py` | DIMOS blueprint composition (sim + skills + MCP server) |
| `run_dimos_mcp.py` | Launch script with low-perf-tier defaults |
| `hermes-config.example.yaml` | Drop-in Hermes config snippet |
| `smoke_test.py` | Standalone JSON-RPC client to verify the bridge |
| `README.md` | This file |

## Tool list (what Hermes will see)

| Tool | Source module | What it does |
|------|---------------|--------------|
| `move` | unitree_skills | Velocity command (linear x/y, angular z, duration) |
| `standup` / `liedown` / `balance_stand` | unitree_skills | Posture commands |
| `speak` | speak_skill | Text-to-speech via OpenAI TTS |
| `navigate_with_text` (full only) | navigation_skill | Semantic navigation by text query |
| `tag_location` (full only) | navigation_skill | Save current pose as a named waypoint |
| `stop_navigation` (full only) | navigation_skill | Cancel current goal |
| `server_status` | McpServer | Introspection: PID, modules, skill count |
| `list_modules` | McpServer | Introspection: deployed module names |
| `agent_send` | McpServer | Send a message to the local DIMOS agent (no-op here) |

## Notes & Caveats

- **Local DIMOS Agent is intentionally omitted.** Hermes is the brain.
  If you want both Hermes and a local LangGraph Agent, add `agent()`
  back to the blueprint and they'll share the skill set.
- **Performance tier is forced to `low`** in `run_dimos_mcp.py` to fit
  on 16 GB laptops. Bump it to `medium` or `high` on a beefier machine
  by editing the script.
- **Speak skill needs `OPENAI_API_KEY`** in the DIMOS process
  environment (it uses OpenAI TTS internally).
- **`agent_send` is a no-op here** — it publishes to `/human_input`
  which the local Agent would consume, but we removed it. Hermes can
  ignore this tool or you can `exclude` it via Hermes config.
