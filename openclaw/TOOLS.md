# Tools - RoboBot

## Tools Used

### Telegram (via OpenClaw channel)
- Receive robot commands from user
- Report command results and status
- Send camera captures as images

### Workspace File System
- Read/write command logs in logs/ directory
- Read configuration and workspace files

### Exec (restricted)
- ONLY for executing scripts/dimos_bridge.py
- Bridge script handles all DimOS communication
- No other exec usage permitted

## Denied Tools
- web_search, web_fetch, browser — no web access needed
- nano-pdf — no PDF generation needed
- nodes, canvas, gateway, spawn — not needed
- Any tool that modifies other agent workspaces
