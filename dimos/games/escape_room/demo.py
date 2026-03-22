"""Robot Escape Room Demo — live puzzle-solving with web dashboard.

CI=1 .venv/bin/python dimos/games/escape_room/demo.py

Opens a web dashboard at http://localhost:8080 showing:
- Live camera feed from the robot
- Radio transcript between Trapped robot and Guide
- Puzzle progress tracker
"""

import itertools
import json
import shutil
import subprocess
import sys
import textwrap
import threading
import time
import webbrowser
from queue import Empty, Queue

import cv2

# ANSI
BOLD = "\033[1m"
DIM = "\033[2m"
RESET = "\033[0m"
CYAN = "\033[36m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
RED = "\033[31m"
MAGENTA = "\033[35m"
WHITE = "\033[37m"
BR_BLACK = "\033[90m"
BR_CYAN = "\033[96m"
BR_GREEN = "\033[92m"
BR_YELLOW = "\033[93m"
BG_YELLOW = "\033[43m"
BG_MAGENTA = "\033[45m"
BG_GREEN = "\033[42m"
BG_RED = "\033[41m"
BG_CYAN = "\033[46m"
BLACK = "\033[30m"
HIDE_CURSOR = "\033[?25l"
SHOW_CURSOR = "\033[?25h"
CLEAR_LINE = "\033[2K\r"

W = 72
_print_lock = threading.Lock()
_clues_found = [0]
_game_won = [False]

# ── Dashboard state ──
DASHBOARD_PORT = 8080
_latest_jpeg = bytearray()
_jpeg_lock = threading.Lock()
_sse_clients: list[Queue] = []
_sse_lock = threading.Lock()

# ═══════════════════════════════════════════════════════════════════
# Dashboard HTML
# ═══════════════════════════════════════════════════════════════════

DASHBOARD_HTML = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Robot Escape Room</title>
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{background:#0a0a0f;color:#e0e0e0;font-family:system-ui,-apple-system,"Segoe UI",sans-serif;overflow:hidden;height:100vh}
.header{display:flex;align-items:center;justify-content:space-between;padding:12px 24px;background:#0e0e18;border-bottom:1px solid rgba(0,229,255,.15)}
.title{font-size:1.3rem;font-weight:700;letter-spacing:.15em;color:#00e5ff}
.title span{color:#e0e0e0;letter-spacing:.05em}
.timer{font-family:"SF Mono",Consolas,monospace;font-size:1.6rem;color:#00e5ff;font-weight:700;min-width:80px;text-align:right}
.main{display:grid;grid-template-columns:1.2fr 1fr;height:calc(100vh - 56px - 64px);gap:0}
.camera-panel{position:relative;background:#000;display:flex;align-items:center;justify-content:center;border-right:1px solid rgba(0,229,255,.1)}
.camera-panel img{width:100%;height:100%;object-fit:contain}
.camera-label{position:absolute;top:12px;left:12px;background:rgba(0,0,0,.7);color:#00e5ff;padding:4px 12px;border-radius:4px;font-size:.75rem;font-weight:600;letter-spacing:.1em;border:1px solid rgba(0,229,255,.3)}
.radio-panel{display:flex;flex-direction:column;background:#0e0e18}
.radio-header{padding:12px 16px;font-size:.85rem;font-weight:600;color:#888;letter-spacing:.1em;border-bottom:1px solid rgba(255,255,255,.06)}
.messages{flex:1;overflow-y:auto;padding:12px 16px;display:flex;flex-direction:column;gap:8px}
.msg{padding:10px 14px;border-radius:12px;font-size:.85rem;line-height:1.5;animation:fadeIn .3s ease;max-width:90%}
.msg.trapped{background:rgba(255,171,64,.08);border:1px solid rgba(255,171,64,.2);color:#ffcc80;align-self:flex-start;border-bottom-left-radius:2px}
.msg.guide{background:rgba(224,64,251,.08);border:1px solid rgba(224,64,251,.2);color:#e1bee7;align-self:flex-end;border-bottom-right-radius:2px}
.msg.event{background:rgba(0,230,118,.1);border:1px solid rgba(0,230,118,.3);color:#a5d6a7;align-self:center;text-align:center;font-weight:600}
.msg .sender{font-size:.7rem;font-weight:700;letter-spacing:.08em;margin-bottom:4px;opacity:.7}
.msg.trapped .sender{color:#ffab40}
.msg.guide .sender{color:#e040fb}
.footer{display:flex;align-items:center;justify-content:center;gap:32px;padding:14px 24px;background:#0e0e18;border-top:1px solid rgba(0,229,255,.1)}
.clue{display:flex;align-items:center;gap:8px;font-size:.85rem}
.clue .dot{width:18px;height:18px;border-radius:50%;border:2px solid #444;transition:all .5s ease}
.clue .dot.found{border-color:#00e676;background:#00e676;box-shadow:0 0 12px rgba(0,230,118,.5)}
.clue .label{color:#888;transition:color .3s}
.clue .label.found{color:#e0e0e0}
.progress-text{font-family:"SF Mono",Consolas,monospace;font-size:.9rem;color:#00e5ff;font-weight:700}
.victory{display:none;position:fixed;inset:0;background:rgba(0,0,0,.85);z-index:100;align-items:center;justify-content:center;flex-direction:column;gap:16px}
.victory.show{display:flex}
.victory h1{font-size:3rem;color:#00e676;letter-spacing:.1em;animation:pulse 1s ease infinite}
.victory p{font-size:1.3rem;color:#e0e0e0}
@keyframes fadeIn{from{opacity:0;transform:translateY(8px)}to{opacity:1;transform:translateY(0)}}
@keyframes pulse{0%,100%{opacity:1}50%{opacity:.7}}
.messages::-webkit-scrollbar{width:4px}
.messages::-webkit-scrollbar-thumb{background:#333;border-radius:2px}
</style>
</head>
<body>
<div class="header">
  <div class="title">ROBOT ESCAPE ROOM <span>// DIMOS</span></div>
  <div class="timer" id="timer">00:00</div>
</div>
<div class="main">
  <div class="camera-panel">
    <div class="camera-label">LIVE CAMERA</div>
    <img id="camera" src="/video_feed" alt="Robot Camera">
  </div>
  <div class="radio-panel">
    <div class="radio-header">RADIO TRANSCRIPT</div>
    <div class="messages" id="messages"></div>
  </div>
</div>
<div class="footer">
  <div class="clue"><div class="dot" id="d0"></div><div class="label" id="l0">Red Sphere</div></div>
  <div class="clue"><div class="dot" id="d1"></div><div class="label" id="l1">Blue Cylinder</div></div>
  <div class="clue"><div class="dot" id="d2"></div><div class="label" id="l2">Green Box</div></div>
  <div class="progress-text" id="prog">0 / 3</div>
</div>
<div class="victory" id="victory">
  <h1>ESCAPED!</h1>
  <p id="vtime"></p>
</div>
<script>
let t0=null,timerInterval=null;
const $=id=>document.getElementById(id);
const msgs=$("messages");
function addMsg(cls,sender,text){
  const d=document.createElement("div");
  d.className="msg "+cls;
  d.innerHTML='<div class="sender">'+sender+"</div>"+text;
  msgs.appendChild(d);
  msgs.scrollTop=msgs.scrollHeight;
}
function startTimer(){
  t0=Date.now();
  timerInterval=setInterval(()=>{
    const s=Math.floor((Date.now()-t0)/1000);
    $("timer").textContent=String(Math.floor(s/60)).padStart(2,"0")+":"+String(s%60).padStart(2,"0");
  },500);
}
function markClue(i,name){
  $("d"+i).classList.add("found");
  $("l"+i).classList.add("found");
  $("prog").textContent=(i+1)+" / 3";
  addMsg("event","","CLUE "+(i+1)+"/3 FOUND: "+name);
}
const es=new EventSource("/events");
es.onmessage=function(e){
  const d=JSON.parse(e.data);
  if(d.type==="radio_trapped")addMsg("trapped","TRAPPED",d.message);
  else if(d.type==="radio_guide")addMsg("guide","GUIDE",d.message);
  else if(d.type==="clue_found")markClue(d.index-1,d.name);
  else if(d.type==="game_won"){
    clearInterval(timerInterval);
    $("vtime").textContent="Escaped in "+d.time+" seconds";
    $("victory").classList.add("show");
  }
  else if(d.type==="game_start"){startTimer();addMsg("event","","GAME STARTED");}
};
startTimer();
</script>
</body>
</html>
"""


# ═══════════════════════════════════════════════════════════════════
# Dashboard server
# ═══════════════════════════════════════════════════════════════════

def _broadcast_sse(event_type, data=None):
    """Push an SSE event to all connected dashboard clients."""
    payload = json.dumps({"type": event_type, **(data or {})})
    with _sse_lock:
        for q in list(_sse_clients):
            try:
                q.put_nowait(payload)
            except Exception:
                pass


def _start_video_subscriber():
    """Subscribe to color_image LCM stream and convert to JPEG."""
    from dimos.core.transport import LCMTransport
    from dimos.msgs.sensor_msgs.Image import Image

    # Image has lcm_encode so autoconnect uses LCMTransport, not pLCMTransport.
    transport = LCMTransport("/color_image", Image)

    def on_frame(img):
        try:
            bgr = img.to_opencv()
            _, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 70])
            with _jpeg_lock:
                _latest_jpeg[:] = buf.tobytes()
        except Exception:
            pass

    transport.subscribe(on_frame)


def _start_dashboard():
    """Launch Flask dashboard server in a daemon thread."""
    from flask import Flask, Response

    app = Flask(__name__)
    # Suppress Flask startup banner and request logs
    import logging as _logging
    _logging.getLogger("werkzeug").setLevel(_logging.ERROR)

    @app.route("/")
    def index():
        return Response(DASHBOARD_HTML, content_type="text/html")

    @app.route("/video_feed")
    def video_feed():
        def generate():
            while True:
                with _jpeg_lock:
                    frame = bytes(_latest_jpeg)
                if frame:
                    yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
                time.sleep(0.1)
        return Response(
            generate(),
            mimetype="multipart/x-mixed-replace; boundary=frame",
        )

    @app.route("/events")
    def events():
        q: Queue = Queue()
        with _sse_lock:
            _sse_clients.append(q)

        def generate():
            try:
                while True:
                    try:
                        msg = q.get(timeout=30)
                        yield f"data: {msg}\n\n"
                    except Empty:
                        yield ": keepalive\n\n"
            finally:
                with _sse_lock:
                    if q in _sse_clients:
                        _sse_clients.remove(q)

        return Response(
            generate(),
            mimetype="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    t = threading.Thread(
        target=lambda: app.run(
            host="0.0.0.0", port=DASHBOARD_PORT, debug=False, threaded=True,
        ),
        daemon=True,
    )
    t.start()


# ═══════════════════════════════════════════════════════════════════
# Terminal UI helpers (keep existing for developer view)
# ═══════════════════════════════════════════════════════════════════

def hline(char="━", color=CYAN):
    return f"  {BOLD}{color}{char * W}{RESET}"


def spinner(msg, done_event, t0):
    frames = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
    for f in itertools.cycle(frames):
        if done_event.is_set():
            break
        elapsed = time.time() - t0
        sys.stdout.write(f"{CLEAR_LINE}  {BOLD}{CYAN}{f}{RESET} {msg} {DIM}[{elapsed:.0f}s]{RESET}")
        sys.stdout.flush()
        time.sleep(0.08)
    sys.stdout.write(CLEAR_LINE)
    sys.stdout.flush()


def progress_bar(pct, width=30, color=CYAN):
    filled = int(width * pct)
    return f"  {BOLD}{color}{'█' * filled}{BR_BLACK}{'░' * (width - filled)}{RESET} {pct * 100:3.0f}%"


def _radio_msg(sender, color, bg, icon, message, t0):
    with _print_lock:
        print()
        if sender == "TRAPPED":
            print(f"  {BOLD}{bg}{BLACK} {icon} {sender} {RESET}  {BR_BLACK}{time.strftime('%H:%M:%S')}{RESET}")
            for line in textwrap.wrap(message, W - 6):
                print(f"  {color}┃{RESET} {line}")
            print(f"  {color}┗{'━' * 50}{RESET}")
        else:
            pad = W - len(sender) - 8
            print(f"  {' ' * pad}{BOLD}{bg}{WHITE} {icon} {sender} {RESET}  {BR_BLACK}{time.strftime('%H:%M:%S')}{RESET}")
            for line in textwrap.wrap(message, W - 6):
                rpad = W - len(line) - 4
                print(f"  {' ' * max(rpad, 2)}{line} {color}┃{RESET}")
            print(f"  {' ' * (W - 52)}{color}{'━' * 50}┛{RESET}")


# ═══════════════════════════════════════════════════════════════════
# LCM subscribers (terminal + dashboard SSE)
# ═══════════════════════════════════════════════════════════════════

def _subscribe_radio(topic, sender, color, bg, icon, t0):
    from dimos.core.transport import pLCMTransport
    transport = pLCMTransport(topic)

    def on_msg(msg):
        if msg and len(msg) > 5:
            _radio_msg(sender, color, bg, icon, msg, t0)
            _broadcast_sse(f"radio_{sender.lower()}", {"message": msg, "sender": sender})

    transport.subscribe(on_msg)


def _subscribe_game_events(t0):
    from dimos.core.transport import pLCMTransport
    transport = pLCMTransport("/game_event")

    def on_event(msg):
        if not msg:
            return
        with _print_lock:
            if msg.startswith("CLUE_FOUND:"):
                parts = msg.split(":")
                found = int(parts[1])
                name = parts[2]
                _clues_found[0] = found
                print()
                print(f"  {BOLD}{BG_GREEN}{BLACK} ✓ CLUE {found}/3 FOUND {RESET}  {GREEN}{name}{RESET}")
                boxes = f"{'■' * found}{'□' * (3 - found)}"
                print(f"  {GREEN}  Progress: [{boxes}]{RESET}")
                print()
                _broadcast_sse("clue_found", {"index": found, "name": name})
            elif msg.startswith("GAME_WON:"):
                secs = msg.split(":")[1]
                _game_won[0] = True
                print()
                print(f"  {BOLD}{BG_GREEN}{BLACK}{'=' * 50}{RESET}")
                print(f"  {BOLD}{BG_GREEN}{BLACK}   ESCAPED IN {secs} SECONDS!   {RESET}")
                print(f"  {BOLD}{BG_GREEN}{BLACK}{'=' * 50}{RESET}")
                print()
                _broadcast_sse("game_won", {"time": secs})
            elif msg == "GAME_START":
                _broadcast_sse("game_start", {})

    transport.subscribe(on_event)


# ═══════════════════════════════════════════════════════════════════
# Preflight & main
# ═══════════════════════════════════════════════════════════════════

def _preflight():
    for name in ["openclaw-gateway", "snap-store", "gnome-software"]:
        try:
            subprocess.run(["pkill", "-f", name], capture_output=True, timeout=3)
        except Exception:
            pass
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    gb = int(line.split()[1]) / 1024 / 1024
                    color = BR_GREEN if gb >= 6 else YELLOW
                    print(f"  {color}{'✓' if gb >= 6 else '⚠'} RAM: {gb:.1f} GB{RESET}")
                    break
    except Exception:
        pass
    try:
        _, _, free = shutil.disk_usage("/")
        free_gb = free / (1024 ** 3)
        color = BR_GREEN if free_gb >= 2 else RED
        print(f"  {color}{'✓' if free_gb >= 2 else '⚠'} Disk: {free_gb:.1f} GB free{RESET}")
    except Exception:
        pass


def main():
    t0 = time.time()
    print(HIDE_CURSOR, end="")
    try:
        _run(t0)
    finally:
        print(SHOW_CURSOR, end="")


def _run(t0):
    _preflight()

    # ── Title ──
    print()
    print(hline())
    print()
    print(f"  {BOLD}{BR_CYAN}  R O B O T   E S C A P E   R O O M{RESET}")
    print(f"  {DIM}  One robot trapped  ·  One guide  ·  3 clues to find{RESET}")
    print()
    print(hline())
    print()

    print(f"  {BOLD}{BG_YELLOW}{BLACK} TRAPPED {RESET} {DIM}G1 humanoid in the maze — must find 3 clues{RESET}")
    print(f"  {BOLD}{BG_MAGENTA}{WHITE}  GUIDE  {RESET} {DIM}gives hints via radio — cannot see the room{RESET}")
    print(f"  {DIM}  Progress: [□□□] — find all 3 to escape{RESET}")
    print()
    time.sleep(1)

    # ── Build ──
    print(f"  {BR_BLACK}{'─' * W}{RESET}")

    from dimos.games.escape_room.blueprint import build_escape_room

    game = build_escape_room()

    done = threading.Event()
    spin = threading.Thread(target=spinner, args=("Launching simulation", done, t0), daemon=True)
    spin.start()

    coordinator = game.build()

    done.set()
    spin.join()
    print(f"  {BR_GREEN}✓ Modules online · 1 MuJoCo sim · 2 LLM agents{RESET}")

    # ── Start dashboard ──
    _start_video_subscriber()
    _start_dashboard()
    print(f"  {BR_GREEN}✓ Dashboard: {BOLD}http://localhost:{DASHBOARD_PORT}{RESET}")

    # Auto-open browser
    try:
        webbrowser.open(f"http://localhost:{DASHBOARD_PORT}")
    except Exception:
        pass

    # ── Warmup ──
    warmup = 8
    print()
    for i in range(warmup + 1):
        pct = i / warmup
        sys.stdout.write(f"{CLEAR_LINE}{progress_bar(pct)}  {DIM}camera warmup{RESET}")
        sys.stdout.flush()
        if i < warmup:
            time.sleep(1)
    print()

    # ── Subscribe to radio + game events ──
    _subscribe_radio("/radio_trapped_out", "TRAPPED", YELLOW, BG_YELLOW, "TRAPPED", t0)
    _subscribe_radio("/radio_guide_out", "GUIDE", MAGENTA, BG_MAGENTA, "GUIDE", t0)
    _subscribe_game_events(t0)

    # ── Start game ──
    print()
    print(hline())
    print(f"  {BOLD}  ESCAPE ROOM — LIVE{RESET}")
    print(hline())

    from dimos.core.transport import pLCMTransport

    time.sleep(2)

    # Kick off Guide
    pLCMTransport("/guide_input").publish(
        "The escape room is ready. Call start_game NOW, then immediately "
        "broadcast the first hint to the Trapped robot. Be concise."
    )

    time.sleep(2)

    # Kick off Trapped robot
    pLCMTransport("/trapped_input").publish(
        "You are in the maze. Call describe_surroundings ONCE to see where you are. "
        "Then IMMEDIATELY call turn_right or move_forward — do NOT describe again "
        "before moving. Keep alternating: look, move, look, move. "
        "If you only see a wall, turn right immediately."
    )

    # ── Wait for game to complete or timeout ──
    game_timeout = 300  # 5 minutes — give agents time to explore the maze
    start_game = time.time()
    last_nudge = start_game
    last_clue_count = 0

    while time.time() - start_game < game_timeout:
        if _game_won[0]:
            time.sleep(3)
            break

        # Nudge: if 15s pass with no new clue, force movement
        now = time.time()
        if now - last_nudge > 15 and _clues_found[0] == last_clue_count:
            nudge_n = int((now - start_game) / 15) % 3
            nudges = [
                "You are stuck! Call turn_around NOW to turn 180 degrees, then move_forward.",
                "MOVE! Call turn_right, then move_forward. Do NOT describe first — just move!",
                "Call move_backward, then turn_around. You need to explore a new area!",
            ]
            pLCMTransport("/trapped_input").publish(nudges[nudge_n])
            last_nudge = now

        if _clues_found[0] > last_clue_count:
            last_clue_count = _clues_found[0]
            last_nudge = now

        time.sleep(1)

    if not _game_won[0]:
        with _print_lock:
            print()
            print(f"  {BOLD}{RED}  TIME'S UP — {_clues_found[0]}/3 clues found{RESET}")
            print()

    # ── Shutdown ──
    print(hline())

    done2 = threading.Event()
    spin2 = threading.Thread(target=spinner, args=("Shutting down", done2, t0), daemon=True)
    spin2.start()

    coordinator.stop()

    done2.set()
    spin2.join()

    elapsed = time.time() - t0
    print(f"  {BR_GREEN}✓ Done{RESET}  {DIM}· {elapsed:.0f}s total · {_clues_found[0]}/3 clues{RESET}")
    print()


if __name__ == "__main__":
    main()
