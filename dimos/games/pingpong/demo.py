"""Ping Pong Championship Demo — sports broadcast with live camera.

CI=1 .venv/bin/python dimos/games/pingpong/demo.py
"""

import json
import itertools
import subprocess
import sys
import threading
import time
import webbrowser
from queue import Empty, Queue

import cv2
import numpy as np

# ANSI
BOLD = "\033[1m"
DIM = "\033[2m"
RESET = "\033[0m"
CYAN = "\033[36m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
RED = "\033[31m"
BR_BLACK = "\033[90m"
BR_GREEN = "\033[92m"
BR_CYAN = "\033[96m"
HIDE_CURSOR = "\033[?25l"
SHOW_CURSOR = "\033[?25h"
CLEAR_LINE = "\033[2K\r"

DASHBOARD_PORT = 8080
_latest_jpeg = bytearray()
_jpeg_lock = threading.Lock()
_sse_clients: list[Queue] = []
_sse_lock = threading.Lock()

# ═══════════════════════════════════════════════════════════════════
# Dashboard HTML — sports broadcast theme
# ═══════════════════════════════════════════════════════════════════

DASHBOARD_HTML = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Robot Ping Pong Championship</title>
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{background:#0a0a0f;color:#e0e0e0;font-family:system-ui,-apple-system,sans-serif;overflow:hidden;height:100vh}
.header{display:flex;align-items:center;justify-content:space-between;padding:10px 24px;background:linear-gradient(90deg,#1a0a2e,#0e1a3a);border-bottom:2px solid #ff6600}
.title{font-size:1.1rem;font-weight:700;letter-spacing:.12em;color:#ff6600}
.title span{color:#e0e0e0;font-weight:400;letter-spacing:.05em}
.timer{font-family:"SF Mono",Consolas,monospace;font-size:1.4rem;color:#ff6600;font-weight:700}
.main{position:relative;height:calc(100vh - 52px)}
.camera{width:100%;height:100%;object-fit:contain;background:#000}
.scoreboard{position:absolute;top:16px;left:50%;transform:translateX(-50%);background:rgba(10,10,20,.9);border:2px solid #ff6600;border-radius:8px;padding:8px 24px;display:flex;align-items:center;gap:20px;backdrop-filter:blur(8px)}
.team{text-align:center;min-width:80px}
.team .name{font-size:.7rem;font-weight:600;letter-spacing:.1em;color:#aaa;margin-bottom:2px}
.team .score{font-size:2rem;font-weight:800;font-family:"SF Mono",Consolas,monospace}
.team.robot .score{color:#00e5ff}
.team.opp .score{color:#ff4444}
.vs{font-size:1rem;color:#666;font-weight:700}
.stats{position:absolute;top:16px;right:16px;background:rgba(10,10,20,.85);border:1px solid rgba(255,102,0,.3);border-radius:8px;padding:10px 16px;backdrop-filter:blur(8px)}
.stat{font-size:.75rem;color:#aaa;margin-bottom:4px}
.stat span{color:#e0e0e0;font-weight:600}
.commentary{position:absolute;bottom:0;left:0;right:0;background:linear-gradient(transparent,rgba(10,10,20,.95));padding:40px 24px 16px}
.comment{font-size:1rem;color:#fff;text-shadow:0 1px 4px rgba(0,0,0,.8);animation:fadeIn .4s ease;max-width:800px}
.comment .label{font-size:.7rem;color:#ff6600;font-weight:700;letter-spacing:.1em;margin-bottom:4px}
.event{position:absolute;top:50%;left:50%;transform:translate(-50%,-50%);font-size:2.5rem;font-weight:800;color:#ff6600;text-shadow:0 0 30px rgba(255,102,0,.5);opacity:0;transition:opacity .3s}
.event.show{opacity:1;animation:pop .5s ease}
@keyframes fadeIn{from{opacity:0;transform:translateY(8px)}to{opacity:1;transform:translateY(0)}}
@keyframes pop{0%{transform:translate(-50%,-50%) scale(0.5)}50%{transform:translate(-50%,-50%) scale(1.1)}100%{transform:translate(-50%,-50%) scale(1)}}
</style>
</head>
<body>
<div class="header">
  <div class="title">ROBOT PING PONG CHAMPIONSHIP <span>// DIMOS + Robomotion</span></div>
  <div class="timer" id="timer">00:00</div>
</div>
<div class="main">
  <img class="camera" id="camera" src="/video_feed" alt="Match Camera">
  <div class="scoreboard">
    <div class="team robot"><div class="name">ROBOT</div><div class="score" id="s_robot">0</div></div>
    <div class="vs">VS</div>
    <div class="team opp"><div class="name">OPPONENT</div><div class="score" id="s_opp">0</div></div>
  </div>
  <div class="stats">
    <div class="stat">SERVES: <span id="serves">0</span></div>
    <div class="stat">SWINGS: <span id="swings">0</span></div>
    <div class="stat">BEST RALLY: <span id="rally">0</span></div>
  </div>
  <div class="commentary">
    <div class="comment" id="comment">
      <div class="label">COMMENTARY</div>
      <div id="comment_text">Waiting for first serve...</div>
    </div>
  </div>
  <div class="event" id="event"></div>
</div>
<script>
let t0=Date.now();
setInterval(()=>{
  const s=Math.floor((Date.now()-t0)/1000);
  document.getElementById("timer").textContent=
    String(Math.floor(s/60)).padStart(2,"0")+":"+String(s%60).padStart(2,"0");
},500);

const COMMENTS={
  serve:["And the serve is in!","Here we go — new point!","The ball is live!"],
  rally:["What a rally!","Back and forth — incredible!","Neither side is giving up!"],
  point_robot:["WHAT A SHOT! Robot takes the point!","Unstoppable return from Robot!","Clinical precision from the machine!"],
  point_opp:["That one goes wide — Opponent scores.","Robot couldn't reach that one.","Point to the Opponent."],
  swing:["Nice swing from Robot!","Robot goes for it!","Big forehand from Robot!"],
};
function rnd(arr){return arr[Math.floor(Math.random()*arr.length)];}
function showEvent(txt){
  const e=document.getElementById("event");
  e.textContent=txt;e.classList.add("show");
  setTimeout(()=>e.classList.remove("show"),1500);
}
function setComment(txt){document.getElementById("comment_text").textContent=txt;}

const es=new EventSource("/events");
es.onmessage=function(e){
  const d=JSON.parse(e.data);
  if(d.type==="score"){
    document.getElementById("s_robot").textContent=d.robot;
    document.getElementById("s_opp").textContent=d.opponent;
    document.getElementById("serves").textContent=d.serves;
    document.getElementById("swings").textContent=d.swings;
    document.getElementById("rally").textContent=d.best_rally;
  }
  if(d.type==="event"){
    if(d.event.startsWith("SERVE")){setComment(rnd(COMMENTS.serve));}
    else if(d.event.startsWith("RALLY")){setComment(rnd(COMMENTS.rally));}
    else if(d.event.startsWith("POINT ROBOT")){showEvent("ROBOT POINT!");setComment(rnd(COMMENTS.point_robot));}
    else if(d.event.startsWith("POINT OPPONENT")){showEvent("OPP POINT");setComment(rnd(COMMENTS.point_opp));}
    else if(d.event.includes("SWING")){setComment(rnd(COMMENTS.swing));}
  }
};
</script>
</body>
</html>
"""


# ═══════════════════════════════════════════════════════════════════
# SSE + Video helpers
# ═══════════════════════════════════════════════════════════════════

def _broadcast_sse(event_type, data=None):
    payload = json.dumps({"type": event_type, **(data or {})})
    with _sse_lock:
        for q in list(_sse_clients):
            try:
                q.put_nowait(payload)
            except Exception:
                pass


def _start_video_subscriber():
    from dimos.core.transport import LCMTransport
    from dimos.msgs.sensor_msgs.Image import Image

    transport = LCMTransport("/color_image", Image)

    def on_frame(img):
        try:
            bgr = img.to_opencv()
            _, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 75])
            with _jpeg_lock:
                _latest_jpeg[:] = buf.tobytes()
        except Exception:
            pass

    transport.subscribe(on_frame)


def _start_dashboard():
    from flask import Flask, Response
    import logging as _logging

    app = Flask(__name__)
    _logging.getLogger("werkzeug").setLevel(_logging.ERROR)

    @app.route("/")
    def index():
        return Response(DASHBOARD_HTML, content_type="text/html")

    @app.route("/video_feed")
    def video_feed():
        def gen():
            while True:
                with _jpeg_lock:
                    frame = bytes(_latest_jpeg)
                if frame:
                    yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
                time.sleep(0.066)
        return Response(gen(), mimetype="multipart/x-mixed-replace; boundary=frame")

    @app.route("/events")
    def events():
        q: Queue = Queue()
        with _sse_lock:
            _sse_clients.append(q)

        def gen():
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

        return Response(gen(), mimetype="text/event-stream",
                        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    threading.Thread(
        target=lambda: app.run(host="0.0.0.0", port=DASHBOARD_PORT, debug=False, threaded=True),
        daemon=True,
    ).start()


# ═══════════════════════════════════════════════════════════════════
# Terminal helpers
# ═══════════════════════════════════════════════════════════════════

def _spinner(msg, done, t0):
    for f in itertools.cycle(["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]):
        if done.is_set():
            break
        e = time.time() - t0
        sys.stdout.write(f"{CLEAR_LINE}  {BOLD}{CYAN}{f}{RESET} {msg} {DIM}[{e:.0f}s]{RESET}")
        sys.stdout.flush()
        time.sleep(0.08)
    sys.stdout.write(CLEAR_LINE)
    sys.stdout.flush()


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════

def main():
    t0 = time.time()
    print(HIDE_CURSOR, end="")
    try:
        _run(t0)
    finally:
        print(SHOW_CURSOR, end="")


def _run(t0):
    # Kill stale RAM consumers
    for p in ["snap-store", "gnome-software"]:
        try:
            subprocess.run(["pkill", "-f", p], capture_output=True, timeout=3)
        except Exception:
            pass

    print()
    print(f"  {BOLD}{BR_CYAN}{'━' * 60}{RESET}")
    print(f"  {BOLD}{YELLOW}  R O B O T   P I N G   P O N G   C H A M P I O N S H I P{RESET}")
    print(f"  {DIM}  G1 humanoid  ·  Table tennis  ·  AI sports broadcast{RESET}")
    print(f"  {BOLD}{BR_CYAN}{'━' * 60}{RESET}")
    print()

    # Build
    from dimos.games.pingpong.blueprint import build_pingpong
    game = build_pingpong()

    done = threading.Event()
    spin = threading.Thread(target=_spinner, args=("Launching simulation", done, t0), daemon=True)
    spin.start()

    coordinator = game.build()

    done.set()
    spin.join()
    print(f"  {BR_GREEN}✓ MuJoCo sim online · G1 tennis · PingPong controller{RESET}")

    # Dashboard
    _start_video_subscriber()
    _start_dashboard()
    print(f"  {BR_GREEN}✓ Dashboard: {BOLD}http://localhost:{DASHBOARD_PORT}{RESET}")

    try:
        webbrowser.open(f"http://localhost:{DASHBOARD_PORT}")
    except Exception:
        pass

    # Warmup
    print()
    for i in range(6):
        pct = i / 5
        filled = int(30 * pct)
        bar = f"{'█' * filled}{'░' * (30 - filled)}"
        sys.stdout.write(f"{CLEAR_LINE}  {BOLD}{CYAN}{bar}{RESET} {pct*100:3.0f}%  {DIM}camera warmup{RESET}")
        sys.stdout.flush()
        time.sleep(1)
    print()

    print()
    print(f"  {BOLD}{BR_CYAN}{'━' * 60}{RESET}")
    print(f"  {BOLD}  MATCH LIVE — watch at http://localhost:{DASHBOARD_PORT}{RESET}")
    print(f"  {BOLD}{BR_CYAN}{'━' * 60}{RESET}")
    print()

    # Match loop — broadcast score updates via SSE every 2 seconds
    match_time = 120  # 2 minute match
    start = time.time()
    last_event = ""

    while time.time() - start < match_time:
        # Push score update (the actual score comes from MuJoCo subprocess)
        # For now, push a heartbeat that the dashboard uses for the timer
        _broadcast_sse("score", {
            "robot": "?",
            "opponent": "?",
            "serves": "?",
            "swings": "?",
            "best_rally": "?",
        })
        time.sleep(2)

    # Shutdown
    print(f"  {BOLD}{BR_CYAN}{'━' * 60}{RESET}")

    done2 = threading.Event()
    spin2 = threading.Thread(target=_spinner, args=("Shutting down", done2, t0), daemon=True)
    spin2.start()
    coordinator.stop()
    done2.set()
    spin2.join()

    elapsed = time.time() - t0
    print(f"  {BR_GREEN}✓ Match complete{RESET} {DIM}· {elapsed:.0f}s total{RESET}")
    print()


if __name__ == "__main__":
    main()
