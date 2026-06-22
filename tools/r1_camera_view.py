#!/usr/bin/env python3
"""Live browser viewer for the R1 front camera.

Streams the robot's camera (via PC1's native video client, the same bridge the
DimOS ``R1PC1Camera`` source uses) as MJPEG. Run it, then open the printed URL in
a browser. No Rerun needed — this sidesteps the DimOS Rerun-viewer version
mismatch entirely.

    python tools/r1_camera_view.py            # then open http://localhost:8088
Env overrides: R1_PC1_HOST, R1_PC1_USER, R1_PC1_PASS, R1_CAM_PORT.
"""

import os
import struct
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

HOST = os.environ.get("R1_PC1_HOST", "192.168.123.164")
USER = os.environ.get("R1_PC1_USER", "unitree")
PW = os.environ.get("R1_PC1_PASS", "123")
BIN = os.environ.get("R1_STREAM_BIN", "/home/unitree/r1_video_stream")
PORT = int(os.environ.get("R1_CAM_PORT", "8088"))

_latest: dict[str, bytes | None] = {"jpg": None}
_stop = threading.Event()


def _reader() -> None:
    import paramiko

    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    c.connect(HOST, username=USER, password=PW, timeout=10)
    _i, stdout, _e = c.exec_command(BIN, timeout=None)
    chan = stdout.channel
    chan.settimeout(5.0)
    buf = b""
    print(f"[r1cam] streaming from {USER}@{HOST}")
    while not _stop.is_set():
        try:
            chunk = chan.recv(65536)
        except Exception:
            continue
        if not chunk:
            break
        buf += chunk
        while len(buf) >= 4:
            ln = struct.unpack(">I", buf[:4])[0]
            if ln <= 0 or ln > 5_000_000:
                buf = b""
                break
            if len(buf) < 4 + ln:
                break
            _latest["jpg"], buf = buf[4 : 4 + ln], buf[4 + ln :]
    try:
        c.exec_command("pkill -f r1_video_stream")
        c.close()
    except Exception:
        pass


class _Handler(BaseHTTPRequestHandler):
    def log_message(self, *a: object) -> None:
        pass

    def do_GET(self) -> None:
        if self.path == "/":
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            self.wfile.write(
                b"<html><body style='margin:0;background:#111'>"
                b"<img src='/stream' style='width:100%;height:100vh;object-fit:contain'>"
                b"</body></html>"
            )
            return
        self.send_response(200)
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        self.end_headers()
        try:
            while not _stop.is_set():
                jpg = _latest["jpg"]
                if jpg:
                    self.wfile.write(
                        b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n"
                    )
                time.sleep(0.04)
        except (BrokenPipeError, ConnectionResetError):
            pass


def main() -> None:
    threading.Thread(target=_reader, daemon=True).start()
    print(f"[r1cam] R1 camera viewer -> open http://localhost:{PORT}  (Ctrl-C to stop)")
    srv = ThreadingHTTPServer(("0.0.0.0", PORT), _Handler)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        _stop.set()


if __name__ == "__main__":
    main()
