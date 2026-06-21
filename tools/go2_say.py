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

"""Make the Go2 speak a phrase from its OWN speaker — standalone AudioHub test.

Path: OpenAI TTS -> WAV -> AudioHub upload -> play_by_uuid. Verbose, so we learn
the robot's audio-list format and confirm playback before wiring it into the
narrator.

Close the Unitree app first (one WebRTC client), and have GO2_AES_KEY set:

    export GO2_AES_KEY=<key>
    .venv/bin/python tools/go2_say.py "hello, I am the Go two robot"
"""

import asyncio
import json
import os
import sys
import tempfile


async def main() -> int:
    text = sys.argv[1] if len(sys.argv) > 1 else "Hello. I am the Go two robot, speaking from my own speaker."
    ip = os.environ.get("ROBOT_IP", "192.168.69.123")
    key = os.environ.get("GO2_AES_KEY")
    if not key:
        print("Set GO2_AES_KEY first: export GO2_AES_KEY=<key>")
        return 2

    # 1. third-party TTS -> WAV file (named so we can find it in the robot's list)
    from openai import OpenAI

    resp = OpenAI().audio.speech.create(
        model="tts-1", voice="onyx", input=text, response_format="wav"
    )
    wav = os.path.join(tempfile.gettempdir(), "dimos_say_hd.wav")
    with open(wav, "wb") as f:
        f.write(resp.read() if hasattr(resp, "read") else resp.content)
    print(f"[tts] wrote {wav} ({os.path.getsize(wav)} bytes)")

    # The robot's audio player expects 44.1 kHz; OpenAI TTS WAV is 24 kHz.
    # Resample in place with the stdlib (no ffmpeg) so the clip is playable.
    import audioop
    import wave

    with wave.open(wav, "rb") as w:
        nch, width, inrate = w.getnchannels(), w.getsampwidth(), w.getframerate()
        frames = w.readframes(w.getnframes())
    if inrate != 44100:
        frames, _ = audioop.ratecv(frames, width, nch, inrate, 44100, None)
        with wave.open(wav, "wb") as w:
            w.setnchannels(nch)
            w.setsampwidth(width)
            w.setframerate(44100)
            w.writeframes(frames)
        print(f"[tts] resampled {inrate} Hz -> 44100 Hz ({os.path.getsize(wav)} bytes)")

    # 2. connect (the proven path)
    from unitree_webrtc_connect.webrtc_audiohub import WebRTCAudioHub
    from unitree_webrtc_connect.webrtc_driver import (
        UnitreeWebRTCConnection,
        WebRTCConnectionMethod,
    )

    conn = UnitreeWebRTCConnection(WebRTCConnectionMethod.LocalSTA, ip=ip, aes_128_key=key)
    await conn.connect()
    print("[conn] connected; opening AudioHub")
    hub = WebRTCAudioHub(conn)

    with wave.open(wav, "rb") as w:
        dur = w.getnframes() / float(w.getframerate())

    # 3. PATH A — library: upload + play_by_uuid
    print("\n[PATH A: library] uploading + play_by_uuid ...")
    await hub.upload_audio_file(wav)
    await asyncio.sleep(1.0)
    uid = None
    try:
        lst = await hub.get_audio_list()
        inner = lst["data"]["data"]
        if isinstance(inner, str):
            inner = json.loads(inner)
        items = inner.get("audio_list", [])
        matches = [it for it in items if "dimos_say" in str(it.get("CUSTOM_NAME", ""))]
        if matches:
            uid = max(matches, key=lambda it: it.get("ADD_TIME", 0)).get("UNIQUE_ID")
        elif items:
            uid = items[-1].get("UNIQUE_ID")
    except Exception as e:  # noqa: BLE001
        print(f"[PATH A] list-parse note: {e!r}")
    if uid:
        print(f"[PATH A] play_by_uuid({uid}) — LISTEN NOW (library playback) ...")
        await hub.play_by_uuid(uid)
        await asyncio.sleep(dur + 3.0)
    else:
        print("[PATH A] no uuid found")

    # 4. PATH B — megaphone: live-announce stream straight to the speaker
    print(f"\n[PATH B: megaphone] enter -> stream {dur:.1f}s -> exit. LISTEN NOW ...")
    try:
        await hub.enter_megaphone()
        await asyncio.sleep(0.5)
        await hub.upload_megaphone(wav)
        await asyncio.sleep(dur + 2.0)
        await hub.exit_megaphone()
    except Exception as e:  # noqa: BLE001
        print(f"[PATH B] megaphone error: {e!r}")

    print("\n[done] Which made the robot speak -- A (library), B (megaphone), or neither?")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
