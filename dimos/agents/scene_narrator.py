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

"""SceneNarrator: the robot narrates what it sees, out loud, via third-party TTS.

A lightweight, autonomous perception→speech loop: it watches the camera
(``color_image``), and every ``period`` seconds asks a vision-language model to
describe the latest frame, then speaks that description through a third-party
TTS engine (OpenAI TTS → local audio out, the same path as :class:`SpeakSkill`).

It needs no human input and no navigation — just a camera — so it pairs with any
robot that publishes ``color_image`` (e.g. a Unitree Go2). Vision uses the
``VLMAgent`` langchain pattern (so the model is configurable: ``gpt-4o-mini`` by
default, or e.g. ``anthropic:claude-haiku-4-5``); speech uses the OpenAI TTS node.
"""

import threading
import time
from typing import TYPE_CHECKING, Any

from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage
from reactivex import Subject
from reactivex.disposable import Disposable

from dimos.constants import DEFAULT_THREAD_JOIN_TIMEOUT
from dimos.core.core import rpc
from dimos.core.module import Module, ModuleConfig
from dimos.core.stream import In
from dimos.msgs.sensor_msgs.Image import Image
from dimos.stream.audio.node_output import SounddeviceAudioOutput
from dimos.stream.audio.tts.node_openai import OpenAITTSNode, Voice
from dimos.utils.logging_config import setup_logger

if TYPE_CHECKING:
    from langchain_core.language_models.chat_models import BaseChatModel

logger = setup_logger()

_DEFAULT_PROMPT = (
    "You are a four-legged robot narrating your surroundings out loud to nearby people. "
    "In one or two short, natural spoken sentences, say what you see in front of you right "
    "now — focus on the most interesting or salient thing. Be specific and conversational. "
    "Do not say 'the image' or 'the camera'; speak as if you are looking at the room yourself."
)


class SceneNarratorConfig(ModuleConfig):
    model: str = "gpt-4o-mini"  # any langchain chat model id, e.g. "anthropic:claude-haiku-4-5"
    period: float = 8.0  # seconds to pause between narrations (on top of model+speech time)
    prompt: str = _DEFAULT_PROMPT


class SceneNarrator(Module):
    """Continuously describe the camera view aloud via a VLM + third-party TTS."""

    config: SceneNarratorConfig

    color_image: In[Image]

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._latest_image: Image | None = None
        self._llm: BaseChatModel | None = None
        self._tts: OpenAITTSNode | None = None
        self._audio: SounddeviceAudioOutput | None = None
        self._audio_lock = threading.Lock()
        self._stop_evt = threading.Event()
        self._narrate_thread: threading.Thread | None = None

    @rpc
    def start(self) -> None:
        super().start()
        self._llm = init_chat_model(self.config.model)
        self._tts = OpenAITTSNode(speed=1.1, voice=Voice.ONYX)
        self._audio = SounddeviceAudioOutput(sample_rate=24000)
        self._audio.consume_audio(self._tts.emit_audio())
        self.register_disposable(Disposable(self.color_image.subscribe(self._on_image)))
        self._narrate_thread = threading.Thread(
            target=self._narrate_loop, daemon=True, name="SceneNarrator"
        )
        self._narrate_thread.start()
        logger.info(f"SceneNarrator started (model={self.config.model}, period={self.config.period}s)")

    @rpc
    def stop(self) -> None:
        self._stop_evt.set()
        if self._narrate_thread is not None:
            self._narrate_thread.join(timeout=DEFAULT_THREAD_JOIN_TIMEOUT)
            self._narrate_thread = None
        if self._tts is not None:
            self._tts.dispose()
            self._tts = None
        if self._audio is not None:
            self._audio.stop()
            self._audio = None
        super().stop()

    def _on_image(self, image: Image) -> None:
        self._latest_image = image

    def _narrate_loop(self) -> None:
        while not self._stop_evt.is_set():
            image = self._latest_image
            if image is None:
                # No frame yet — check again shortly rather than burning a full period.
                self._stop_evt.wait(0.5)
                continue
            try:
                description = self._describe(image)
                if description:
                    logger.info(f"[narrator] {description}")
                    self._speak(description)
            except Exception as e:
                logger.warning(f"[narrator] narration step failed: {e}")
            self._stop_evt.wait(self.config.period)

    def _describe(self, image: Image) -> str:
        assert self._llm is not None
        content = [{"type": "text", "text": self.config.prompt}, *image.agent_encode()]
        response = self._llm.invoke([HumanMessage(content=content)])
        text = response.content
        return text.strip() if isinstance(text, str) else str(text).strip()

    def _speak(self, text: str) -> None:
        assert self._tts is not None
        # Mirrors SpeakSkill._speak_blocking: feed text to the TTS node and wait
        # for synthesis to complete so narrations don't overlap.
        with self._audio_lock:
            text_subject: Subject[str] = Subject()
            done = threading.Event()
            self._tts.consume_text(text_subject)
            subscription = self._tts.emit_text().subscribe(
                on_next=lambda _t: done.set(),
                on_error=lambda _e: done.set(),
            )
            text_subject.on_next(text)
            text_subject.on_completed()
            if not done.wait(timeout=max(5.0, len(text) * 0.1)):
                logger.warning(f"[narrator] TTS timeout while speaking: {text}")
            else:
                time.sleep(0.3)  # let the audio buffer flush
            subscription.dispose()


__all__ = ["SceneNarrator", "SceneNarratorConfig"]
