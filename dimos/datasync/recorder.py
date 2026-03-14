# Copyright 2025-2026 Dimensional Inc.
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
"""DataRecorderModule — record all LCM topics to a Session."""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from reactivex.disposable import Disposable

from dimos.core.core import rpc
from dimos.core.module import Module, ModuleConfig
from dimos.datasync.session import Session
from dimos.protocol.pubsub.impl.lcmpubsub import LCM
from dimos.utils.logging_config import setup_logger

if TYPE_CHECKING:
    from dimos.protocol.pubsub.spec import SubscribeAllCapable

logger = setup_logger()

_SANITIZE_RE = re.compile(r"[^a-zA-Z0-9_]")


def _topic_to_key(topic: Any) -> str:
    name: str = getattr(topic, "name", None) or str(topic)
    name = name.split("#")[0]
    name = name.replace("/", "_")
    name = _SANITIZE_RE.sub("", name)
    name = name.lstrip("_")
    return name or "unknown"


@dataclass
class RecorderConfig(ModuleConfig):
    pubsubs: list[SubscribeAllCapable[Any, Any]] = field(default_factory=lambda: [LCM()])
    session_id: str | None = None
    robot_type: str = ""
    tags: list[str] = field(default_factory=list)
    topic_filter: re.Pattern[str] | None = None
    max_rate_hz: float = 0.0


class DataRecorderModule(Module):
    """Record all messages from pubsubs to a Session on disk."""

    default_config = RecorderConfig
    config: RecorderConfig

    @rpc
    def start(self) -> None:
        super().start()
        self._session = Session.create(
            session_id=self.config.session_id,
            robot_type=self.config.robot_type,
            tags=self.config.tags,
        )
        self._last_record: dict[str, float] = {}
        self._msg_count = 0
        logger.info("DataRecorder starting", session_id=self._session.session_id)
        for pubsub in self.config.pubsubs:
            if hasattr(pubsub, "start"):
                pubsub.start()  # type: ignore[union-attr]
            unsub = pubsub.subscribe_all(self._on_message)
            self._disposables.add(Disposable(unsub))
        for pubsub in self.config.pubsubs:
            if hasattr(pubsub, "stop"):
                self._disposables.add(Disposable(pubsub.stop))  # type: ignore[union-attr]

    def _on_message(self, msg: Any, topic: Any) -> None:
        topic_key = _topic_to_key(topic)
        if self.config.topic_filter is not None:
            topic_name: str = getattr(topic, "name", None) or str(topic)
            if not self.config.topic_filter.search(topic_name):
                return
        if self.config.max_rate_hz > 0:
            now = time.monotonic()
            last = self._last_record.get(topic_key, 0.0)
            if now - last < 1.0 / self.config.max_rate_hz:
                return
            self._last_record[topic_key] = now
        store = self._session.get_store(topic_key)
        try:
            store.save(msg)
            self._msg_count += 1
        except Exception:
            logger.warning(f"Failed to save message on topic {topic_key}", exc_info=True)

    @rpc
    def stop(self) -> None:
        logger.info("DataRecorder stopping", session_id=self._session.session_id, messages_recorded=self._msg_count)
        self._session.close()
        super().stop()
