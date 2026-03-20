"""RadioSkill and RadioBridge — inter-robot communication modules.

RadioSkill gives each robot's Agent the ability to broadcast messages.
RadioBridge routes messages between robots by injecting them into each
other's Agent human_input stream.
"""

import time

from dimos.agents.annotation import skill
from dimos.core.core import rpc
from dimos.core.module import Module
from dimos.core.stream import In, Out
from dimos.utils.logging_config import setup_logger
from reactivex.disposable import Disposable

logger = setup_logger()


class RadioSkill(Module):
    """Gives an Agent the ability to broadcast messages to the other robot."""

    radio_out: Out[str]

    @skill
    def broadcast(self, message: str) -> str:
        """Send a radio message to your partner robot. Use this to share
        observations, report what you see, or coordinate actions.

        Args:
            message: The message to broadcast.

        Returns:
            Confirmation that the message was sent.
        """
        ts = time.strftime("%H:%M:%S")
        formatted = f"[{ts}] {message}"
        self.radio_out.publish(formatted)
        logger.info(f"[RADIO] Broadcast: {formatted}")
        return f"Message broadcast: {message}"

    @skill
    def request_help(self, description: str, urgency: str = "normal") -> str:
        """Request help from your partner robot. Use this for emergencies
        or when you need assistance at your location.

        Args:
            description: What you need help with.
            urgency: How urgent — "normal", "high", or "critical".

        Returns:
            Confirmation that the help request was sent.
        """
        ts = time.strftime("%H:%M:%S")
        tag = "HELP REQUEST" if urgency == "normal" else f"URGENT ({urgency.upper()})"
        formatted = f"[{ts}] [{tag}] {description}"
        self.radio_out.publish(formatted)
        logger.info(f"[RADIO] {tag}: {description}")
        return f"Help request sent (urgency={urgency}): {description}"


radio_skill = RadioSkill.blueprint


class RadioBridge(Module):
    """Routes radio messages between robots.

    Subscribes to each robot's radio output and injects formatted messages
    into the other robot's Agent human_input stream.
    """

    radio_a_in: In[str]
    radio_c_in: In[str]
    inject_to_a: Out[str]
    inject_to_c: Out[str]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._last_a_time = 0.0
        self._last_c_time = 0.0
        self._throttle_seconds = 3.0  # min gap between injections

    @rpc
    def start(self) -> None:
        super().start()
        self._disposables.add(Disposable(self.radio_a_in.subscribe(self._on_a_radio)))
        self._disposables.add(Disposable(self.radio_c_in.subscribe(self._on_c_radio)))

    def _on_a_radio(self, message: str) -> None:
        now = time.time()
        if now - self._last_c_time < self._throttle_seconds:
            return
        self._last_c_time = now
        formatted = f"[RADIO from Alpha] {message}"
        logger.info(f"[BRIDGE] A->C: {formatted}")
        self.inject_to_c.publish(formatted)

    def _on_c_radio(self, message: str) -> None:
        now = time.time()
        if now - self._last_a_time < self._throttle_seconds:
            return
        self._last_a_time = now
        formatted = f"[RADIO from Charlie] {message}"
        logger.info(f"[BRIDGE] C->A: {formatted}")
        self.inject_to_a.publish(formatted)


radio_bridge = RadioBridge.blueprint

__all__ = ["RadioSkill", "radio_skill", "RadioBridge", "radio_bridge"]
