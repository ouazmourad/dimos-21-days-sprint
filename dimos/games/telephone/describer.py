"""TelephoneDescriber: Robot A sees an object and produces a text description."""

from dimos.core.core import rpc
from dimos.core.module import Module
from dimos.core.stream import In, Out
from dimos.msgs.sensor_msgs.Image import Image
from dimos.utils.logging_config import setup_logger
from reactivex.disposable import Disposable

logger = setup_logger()


class TelephoneDescriber(Module):
    """Robot A: sees an object and produces a text description via VLM."""

    color_image: In[Image]
    game_command: In[str]
    telephone_out: Out[str]
    transcript: Out[str]

    vlm_rpc: str = "VLMAgent.query"
    rpc_calls: list[str] = ["VLMAgent.query"]

    @rpc
    def start(self) -> None:
        super().start()
        self._disposables.add(
            Disposable(self.game_command.subscribe(self._on_command))
        )

    def _on_command(self, cmd: str) -> None:
        if cmd != "DESCRIBE":
            return

        logger.info("[DESCRIBER] Received DESCRIBE command, querying VLM...")

        try:
            vlm_query = self.get_rpc_calls(self.vlm_rpc)
        except Exception:
            logger.error("VLMAgent not connected")
            return

        response = vlm_query(
            "You are playing a game of telephone. Look at the scene and pick "
            "the most prominent or interesting object you can see. Describe it "
            "in detail for someone who cannot see it: mention its color, shape, "
            "size, texture, and position relative to other objects. "
            "Do NOT name the object directly - only describe its visual properties. "
            "Be vivid but concise (2-3 sentences max)."
        )

        logger.info(f"[DESCRIBER] Description: {response}")
        self.telephone_out.publish(response)
        self.transcript.publish(response)


telephone_describer = TelephoneDescriber.blueprint

__all__ = ["TelephoneDescriber", "telephone_describer"]
