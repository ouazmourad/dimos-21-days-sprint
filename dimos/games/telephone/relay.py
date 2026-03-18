"""TelephoneRelay: Robot B receives a description, reinterprets it, passes it on."""

from dimos.core.core import rpc
from dimos.core.module import Module
from dimos.core.stream import In, Out
from dimos.msgs.sensor_msgs.Image import Image
from dimos.utils.logging_config import setup_logger
from reactivex.disposable import Disposable

logger = setup_logger()


class TelephoneRelay(Module):
    """Robot B: receives description from A, reinterprets using its own
    visual context, and relays to C."""

    telephone_in: In[str]
    telephone_out: Out[str]
    transcript: Out[str]
    color_image: In[Image]

    rpc_calls: list[str] = ["VLMAgent.query"]

    @rpc
    def start(self) -> None:
        super().start()
        self._disposables.add(
            Disposable(self.telephone_in.subscribe(self._on_receive))
        )

    def _on_receive(self, description: str) -> None:
        logger.info(f"[RELAY] Received description: {description}")

        try:
            vlm_query = self.get_rpc_calls("VLMAgent.query")
        except Exception:
            logger.error("VLMAgent not connected")
            self.telephone_out.publish(description)
            self.transcript.publish(description)
            return

        response = vlm_query(
            f"You are playing a game of telephone. Someone just told you "
            f'about an object they saw: "{description}"\n\n'
            f"Look at what YOU can see right now. Rewrite this description "
            f"in your own words for the next person in the chain, who needs "
            f"to find this object. Stay faithful to the original but use "
            f"your own visual context to clarify or adjust. "
            f"Keep it to 2-3 sentences."
        )

        logger.info(f"[RELAY] Relayed as: {response}")
        self.telephone_out.publish(response)
        self.transcript.publish(response)


telephone_relay = TelephoneRelay.blueprint

__all__ = ["TelephoneRelay", "telephone_relay"]
