"""PatrolObserver — VLM-based scene description skill for patrol robots.

Gives each robot's Agent the ability to look around and describe what
it sees through its camera, using the VLMAgent's visual_query RPC.
"""

from dimos.agents.annotation import skill
from dimos.core.core import rpc
from dimos.core.module import Module
from dimos.core.stream import In
from dimos.msgs.sensor_msgs.Image import Image
from dimos.utils.logging_config import setup_logger

logger = setup_logger()


class PatrolObserver(Module):
    """Provides a describe_surroundings skill that queries the VLM
    with the robot's current camera frame."""

    color_image: In[Image]

    vlm_rpc: str = "VLMAgent.visual_query"
    rpc_calls: list[str] = ["VLMAgent.visual_query"]

    @skill
    def describe_surroundings(self) -> str:
        """Look around and describe what you see through your camera.
        Use this to observe your environment during patrol.

        Returns:
            A detailed description of the current scene.
        """
        try:
            vlm_query = self.get_rpc_calls(self.vlm_rpc)
        except Exception:
            return "VLM not available — cannot observe."

        response = vlm_query(
            "You are a patrol robot. Describe what you see right now in detail. "
            "Focus on: people (their state — standing, sitting, lying down), "
            "objects out of place, anything unusual or concerning, and notable "
            "landmarks that help identify your location. Be specific and concise."
        )
        logger.info(f"[OBSERVER] Observation: {response}")
        return response


patrol_observer = PatrolObserver.blueprint

__all__ = ["PatrolObserver", "patrol_observer"]
