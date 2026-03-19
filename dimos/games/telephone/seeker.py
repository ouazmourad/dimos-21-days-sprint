"""TelephoneSeeker: Robot C receives the final description and navigates to find the object."""

import time
from threading import Thread

from dimos.core.core import rpc
from dimos.core.module import Module
from dimos.core.stream import In, Out
from dimos.msgs.geometry_msgs.PoseStamped import PoseStamped
from dimos.msgs.geometry_msgs.Quaternion import Quaternion
from dimos.msgs.geometry_msgs.Vector3 import Vector3
from dimos.msgs.sensor_msgs.Image import Image
from dimos.utils.logging_config import setup_logger
from reactivex.disposable import Disposable

logger = setup_logger()


class TelephoneSeeker(Module):
    """Robot C: receives the relayed description and tries to find the object."""

    telephone_in: In[str]
    color_image: In[Image]
    odom: In[PoseStamped]
    search_result: Out[str]

    vlm_rpc: str = "VLMAgent.query"
    rpc_calls: list[str] = [
        "VLMAgent.query",
        "NavigationInterface.set_goal",
        "NavigationInterface.is_goal_reached",
        "NavigationInterface.cancel_goal",
        "SpatialMemory.query_by_text",
    ]

    @rpc
    def start(self) -> None:
        super().start()
        self._disposables.add(
            Disposable(self.telephone_in.subscribe(self._on_seek))
        )

    def _on_seek(self, description: str) -> None:
        logger.info(f"[SEEKER] Received description: {description}")
        thread = Thread(
            target=self._seek_object,
            args=(description,),
            daemon=True,
        )
        thread.start()

    def _seek_object(self, description: str) -> None:
        # Step 1: Query spatial memory for candidate locations
        try:
            spatial_query = self.get_rpc_calls("SpatialMemory.query_by_text")
            candidates = spatial_query(description)
            logger.info(f"[SEEKER] Found {len(candidates)} spatial memory candidates")
        except Exception:
            logger.warning("SpatialMemory not available, will search visually")
            candidates = []

        # Step 2: Navigate to best candidate
        if candidates:
            best = candidates[0]
            metadata = best.get("metadata", [{}])
            if metadata:
                meta = metadata[0] if isinstance(metadata, list) else metadata
                pos_x = meta.get("pos_x", 0)
                pos_y = meta.get("pos_y", 0)

                logger.info(f"[SEEKER] Navigating to candidate at ({pos_x:.2f}, {pos_y:.2f})")

                try:
                    goal = PoseStamped(
                        position=Vector3(pos_x, pos_y, 0),
                        orientation=Quaternion(0, 0, 0, 1),
                        frame_id="map",
                    )

                    set_goal = self.get_rpc_calls("NavigationInterface.set_goal")
                    set_goal(goal)

                    is_reached = self.get_rpc_calls("NavigationInterface.is_goal_reached")
                    for _ in range(60):
                        time.sleep(0.5)
                        if is_reached():
                            logger.info("[SEEKER] Arrived at candidate location")
                            break
                    else:
                        logger.warning("[SEEKER] Navigation timeout, verifying anyway")

                except Exception as e:
                    logger.error(f"[SEEKER] Navigation error: {e}")

        # Step 3: Verify with VLM
        try:
            vlm_query = self.get_rpc_calls(self.vlm_rpc)
            verdict = vlm_query(
                f"You are playing a game of telephone. Someone described an "
                f'object as: "{description}"\n\n'
                f"Look at what you can see right now. Does anything in your "
                f"view match that description? If so, name the object you "
                f"think they were describing and explain why it matches. "
                f"If nothing matches, say so."
            )
            logger.info(f"[SEEKER] Verdict: {verdict}")
            self.search_result.publish(verdict)
        except Exception as e:
            logger.error(f"[SEEKER] VLM query failed: {e}")
            self.search_result.publish(f"Error: could not verify - {e}")


telephone_seeker = TelephoneSeeker.blueprint

__all__ = ["TelephoneSeeker", "telephone_seeker"]
