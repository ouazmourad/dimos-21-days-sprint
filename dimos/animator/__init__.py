"""Robot Technical Animator — character performance layer for DimOS.

v1 scope: a Go2-only blueprint that turns intent ("notice the guest",
"act curious") into expressive joint commands biased by a personality
profile. See ``docs/ROBOT_ANIMATOR_V1_PROPOSAL.md`` for the full design
and the explicit failure modes this code is built against.

The public surface is intentionally small:

    from dimos.animator import CharacterRig, Personality, BehaviorMixer
    from dimos.animator.retargeter import Go2Retargeter
    from dimos.animator.intents import notice_guest, curious_head_tilt
"""

from dimos.animator.mixer import BehaviorMixer, ChannelState, MixedMotion
from dimos.animator.personality import Personality
from dimos.animator.rig import CharacterRig, JointRole

__all__ = [
    "BehaviorMixer",
    "CharacterRig",
    "ChannelState",
    "JointRole",
    "MixedMotion",
    "Personality",
]
