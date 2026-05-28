"""Intent library — short, scripted performance beats.

Each intent is a coroutine that drives the channels for a finite
duration. Driven by a personality, the same intent should produce
visibly different motion (the v1 success criterion).

The intents are plain Python coroutines, not DimOS skills, so they
can be unit-tested in isolation. Wrapping them in @skill for agent
calling is one wrapper file away.
"""

from dimos.animator.intents.curious_head_tilt import curious_head_tilt
from dimos.animator.intents.notice_guest import notice_guest
from dimos.animator.intents.proud_chest_lift import proud_chest_lift
from dimos.animator.intents.search_room import search_room

__all__ = [
    "curious_head_tilt",
    "notice_guest",
    "proud_chest_lift",
    "search_room",
]


# Intent registry: name → callable. Useful for blueprint dispatch.
INTENTS = {
    "notice_guest": notice_guest,
    "curious_head_tilt": curious_head_tilt,
    "proud_chest_lift": proud_chest_lift,
    "search_room": search_room,
}
