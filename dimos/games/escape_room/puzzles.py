"""Puzzle definitions for the Robot Escape Room.

Each puzzle has:
- A clue the Trapped robot must find (described by what VLM would see)
- A hint the Guide gives to help locate it
- Keywords that validate discovery (if VLM description contains these)
"""

from dataclasses import dataclass


@dataclass
class Puzzle:
    name: str
    description: str  # what the object actually is
    hint: str  # cryptic hint the Guide gives
    keywords: list[str]  # any of these in VLM response = found
    location_hint: str  # where in the office scene


# The 3 puzzles use objects already in scene_office1.xml
PUZZLES = [
    Puzzle(
        name="Clue 1: The Hidden Seat",
        description="A black office chair tucked under a desk",
        hint=(
            "Your first clue is something people sit on, but it's hiding. "
            "Look for something dark and low, tucked away where people work. "
            "Move toward the desks and look carefully underneath."
        ),
        keywords=["chair", "seat", "stool", "sitting", "wheels", "rolling"],
        location_hint="near the desks",
    ),
    Puzzle(
        name="Clue 2: The Light Barrier",
        description="The curtains or window panels",
        hint=(
            "Your second clue hangs vertically and blocks the outside world. "
            "It's soft, it drapes, and it controls how much light enters the room. "
            "Turn toward where the light comes from and look for fabric."
        ),
        keywords=["curtain", "drape", "fabric", "window", "blind", "panel", "hanging"],
        location_hint="near the windows",
    ),
    Puzzle(
        name="Clue 3: The Gathering Surface",
        description="The large meeting/conference table",
        hint=(
            "Your final clue is where groups come together. It's flat, wide, "
            "and surrounded by the first clue you found. It dominates the center "
            "of the space. Describe this large surface to complete your escape."
        ),
        keywords=["table", "desk", "surface", "conference", "meeting", "flat", "long"],
        location_hint="center of the room",
    ),
]


def check_discovery(vlm_description: str, puzzle: Puzzle) -> bool:
    """Check if the VLM description matches the puzzle keywords."""
    desc_lower = vlm_description.lower()
    return any(kw in desc_lower for kw in puzzle.keywords)
