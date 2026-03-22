"""Puzzle definitions for the Robot Escape Room maze.

Each puzzle has:
- A clue the Trapped robot must find (a colored object in the maze)
- A hint the Guide gives to help locate it
- Keywords that validate discovery (if VLM description contains these)
"""

from dataclasses import dataclass


@dataclass
class Puzzle:
    name: str
    description: str
    hint: str
    keywords: list[str]
    location_hint: str


# Clues ordered nearest-to-spawn first: Blue (spawn room) → Green (adjacent) → Red (far)
PUZZLES = [
    Puzzle(
        name="Clue 1: The Blue Cylinder",
        description="A glowing blue cylinder on a pedestal in the lower-left area",
        hint=(
            "Your first clue is a glowing blue tube on a grey pedestal. "
            "It's nearby — look around you in the lower-left area. "
            "Turn and walk until you see something bright blue on a stand."
        ),
        keywords=["blue", "cylinder", "tube", "can", "cylindrical", "pillar"],
        location_hint="lower-left pocket near spawn",
    ),
    Puzzle(
        name="Clue 2: The Green Box",
        description="A glowing green cube on a pedestal marking the exit",
        hint=(
            "Your second clue is a glowing green cube on a grey stand. "
            "It's in the upper-right area. Head north through the gap "
            "in the wall, then look to your right for the green box."
        ),
        keywords=["green", "box", "cube", "square", "block", "prism", "rectangle", "rectangular"],
        location_hint="upper-right corner",
    ),
    Puzzle(
        name="Clue 3: The Red Sphere",
        description="A glowing red sphere on a pedestal in the left room",
        hint=(
            "Your final clue is a glowing red ball on a grey pedestal. "
            "It's on the far LEFT side of the maze. Keep heading north "
            "past the center wall opening, then go left. "
            "Find the bright red sphere and you're FREE!"
        ),
        keywords=["red", "sphere", "ball", "round", "circular", "orb"],
        location_hint="left room behind the center divider",
    ),
]


_COLOR_WORDS = {"red", "blue", "green"}
_SHAPE_WORDS = {
    "sphere", "ball", "round", "circular", "orb",
    "cylinder", "tube", "can", "cylindrical", "pillar",
    "box", "cube", "square", "block", "prism", "rectangle", "rectangular",
}


def check_discovery(vlm_description: str, puzzle: Puzzle) -> bool:
    """Check if the VLM description matches both a color AND shape keyword.

    Requiring two keyword categories prevents false positives like
    'red wall' matching the Red Sphere puzzle.
    """
    desc_lower = vlm_description.lower()
    has_color = any(kw in desc_lower for kw in puzzle.keywords if kw in _COLOR_WORDS)
    has_shape = any(kw in desc_lower for kw in puzzle.keywords if kw in _SHAPE_WORDS)
    return has_color and has_shape
