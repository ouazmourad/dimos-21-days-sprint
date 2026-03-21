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


# The 3 clues are colored objects placed in the maze
PUZZLES = [
    Puzzle(
        name="Clue 1: The Red Sphere",
        description="A bright red ball on the ground in a dead-end corridor",
        hint=(
            "Your first clue is round and red, like a small ball. "
            "It's hiding in a dead end on the left side of the maze. "
            "Turn left and follow the wall — look for something bright "
            "on the ground. Ignore any red cubes — you need a SPHERE."
        ),
        keywords=["red", "sphere", "ball", "round", "circular"],
        location_hint="dead end on the left side",
    ),
    Puzzle(
        name="Clue 2: The Blue Cylinder",
        description="A blue cylinder tucked behind a corner on the right side",
        hint=(
            "Your second clue is blue and shaped like a tube or can. "
            "It's on the right side of the maze, hidden behind a corner. "
            "Navigate to the right corridors and look for something blue "
            "and cylindrical near the wall."
        ),
        keywords=["blue", "cylinder", "tube", "can", "cylindrical", "pillar"],
        location_hint="right corridor behind a corner",
    ),
    Puzzle(
        name="Clue 3: The Green Box",
        description="A green cube near the upper-right corner — the exit",
        hint=(
            "Your final clue is green and square, like a small box or cube. "
            "It marks the exit of the maze in the upper-right area. "
            "Navigate toward the far corner — when you find the green box, "
            "describe it and you're FREE!"
        ),
        keywords=["green", "box", "cube", "square", "block"],
        location_hint="upper-right corner near the exit",
    ),
]


def check_discovery(vlm_description: str, puzzle: Puzzle) -> bool:
    """Check if the VLM description matches the puzzle keywords."""
    desc_lower = vlm_description.lower()
    return any(kw in desc_lower for kw in puzzle.keywords)
