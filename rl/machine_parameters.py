"""Data models for fixture layout optimization."""

from typing import Dict, Tuple
from dataclasses import dataclass

HORIZONTAL_SECUIRITY_DISTANCE = 200.0
VERTICAL_SECUIRITY_DISTANCE = 99.0

FIXTURE_AVAILABILITY = {
    1: 24,
    2: 12,
}

FIXTURE_DIMENSIONS: Dict[int, Tuple[float, float]] = {
    0: (0.0, 0.0),
    1: (145.0, 145.0),
    2: (180.0, 65.0),
}

@dataclass
class FixtureState:
    """Represents the state of a single fixture (position, orientation, type)."""
    x: float
    y: float
    angle: float
    type_id: int
