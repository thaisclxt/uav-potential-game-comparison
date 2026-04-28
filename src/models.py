from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass
class Depot:
    location: Tuple[float, float]

    @property
    def x(self):
        return self.location[0]

    @property
    def y(self):
        return self.location[1]


@dataclass
class Waypoint:
    x: float
    y: float
    wid: int
    revenue: float


@dataclass
class UAV:
    uid: int
    m_j: int = 0 # Number of repetitions
    sequence: List[Waypoint] = field(default_factory=list)

    def reset(self) -> None:
        self.sequence.clear()
        self.m_j = 0
