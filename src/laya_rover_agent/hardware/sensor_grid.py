"""Local terrain oracle built from real sensor beams.

The safety guardian is deterministic code that asks a terrain oracle three
questions: is this cell in bounds, is it an obstacle, is it a hazardous slope?
In simulation the oracle is the generated map. On a real rover there is no map,
so this module synthesises an equivalent oracle from the sensor frame - which is
why the *same* guardian class supervises both, with no forking of safety logic.

Assumption to be explicit about: cells the beams did not cover are treated as
traversable (``unknown_is_blocked=False`` by default). A beam that reports
"nothing within ``sensor_range``" is positive evidence for the cells it swept;
everything outside the beam fan is genuinely unknown. A mission-qualified
system would layer persistent occupancy mapping plus a hard "never drive into
unknown terrain at speed" policy on top - that is deliberately out of scope
here (see ``docs/HARDWARE_DEPLOYMENT.md``).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..types import CellType, Heading, heading_delta, turn_left, turn_right
from .base import SensorFrame

BEAM_SEQUENCE: tuple[str, ...] = ("front", "front_left", "front_right", "left", "right")


@dataclass
class SensorGridFacade:
    """Duck-typed terrain oracle: ``in_bounds`` / ``is_obstacle`` / ``is_hazard``."""

    width: int
    height: int
    origin: tuple[int, int]
    start: tuple[int, int]
    goal: tuple[int, int]
    unknown_is_blocked: bool = False
    obstacles: set[tuple[int, int]] = field(default_factory=set)
    hazards: set[tuple[int, int]] = field(default_factory=set)
    free: set[tuple[int, int]] = field(default_factory=set)

    # -- construction ------------------------------------------------------- #

    @classmethod
    def from_frame(
        cls,
        frame: SensorFrame,
        goal_absolute: tuple[int, int] | None = None,
        unknown_is_blocked: bool = False,
    ) -> "SensorGridFacade":
        width = max(8, 2 * frame.sensor_range + 4)
        height = max(8, 2 * frame.sensor_range + 4)
        goal = goal_absolute or (frame.x + frame.goal_relative[0], frame.y + frame.goal_relative[1])
        facade = cls(
            width=width,
            height=height,
            origin=(frame.x - width // 2, frame.y - height // 2),
            start=(frame.x, frame.y),
            goal=goal,
            unknown_is_blocked=unknown_is_blocked,
        )
        facade.update(frame)
        return facade

    # -- oracle API --------------------------------------------------------- #

    def in_bounds(self, x: int, y: int) -> bool:
        lx, ly = self.to_local(x, y)
        return 0 <= lx < self.width and 0 <= ly < self.height

    def is_obstacle(self, x: int, y: int) -> bool:
        if (x, y) in self.obstacles:
            return True
        if not self.in_bounds(x, y):
            return True
        if self.unknown_is_blocked:
            return (x, y) not in self.free
        return False

    def is_hazard(self, x: int, y: int) -> bool:
        return (x, y) in self.hazards

    def cell(self, x: int, y: int) -> CellType:
        if self.is_obstacle(x, y):
            return CellType.OBSTACLE
        if self.is_hazard(x, y):
            return CellType.HAZARD_SLOPE
        return CellType.FREE

    # -- maintenance -------------------------------------------------------- #

    def to_local(self, x: int, y: int) -> tuple[int, int]:
        return (x - self.origin[0], y - self.origin[1])

    def to_absolute(self, lx: int, ly: int) -> tuple[int, int]:
        return (lx + self.origin[0], ly + self.origin[1])

    def update(self, frame: SensorFrame) -> None:
        """Fold one frame's beams into the occupancy sets."""
        self.start = (frame.x, frame.y)
        origin = (frame.x - self.width // 2, frame.y - self.height // 2)
        if origin != self.origin:
            shift_x = self.origin[0] - origin[0]
            shift_y = self.origin[1] - origin[1]
            self.obstacles = {(x + shift_x, y + shift_y) for (x, y) in self.obstacles}
            self.hazards = {(x + shift_x, y + shift_y) for (x, y) in self.hazards}
            self.free = {(x + shift_x, y + shift_y) for (x, y) in self.free}
            self.origin = origin

        for beam in BEAM_SEQUENCE:
            distance = int(frame.obstacles.get(beam, frame.sensor_range + 1))
            heading = _beam_heading(frame.heading, beam)
            self._mark_beam(frame.x, frame.y, heading, distance)

        for beam, present in frame.hazards.items():
            if not present:
                continue
            heading = _beam_heading(frame.heading, beam)
            dx, dy = heading_delta(heading)
            for step in range(1, frame.sensor_range + 1):
                self.hazards.add((frame.x + dx * step, frame.y + dy * step))

    def _mark_beam(self, x: int, y: int, heading: Heading, distance: int) -> None:
        dx, dy = heading_delta(heading)
        for step in range(1, distance):
            self.free.add((x + dx * step, y + dy * step))
        self.obstacles.add((x + dx * distance, y + dy * distance))


def _beam_heading(heading: Heading, beam: str) -> Heading:
    if beam == "front":
        return heading
    if beam == "left":
        return turn_left(heading)
    if beam == "right":
        return turn_right(heading)
    if beam == "front_left":
        return turn_left(heading)
    if beam == "front_right":
        return turn_right(heading)
    raise ValueError(f"unknown beam {beam!r}")
