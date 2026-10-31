"""Perception (FR-4).

Converts world state into the compact local observation handed to Laya, the
fallback controller, and the baseline controller. All three see *exactly* the
same geometry - a fixed forward-facing local window of five beams
(``front``, ``front_left``, ``front_right``, ``left``, ``right``) capped at
``sensor_range`` - so no controller has privileged information and the
comparison between them is fair.

The full map is never exposed to a controller. Distances are integers in cells;
the sentinel ``sensor_range + 1`` means "nothing detected", never ``null``.
"""

from __future__ import annotations

from .config import RoverConfig, WorldConfig
from .types import (
    Action,
    CellType,
    GoalObservation,
    GridMap,
    HazardObservation,
    Observation,
    ObstacleObservation,
    RoverState,
    euclidean,
    heading_delta,
    turn_left,
    turn_right,
)

#: The five beams of the forward-facing window, in canonical (tie-break) order.
RAY_ORDER: tuple[str, ...] = ("front", "front_left", "front_right", "left", "right")

#: Beams that are reported as booleans in ``hazards`` (FR-4).
HAZARD_BEAMS: tuple[str, ...] = ("front", "front_left", "front_right")


class Perception:
    """Fixed forward-facing local window (FR-4).

    Beam geometry, in the pinned frame (x east, y south):

    ===============  ============  =========================================
    beam             origin        direction
    ===============  ============  =========================================
    ``front``        rover cell    heading
    ``front_left``   one cell to   heading (parallel offset beam)
                     the left
    ``front_right``  one cell to   heading (parallel offset beam)
                     the right
    ``left``         rover cell    ``turn_left(heading)``
    ``right``        rover cell    ``turn_right(heading)``
    ===============  ============  =========================================

    The two offset beams are what make the window genuinely *forward-facing*
    rather than a pair of duplicate side rays: they describe the corridor ahead
    rather than the rover's flanks.
    """

    def __init__(self, world_config: WorldConfig, rover_config: RoverConfig) -> None:
        self.world = world_config
        self.rover = rover_config
        self.sensor_range = world_config.sensor_range

    # -- beam geometry ------------------------------------------------------ #

    def _beam(self, heading: str, ray: str) -> tuple[tuple[int, int], tuple[int, int]]:
        """Return ``(origin_offset, direction)`` for a beam."""
        forward = heading_delta(heading)
        if ray == "front":
            return (0, 0), forward
        if ray == "front_left":
            return heading_delta(turn_left(heading)), forward
        if ray == "front_right":
            return heading_delta(turn_right(heading)), forward
        if ray == "left":
            return (0, 0), heading_delta(turn_left(heading))
        if ray == "right":
            return (0, 0), heading_delta(turn_right(heading))
        raise ValueError(f"unknown ray {ray!r}")

    def _cast(
        self,
        grid: GridMap,
        x: int,
        y: int,
        heading: str,
        ray: str,
        *,
        kind: CellType = CellType.OBSTACLE,
    ) -> int:
        """Distance in cells to the first cell of ``kind`` along a beam.

        Returns the sentinel ``sensor_range + 1`` when nothing is detected.
        """
        (ox, oy), (dx, dy) = self._beam(heading, ray)
        px, py = x + ox, y + oy
        for step in range(1, self.sensor_range + 1):
            px, py = px + dx, py + dy
            if not grid.in_bounds(px, py):
                return step
            if kind is CellType.HAZARD_SLOPE:
                if grid.cell(px, py) is CellType.HAZARD_SLOPE:
                    return step
                continue
            if grid.cell(px, py) is CellType.OBSTACLE:
                return step
        return self.sensor_range + 1

    def ray_distances(self, grid: GridMap, x: int, y: int, heading: str) -> dict[str, int]:
        """Obstacle distances for all five beams (used by the CLI map view)."""
        return {
            ray: self._cast(grid, x, y, heading, ray) for ray in RAY_ORDER
        }

    # -- observation -------------------------------------------------------- #

    def observe(
        self,
        grid: GridMap,
        state: RoverState,
        safe_actions: list[Action] | tuple[Action, ...],
        previous_action: Action | None,
    ) -> Observation:
        """Build the FR-4 observation for ``state``.

        ``safe_actions`` is the guardian's pre-filter, so the controller's view
        of what is legal and the guardian's view cannot drift apart.
        """
        heading = state.heading

        distances = {
            ray: self._cast(grid, state.x, state.y, heading, ray) for ray in RAY_ORDER
        }
        obstacles = ObstacleObservation(
            front=distances["front"],
            front_left=distances["front_left"],
            front_right=distances["front_right"],
            left=distances["left"],
            right=distances["right"],
        )

        hazard_results = {
            ray: self._cast(grid, state.x, state.y, heading, ray, kind=CellType.HAZARD_SLOPE)
            for ray in HAZARD_BEAMS
        }
        hazards = HazardObservation(
            front=hazard_results["front"] <= self.sensor_range,
            front_left=hazard_results["front_left"] <= self.sensor_range,
            front_right=hazard_results["front_right"] <= self.sensor_range,
        )

        gx, gy = grid.goal
        goal = GoalObservation(
            relative_x=gx - state.x,
            relative_y=gy - state.y,
            distance=euclidean(state.x, state.y, gx, gy),
        )

        return Observation(
            goal=goal,
            obstacles=obstacles,
            hazards=hazards,
            heading=heading,
            battery_fraction=state.battery,
            energy_used=state.energy_used,
            steps=state.steps,
            previous_action=previous_action.value if previous_action else None,
            recent_stops=state.consecutive_stops,
            safe_actions=[a.value for a in safe_actions],
        )
