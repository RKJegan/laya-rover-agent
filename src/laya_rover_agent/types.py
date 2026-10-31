"""Core domain types for the Laya Rover Agent.

Pins the conventions that the spec demands be explicit rather than inferred:

* Coordinate frame: origin ``(0, 0)`` is the **top-left** cell; ``x`` increases
  eastward, ``y`` increases southward. Heading ``N`` decreases ``y``,
  ``E`` increases ``x``, ``S`` increases ``y``, ``W`` decreases ``x``.
* ``front_left`` / ``front_right`` are the headings 90 degrees from the current
  heading, relative to the direction of travel, in the same frame.
* Distances are **Euclidean** grid-cell distances everywhere.
* The "nothing detected" sentinel is the integer ``sensor_range + 1`` (never
  ``null``) so the observation schema stays strictly numeric.
"""

from __future__ import annotations

import math
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

# --------------------------------------------------------------------------- #
# Enumerations
# --------------------------------------------------------------------------- #


class Action(str, Enum):
    """The finite action set (FR-3). No free-text, no raw actuator values."""

    MOVE_FORWARD = "MOVE_FORWARD"
    TURN_LEFT_AND_MOVE = "TURN_LEFT_AND_MOVE"
    TURN_RIGHT_AND_MOVE = "TURN_RIGHT_AND_MOVE"
    REVERSE = "REVERSE"
    STOP_AND_REPLAN = "STOP_AND_REPLAN"

    def __str__(self) -> str:  # pragma: no cover - display helper
        return self.value


ALL_ACTIONS: tuple[Action, ...] = (
    Action.MOVE_FORWARD,
    Action.TURN_LEFT_AND_MOVE,
    Action.TURN_RIGHT_AND_MOVE,
    Action.REVERSE,
    Action.STOP_AND_REPLAN,
)

#: Translation-only actions (STOP_AND_REPLAN never moves the rover).
MOVING_ACTIONS: tuple[Action, ...] = (
    Action.MOVE_FORWARD,
    Action.TURN_LEFT_AND_MOVE,
    Action.TURN_RIGHT_AND_MOVE,
    Action.REVERSE,
)

Heading = Literal["N", "E", "S", "W"]
HEADINGS: tuple[Heading, ...] = ("N", "E", "S", "W")

#: (dx, dy) deltas in the pinned coordinate frame.
HEADING_DELTAS: dict[str, tuple[int, int]] = {
    "N": (0, -1),
    "E": (1, 0),
    "S": (0, 1),
    "W": (-1, 0),
}


def turn_left(heading: str) -> str:
    """Rotate a heading 90 degrees counter-clockwise in the pinned frame."""
    return HEADINGS[(HEADINGS.index(heading) - 1) % 4]


def turn_right(heading: str) -> str:
    """Rotate a heading 90 degrees clockwise in the pinned frame."""
    return HEADINGS[(HEADINGS.index(heading) + 1) % 4]


def heading_delta(heading: str) -> tuple[int, int]:
    return HEADING_DELTAS[heading]


def euclidean(ax: float, ay: float, bx: float, by: float) -> float:
    """Canonical distance function used *everywhere* (goal.distance, fallback
    comparisons, metrics) so no two code paths disagree."""
    return math.hypot(ax - bx, ay - by)


class CellType(str, Enum):
    FREE = "FREE"
    OBSTACLE = "OBSTACLE"
    HAZARD_SLOPE = "HAZARD_SLOPE"
    START = "START"
    GOAL = "GOAL"


class Status(str, Enum):
    ACTIVE = "active"
    GOAL = "goal"
    COLLISION = "collision"
    BATTERY_DEPLETED = "battery_depleted"
    STUCK = "stuck"
    TIMEOUT = "timeout"


class ActionSource(str, Enum):
    LAYA = "laya"
    FALLBACK = "fallback"
    GUARDIAN_OVERRIDE = "guardian_override"
    BASELINE = "baseline"


class GuardianReason(str, Enum):
    """Machine-readable reason codes for every block / override (FR-5)."""

    # Structured API failure classes (never a reason to move).
    MALFORMED_RESPONSE = "MALFORMED_RESPONSE"
    SCHEMA_VIOLATION = "SCHEMA_VIOLATION"
    LOW_CONFIDENCE = "LOW_CONFIDENCE"
    DECISION_TIMEOUT = "DECISION_TIMEOUT"
    API_ERROR = "API_ERROR"
    AUTH_ERROR = "AUTH_ERROR"
    CONNECTION_ERROR = "CONNECTION_ERROR"
    RATE_LIMITED = "RATE_LIMITED"
    NO_CREDITS = "NO_CREDITS"
    # Guardian hard-rule rejections.
    BLOCKED_OBSTACLE = "BLOCKED_OBSTACLE"
    BLOCKED_OUT_OF_BOUNDS = "BLOCKED_OUT_OF_BOUNDS"
    BLOCKED_HAZARD_BATTERY = "BLOCKED_HAZARD_BATTERY"
    BLOCKED_HAZARD_PREFERENCE = "BLOCKED_HAZARD_PREFERENCE"
    BLOCKED_BATTERY_RESERVE = "BLOCKED_BATTERY_RESERVE"
    NOT_IN_SAFE_ACTIONS = "NOT_IN_SAFE_ACTIONS"
    STOP_CAP_REACHED = "STOP_CAP_REACHED"
    NO_LEGAL_ACTION = "NO_LEGAL_ACTION"
    # Diagnostic.
    EPISODE_WALL_CLOCK_TIMEOUT = "EPISODE_WALL_CLOCK_TIMEOUT"
    # Success (no override happened).
    APPROVED = "APPROVED"


# --------------------------------------------------------------------------- #
# State models
# --------------------------------------------------------------------------- #


class RoverState(BaseModel):
    """FR-2 rover state."""

    model_config = ConfigDict(frozen=True)

    x: int
    y: int
    heading: Heading
    battery: float = Field(ge=0.0, le=1.0, description="normalized 0..1")
    energy_used: float = Field(ge=0.0, description="raw energy units consumed")
    steps: int = Field(ge=0)
    status: Status = Status.ACTIVE
    consecutive_stops: int = Field(default=0, ge=0)


class GoalObservation(BaseModel):
    relative_x: int
    relative_y: int
    distance: float = Field(ge=0.0, description="Euclidean distance in cells")


class ObstacleObservation(BaseModel):
    """Obstacle distance per direction. Sentinel = sensor_range + 1."""

    front: int = Field(ge=0)
    front_left: int = Field(ge=0)
    front_right: int = Field(ge=0)
    left: int = Field(ge=0)
    right: int = Field(ge=0)


class HazardObservation(BaseModel):
    """Whether a HAZARD_SLOPE cell lies within the forward window per direction."""

    front: bool
    front_left: bool
    front_right: bool


class Observation(BaseModel):
    """FR-4 observation schema handed to Laya / fallback / baseline verbatim."""

    model_config = ConfigDict(frozen=True)

    goal: GoalObservation
    obstacles: ObstacleObservation
    hazards: HazardObservation
    heading: Heading
    battery_fraction: float = Field(ge=0.0, le=1.0)
    energy_used: float
    steps: int
    previous_action: str | None
    recent_stops: int = Field(ge=0)
    safe_actions: list[str]

    def to_decision_state(self) -> dict[str, object]:
        """Compact, self-describing state blob sent as Laya's ``state`` field.

        Kept strictly numeric/boolean with explicit conventions so the
        decision model is never asked to guess the map frame.
        """
        return {
            "task": "planetary_rover_local_next_action_selection",
            "coordinate_frame": "origin top-left, x east, y south, N=-y E=+x S=+y W=-x",
            "heading": self.heading,
            "goal": {
                "relative_x_east_positive": self.goal.relative_x,
                "relative_y_south_positive": self.goal.relative_y,
                "euclidean_distance_cells": round(self.goal.distance, 3),
            },
            "nearest_obstacle_distance_cells": self.obstacles.model_dump(),
            "no_obstacle_sentinel": "sensor_range + 1 (value 7 in this configuration)",
            "hazard_slope_nearby": self.hazards.model_dump(),
            "battery_fraction": round(self.battery_fraction, 4),
            "energy_used_units": round(self.energy_used, 3),
            "steps_taken": self.steps,
            "previous_action": self.previous_action,
            "recent_stop_actions": self.recent_stops,
            "allowed_actions": list(self.safe_actions),
            "notes": (
                "HAZARD_SLOPE cells are traversable but cost 1.8x energy and carry "
                "a stall probability. Obstacle distances use the sentinel above for "
                "'nothing detected'. Choose exactly one allowed action."
            ),
        }


class GridMap(BaseModel):
    """FR-1 deterministic 2D terrain map."""

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    width: int
    height: int
    start: tuple[int, int]
    goal: tuple[int, int]
    cells: list[list[str]] = Field(description="row-major cell types, y then x")
    #: Bounding boxes of the engineered concave pockets (x, y, w, h, bricks),
    #: recorded so tests, figures and the report can point at the trap geometry.
    trap_clusters: tuple[dict[str, int], ...] = ()
    preset: str = "random"
    difficulty: str = "medium"
    seed: int = 0

    # -- accessors ---------------------------------------------------------- #

    def in_bounds(self, x: int, y: int) -> bool:
        return 0 <= x < self.width and 0 <= y < self.height

    def cell(self, x: int, y: int) -> CellType:
        if not self.in_bounds(x, y):
            return CellType.OBSTACLE  # out of bounds reads as blocked, never crashes
        return CellType(self.cells[y][x])

    def is_obstacle(self, x: int, y: int) -> bool:
        return self.cell(x, y) is CellType.OBSTACLE

    def is_hazard(self, x: int, y: int) -> bool:
        return self.cell(x, y) is CellType.HAZARD_SLOPE

    def is_traversable(self, x: int, y: int) -> bool:
        """Traversable == inside the grid and not an obstacle (hazards allowed)."""
        return self.in_bounds(x, y) and not self.is_obstacle(x, y)

    def rows(self) -> list[list[str]]:
        return self.cells


class DecisionResult(BaseModel):
    """Validated decision payload (FR-6 / Section 15.4).

    ``reason_code`` is optional because the live Laya contract returns
    ``choice`` + ``confidence`` + ``probabilities`` only; the spec's
    ``reason_code`` is preserved when a provider supplies it (or when the
    mock emits it) and is otherwise derived locally by the guardian.
    """

    model_config = ConfigDict(extra="ignore")

    chosen_action: Action
    confidence: float = Field(ge=0.0, le=1.0)
    reason_code: str | None = None
    probabilities: dict[str, float] | None = None


class DecisionCall(BaseModel):
    """Outcome envelope from any adapter. ``valid=False`` never means "may move"."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    valid: bool
    decision: DecisionResult | None = None
    reason_code: GuardianReason | None = None
    latency_ms: float = 0.0
    http_status: int | None = None
    raw_response: str | None = None
    raw_request: str | None = None
    provider_model: str | None = None
    #: Laya reports ``input_tokens``/``output_tokens``/``state_tokens`` as numbers but
    #: ``truncated`` (bool) and ``truncated_questions`` (list[str]) alongside them, so
    #: this is deliberately not ``dict[str, float]``.
    usage: dict[str, Any] | None = None
    #: True when the served model reports that it did not read the whole state. A
    #: truncated state means the decision was made on partial evidence, which is
    #: recorded per step for the safety case rather than silently ignored.
    truncated: bool = False
