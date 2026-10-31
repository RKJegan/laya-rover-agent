"""Transport-agnostic rover interface.

The agent core never talks to a motor. It consumes a :class:`SensorFrame` and
emits one :class:`Action`; the *backend* is the only thing that knows whether
that action ends up as a simulated step, a ROS 2 topic message, or a serial
line. Swapping backends therefore requires **no change to the agent, guardian,
Laya adapter or logging code** - only the ``--backend`` selection.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from ..types import Action, Heading, RoverState


@dataclass(frozen=True)
class SensorFrame:
    """One fused sensor sample, already reduced to the observation contract.

    ``obstacles`` uses the same five relative beams and the same integer
    sentinel (``sensor_range + 1``) as :class:`~laya_rover_agent.types.Observation`,
    so a real rover and the simulator feed the agent identical structure.
    """

    x: int
    y: int
    heading: Heading
    battery: float
    obstacles: dict[str, int]
    hazards: dict[str, bool]
    goal_relative: tuple[int, int]
    goal_distance: float
    sensor_range: int
    timestamp_s: float
    raw: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class ExecutionReport:
    """Outcome of one commanded action, as reported by the hardware."""

    moved: bool
    stalled: bool = False
    collision: bool = False
    reached_goal: bool = False
    energy_used: float = 0.0
    battery: float = 1.0
    position: tuple[int, int] | None = None
    heading: Heading | None = None
    latency_ms: float = 0.0
    raw: dict[str, object] = field(default_factory=dict)


@runtime_checkable
class RoverBackend(Protocol):
    """What every rover transport must provide."""

    name: str

    def connect(self) -> None:  # pragma: no cover - protocol
        ...

    def read_sensors(self) -> SensorFrame:  # pragma: no cover - protocol
        ...

    def execute(self, action: Action) -> ExecutionReport:  # pragma: no cover - protocol
        ...

    def emergency_stop(self) -> None:  # pragma: no cover - protocol
        ...

    def close(self) -> None:  # pragma: no cover - protocol
        ...


class BackendUnavailableError(RuntimeError):
    """Raised when a backend's optional dependency (rclpy, pyserial) is missing."""


def frame_to_state(frame: SensorFrame, rover_state: RoverState) -> RoverState:
    """Refresh the parts of a RoverState that hardware can measure."""
    return rover_state.model_copy(
        update={
            "x": frame.x,
            "y": frame.y,
            "heading": frame.heading,
            "battery": max(0.0, min(1.0, frame.battery)),
        }
    )
