"""Virtual test simulator backend.

Presents the *exact same* :class:`~laya_rover_agent.hardware.base.RoverBackend`
interface as the ROS 2 and serial backends, but underneath it drives the
discrete simulator on a generated map. This is how the rover-ready stack is
validated end to end before any hardware is attached - and how you can
rehearse a mission profile offline.
"""

from __future__ import annotations

import random
import time

from ..config import ProjectConfig
from ..dynamics import RoverSimulator, action_target
from ..perception import Perception
from ..types import Action, CellType, GridMap, RoverState, Status
from ..world import MapSpec, WorldGenerator, stream_seed
from .base import ExecutionReport, SensorFrame


class SimBackend:
    """Simulated rover: same interface, virtual hardware."""

    name = "sim"

    def __init__(
        self,
        config: ProjectConfig,
        seed: int = 0,
        difficulty: str = "medium",
        preset: str = "random",
        clock=time.perf_counter,
    ) -> None:
        self.config = config
        self.clock = clock
        self.map_spec = MapSpec(seed=seed, difficulty=difficulty, preset=preset)
        self.grid: GridMap = WorldGenerator(config.world).generate(self.map_spec)
        self.sim = RoverSimulator(
            world=self.grid,
            config=config.rover,
            difficulty=difficulty,
            stall_rng=random.Random(stream_seed(seed * 17 + 5, "hazard_stalls")),
            clock=clock,
        )
        self.perception = Perception(config.world, config.rover)
        self.state: RoverState | None = None
        self._connected = False

    # -- lifecycle ---------------------------------------------------------- #

    def connect(self) -> None:
        if self.state is None:
            self.state = self.sim.initial_state()
            self.sim.note_position(self.state)
        self._connected = True

    def close(self) -> None:
        self._connected = False

    def emergency_stop(self) -> None:
        """Simulated e-stop: freezes the rover and marks the episode aborted."""
        if self.state is not None:
            self.state = self.state.model_copy(update={"status": Status.STUCK})

    # -- I/O ---------------------------------------------------------------- #

    def read_sensors(self) -> SensorFrame:
        if self.state is None:
            raise RuntimeError("connect() must be called before read_sensors()")
        from ..safety import SafetyGuardian

        guardian = SafetyGuardian(self.grid, self.config.world, self.config.rover)
        safe = guardian.safe_actions(self.state)
        observation = self.perception.observe(self.grid, self.state, safe.actions, None)
        return SensorFrame(
            x=self.state.x,
            y=self.state.y,
            heading=self.state.heading,
            battery=self.state.battery,
            obstacles=observation.obstacles.model_dump(),
            hazards=observation.hazards.model_dump(),
            goal_relative=(observation.goal.relative_x, observation.goal.relative_y),
            goal_distance=observation.goal.distance,
            sensor_range=self.config.world.sensor_range,
            timestamp_s=self.clock(),
            raw={"state": self.state.model_dump(mode="json")},
        )

    def execute(self, action: Action) -> ExecutionReport:
        if self.state is None:
            raise RuntimeError("connect() must be called before execute()")
        started = self.clock()
        outcome = self.sim.step(self.state, action)
        self.state = outcome.state
        return ExecutionReport(
            moved=outcome.moved,
            stalled=outcome.stalled,
            collision=outcome.collision,
            reached_goal=outcome.reached_goal,
            energy_used=outcome.state.energy_used,
            battery=outcome.state.battery,
            position=(outcome.state.x, outcome.state.y),
            heading=outcome.state.heading,
            latency_ms=(self.clock() - started) * 1000.0,
            raw={"status": outcome.state.status.value, "reason": outcome.reason},
        )

    # -- helpers ------------------------------------------------------------ #

    def would_enter(self, action: Action) -> tuple[bool, bool]:
        if self.state is None:
            return (False, False)
        return self.sim.would_enter(self.state, action)

    @property
    def status(self) -> Status:
        return self.state.status if self.state else Status.ACTIVE

    def goal(self) -> tuple[int, int]:
        return self.grid.goal

    def cell_is_hazard(self, action: Action) -> bool:
        if self.state is None:
            return False
        return self.grid.is_hazard(*action_target(self.state.x, self.state.y, self.state.heading, action))

    def goal_cell_type(self) -> CellType:
        return self.grid.cell(*self.grid.goal)
