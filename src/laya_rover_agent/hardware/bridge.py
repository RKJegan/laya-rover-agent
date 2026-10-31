"""Rover bridge: the same agent, running on a rover.

This is the module that makes the project *rover ready*. It runs the identical
decision loop as the simulator - safety pre-filter, Laya decision, guardian
approval, typed action - but the loop is driven by a hardware backend
(:mod:`sim_backend`, :mod:`ros2_backend`, :mod:`serial_backend`).

Safety posture on hardware:

* the guardian is the same deterministic code path as in the experiments, so a
  veto on the rover and a veto in simulation are the same decision;
* unknown terrain is configurable per backend (``unknown_is_blocked``);
* an emergency stop is published on any terminal condition, on battery below
  the reserve, and whenever the decision layer is unavailable and the fallback
  also has no legal action;
* ``dry_run=True`` (default) lets you rehearse a mission with **zero motion**:
  every action is computed, guarded and logged, but never sent to the wheels.
"""

from __future__ import annotations

import asyncio
import json
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from ..config import ProjectConfig
from ..controllers.base import Controller, ControllerContext
from ..controllers.fallback import FallbackController, FallbackPolicy
from ..safety import SafetyGuardian
from ..types import (
    Action,
    ActionSource,
    GoalObservation,
    GuardianReason,
    HazardObservation,
    Observation,
    ObstacleObservation,
    RoverState,
    Status,
)
from .base import RoverBackend, SensorFrame
from .sensor_grid import SensorGridFacade


@dataclass
class MissionSummary:
    backend: str
    controller: str
    steps: int
    actions: dict[str, int] = field(default_factory=dict)
    guardian_overrides: int = 0
    fallback_count: int = 0
    laya_calls: int = 0
    laya_invalid: int = 0
    reached_goal: bool = False
    stopped_reason: str = "max_steps"
    wall_clock_s: float = 0.0
    started_at: str = ""
    finished_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "controller": self.controller,
            "steps": self.steps,
            "actions": self.actions,
            "guardian_overrides": self.guardian_overrides,
            "fallback_count": self.fallback_count,
            "laya_calls": self.laya_calls,
            "laya_invalid": self.laya_invalid,
            "reached_goal": self.reached_goal,
            "stopped_reason": self.stopped_reason,
            "wall_clock_s": round(self.wall_clock_s, 3),
            "started_at": self.started_at,
            "finished_at": self.finished_at,
        }


class RoverBridge:
    """Drives a backend with the guarded Laya policy."""

    def __init__(
        self,
        config: ProjectConfig,
        backend: RoverBackend,
        controller: Controller,
        guardian_factory: Callable[[RoverState, SensorGridFacade], SafetyGuardian] | None = None,
        fallback: FallbackController | None = None,
        clock: Callable[[], float] = time.perf_counter,
        log_path: Path | None = None,
    ) -> None:
        self.config = config
        self.backend = backend
        self.controller = controller
        self.fallback = fallback or FallbackController(
            FallbackPolicy(**config.controllers.fallback.model_dump())
        )
        self.guardian_factory = guardian_factory
        self.clock = clock
        self.log_path = log_path
        self._log_handle = None

    # ------------------------------------------------------------------ #

    def run_mission(
        self,
        max_steps: int | None = None,
        dry_run: bool = True,
        wall_clock_budget_s: float = 600.0,
    ) -> MissionSummary:
        """Blocking entry point (CLI / scripts)."""
        return asyncio.run(self.run_mission_async(max_steps, dry_run, wall_clock_budget_s))

    async def run_mission_async(
        self,
        max_steps: int | None = None,
        dry_run: bool = True,
        wall_clock_budget_s: float = 600.0,
    ) -> MissionSummary:
        """Mission loop. Awaitable so the live Laya adapter's client and its
        concurrency limits are reused exactly as in the experiment runner."""
        config = self.config
        max_steps = max_steps or config.rover.max_steps
        started = datetime.now(timezone.utc)
        t0 = self.clock()

        self.backend.connect()
        if self.log_path is not None:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            self._log_handle = self.log_path.open("w", encoding="utf-8")

        summary = MissionSummary(
            backend=self.backend.name,
            controller=getattr(self.controller, "name", "controller"),
            steps=0,
            started_at=started.isoformat(),
        )
        previous_action: Action | None = None
        energy_used = 0.0
        recent_stops = 0
        recent_positions: deque[tuple[int, int]] = deque(maxlen=64)

        try:
            while summary.steps < max_steps:
                if self.clock() - t0 > wall_clock_budget_s:
                    summary.stopped_reason = "wall_clock_budget"
                    break
                if self.backend.name == "sim" and getattr(self.backend, "status", None) is not None:
                    status = self.backend.status  # type: ignore[attr-defined]
                    if status is not Status.ACTIVE:
                        summary.stopped_reason = status.value
                        summary.reached_goal = status is Status.GOAL
                        break

                frame = self.backend.read_sensors()
                if frame.battery < config.rover.battery_reserve:
                    summary.stopped_reason = "battery_reserve"
                    self.backend.emergency_stop()
                    break

                facade = SensorGridFacade.from_frame(frame)
                state = RoverState(
                    x=frame.x,
                    y=frame.y,
                    heading=frame.heading,
                    battery=frame.battery,
                    energy_used=energy_used,
                    steps=summary.steps,
                    status=Status.ACTIVE,
                    consecutive_stops=recent_stops,
                )
                guardian = self._guardian(state, facade)
                safe = guardian.safe_actions(state)
                observation = _observation_from_frame(
                    frame, safe.as_values(), previous_action, energy_used, summary.steps, recent_stops
                )
                context = ControllerContext(
                    current_position=(state.x, state.y),
                    recent_positions=tuple(recent_positions),
                    previous_action=previous_action,
                    sensor_sentinel=frame.sensor_range + 1,
                )
                fallback_action = self.fallback.choose(observation, safe, context)
                proposal = await self.controller.propose(observation, state, safe, summary.steps, context)
                if proposal.valid and proposal.action is not None:
                    approval = guardian.approve(
                        proposed=proposal.action,
                        proposed_source=proposal.source,
                        safe=safe,
                        state=state,
                        fallback_action=fallback_action,
                    )
                    source = ActionSource.GUARDIAN_OVERRIDE if approval.overridden else proposal.source
                    routing_reason = approval.reason_code if approval.overridden else None
                else:
                    reason = proposal.failure_reason or GuardianReason.MALFORMED_RESPONSE
                    approval = guardian.approve(
                        proposed=None,
                        proposed_source=proposal.source,
                        safe=safe,
                        state=state,
                        failure_reason=reason,
                        fallback_action=fallback_action,
                    )
                    source = ActionSource.FALLBACK
                    routing_reason = reason

                summary.laya_calls += 1 if getattr(self.controller, "name", "") == "laya_agent" else 0
                if not proposal.valid:
                    summary.laya_invalid += 1
                if source is ActionSource.FALLBACK:
                    summary.fallback_count += 1
                elif source is ActionSource.GUARDIAN_OVERRIDE:
                    summary.guardian_overrides += 1

                if dry_run:
                    report = None
                    moved = False
                else:
                    report = self.backend.execute(approval.final_action)
                    energy_used = report.energy_used
                    moved = report.moved
                    if report.reached_goal:
                        summary.reached_goal = True

                recent_positions.append((state.x, state.y))
                summary.steps += 1
                summary.actions[approval.final_action.value] = (
                    summary.actions.get(approval.final_action.value, 0) + 1
                )
                if approval.final_action is Action.STOP_AND_REPLAN:
                    recent_stops += 1
                else:
                    recent_stops = 0
                previous_action = approval.final_action

                self._log(
                    {
                        "backend": self.backend.name,
                        "dry_run": dry_run,
                        "step": summary.steps - 1,
                        "state_before": state.model_dump(mode="json"),
                        "observation": observation.model_dump(mode="json"),
                        "raw_laya_action": proposal.action.value if proposal.action else None,
                        "raw_laya_confidence": proposal.confidence,
                        "action_source": source.value,
                        "routing_reason_code": routing_reason.value if routing_reason else None,
                        "guardian_reason_code": approval.reason_code.value,
                        "final_action": approval.final_action.value,
                        "counterfactual_collision": approval.counterfactual_collision,
                        "counterfactual_hazard_entry": approval.counterfactual_hazard_entry,
                        "moved": moved,
                        "latency_ms": round(proposal.latency_ms, 3),
                        "sensor_frame": frame.raw or {"x": frame.x, "y": frame.y},
                    }
                )

                if summary.reached_goal:
                    summary.stopped_reason = "goal"
                    break
        except Exception as exc:  # noqa: BLE001 - hardware must fail safe
            summary.stopped_reason = f"error:{type(exc).__name__}"
            try:
                self.backend.emergency_stop()
            except Exception:  # noqa: BLE001
                pass
        finally:
            close = getattr(self.controller, "aclose", None)
            if close is not None:
                try:
                    await close()
                except Exception:  # noqa: BLE001
                    pass
            if not dry_run:
                try:
                    self.backend.emergency_stop()  # always leave the rover stopped
                except Exception:  # noqa: BLE001
                    pass
            self.backend.close()
            if self._log_handle is not None:
                self._log_handle.close()
                self._log_handle = None

        summary.wall_clock_s = self.clock() - t0
        summary.finished_at = datetime.now(timezone.utc).isoformat()
        if self.log_path is not None:
            summary_path = self.log_path.with_suffix(".summary.json")
            summary_path.write_text(json.dumps(summary.to_dict(), indent=2, default=str), encoding="utf-8")
        return summary

    # ------------------------------------------------------------------ #

    def _guardian(self, state: RoverState, facade: SensorGridFacade) -> SafetyGuardian:
        if self.guardian_factory is not None:
            return self.guardian_factory(state, facade)
        return SafetyGuardian(facade, self.config.world, self.config.rover)  # type: ignore[arg-type]

    def _log(self, row: dict[str, Any]) -> None:
        if self._log_handle is None:
            return
        self._log_handle.write(json.dumps(row, sort_keys=True, default=str) + "\n")


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #


def _observation_from_frame(
    frame: SensorFrame,
    safe_actions: list[str],
    previous_action: Action | None,
    energy_used: float,
    steps: int,
    recent_stops: int,
) -> Observation:
    """Identical observation contract to the simulator's - by construction."""
    return Observation(
        goal=GoalObservation(
            relative_x=frame.goal_relative[0],
            relative_y=frame.goal_relative[1],
            distance=frame.goal_distance,
        ),
        obstacles=ObstacleObservation(
            front=int(frame.obstacles.get("front", frame.sensor_range + 1)),
            front_left=int(frame.obstacles.get("front_left", frame.sensor_range + 1)),
            front_right=int(frame.obstacles.get("front_right", frame.sensor_range + 1)),
            left=int(frame.obstacles.get("left", frame.sensor_range + 1)),
            right=int(frame.obstacles.get("right", frame.sensor_range + 1)),
        ),
        hazards=HazardObservation(
            front=bool(frame.hazards.get("front", False)),
            front_left=bool(frame.hazards.get("front_left", False)),
            front_right=bool(frame.hazards.get("front_right", False)),
        ),
        heading=frame.heading,
        battery_fraction=frame.battery,
        energy_used=energy_used,
        steps=steps,
        previous_action=previous_action.value if previous_action else None,
        recent_stops=recent_stops,
        safe_actions=safe_actions,
    )


