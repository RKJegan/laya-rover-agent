"""Per-episode control loop (Section 5.2 decision flow).

    world -> perception -> safety pre-filter -> controller decision
          -> output validation -> guardian approval/override -> simulator
          -> logging -> next timestep

Every timestep produces exactly one JSONL record with the full audit trail
(Section FR-11). Nothing here mutates module-level state, and every stochastic
source is derived from ``(episode seed, stream tag)``, so an episode is
bit-reproducible: same seed + same config -> same action sequence.
"""

from __future__ import annotations

import random
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from .config import ProjectConfig
from .controllers.base import Controller, ControllerContext, Proposal
from .controllers.baseline import BaselineController
from .controllers.fallback import FallbackController, FallbackPolicy
from .dynamics import EpisodeOutcome, RoverSimulator
from .perception import Perception
from .safety import SafetyGuardian
from .types import (
    Action,
    ActionSource,
    GuardianReason,
    RoverState,
    Status,
)
from .world import MapSpec, WorldGenerator, stream_seed

#: Reason codes that mean "the guardian stopped something unsafe", as opposed
#: to "the decision source failed and we fell back".
HARD_RULE_REASONS: frozenset[GuardianReason] = frozenset(
    {
        GuardianReason.BLOCKED_OBSTACLE,
        GuardianReason.BLOCKED_OUT_OF_BOUNDS,
        GuardianReason.BLOCKED_HAZARD_BATTERY,
        GuardianReason.BLOCKED_HAZARD_PREFERENCE,
        GuardianReason.BLOCKED_BATTERY_RESERVE,
        GuardianReason.NOT_IN_SAFE_ACTIONS,
        GuardianReason.STOP_CAP_REACHED,
    }
)

#: Boxed-in states are recorded separately: no proposal was rejected, the
#: guardian simply had nothing legal to offer.
BOXED_IN_REASONS: frozenset[GuardianReason] = frozenset({GuardianReason.NO_LEGAL_ACTION})


@dataclass(frozen=True)
class EpisodeSpec:
    """One cell of the experiment matrix."""

    map_spec: MapSpec
    controller: str                     # "laya_agent" | "baseline"
    confidence_threshold: float
    guardian_mode: str = "enforced"
    index: int = 0

    @property
    def episode_id(self) -> str:
        return (
            f"{self.map_spec.difficulty}-s{self.map_spec.seed:03d}-{self.map_spec.preset}"
            f"-{self.controller}-t{int(round(self.confidence_threshold * 100))}"
        )


@dataclass
class EpisodeResult:
    spec: EpisodeSpec
    outcome: EpisodeOutcome
    timesteps: list[dict[str, object]] = field(default_factory=list)
    summary: dict[str, object] = field(default_factory=dict)
    map_key: str = ""
    trajectory: list[tuple[int, int]] = field(default_factory=list)
    map_cells: list[list[str]] | None = None
    map_start: tuple[int, int] | None = None
    map_goal: tuple[int, int] | None = None


class EpisodeRunner:
    """Runs one episode. Reusable; holds no per-episode mutable state."""

    def __init__(
        self,
        config: ProjectConfig,
        controller_factory: Callable[[EpisodeSpec], Controller],
        clock: Callable[[], float] = time.perf_counter,
        wall_clock_budget_s: float | None = None,
    ) -> None:
        self.config = config
        self.controller_factory = controller_factory
        self.clock = clock
        self.wall_clock_budget_s = (
            wall_clock_budget_s
            if wall_clock_budget_s is not None
            else config.rover.episode_wall_clock_timeout_s
        )

    # ------------------------------------------------------------------ #

    async def run(self, spec: EpisodeSpec, capture_map: bool = False) -> EpisodeResult:
        config = self.config
        world = WorldGenerator(config.world).generate(spec.map_spec)

        stall_rng = random.Random(
            stream_seed(spec.map_spec.seed * 31 + spec.index, "hazard_stalls")
        )
        sim = RoverSimulator(
            world=world,
            config=config.rover,
            difficulty=spec.map_spec.difficulty,
            stall_rng=stall_rng,
            clock=self.clock,
        )
        perception = Perception(config.world, config.rover)
        guardian = SafetyGuardian(world, config.world, config.rover)
        fallback = FallbackController(FallbackPolicy(**config.controllers.fallback.model_dump()))
        controller = self.controller_factory(spec)

        state = sim.initial_state()
        sim.note_position(state)
        previous_action: Action | None = None
        trajectory: list[tuple[int, int]] = [(state.x, state.y)]

        timesteps: list[dict[str, object]] = []
        latencies: list[float] = []
        confidence_records: list[dict[str, object]] = []
        counters = {
            "collisions": 0,
            "guardian_overrides": 0,
            "unsafe_actions_prevented": 0,
            "collisions_prevented": 0,
            "hazard_entries_prevented": 0,
            "fallback_count": 0,
            "laya_calls": 0,
            "laya_invalid": 0,
            "stalls": 0,
            "hazard_entries": 0,
            "stop_actions": 0,
        }
        path_length = 0.0
        recent_positions: deque[tuple[int, int]] = deque([(state.x, state.y)], maxlen=64)
        started = self.clock()
        aborted_reason: str | None = None
        terminated_by_wall_clock = False

        # ---- wall-clock guard (distinct from max_steps) ------------------- #
        while state.status is Status.ACTIVE and state.steps < config.rover.max_steps:
            if self.clock() - started > self.wall_clock_budget_s:
                aborted_reason = GuardianReason.EPISODE_WALL_CLOCK_TIMEOUT.value
                terminated_by_wall_clock = True
                break

            step = state.steps
            state_before = state
            safe = guardian.safe_actions(state)
            if not safe.actions:
                # Boxed in: every action is blocked (obstacles / out of bounds /
                # battery floor / stop cap). Nothing the decision layer says can
                # move the rover, so end the episode now as `stuck` instead of
                # burning the step budget on idle stops.
                state = state.model_copy(update={"status": Status.STUCK})
                aborted_reason = GuardianReason.NO_LEGAL_ACTION.value
                break
            observation = perception.observe(world, state, safe.actions, previous_action)
            context = ControllerContext(
                current_position=(state.x, state.y),
                recent_positions=tuple(recent_positions),
                previous_action=previous_action,
                sensor_sentinel=perception.sensor_range + 1,
            )
            fallback_action = fallback.choose(observation, safe, context)

            proposal = await controller.propose(observation, state, safe, step, context)
            approval, routed_source, routing_reason = self._route_and_approve(
                spec=spec,
                proposal=proposal,
                safe=safe,
                state=state,
                guardian=guardian,
                fallback_action=fallback_action,
            )

            outcome = sim.step(state, approval.final_action)
            state_after = outcome.state

            # ---- counters ------------------------------------------------- #
            if outcome.collision:
                counters["collisions"] += 1
            if outcome.stalled:
                counters["stalls"] += 1
            if outcome.entered_hazard and outcome.moved:
                counters["hazard_entries"] += 1
            if approval.final_action is Action.STOP_AND_REPLAN:
                counters["stop_actions"] += 1
            if outcome.moved:
                path_length += 1.0

            if routed_source is ActionSource.FALLBACK:
                counters["fallback_count"] += 1
            elif routed_source is ActionSource.GUARDIAN_OVERRIDE:
                counters["guardian_overrides"] += 1
            if approval.reason_code in HARD_RULE_REASONS:
                counters["unsafe_actions_prevented"] += 1
                if approval.counterfactual_collision:
                    counters["collisions_prevented"] += 1
                if approval.counterfactual_hazard_entry:
                    counters["hazard_entries_prevented"] += 1

            if spec.controller == "laya_agent":
                counters["laya_calls"] += 1
                if not proposal.valid:
                    counters["laya_invalid"] += 1
                latencies.append(proposal.latency_ms)

            # ---- calibration record (FR-10.5) ----------------------------- #
            if spec.controller == "laya_agent" and proposal.valid and proposal.action is not None:
                agreed = proposal.action is fallback_action
                confidence_records.append(
                    {
                        "step": step,
                        "confidence_threshold": spec.confidence_threshold,
                        "confidence": proposal.confidence,
                        "laya_action": proposal.action.value,
                        "fallback_action": fallback_action.value,
                        "agreed_with_fallback": agreed,
                        "approved_without_override": not approval.overridden,
                        "guardian_reason_code": approval.reason_code.value,
                    }
                )

            # ---- per-timestep log ---------------------------------------- #
            timesteps.append(
                {
                    "episode_id": spec.episode_id,
                    "map_key": spec.map_spec.key,
                    "seed": spec.map_spec.seed,
                    "difficulty": spec.map_spec.difficulty,
                    "preset": spec.map_spec.preset,
                    "controller": spec.controller,
                    "confidence_threshold": spec.confidence_threshold,
                    "guardian_mode": spec.guardian_mode,
                    "step": step,
                    "state_before": state_before.model_dump(mode="json"),
                    "observation": observation.model_dump(mode="json"),
                    "action_source": routed_source.value,
                    "raw_laya_action": (
                        proposal.action.value if (spec.controller == "laya_agent" and proposal.action) else None
                    ),
                    "raw_laya_confidence": proposal.confidence if spec.controller == "laya_agent" else None,
                    "raw_laya_valid": proposal.valid if spec.controller == "laya_agent" else None,
                    "mock_branch": proposal.diagnostics.get("mock_branch") if spec.controller == "laya_agent" else None,
                    "fallback_action": fallback_action.value,
                    "routing_reason_code": routing_reason.value if routing_reason else None,
                    "final_action": approval.final_action.value,
                    "guardian_reason_code": approval.reason_code.value,
                    "guardian_overridden": approval.overridden,
                    "counterfactual_collision": approval.counterfactual_collision,
                    "counterfactual_hazard_entry": approval.counterfactual_hazard_entry,
                    # Shadow/ablation bookkeeping only (spec 5.3): the row records
                    # which action the decision layer proposed, while
                    # ``guardian_reason_code`` / ``counterfactual_*`` record what the
                    # guardian would have had to stop. The rover is *never* moved into
                    # an unsafe cell in either mode.
                    "ablation_counterfactual_action": (
                        proposal.action.value
                        if (spec.guardian_mode == "log_only_ablation" and proposal.action is not None)
                        else None
                    ),
                    "moved": outcome.moved,
                    "stalled": outcome.stalled,
                    "entered_hazard": outcome.entered_hazard,
                    "collision": outcome.collision,
                    "reached_goal": outcome.reached_goal,
                    "state_after": state_after.model_dump(mode="json"),
                    "latency_ms": round(proposal.latency_ms, 3),
                    "usage": proposal.usage,
                    # Which checkpoint actually answered (matters when the request
                    # leaves ``model`` on ``auto``) and whether it read the whole
                    # state - both are audit facts, not tuning knobs.
                    "provider_model": proposal.provider_model,
                    "decision_truncated": proposal.truncated,
                    "raw_response": proposal.raw_response,
                }
            )

            if outcome.moved:
                trajectory.append((state_after.x, state_after.y))
            recent_positions.append((state_after.x, state_after.y))
            state = state_after
            previous_action = approval.final_action

        # ---- finalise ------------------------------------------------------ #
        terminal_status = state.status
        if terminated_by_wall_clock and terminal_status is Status.ACTIVE:
            state = state.model_copy(update={"status": Status.TIMEOUT})

        success = terminal_status is Status.GOAL
        elapsed = self.clock() - started
        outcome_summary = EpisodeOutcome(
            success=success,
            status=state.status.value,
            terminal=_terminal_label(state.status),
            steps=state.steps,
            energy_used=round(state.energy_used, 4),
            battery_remaining=round(state.battery, 6),
            final_distance_to_goal=round(_goal_distance(world, state), 4),
            path_length=round(path_length, 4),
            collisions=counters["collisions"],
            guardian_overrides=counters["guardian_overrides"],
            unsafe_actions_prevented=counters["unsafe_actions_prevented"],
            collisions_prevented=counters["collisions_prevented"],
            hazard_entries_prevented=counters["hazard_entries_prevented"],
            fallback_count=counters["fallback_count"],
            laya_calls=counters["laya_calls"],
            laya_invalid=counters["laya_invalid"],
            stalls=counters["stalls"],
            hazard_entries=counters["hazard_entries"],
            stop_actions=counters["stop_actions"],
            wall_clock_s=round(elapsed, 4),
            latencies_ms=latencies,
            confidence_records=confidence_records,
            aborted_reason=aborted_reason,
        )
        result = EpisodeResult(
            spec=spec,
            outcome=outcome_summary,
            timesteps=timesteps,
            map_key=spec.map_spec.key,
            trajectory=trajectory,
            map_cells=[list(row) for row in world.rows()] if capture_map else None,
            map_start=world.start if capture_map else None,
            map_goal=world.goal if capture_map else None,
        )
        result.summary = self._summary_row(result)
        return result

    # ------------------------------------------------------------------ #

    def _route_and_approve(
        self,
        spec: EpisodeSpec,
        proposal: Proposal,
        safe,
        state: RoverState,
        guardian: SafetyGuardian,
        fallback_action: Action,
    ):
        """Apply the FR-6 routing rules, then the guardian.

        Returns ``(approval, action_source, routing_reason)``.
        """
        if spec.controller != "laya_agent":
            approval = guardian.approve(
                proposed=proposal.action,
                proposed_source=proposal.source,
                safe=safe,
                state=state,
                failure_reason=proposal.failure_reason,
                fallback_action=fallback_action,
            )
            if approval.overridden:
                return approval, ActionSource.GUARDIAN_OVERRIDE, approval.reason_code
            return approval, proposal.source, None

        # ---- laya_agent routing rules -------------------------------------- #
        if not proposal.valid or proposal.action is None:
            reason = proposal.failure_reason or GuardianReason.MALFORMED_RESPONSE
            approval = guardian.approve(
                proposed=None,
                proposed_source=proposal.source,
                safe=safe,
                state=state,
                failure_reason=reason,
                fallback_action=fallback_action,
            )
            return approval, ActionSource.FALLBACK, reason

        if proposal.latency_ms > self.config.agent.decision_timeout_ms:
            approval = guardian.approve(
                proposed=None,
                proposed_source=proposal.source,
                safe=safe,
                state=state,
                failure_reason=GuardianReason.DECISION_TIMEOUT,
                fallback_action=fallback_action,
            )
            return approval, ActionSource.FALLBACK, GuardianReason.DECISION_TIMEOUT

        confidence = proposal.confidence if proposal.confidence is not None else 0.0
        if confidence < spec.confidence_threshold:
            approval = guardian.approve(
                proposed=None,
                proposed_source=proposal.source,
                safe=safe,
                state=state,
                failure_reason=GuardianReason.LOW_CONFIDENCE,
                fallback_action=fallback_action,
            )
            return approval, ActionSource.FALLBACK, GuardianReason.LOW_CONFIDENCE

        approval = guardian.approve(
            proposed=proposal.action,
            proposed_source=proposal.source,
            safe=safe,
            state=state,
            fallback_action=fallback_action,
        )
        if approval.overridden:
            return approval, ActionSource.GUARDIAN_OVERRIDE, approval.reason_code
        return approval, ActionSource.LAYA, None

    # ------------------------------------------------------------------ #

    def _summary_row(self, result: EpisodeResult) -> dict[str, object]:
        outcome = result.outcome
        spec = result.spec
        latencies = sorted(outcome.latencies_ms)
        record: dict[str, object] = {
            "episode_id": spec.episode_id,
            "map_key": result.map_key,
            "seed": spec.map_spec.seed,
            "difficulty": spec.map_spec.difficulty,
            "preset": spec.map_spec.preset,
            "controller": spec.controller,
            "confidence_threshold": spec.confidence_threshold,
            "guardian_mode": spec.guardian_mode,
            "success": int(outcome.success),
            "status": outcome.status,
            "terminal": outcome.terminal,
            "steps": outcome.steps,
            "energy_used": outcome.energy_used,
            "battery_remaining": outcome.battery_remaining,
            "path_length": outcome.path_length,
            "final_distance_to_goal": outcome.final_distance_to_goal,
            "collisions": outcome.collisions,
            "guardian_overrides": outcome.guardian_overrides,
            "unsafe_actions_prevented": outcome.unsafe_actions_prevented,
            "collisions_prevented": outcome.collisions_prevented,
            "hazard_entries_prevented": outcome.hazard_entries_prevented,
            "fallback_count": outcome.fallback_count,
            "laya_calls": outcome.laya_calls,
            "laya_invalid": outcome.laya_invalid,
            "fallback_rate": round(outcome.fallback_count / outcome.steps, 4) if outcome.steps else 0.0,
            "stalls": outcome.stalls,
            "hazard_entries": outcome.hazard_entries,
            "stop_actions": outcome.stop_actions,
            "wall_clock_s": outcome.wall_clock_s,
            "latency_mean_ms": round(sum(latencies) / len(latencies), 3) if latencies else 0.0,
            "latency_p50_ms": _percentile(latencies, 50),
            "latency_p95_ms": _percentile(latencies, 95),
            "aborted_reason": outcome.aborted_reason,
        }
        return record


def _terminal_label(status: Status) -> str:
    mapping = {
        Status.GOAL: "goal",
        Status.COLLISION: "collision",
        Status.BATTERY_DEPLETED: "battery_depleted",
        Status.STUCK: "stuck",
        Status.TIMEOUT: "timeout",
        Status.ACTIVE: "active",
    }
    return mapping[status]


def _goal_distance(world, state: RoverState) -> float:
    from .types import euclidean

    return euclidean(state.x, state.y, world.goal[0], world.goal[1])


def _percentile(sorted_values: list[float], pct: float) -> float:
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return round(sorted_values[0], 3)
    rank = (pct / 100.0) * (len(sorted_values) - 1)
    low = int(rank)
    high = min(low + 1, len(sorted_values) - 1)
    frac = rank - low
    value = sorted_values[low] + (sorted_values[high] - sorted_values[low]) * frac
    return round(value, 3)


def write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    import json

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, default=str) + "\n")
