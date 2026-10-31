"""Rover dynamics + episode simulator (FR-2, FR-3).

Implements discretised rover motion with explicit energy accounting, hazard
stalls, collisions, the stop cap, and the two "give up" conditions:

* ``stuck`` - the loop/oscillation detector (net progress toward the goal over
  the last ``stuck_window`` positions is ~0 despite movement actions), or the
  rover being boxed in.
* ``timeout`` - ``max_steps`` exhausted, or the per-episode wall-clock guard
  firing (Section 7 quality bar).

``RoverState`` is immutable: every step returns a new state and nothing outside
it is mutated, which makes episodes trivially replayable and audit-friendly.
Wall-clock is measured through an injectable ``clock`` so tests are
deterministic and no episode silently depends on real time.
"""

from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass, field
from typing import Callable, Literal

from .config import RoverConfig
from .types import (
    Action,
    GridMap,
    RoverState,
    Status,
    euclidean,
    heading_delta,
    turn_left,
    turn_right,
)


@dataclass(frozen=True)
class StepOutcome:
    """Result of applying one action."""

    state: RoverState
    moved: bool
    stalled: bool = False
    collision: bool = False
    entered_hazard: bool = False
    reached_goal: bool = False
    blocked: bool = False          # action was illegal (obstacle / out of bounds)
    reason: str | None = None


def next_heading(heading: str, action: Action) -> str:
    if action is Action.TURN_LEFT_AND_MOVE:
        return turn_left(heading)
    if action is Action.TURN_RIGHT_AND_MOVE:
        return turn_right(heading)
    return heading


def action_target(x: int, y: int, heading: str, action: Action) -> tuple[int, int]:
    """Cell the action would enter (== current cell for STOP_AND_REPLAN)."""
    if action is Action.STOP_AND_REPLAN:
        return x, y
    new_heading = next_heading(heading, action)
    dx, dy = heading_delta(new_heading)
    if action is Action.REVERSE:
        return x - dx, y - dy
    return x + dx, y + dy


class RoverSimulator:
    """Applies one action to a rover state. No global mutable state."""

    def __init__(
        self,
        world: GridMap,
        config: RoverConfig,
        difficulty: str,
        stall_rng: random.Random,
        clock: Callable[[], float] | None = None,
    ) -> None:
        self.world = world
        self.config = config
        self.difficulty = difficulty
        self.stall_rng = stall_rng
        self.clock = clock
        self._history: list[tuple[int, int]] = []
        self._first_seen: dict[tuple[int, int], int] = {}

    # -- state helpers ------------------------------------------------------ #

    def initial_state(self) -> RoverState:
        return RoverState(
            x=self.world.start[0],
            y=self.world.start[1],
            heading="N",
            battery=self.config.initial_battery,
            energy_used=0.0,
            steps=0,
            status=Status.ACTIVE,
        )

    def note_position(self, state: RoverState) -> None:
        position = (state.x, state.y)
        self._history.append(position)
        self._first_seen.setdefault(position, len(self._history) - 1)

    # -- energy ------------------------------------------------------------- #

    def projected_cost(self, state: RoverState, action: Action) -> float:
        """Energy the action would consume *if* it enters a hazard cell."""
        target = action_target(state.x, state.y, state.heading, action)
        hazard = action is not Action.STOP_AND_REPLAN and self.world.is_hazard(*target)
        return self.config.cost_of(action, hazard=hazard)

    def _battery_after(self, state: RoverState, cost: float) -> float:
        return round(max(0.0, state.battery - cost / self.config.energy_capacity), 10)

    # -- termination rules -------------------------------------------------- #

    def _stall_probability(self) -> float:
        return self.config.hazard_stall_probability.get(self.difficulty, 0.15)

    def termination_status(self, state: RoverState) -> Status:
        """Post-move status classification (pure: returns, never mutates)."""
        if state.status is Status.GOAL:
            return Status.GOAL
        if state.battery <= 0.0:
            return Status.BATTERY_DEPLETED
        if state.steps >= self.config.max_steps:
            return Status.TIMEOUT
        if self._loop_detected():
            return Status.STUCK
        return Status.ACTIVE

    def _loop_detected(self) -> bool:
        """Stuck / oscillation detection: "going nowhere while discovering nothing".

        Over the last ``stuck_window`` occupied cells the rover is declared
        stuck when **both** hold:

        * net progress toward the goal is below ``stuck_displacement_epsilon``, and
        * at most ``stuck_min_new_cells`` of those cells were first discovered
          inside that same window.

        Compared with the spec's literal "net displacement over the last N
        positions is ~0", this avoids two false positives that would otherwise
        end healthy episodes: a wide but legitimate detour around a large
        obstacle cluster (no net progress over 8 steps, but it is discovering
        new cells every step), and a single probe into a one-cell alcove
        followed by backing out. What it still catches - which is exactly what
        the spec targets - is an A-B-A-B bounce, a left/right wiggle at a wall,
        and a rover circling inside a concave pocket: no progress, no new
        ground.
        """
        count = len(self._history)
        if count < self.config.stuck_window:
            return False
        window_start = count - self.config.stuck_window
        window = self._history[window_start:]
        if len(set(window)) <= 2:
            # A rover that has spent the whole window inside one or two cells is
            # oscillating on the spot whatever the goal geometry says (a bounce
            # *along* the goal axis can otherwise hide behind the progress test).
            return True
        new_cells = sum(1 for position in window if self._first_seen[position] >= window_start)
        if new_cells > self.config.stuck_min_new_cells:
            return False
        gx, gy = self.world.goal
        oldest, newest = window[0], window[-1]
        progress = euclidean(oldest[0], oldest[1], gx, gy) - euclidean(newest[0], newest[1], gx, gy)
        return progress < self.config.stuck_displacement_epsilon

    # -- step --------------------------------------------------------------- #

    def step(self, state: RoverState, action: Action) -> StepOutcome:
        if state.status is not Status.ACTIVE:
            return StepOutcome(state=state, moved=False, reason="episode_finished")

        cost = self.projected_cost(state, action)
        battery = self._battery_after(state, cost)
        energy_used = round(state.energy_used + cost, 10)
        steps = state.steps + 1

        # --- idle / replan -------------------------------------------------- #
        if action is Action.STOP_AND_REPLAN:
            new_state = state.model_copy(
                update={
                    "battery": battery,
                    "energy_used": energy_used,
                    "steps": steps,
                    "consecutive_stops": state.consecutive_stops + 1,
                }
            )
            # An idle step still counts as "occupying" this cell: without this a
            # rover that stops forever never trips the loop detector and burns the
            # whole step budget doing nothing.
            self.note_position(new_state)
            new_state = new_state.model_copy(update={"status": self.termination_status(new_state)})
            reason = "battery_depleted" if new_state.status is Status.BATTERY_DEPLETED else "stopped"
            return StepOutcome(state=new_state, moved=False, reason=reason)

        target = action_target(state.x, state.y, state.heading, action)
        new_heading = next_heading(state.heading, action)
        tx, ty = target

        # --- illegal motion: out of bounds or obstacle ---------------------- #
        if not self.world.in_bounds(tx, ty) or self.world.is_obstacle(tx, ty):
            new_state = state.model_copy(
                update={
                    "battery": battery,
                    "energy_used": energy_used,
                    "steps": steps,
                    "consecutive_stops": 0,
                    "status": Status.COLLISION,
                }
            )
            return StepOutcome(
                state=new_state,
                moved=False,
                collision=True,
                blocked=True,
                reason="blocked_obstacle" if self.world.in_bounds(tx, ty) else "blocked_out_of_bounds",
            )

        # --- hazard stall --------------------------------------------------- #
        entering_hazard = self.world.is_hazard(tx, ty)
        stalled = entering_hazard and self.stall_rng.random() < self._stall_probability()

        if stalled:
            # Action is absorbed: energy spent, steps consumed, no advance, no
            # heading change (the manoeuvre aborted).
            new_state = state.model_copy(
                update={
                    "battery": battery,
                    "energy_used": energy_used,
                    "steps": steps,
                    "consecutive_stops": 0,
                }
            )
            self.note_position(new_state)
            new_state = new_state.model_copy(update={"status": self.termination_status(new_state)})
            return StepOutcome(
                state=new_state,
                moved=False,
                stalled=True,
                entered_hazard=True,
                reason="hazard_stall",
            )

        # --- normal move ---------------------------------------------------- #
        reached_goal = (tx, ty) == self.world.goal
        new_state = state.model_copy(
            update={
                "x": tx,
                "y": ty,
                "heading": new_heading,
                "battery": battery,
                "energy_used": energy_used,
                "steps": steps,
                "consecutive_stops": 0,
                "status": Status.GOAL if reached_goal else Status.ACTIVE,
            }
        )
        self.note_position(new_state)
        if not reached_goal:
            new_state = new_state.model_copy(update={"status": self.termination_status(new_state)})
        return StepOutcome(
            state=new_state,
            moved=True,
            entered_hazard=entering_hazard,
            reached_goal=reached_goal,
        )

    # -- counterfactual probe (guardian ablation, Section 5.3) -------------- #

    def would_enter(self, state: RoverState, action: Action) -> tuple[bool, bool]:
        """What *would* happen if this action executed, without executing it.

        Returns ``(is_collision, is_hazard_entry)``. Used only for ablation
        logging / safety-case accounting; the rover is never actually moved into
        a blocked cell.
        """
        if action is Action.STOP_AND_REPLAN:
            return False, False
        tx, ty = action_target(state.x, state.y, state.heading, action)
        if not self.world.in_bounds(tx, ty) or self.world.is_obstacle(tx, ty):
            return True, False
        return False, self.world.is_hazard(tx, ty)


@dataclass
class EpisodeOutcome:
    """Per-episode summary row."""

    success: bool
    status: str
    steps: int
    energy_used: float
    battery_remaining: float
    final_distance_to_goal: float
    path_length: float
    collisions: int
    guardian_overrides: int
    unsafe_actions_prevented: int
    collisions_prevented: int
    hazard_entries_prevented: int
    fallback_count: int
    laya_calls: int
    laya_invalid: int
    stalls: int
    hazard_entries: int
    stop_actions: int
    wall_clock_s: float
    latencies_ms: list[float] = field(default_factory=list)
    confidence_records: list[dict[str, object]] = field(default_factory=list)
    aborted_reason: str | None = None
    terminal: Literal["goal", "collision", "battery_depleted", "stuck", "timeout", "active"] = "active"
