"""Fallback controller (FR-7).

Deterministic, conservative, goal-distance-minimising - the safety net used
when Laya is unavailable, uncertain, malformed, or rejected. It is *not* the
comparison baseline (Section 5.4); see ``baseline.py`` for the deliberately
naive greedy heuristic it is contrasted with.

Candidate actions are ranked by one conservative score:

    score(action) = distance_to_goal(next cell)
                  + hazard_aversion_margin   if the entry is a HAZARD_SLOPE
                  + pocket_penalty           if the move drives into a pocket
                  + memory_penalty * recency if the cell was recently occupied

FR-7's rules are preserved as the structure of the decision:

1. forward is safe and progresses toward the goal with **no penalty attached**
   -> ``MOVE_FORWARD`` (the literal FR-7 rule 1), or the move reaches the goal
2. otherwise the safe action with the lowest conservative score - i.e. the
   least-distant safe turn, unless a turn would drive into a pocket or a hazard
   that a slightly longer alternative avoids
3. otherwise ``REVERSE``
4. otherwise ``STOP_AND_REPLAN``

Every input comes from the fixed local observation window or the rover's own
recent-pose history: no map knowledge, no global planning, fully deterministic.

The pocket rule is what makes this controller genuinely *conservative*: a wall
within ``dead_end_clearance_threshold`` cells ahead means the rover is looking
at a shallow enclosed pocket, and a pocket the rover never enters is a pocket it
never has to escape - so it will take up to ``pocket_penalty`` cells of detour
to stay out. The memory term is a short-term visited-cell memory that stops an
A-B-A bounce and makes the rover prefer the least recently visited neighbour
when it has to retrace ground. The baseline has neither.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..safety import SafeActionSet, SafetyGuardian
from ..types import (
    Action,
    ActionSource,
    GuardianReason,
    Observation,
    RoverState,
    heading_delta,
    turn_left,
    turn_right,
)
from .base import ControllerContext, Proposal


@dataclass(frozen=True)
class FallbackPolicy:
    """Tunables for the conservative controller (all deterministic)."""

    dead_end_avoidance: bool = True
    dead_end_clearance_threshold: int = 3
    hazard_aversion_margin: float = 0.75
    pocket_penalty: float = 3.0
    memory_cells: int = 12
    memory_penalty: float = 1.0
    memory_floor_fraction: float = 0.5
    backtrack_penalty: float = 2.0
    escape_window: int = 6
    escape_max_distinct: int = 4
    escape_multiplier: float = 3.0


#: Canonical scan order for deterministic tie-breaking (documented).
CONSERVATIVE_ORDER: tuple[Action, ...] = (
    Action.MOVE_FORWARD,
    Action.TURN_LEFT_AND_MOVE,
    Action.TURN_RIGHT_AND_MOVE,
    Action.REVERSE,
    Action.STOP_AND_REPLAN,
)


class FallbackController:
    name = "fallback"

    def __init__(self, policy: FallbackPolicy | None = None) -> None:
        self.policy = policy or FallbackPolicy()

    # ------------------------------------------------------------------ #

    def choose(
        self,
        observation: Observation,
        safe: SafeActionSet,
        context: ControllerContext | None = None,
    ) -> Action:
        """Pure, synchronous decision - also the calibration reference (FR-10.5)."""
        policy = self.policy
        goal_rel = (observation.goal.relative_x, observation.goal.relative_y)
        current_distance = observation.goal.distance
        origin = context.current_position if context else None

        def target_offset(action: Action) -> tuple[int, int] | None:
            if action is Action.STOP_AND_REPLAN:
                return (0, 0)
            if action is Action.REVERSE:
                dx, dy = heading_delta(observation.heading)
                return (-dx, -dy)
            if action is Action.TURN_LEFT_AND_MOVE:
                return heading_delta(turn_left(observation.heading))
            if action is Action.TURN_RIGHT_AND_MOVE:
                return heading_delta(turn_right(observation.heading))
            return heading_delta(observation.heading)

        def distance(action: Action) -> float:
            return SafetyGuardian.next_cell_distance(goal_rel, observation.heading, action)

        # Escape urgency: if the rover's own recent poses describe a cycle
        # (few distinct cells), visited cells are penalised much harder for as
        # long as the cycle lasts. This is "escape mode" implemented as soft
        # pressure rather than a mode flag: the controller keeps minimising
        # goal distance, it just stops re-entering ground it has already worn
        # out, which is what eventually breaks a concave-region loop.
        escape = 1.0
        if context and len(context.recent_positions) >= policy.escape_window:
            recent = context.recent_positions[-policy.escape_window :]
            if len(set(recent)) <= policy.escape_max_distinct:
                escape = policy.escape_multiplier

        def memory_penalty(action: Action) -> float:
            """Penalty for moving onto a recently occupied cell.

            Two components, both purely local history:

            * ``memory_penalty`` scaled by recency - discourages retracing and
              breaks A-B-A ties in favour of fresh ground;
            * ``backtrack_penalty`` when the target is the cell the rover has
              just left, i.e. an immediate undo of the previous move. Without
              it the rover ping-pongs: reversing one cell shaves a little
              distance off, the memory term is not enough to outbid it, and the
              same two cells are re-visited until the stuck detector fires.
              A real rover never undoes its last move by default.
            """
            if not context or origin is None or not context.recent_positions:
                return 0.0
            offset = target_offset(action)
            if offset is None:
                return 0.0
            target = (origin[0] + offset[0], origin[1] + offset[1])
            window = list(context.recent_positions)[-policy.memory_cells :]

            penalty = 0.0
            if policy.memory_penalty > 0.0:
                count = len(window)
                for index in range(count - 1, -1, -1):  # newest first
                    if window[index] == target:
                        recency = (index + 1) / count
                        floor = policy.memory_floor_fraction
                        penalty = policy.memory_penalty * (floor + (1.0 - floor) * recency)
                        break
            # "the cell the rover just left": the most recent entry that is not
            # the cell it is standing on. Written this way so the rule holds
            # whichever convention the caller uses for the history (current
            # position included or not).
            for position in reversed(context.recent_positions):
                if origin is not None and position == origin:
                    continue
                if position == target:
                    penalty = max(penalty, policy.backtrack_penalty)
                break
            return penalty * escape

        def is_hazard(action: Action) -> bool:
            return bool(safe.enters_hazard.get(action, False))

        def clearance(action: Action) -> int:
            """Forward clearance in cells at the cell this action enters.

            Only MOVE_FORWARD can be judged directly from the current window;
            for turns the rover simply re-evaluates on the next timestep, which
            is why a turn is never pocket-penalised here.
            """
            if action is Action.MOVE_FORWARD:
                return observation.obstacles.front
            return context.sensor_sentinel if context else observation.obstacles.front

        def is_pocket(action: Action) -> bool:
            if not policy.dead_end_avoidance:
                return False
            return action is Action.MOVE_FORWARD and clearance(action) <= policy.dead_end_clearance_threshold

        def penalty(action: Action) -> float:
            value = 0.0
            if is_hazard(action):
                value += policy.hazard_aversion_margin
            if is_pocket(action):
                value += policy.pocket_penalty
            return value + memory_penalty(action)

        def score(action: Action) -> float:
            return distance(action) + penalty(action)

        candidates = [a for a in safe.actions if a is not Action.STOP_AND_REPLAN]
        if not candidates:
            return Action.STOP_AND_REPLAN

        # 1. FR-7 rule 1: unpenalised, distance-reducing forward, or goal entry.
        if Action.MOVE_FORWARD in candidates:
            forward_distance = distance(Action.MOVE_FORWARD)
            reaches_goal = forward_distance <= 1.0 + 1e-9
            if reaches_goal:
                return Action.MOVE_FORWARD
            if forward_distance < current_distance and penalty(Action.MOVE_FORWARD) == 0.0:
                return Action.MOVE_FORWARD

        # 2. best conservative action (covers FR-7 rule 2, the least-distant turn).
        return min(candidates, key=lambda a: (score(a), CONSERVATIVE_ORDER.index(a)))

    # ------------------------------------------------------------------ #

    async def propose(
        self,
        observation: Observation,
        state: RoverState,
        safe: SafeActionSet,
        step: int,
        context: ControllerContext | None = None,
    ) -> Proposal:
        action = self.choose(observation, safe, context)
        return Proposal(
            action=action,
            source=ActionSource.FALLBACK,
            confidence=1.0,
            valid=True,
            reason_code=GuardianReason.APPROVED.value,
            diagnostics={"controller": self.name},
        )
