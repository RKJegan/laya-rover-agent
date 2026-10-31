"""Safety guardian (FR-5, Section 5.3).

Two responsibilities:

1. **Pre-filter** - compute the set of actions the rover may legally take from
   this state (the ``safe_actions`` field of the observation). Because every
   controller receives the *same* pre-filtered set, no controller can be
   "accidentally safe" or "accidentally unsafe" by construction.
2. **Approval** - after a controller proposes an action, deterministically
   accept or reject it, log a machine-readable reason code, and - when
   rejecting - substitute a conservative action.

The guardian also implements the **counterfactual/ablation probe**: when it
blocks an action it reports what *would* have happened (collision / hazard
entry) had the action executed, without ever executing it. That yields the
"unsafe entries prevented by the guardian" safety-case metric required by
Section 5.3.

HAZARD_SLOPE semantics (FR-1, revised): hazards are traversable-but-risky, not
obstacles. They are blocked only when

* the projected post-move battery would fall below ``battery_reserve * 2``, or
* an *equally good* safe non-hazard action exists (distance-to-goal of the
  alternative's next cell <= that of the proposed hazard entry).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .config import RoverConfig, WorldConfig
from .dynamics import RoverSimulator, action_target
from .types import (
    ALL_ACTIONS,
    Action,
    ActionSource,
    GridMap,
    Heading,
    GuardianReason,
    RoverState,
    euclidean,
    heading_delta,
    turn_left,
    turn_right,
)

EQUALLY_GOOD_EPSILON = 1e-9


@dataclass(frozen=True)
class SafeActionSet:
    """Result of the pre-filter for one timestep."""

    actions: tuple[Action, ...]
    blocked: dict[Action, GuardianReason] = field(default_factory=dict)
    enters_hazard: dict[Action, bool] = field(default_factory=dict)

    def as_values(self) -> list[str]:
        return [a.value for a in self.actions]

    def __contains__(self, action: object) -> bool:
        return action in self.actions


@dataclass(frozen=True)
class Approval:
    """Guardian verdict for one proposal."""

    final_action: Action
    overridden: bool
    reason_code: GuardianReason
    counterfactual_collision: bool = False
    counterfactual_hazard_entry: bool = False
    note: str | None = None


class SafetyGuardian:
    """Deterministic hard constraints. No randomness, no LLM involvement."""

    def __init__(self, world: GridMap, world_config: WorldConfig, rover_config: RoverConfig) -> None:
        self.world = world
        self.world_config = world_config
        self.rover = rover_config

    # ------------------------------------------------------------------ #
    # Pre-filter
    # ------------------------------------------------------------------ #

    def safe_actions(self, state: RoverState) -> SafeActionSet:
        blocked: dict[Action, GuardianReason] = {}
        hazard_map: dict[Action, bool] = {}
        candidates: list[Action] = []

        for action in ALL_ACTIONS:
            reason = self._blocked_reason(state, action)
            if reason is not None:
                blocked[action] = reason
                continue
            candidates.append(action)
            hazard_map[action] = self._enters_hazard(state, action)

        # Hazard preference rule: drop a hazard entry when an equally good safe
        # non-hazard action exists (deterministic, applies to every controller).
        preferred: list[Action] = []
        for action in candidates:
            if hazard_map.get(action) and self._has_equal_non_hazard_alternative(
                state, action, candidates, hazard_map
            ):
                blocked[action] = GuardianReason.BLOCKED_HAZARD_PREFERENCE
                continue
            preferred.append(action)

        ordered = tuple(a for a in ALL_ACTIONS if a in preferred)
        return SafeActionSet(
            actions=ordered,
            blocked=blocked,
            enters_hazard={a: hazard_map.get(a, False) for a in ordered},
        )

    def _blocked_reason(self, state: RoverState, action: Action) -> GuardianReason | None:
        if action is Action.STOP_AND_REPLAN:
            if state.consecutive_stops >= self.rover.max_consecutive_stops:
                return GuardianReason.STOP_CAP_REACHED
            if state.battery - self._cost_fraction(state, action) < 0.0:
                return GuardianReason.BLOCKED_BATTERY_RESERVE
            return None

        tx, ty = action_target(state.x, state.y, state.heading, action)
        if not self.world.in_bounds(tx, ty):
            return GuardianReason.BLOCKED_OUT_OF_BOUNDS
        if self.world.is_obstacle(tx, ty):
            return GuardianReason.BLOCKED_OBSTACLE

        reaches_goal = (tx, ty) == self.world.goal
        cost = self._cost_fraction(state, action)
        # Hard floor: never spend below empty (tie-break from FR-5).
        if state.battery - cost < 0.0:
            return GuardianReason.BLOCKED_BATTERY_RESERVE
        # Reserve rule: no discretionary motion below the reserve unless the
        # action reaches the goal.
        if state.battery < self.rover.battery_reserve and not reaches_goal:
            return GuardianReason.BLOCKED_BATTERY_RESERVE
        # Conditional hazard rule #1 (battery): hazard entry needs headroom.
        if self.world.is_hazard(tx, ty):
            if state.battery - cost < self.rover.battery_reserve * 2:
                return GuardianReason.BLOCKED_HAZARD_BATTERY
        return None

    def _cost_fraction(self, state: RoverState, action: Action) -> float:
        tx, ty = action_target(state.x, state.y, state.heading, action)
        hazard = action is not Action.STOP_AND_REPLAN and self.world.is_hazard(tx, ty)
        return self.rover.cost_of(action, hazard=hazard) / self.rover.energy_capacity

    def _enters_hazard(self, state: RoverState, action: Action) -> bool:
        if action is Action.STOP_AND_REPLAN:
            return False
        return self.world.is_hazard(*action_target(state.x, state.y, state.heading, action))

    def _has_equal_non_hazard_alternative(
        self,
        state: RoverState,
        hazard_action: Action,
        candidates: list[Action],
        hazard_map: dict[Action, bool],
    ) -> bool:
        gx, gy = self.world.goal
        hx, hy = action_target(state.x, state.y, state.heading, hazard_action)
        hazard_distance = euclidean(hx, hy, gx, gy)
        for other in candidates:
            if other is hazard_action or hazard_map.get(other):
                continue
            ox, oy = action_target(state.x, state.y, state.heading, other)
            if euclidean(ox, oy, gx, gy) <= hazard_distance + EQUALLY_GOOD_EPSILON:
                return True
        return False

    # ------------------------------------------------------------------ #
    # Approval / override
    # ------------------------------------------------------------------ #

    def approve(
        self,
        proposed: Action | None,
        proposed_source: ActionSource,
        safe: SafeActionSet,
        state: RoverState,
        failure_reason: GuardianReason | None = None,
        fallback_action: Action | None = None,
    ) -> Approval:
        """Accept or reject a proposal. ``proposed=None`` means the decision
        source failed outright (invalid/timeout/unavailable)."""
        if not safe.actions:
            # Boxed in: nothing legal. STOP regardless of the cap so the episode
            # terminates cleanly (the simulator reports a permitted idle step and
            # the loop/step guard ends the episode).
            return Approval(
                final_action=Action.STOP_AND_REPLAN,
                overridden=True,
                reason_code=GuardianReason.NO_LEGAL_ACTION,
                note="no legal action available - boxed in",
            )

        if proposed is None:
            resolved = self._resolve_override(state, safe, fallback_action)
            return Approval(
                final_action=resolved,
                overridden=True,
                reason_code=failure_reason or GuardianReason.MALFORMED_RESPONSE,
                note="proposal unavailable - conservative resolution applied",
            )

        if proposed not in safe.actions:
            reason = safe.blocked.get(proposed, GuardianReason.NOT_IN_SAFE_ACTIONS)
            resolved = self._resolve_override(state, safe, fallback_action)
            collision, hazard = self.counterfactual(state, proposed)
            return Approval(
                final_action=resolved,
                overridden=True,
                reason_code=reason,
                counterfactual_collision=collision,
                counterfactual_hazard_entry=hazard,
                note=f"{proposed.value} rejected",
            )

        return Approval(final_action=proposed, overridden=False, reason_code=GuardianReason.APPROVED)

    def counterfactual(self, state: RoverState, action: Action) -> tuple[bool, bool]:
        """(would_collide, would_enter_hazard) - ablation probe, never executes."""
        if action is Action.STOP_AND_REPLAN:
            return False, False
        tx, ty = action_target(state.x, state.y, state.heading, action)
        if not self.world.in_bounds(tx, ty) or self.world.is_obstacle(tx, ty):
            return True, False
        return False, self.world.is_hazard(tx, ty)

    def _resolve_override(
        self,
        state: RoverState,
        safe: SafeActionSet,
        fallback_action: Action | None,
    ) -> Action:
        """Deterministic conservative resolution.

        Prefers the fallback controller's choice when it is legal; otherwise the
        legal action minimising distance-to-goal, tie-broken by the canonical
        action order so behaviour is fully reproducible.
        """
        if fallback_action is not None and fallback_action in safe.actions:
            return fallback_action
        gx, gy = self.world.goal
        best = safe.actions[0]
        best_key = (euclidean(*action_target(state.x, state.y, state.heading, best), gx, gy), ALL_ACTIONS.index(best))
        for action in safe.actions[1:]:
            tx, ty = action_target(state.x, state.y, state.heading, action)
            key = (euclidean(tx, ty, gx, gy), ALL_ACTIONS.index(action))
            if key < best_key:
                best, best_key = action, key
        return best

    # ------------------------------------------------------------------ #
    # Geometry helper shared with controllers (one-step lookahead on the
    # observation alone, so controllers never need the full map)
    # ------------------------------------------------------------------ #

    @staticmethod
    def next_cell_distance(
        observation_goal_rel: tuple[int, int],
        heading: Heading,
        action: Action,
    ) -> float:
        """Distance from the cell this action would enter to the goal, computed
        purely from the observation (relative goal + heading)."""
        goal_x, goal_y = observation_goal_rel

        def delta(h: str) -> tuple[int, int]:
            return heading_delta(h)

        if action is Action.STOP_AND_REPLAN:
            return euclidean(0.0, 0.0, goal_x, goal_y)
        if action is Action.TURN_LEFT_AND_MOVE:
            dx, dy = delta(turn_left(heading))
        elif action is Action.TURN_RIGHT_AND_MOVE:
            dx, dy = delta(turn_right(heading))
        elif action is Action.REVERSE:
            dx, dy = delta(heading)
            dx, dy = -dx, -dy
        else:
            dx, dy = delta(heading)
        return euclidean(dx, dy, goal_x, goal_y)


__all__ = [
    "SafetyGuardian",
    "SafeActionSet",
    "Approval",
    "EQUALLY_GOOD_EPSILON",
]
