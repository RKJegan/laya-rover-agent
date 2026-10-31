"""Baseline rule-based controller (FR-8, Section 5.4).

Deliberately **algorithmically distinct** from the fallback controller, so the
experiment can actually detect a difference (the original v1 spec's baseline and
fallback were near-identical, which would have structurally biased the result
toward "no measurable difference"):

* one-step steepest descent on Euclidean distance to goal,
* **no lookahead** beyond the current single-step observation,
* **no memory** of previous actions (cannot notice it is oscillating),
* **hazard-blind**: a hazard entry is scored exactly like free terrain, so it
  happily walks through risky slopes and eats the energy penalty,
* fixed tie-break priority ``front > left > right > reverse``,
* ``REVERSE`` is a last resort, used only when no forward/turn action is safe -
  a standard reactive-robot heuristic, which is what makes the baseline a
  credible classical controller on open terrain while leaving it structurally
  unable to notice that it is cycling inside a concave trap.

That combination is a legitimate classical reactive controller and is exactly
the kind of local-minimum-prone policy a context-aware decision layer is
expected to beat near concave (U-shaped / canyon) obstacle clusters.
"""

from __future__ import annotations

from ..safety import SafeActionSet, SafetyGuardian
from ..types import Action, ActionSource, GuardianReason, Observation, RoverState
from .base import ControllerContext, Proposal

#: Fixed deterministic tie-break priority (Section 5.4).
BASELINE_PRIORITY: tuple[Action, ...] = (
    Action.MOVE_FORWARD,
    Action.TURN_LEFT_AND_MOVE,
    Action.TURN_RIGHT_AND_MOVE,
    Action.REVERSE,
)


class BaselineController:
    name = "baseline"

    def choose(self, observation: Observation, safe: SafeActionSet) -> Action:
        goal_rel = (observation.goal.relative_x, observation.goal.relative_y)
        # forward / turns first; reverse only when nothing else is legal
        candidates = [a for a in BASELINE_PRIORITY if a is not Action.REVERSE and a in safe]
        if not candidates:
            if Action.REVERSE in safe:
                return Action.REVERSE
            return Action.STOP_AND_REPLAN
        return min(
            candidates,
            key=lambda a: (
                SafetyGuardian.next_cell_distance(goal_rel, observation.heading, a),
                BASELINE_PRIORITY.index(a),
            ),
        )

    async def propose(
        self,
        observation: Observation,
        state: RoverState,
        safe: SafeActionSet,
        step: int,
        context: ControllerContext | None = None,
    ) -> Proposal:
        # ``context`` is deliberately ignored: the baseline is memoryless.
        action = self.choose(observation, safe)
        return Proposal(
            action=action,
            source=ActionSource.BASELINE,
            confidence=1.0,
            valid=True,
            reason_code=GuardianReason.APPROVED.value,
            diagnostics={"controller": self.name},
        )
