"""Controller protocol + shared proposal/context envelopes."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from ..safety import SafeActionSet
from ..types import Action, ActionSource, GuardianReason, Observation, RoverState


@dataclass(frozen=True)
class ControllerContext:
    """Non-observation context offered to controllers.

    Deliberately **not** part of the Laya observation schema (FR-4 is frozen):
    it carries the rover's own local pose history, which is legitimate onboard
    state that any flight software would keep in its own blackboard.

    The baseline controller ignores this by design (Section 5.4: memoryless,
    prone to oscillation); the conservative fallback uses it as a short-term
    visited-cell memory so the safety net does not get trapped in a concave
    local minimum. The Laya decision layer never sees it - it gets the fixed
    observation window plus the allowed action set.
    """

    current_position: tuple[int, int] | None = None
    recent_positions: tuple[tuple[int, int], ...] = ()
    previous_action: Action | None = None
    #: "nothing detected" sentinel (sensor_range + 1); kept out of the frozen
    #: FR-4 observation schema so the Laya payload is exactly as specified.
    sensor_sentinel: int = 7


@dataclass(frozen=True)
class Proposal:
    """A controller's proposed action plus everything the guardian and the
    logger need in order to audit it."""

    action: Action | None                     # None => controller produced nothing usable
    source: ActionSource
    confidence: float | None = None
    valid: bool = True
    latency_ms: float = 0.0
    reason_code: str | None = None            # provider-supplied reason (optional)
    failure_reason: GuardianReason | None = None   # why the proposal is unusable
    raw_response: str | None = None           # redacted, truncated
    raw_request: str | None = None            # redacted, truncated
    provider_model: str | None = None
    probabilities: dict[str, float] | None = None
    #: Laya's usage record mixes numeric tokens with a bool truncation flag and a
    #: list of truncated question ids, so this is not ``dict[str, float]``.
    usage: dict[str, Any] | None = None
    #: The served model reported that it did not read the whole state. Recorded
    #: per step because a decision on partial evidence is a safety-relevant fact.
    truncated: bool = False
    diagnostics: dict[str, object] = field(default_factory=dict)


@runtime_checkable
class Controller(Protocol):
    """Anything that can propose the next action from an observation."""

    name: str

    async def propose(
        self,
        observation: Observation,
        state: RoverState,
        safe: SafeActionSet,
        step: int,
        context: ControllerContext | None = None,
    ) -> Proposal:  # pragma: no cover - protocol
        ...
