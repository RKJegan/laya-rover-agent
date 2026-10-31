"""The decision-layer interface, independent of any one provider.

This module is the seam the rest of the agent talks to. It exists so that the
model behind it is a *configuration choice* rather than a structural assumption:
the safety guardian, the fallback controller, the episode loop and the whole
experiment matrix only ever see this protocol, never a provider name.

    DecisionAdapter   the remote decision-service seam: adecide() / decide()
    DecisionResult    the validated payload a decision service must produce
    DecisionCall      the outcome envelope (validity, reason code, latency, usage)
    ACTION_QUESTION_ID / REASON_QUESTION_ID   the typed-question ids used on the wire

Two decision layers ship today, and they fill **different** seams on purpose:

* :class:`~laya_rover_agent.laya.api.LayaApiAdapter` - a self-hosted
  ``laya-serve`` over HTTP. It implements :class:`DecisionAdapter`: it is asked a
  question and answers with a :class:`DecisionCall`, exactly like the wire.
* :class:`~laya_rover_agent.laya.mock.MockLayaAdapter` - the deterministic
  offline stand-in. It implements the controller-facing ``propose()`` seam
  instead, because its whole definition is *"what would the fallback controller
  have chosen, and how does the mock disagree with it"* - that needs the current
  rover state and the guardian's safe set, which a wire-level signature cannot
  express. It is wired in by ``controllers/laya_agent.py``.

So a new *remote* backend implements :class:`DecisionAdapter` and is drop-in; a
new *local policy* implements ``Controller.propose()``. Either way, every
guarantee the guardian relies on (``valid=False`` never means "may move", the
chosen action always comes from the supplied safe set) is enforced *after* the
adapter, so a new backend cannot weaken them by mistake.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from ..types import DecisionCall, DecisionResult, Observation
from .api import LayaApiAdapter
from .mock import MockLayaAdapter
from .prompt import ACTION_QUESTION_ID, REASON_QUESTION_ID, build_questions

__all__ = [
    "DecisionAdapter",
    "DecisionCall",
    "DecisionResult",
    "LayaApiAdapter",
    "MockLayaAdapter",
    "ACTION_QUESTION_ID",
    "REASON_QUESTION_ID",
    "build_questions",
]


@runtime_checkable
class DecisionAdapter(Protocol):
    """The only shape the control loop requires of a decision backend.

    ``safe_actions`` is the guardian's pre-filtered, already-legal action set.
    An adapter may only ever return a member of it: anything else is rejected by
    the guardian before execution, so an out-of-set answer cannot move the rover
    even if a backend misbehaves. Adapters must not raise into the control loop -
    a failure is ``valid=False``, which routes to the fallback controller.
    """

    name: str

    async def adecide(self, observation: Observation, safe_actions: list[str]) -> DecisionCall:
        """Async decision - the path the experiment runner and bridge use."""
        ...

    def decide(self, observation: Observation, safe_actions: list[str] | None = None) -> DecisionCall:
        """Blocking decision - for scripts and single-threaded tooling."""
        ...
