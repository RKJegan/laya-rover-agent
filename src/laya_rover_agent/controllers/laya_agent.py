"""The Laya-backed agent controller: wraps any Laya adapter as a Controller."""

from __future__ import annotations

from ..laya.api import LayaApiAdapter
from ..laya.mock import MockLayaAdapter
from ..safety import SafeActionSet
from ..types import ActionSource, GuardianReason, Observation, RoverState
from .base import ControllerContext, Proposal


class LayaAgentController:
    """Proposes actions from Laya. Never decides safety - the guardian does."""

    name = "laya_agent"

    def __init__(self, adapter: MockLayaAdapter | LayaApiAdapter) -> None:
        self.adapter = adapter

    async def propose(
        self,
        observation: Observation,
        state: RoverState,
        safe: SafeActionSet,
        step: int,
        context: ControllerContext | None = None,
    ) -> Proposal:
        if isinstance(self.adapter, MockLayaAdapter):
            return await self.adapter.propose(observation, state, safe, step, context)

        result = await self.adapter.adecide(observation, observation.safe_actions)
        if not result.valid or result.decision is None:
            return Proposal(
                action=None,
                source=ActionSource.LAYA,
                confidence=None,
                valid=False,
                latency_ms=result.latency_ms,
                failure_reason=result.reason_code or GuardianReason.MALFORMED_RESPONSE,
                raw_response=result.raw_response,
                raw_request=result.raw_request,
                provider_model=result.provider_model,
                usage=result.usage,
                truncated=result.truncated,
                diagnostics={"http_status": result.http_status or 0},
            )
        decision = result.decision
        return Proposal(
            action=decision.chosen_action,
            source=ActionSource.LAYA,
            confidence=decision.confidence,
            valid=True,
            latency_ms=result.latency_ms,
            reason_code=decision.reason_code,
            raw_response=result.raw_response,
            raw_request=result.raw_request,
            provider_model=result.provider_model,
            probabilities=decision.probabilities,
            usage=result.usage,
            truncated=result.truncated,
            diagnostics={"http_status": result.http_status or 0},
        )

    async def aclose(self) -> None:
        aclose = getattr(self.adapter, "aclose", None)
        if aclose is not None:
            await aclose()
