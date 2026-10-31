"""Adapter construction: mock vs live, one place, no hidden globals."""

from __future__ import annotations

from typing import Callable

from ..config import ProjectConfig
from ..controllers.baseline import BaselineController
from ..controllers.fallback import FallbackController, FallbackPolicy
from ..controllers.laya_agent import LayaAgentController
from ..episode import EpisodeSpec
from ..security import SecretRedactor
from ..types import ActionSource
from .api import LayaApiAdapter
from .mock import MockLayaAdapter


class ControllerFactory:
    """Builds the controller for one episode.

    The live API adapter is **shared across episodes** (one connection pool, one
    concurrency semaphore) while the mock adapter is **per episode** because its
    determinism is a function of the episode seed.
    """

    def __init__(
        self,
        config: ProjectConfig,
        api_key: str | None = None,
        redactor: SecretRedactor | None = None,
        request_style: str = "typed",
    ) -> None:
        self.config = config
        self.api_key = api_key
        self.redactor = redactor or SecretRedactor([api_key])
        self.request_style = request_style
        self._api_adapter: LayaApiAdapter | None = None

    # ------------------------------------------------------------------ #

    @property
    def api_adapter(self) -> LayaApiAdapter:
        if self._api_adapter is None:
            api = self.config.agent.laya_api
            self._api_adapter = LayaApiAdapter(
                config=api,
                api_key=self.api_key,
                decision_timeout_ms=self.config.agent.decision_timeout_ms,
                redactor=self.redactor,
                request_style=self.request_style,
            )
        return self._api_adapter

    def __call__(self, spec: EpisodeSpec):
        if spec.controller == "baseline":
            return BaselineController()
        if spec.controller != "laya_agent":
            raise ValueError(f"unknown controller {spec.controller!r}")

        if self.config.agent.mode == "mock_laya":
            return LayaAgentController(
                MockLayaAdapter(
                    profile=self.config.agent.mock_laya_profile,
                    episode_seed=spec.map_spec.seed * 97 + spec.index,
                    fallback=FallbackController(self._fallback_policy()),
                )
            )
        return LayaAgentController(self.api_adapter)

    def _fallback_policy(self) -> FallbackPolicy:
        return FallbackPolicy(**self.config.controllers.fallback.model_dump())

    async def aclose(self) -> None:
        if self._api_adapter is not None:
            await self._api_adapter.aclose()


def make_single_controller(
    config: ProjectConfig,
    controller_name: str,
    episode_seed: int = 0,
    api_key: str | None = None,
    redactor: SecretRedactor | None = None,
) -> Callable[[int], object]:
    """Helper used by tests to obtain a controller without an EpisodeSpec."""
    factory = ControllerFactory(config, api_key=api_key, redactor=redactor)

    def build(index: int = 0):
        from ..world import MapSpec

        spec = EpisodeSpec(
            map_spec=MapSpec(seed=episode_seed, difficulty="medium", preset="random"),
            controller=controller_name,
            confidence_threshold=config.agent.confidence_threshold,
            index=index,
        )
        return factory(spec)

    return build


__all__ = ["ControllerFactory", "make_single_controller", "ActionSource"]
