"""Shared fixtures for the offline test suite.

Everything here runs with **zero network access**. Layer 4 (live Laya) lives in
``scripts/smoke_test_laya_api.py`` and is never imported by pytest.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:  # allow `pytest` without an editable install
    sys.path.insert(0, str(SRC))

from laya_rover_agent.config import ProjectConfig, load_config  # noqa: E402
from laya_rover_agent.controllers.base import ControllerContext  # noqa: E402
from laya_rover_agent.controllers.fallback import FallbackController, FallbackPolicy  # noqa: E402
from laya_rover_agent.dynamics import RoverSimulator  # noqa: E402
from laya_rover_agent.perception import Perception  # noqa: E402
from laya_rover_agent.safety import SafetyGuardian  # noqa: E402
from laya_rover_agent.types import Action, GridMap, Observation, RoverState  # noqa: E402
from laya_rover_agent.world import MapSpec, WorldGenerator  # noqa: E402


@pytest.fixture(scope="session")
def project_root() -> Path:
    return PROJECT_ROOT


@pytest.fixture(scope="session")
def quick_config() -> ProjectConfig:
    """The small offline configuration (30x30 maps, short episodes)."""
    return load_config(PROJECT_ROOT / "configs" / "quick.yaml")


@pytest.fixture()
def generator(quick_config: ProjectConfig) -> WorldGenerator:
    return WorldGenerator(quick_config.world)


def make_map(generator: WorldGenerator, preset: str = "random", seed: int = 0, difficulty: str = "easy") -> GridMap:
    return generator.generate(MapSpec(seed=seed, difficulty=difficulty, preset=preset))


@pytest.fixture()
def grid(generator: WorldGenerator) -> GridMap:
    return make_map(generator, preset="random", seed=1)


@pytest.fixture()
def trap_grid(generator: WorldGenerator) -> GridMap:
    return make_map(generator, preset="trap_u", seed=1)


@pytest.fixture()
def perception(quick_config: ProjectConfig) -> Perception:
    return Perception(quick_config.world, quick_config.rover)


@pytest.fixture()
def guardian(quick_config: ProjectConfig, grid: GridMap) -> SafetyGuardian:
    return SafetyGuardian(grid, quick_config.world, quick_config.rover)


@pytest.fixture()
def simulator(quick_config: ProjectConfig, grid: GridMap) -> RoverSimulator:
    import random

    return RoverSimulator(
        world=grid,
        config=quick_config.rover,
        difficulty="easy",
        stall_rng=random.Random(1234),
        clock=lambda: 0.0,
    )


@pytest.fixture()
def fallback(quick_config: ProjectConfig) -> FallbackController:
    return FallbackController(FallbackPolicy(**quick_config.controllers.fallback.model_dump()))


def observe(perception: Perception, guardian: SafetyGuardian, grid: GridMap, state: RoverState):
    """Guardian pre-filter -> observation, exactly as the episode loop does it."""
    safe = guardian.safe_actions(state)
    observation = perception.observe(grid, state, safe.actions, None)
    return observation, safe


__all__ = [
    "Action",
    "ControllerContext",
    "FallbackController",
    "MapSpec",
    "Observation",
    "ProjectConfig",
    "RoverState",
    "make_map",
    "observe",
]
