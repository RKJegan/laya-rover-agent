"""Layer 1 - controller differentiation (FR-7 / Section 5.4).

The structural claim the report leans on is that the baseline is *algorithmically
distinct* from the fallback: memoryless + hazard-blind + fixed tie-break, versus
conservative + hazard-averse + short-term memory. These tests hold that claim.
"""

from __future__ import annotations

import random

from conftest import make_map, observe
from laya_rover_agent.controllers.baseline import BaselineController
from laya_rover_agent.controllers.fallback import FallbackController, FallbackPolicy
from laya_rover_agent.dynamics import RoverSimulator
from laya_rover_agent.safety import SafetyGuardian
from laya_rover_agent.types import Action, CellType, GridMap, RoverState, Status
from laya_rover_agent.world import MapSpec, WorldGenerator


def make_state(**overrides) -> RoverState:
    base = dict(x=5, y=5, heading="E", battery=1.0, energy_used=0.0, steps=0, status=Status.ACTIVE, consecutive_stops=0)
    base.update(overrides)
    return RoverState(**base)


def _pocket_map(width: int = 16, height: int = 14) -> GridMap:
    """A real 3-deep pocket (the generator's own glyph), goal beyond it.

    The rover starts at the mouth facing in, so a purely greedy controller walks
    into the pocket and oscillates at the back wall, while a controller that can
    see the enclosing wall refuses the entrance and goes around.
    """
    from laya_rover_agent.world import trap_wall

    cells = [[CellType.FREE.value] * width for _ in range(height)]
    ox, oy = 5, 4
    for dy, row in enumerate(trap_wall("trap_u", 5, 4, "north")):
        for dx, ch in enumerate(row):
            if ch == "#":
                cells[oy + dy][ox + dx] = CellType.OBSTACLE.value
    start, goal = (7, 3), (7, height - 1)
    cells[start[1]][start[0]] = CellType.START.value
    cells[goal[1]][goal[0]] = CellType.GOAL.value
    return GridMap(width=width, height=height, start=start, goal=goal, cells=cells)


def _open_map(width: int = 12, height: int = 14) -> GridMap:
    cells = [[CellType.FREE.value] * width for _ in range(height)]
    start, goal = (6, 3), (6, height - 1)
    cells[start[1]][start[0]] = CellType.START.value
    cells[goal[1]][goal[0]] = CellType.GOAL.value
    return GridMap(width=width, height=height, start=start, goal=goal, cells=cells)


def test_baseline_is_greedy_and_memoryless(quick_config):
    grid = _pocket_map()
    guardian = SafetyGuardian(grid, quick_config.world, quick_config.rover)
    baseline = BaselineController()
    state = make_state(x=7, y=4, heading="S")
    safe = guardian.safe_actions(state)
    # Forward enters the pocket and strictly reduces goal distance -> greedy takes it.
    assert baseline.choose(_observation(grid, state, safe, quick_config), safe) is Action.MOVE_FORWARD
    # The baseline exposes no policy object and no history hook at all: it is
    # memoryless by construction, which is what makes it the naive comparison.
    assert not hasattr(baseline, "policy")
    import inspect

    params = list(inspect.signature(baseline.choose).parameters)
    assert params == ["observation", "safe"], "baseline must not accept any context/history"


def _observation(grid, state, safe, config):
    from laya_rover_agent.perception import Perception

    return Perception(config.world, config.rover).observe(grid, state, safe.actions, None)


def test_fallback_refuses_to_repeat_a_just_visited_cell(quick_config):
    """The conservative controller is the one with short-term memory.

    Straight corridor, goal directly ahead: forward is unambiguously the best
    move, so the memoryless baseline takes it - and so does the fallback when it
    has no history. Give the fallback a history showing that the cell ahead was
    just occupied and it steps aside instead of re-treading it.
    """
    grid = _open_map()
    guardian = SafetyGuardian(grid, quick_config.world, quick_config.rover)
    from laya_rover_agent.controllers.base import ControllerContext
    from laya_rover_agent.perception import Perception

    perception = Perception(quick_config.world, quick_config.rover)
    fallback = FallbackController(FallbackPolicy(**quick_config.controllers.fallback.model_dump()))
    baseline = BaselineController()
    state = make_state(x=6, y=4, heading="S")
    safe = guardian.safe_actions(state)
    observation = perception.observe(grid, state, safe.actions, None)

    assert baseline.choose(observation, safe) is Action.MOVE_FORWARD
    fresh = fallback.choose(observation, safe, ControllerContext(current_position=(6, 4), recent_positions=((6, 4),)))
    revisited = fallback.choose(
        observation,
        safe,
        ControllerContext(
            current_position=(6, 4),
            recent_positions=((6, 4), (6, 5)),
            previous_action=Action.MOVE_FORWARD,
        ),
    )
    assert fresh is Action.MOVE_FORWARD
    assert revisited is not Action.MOVE_FORWARD, "memory must break the re-entry"


def test_controllers_are_deterministic(quick_config, generator):
    grid = generator.generate(MapSpec(seed=6, difficulty="easy", preset="trap_twincup"))
    guardian = SafetyGuardian(grid, quick_config.world, quick_config.rover)
    from laya_rover_agent.perception import Perception

    perception = Perception(quick_config.world, quick_config.rover)
    fallback = FallbackController(FallbackPolicy(**quick_config.controllers.fallback.model_dump()))
    baseline = BaselineController()
    state = make_state(x=grid.start[0], y=grid.start[1])
    safe = guardian.safe_actions(state)
    observation = perception.observe(grid, state, safe.actions, None)
    assert fallback.choose(observation, safe) is fallback.choose(observation, safe)
    assert baseline.choose(observation, safe) is baseline.choose(observation, safe)


def test_fallback_avoids_visible_pocket_that_baseline_enters(quick_config):
    """Head-to-head on the same local observation: the designed contrast."""
    grid = _pocket_map()
    guardian = SafetyGuardian(grid, quick_config.world, quick_config.rover)
    from laya_rover_agent.controllers.base import ControllerContext
    from laya_rover_agent.perception import Perception

    perception = Perception(quick_config.world, quick_config.rover)
    fallback = FallbackController(FallbackPolicy(**quick_config.controllers.fallback.model_dump()))
    baseline = BaselineController()
    state = make_state(x=7, y=4, heading="S")  # standing in the mouth, facing the back wall
    safe = guardian.safe_actions(state)
    observation = perception.observe(grid, state, safe.actions, None)

    baseline_action = baseline.choose(observation, safe)
    fallback_action = fallback.choose(
        observation, safe, ControllerContext(current_position=(7, 4), recent_positions=((7, 4),))
    )
    assert baseline_action is Action.MOVE_FORWARD
    assert fallback_action is not Action.MOVE_FORWARD


def test_baseline_oscillates_but_fallback_does_not(quick_config):
    """End-to-end on the pocket map: baseline dies as `stuck`, fallback escapes."""
    grid = _pocket_map()
    guardian = SafetyGuardian(grid, quick_config.world, quick_config.rover)
    from laya_rover_agent.controllers.base import ControllerContext
    from laya_rover_agent.perception import Perception

    perception = Perception(quick_config.world, quick_config.rover)
    fallback = FallbackController(FallbackPolicy(**quick_config.controllers.fallback.model_dump()))
    baseline = BaselineController()

    def play(controller, use_context: bool):
        sim = RoverSimulator(world=grid, config=quick_config.rover, difficulty="easy", stall_rng=random.Random(0))
        state = sim.initial_state()
        sim.note_position(state)
        history: list[tuple[int, int]] = [(state.x, state.y)]
        for _ in range(60):
            safe = guardian.safe_actions(state)
            if not safe.actions:
                break
            observation = perception.observe(grid, state, safe.actions, None)
            context = ControllerContext(
                current_position=(state.x, state.y),
                recent_positions=tuple(history[-12:]),
            )
            action = (
                fallback.choose(observation, safe, context)
                if use_context
                else controller.choose(observation, safe)
            )
            outcome = sim.step(state, action)
            state = outcome.state
            history.append((state.x, state.y))
            if state.status is not Status.ACTIVE:
                break
        return state

    baseline_final = play(baseline, use_context=False)
    fallback_final = play(fallback, use_context=True)
    assert baseline_final.status is Status.STUCK
    assert fallback_final.status is Status.GOAL, f"fallback ended {fallback_final.status}"
