"""Layer 1 - rover kinematics, energy accounting, hazard stalls, loop detection (FR-2/FR-3)."""

from __future__ import annotations

import random

import pytest

from conftest import make_map
from laya_rover_agent.dynamics import RoverSimulator, action_target, next_heading
from laya_rover_agent.types import Action, RoverState, Status


def make_state(**overrides) -> RoverState:
    base = dict(x=5, y=5, heading="E", battery=1.0, energy_used=0.0, steps=0, status=Status.ACTIVE, consecutive_stops=0)
    base.update(overrides)
    return RoverState(**base)


def test_heading_semantics_are_pinned():
    assert next_heading("N", Action.MOVE_FORWARD) == "N"
    assert next_heading("N", Action.TURN_LEFT_AND_MOVE) == "W"
    assert next_heading("N", Action.TURN_RIGHT_AND_MOVE) == "E"
    assert next_heading("N", Action.REVERSE) == "N"
    assert action_target(5, 5, "N", Action.MOVE_FORWARD) == (5, 4)  # north = -y (screen frame)
    assert action_target(5, 5, "N", Action.REVERSE) == (5, 6)


def test_energy_costs_match_the_spec(quick_config, grid, simulator):
    costs = quick_config.rover.energy_cost
    assert costs["MOVE_FORWARD"] == 1.0
    assert costs["TURN_LEFT_AND_MOVE"] == 1.3
    assert costs["TURN_RIGHT_AND_MOVE"] == 1.3
    assert costs["REVERSE"] == 1.5
    assert costs["STOP_AND_REPLAN"] == 0.1

    state = make_state(x=grid.start[0], y=grid.start[1], heading="E")
    forward = simulator.step(state, Action.MOVE_FORWARD)
    assert forward.state.energy_used - state.energy_used == pytest.approx(1.0)


def test_obstacle_entry_is_a_collision(quick_config, generator):
    grid = make_map(generator, preset="random", seed=1)
    blocked = None
    for y in range(grid.height):
        for x in range(grid.width):
            if grid.is_obstacle(x + 1, y) and not grid.is_obstacle(x, y):
                blocked = (x, y)
                break
        if blocked:
            break
    assert blocked, "test map needs at least one obstacle"
    sim = RoverSimulator(world=grid, config=quick_config.rover, difficulty="easy", stall_rng=random.Random(0))
    outcome = sim.step(make_state(x=blocked[0], y=blocked[1], heading="E"), Action.MOVE_FORWARD)
    assert outcome.collision and outcome.blocked
    assert outcome.state.status is Status.COLLISION


def test_hazard_entry_costs_multiplier_and_can_stall():
    """A stall consumes energy and a step but does not advance the rover (FR-1)."""
    from laya_rover_agent.config import RoverConfig

    sim = RoverSimulator(world=_hazard_world(), config=RoverConfig(), difficulty="hard", stall_rng=random.Random(0))
    sim_stall = RoverSimulator(world=_hazard_world(), config=RoverConfig(), difficulty="hard", stall_rng=_AlwaysStall())
    state = make_state(x=5, y=5, heading="E")
    moving = sim.step(state, Action.MOVE_FORWARD)
    assert moving.entered_hazard and moving.moved
    assert moving.state.energy_used == pytest.approx(1.0 * 1.8)

    stalled = sim_stall.step(state, Action.MOVE_FORWARD)
    assert stalled.stalled and not stalled.moved
    assert (stalled.state.x, stalled.state.y) == (5, 5)
    assert stalled.state.steps == 1


def _hazard_world():
    from laya_rover_agent.types import GridMap

    cells = [["FREE"] * 12 for _ in range(12)]
    cells[5][6] = "HAZARD_SLOPE"
    cells[0][0] = "START"
    cells[11][11] = "GOAL"
    return GridMap(width=12, height=12, start=(0, 0), goal=(11, 11), cells=cells)


class _AlwaysStall(random.Random):
    def random(self) -> float:  # noqa: D102 - deliberately always below the stall probability
        return 0.0


def test_stop_cap_counts_consecutive_stops(quick_config, grid):
    sim = RoverSimulator(world=grid, config=quick_config.rover, difficulty="easy", stall_rng=random.Random(0))
    state = make_state(x=grid.start[0], y=grid.start[1])
    for expected in range(1, 4):
        state = sim.step(state, Action.STOP_AND_REPLAN).state
        assert state.consecutive_stops == expected
    state = sim.step(state, Action.TURN_LEFT_AND_MOVE).state
    assert state.consecutive_stops == 0


def test_loop_detector_fires_on_abab_bounce(quick_config, grid):
    """A-B-A-B with no net progress and no new cells is `stuck` (FR-3)."""
    sim = RoverSimulator(world=grid, config=quick_config.rover, difficulty="easy", stall_rng=random.Random(0))
    run = _find_free_horizontal_run(grid, length=3)
    assert run is not None, "test map needs a free horizontal run"
    x, y = run
    state = make_state(x=x, y=y, heading="E")
    sim.note_position(state)

    bounced = False
    for index in range(quick_config.rover.stuck_window + 4):
        # Turn around in place, then advance: a pure two-cell ping-pong.
        heading = "E" if index % 2 == 0 else "W"
        state = state.model_copy(update={"heading": heading})
        outcome = sim.step(state, Action.MOVE_FORWARD)
        state = outcome.state
        if state.status is Status.STUCK:
            bounced = True
            break
        assert not outcome.collision, "bounce must stay on free cells"
    assert bounced, "A-B-A oscillation was not detected"


def _find_free_horizontal_run(grid, length: int = 3):
    for y in range(1, grid.height - 1):
        for x in range(1, grid.width - length - 1):
            if all(not grid.is_obstacle(x + offset, y) and not grid.is_hazard(x + offset, y) for offset in range(length + 1)):
                return (x, y)
    return None


def test_loop_detector_ignores_steady_progress(quick_config, grid):
    sim = RoverSimulator(world=grid, config=quick_config.rover, difficulty="easy", stall_rng=random.Random(0))
    state = make_state(x=2, y=grid.goal[1], heading="E")
    sim.note_position(state)
    for _ in range(quick_config.rover.stuck_window + 4):
        outcome = sim.step(state, Action.MOVE_FORWARD)
        state = outcome.state
        if state.status is not Status.ACTIVE:
            break
    assert state.status is not Status.STUCK


def test_battery_depletion_terminates(quick_config, grid):
    sim = RoverSimulator(world=grid, config=quick_config.rover, difficulty="easy", stall_rng=random.Random(0))
    state = make_state(x=grid.start[0], y=grid.start[1], battery=0.005)
    outcome = sim.step(state, Action.MOVE_FORWARD)
    assert outcome.state.status is Status.BATTERY_DEPLETED


def test_goal_reached_terminates(quick_config, grid):
    sim = RoverSimulator(world=grid, config=quick_config.rover, difficulty="easy", stall_rng=random.Random(0))
    gx, gy = grid.goal
    state = make_state(x=gx - 1, y=gy, heading="E")
    outcome = sim.step(state, Action.MOVE_FORWARD)
    assert outcome.reached_goal
    assert outcome.state.status is Status.GOAL


def test_replay_is_deterministic(quick_config, grid):
    def play():
        sim = RoverSimulator(world=grid, config=quick_config.rover, difficulty="easy", stall_rng=random.Random(99))
        state = sim.initial_state()
        trace = []
        for _ in range(30):
            outcome = sim.step(state, Action.MOVE_FORWARD)
            state = outcome.state
            trace.append((state.x, state.y, round(state.energy_used, 6), state.status.value))
            if state.status is not Status.ACTIVE:
                break
        return trace

    assert play() == play()
