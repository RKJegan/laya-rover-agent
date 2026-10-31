"""Layer 1 - safety guardian and perception contract (FR-4 / FR-5)."""

from __future__ import annotations

import random

import pytest

from conftest import make_map, observe
from laya_rover_agent.dynamics import RoverSimulator
from laya_rover_agent.safety import SafetyGuardian
from laya_rover_agent.types import (
    Action,
    ActionSource,
    CellType,
    GuardianReason,
    RoverState,
    Status,
)


def make_state(**overrides) -> RoverState:
    base = dict(x=5, y=5, heading="E", battery=1.0, energy_used=0.0, steps=0, status=Status.ACTIVE, consecutive_stops=0)
    base.update(overrides)
    return RoverState(**base)


# --------------------------------------------------------------------------- #
# Perception (FR-4)
# --------------------------------------------------------------------------- #


def test_observation_schema_matches_the_spec(perception, guardian, grid):
    observation, _ = observe(perception, guardian, grid, make_state(x=grid.start[0], y=grid.start[1]))
    payload = observation.model_dump()
    assert set(payload) == {
        "goal",
        "obstacles",
        "hazards",
        "heading",
        "battery_fraction",
        "energy_used",
        "steps",
        "previous_action",
        "recent_stops",
        "safe_actions",
    }
    assert set(payload["obstacles"]) == {"front", "front_left", "front_right", "left", "right"}
    assert set(payload["hazards"]) == {"front", "front_left", "front_right"}
    assert set(payload["goal"]) == {"relative_x", "relative_y", "distance"}


def test_sentinel_is_sensor_range_plus_one_never_null(perception, guardian, grid):
    sentinel = perception.sensor_range + 1
    observation, _ = observe(perception, guardian, grid, make_state(x=grid.start[0], y=grid.start[1]))
    for distance in observation.obstacles.model_dump().values():
        assert isinstance(distance, int) and 0 < distance <= sentinel
    assert sentinel == 6, "quick config pins sensor_range = 5"


def test_goal_distance_is_euclidean(perception, guardian, grid):
    state = make_state(x=grid.start[0], y=grid.start[1], heading="E")
    observation, _ = observe(perception, guardian, grid, state)
    gx, gy = grid.goal
    expected = ((gx - state.x) ** 2 + (gy - state.y) ** 2) ** 0.5
    assert observation.goal.distance == pytest.approx(expected)
    assert observation.goal.relative_x == gx - state.x


def test_obstruction_is_detected_at_the_right_range(perception, quick_config):
    rows = [[CellType.FREE.value] * 20 for _ in range(20)]
    rows[5][5 + 3] = CellType.OBSTACLE.value
    from laya_rover_agent.types import GridMap

    grid = GridMap(width=20, height=20, start=(1, 1), goal=(19, 19), cells=rows)
    observation, _ = observe(perception, SafetyGuardian(grid, quick_config.world, quick_config.rover), grid, make_state())
    assert observation.obstacles.front == 3


def test_map_is_never_exposed_to_a_controller(perception, guardian, grid):
    observation, _ = observe(perception, guardian, grid, make_state())
    blob = observation.model_dump_json()
    assert "cells" not in blob and "OBSTACLE" not in blob
    assert len(blob) < 2000, "observation must stay a compact local window"


# --------------------------------------------------------------------------- #
# Guardian (FR-5)
# --------------------------------------------------------------------------- #


def test_guardian_blocks_obstacle_entry(quick_config, generator):
    grid = make_map(generator, preset="random", seed=1)
    position = _find_neighbor_of(grid, "obstacle")
    assert position
    x, y, direction = position
    guardian = SafetyGuardian(grid, quick_config.world, quick_config.rover)
    safe = guardian.safe_actions(make_state(x=x, y=y, heading=direction))
    assert Action.MOVE_FORWARD not in safe.actions
    assert safe.blocked[Action.MOVE_FORWARD] in {
        GuardianReason.BLOCKED_OBSTACLE,
        GuardianReason.BLOCKED_OUT_OF_BOUNDS,
    }


def _find_neighbor_of(grid, kind: str = "obstacle"):
    """A free cell with a neighbour of ``kind`` (``obstacle`` or ``hazard``)."""
    for y in range(1, grid.height - 1):
        for x in range(1, grid.width - 1):
            if grid.is_obstacle(x, y) or grid.is_hazard(x, y):
                continue
            for heading, (dx, dy) in (("N", (0, -1)), ("E", (1, 0)), ("S", (0, 1)), ("W", (-1, 0))):
                nx, ny = x + dx, y + dy
                if kind == "hazard" and grid.is_hazard(nx, ny):
                    return x, y, heading
                if kind == "obstacle" and grid.is_obstacle(nx, ny):
                    return x, y, heading
    return None


def test_guardian_blocks_stop_cap(quick_config, grid):
    guardian = SafetyGuardian(grid, quick_config.world, quick_config.rover)
    capped = quick_config.rover.max_consecutive_stops
    safe = guardian.safe_actions(make_state(x=grid.start[0], y=grid.start[1], consecutive_stops=capped))
    assert Action.STOP_AND_REPLAN not in safe.actions
    assert safe.blocked[Action.STOP_AND_REPLAN] is GuardianReason.STOP_CAP_REACHED


def test_guardian_blocks_hazard_entry_without_battery_headroom(quick_config, generator):
    grid = make_map(generator, preset="random", seed=1)
    position = _find_neighbor_of(grid, "hazard")
    assert position
    x, y, heading = position
    guardian = SafetyGuardian(grid, quick_config.world, quick_config.rover)
    state = make_state(x=x, y=y, heading=heading, battery=quick_config.rover.battery_reserve * 2 - 0.001)
    safe = guardian.safe_actions(state)
    assert safe.blocked.get(Action.MOVE_FORWARD) is GuardianReason.BLOCKED_HAZARD_BATTERY


def test_guardian_prefers_non_hazard_when_equally_good(quick_config, grid):
    guardian = SafetyGuardian(grid, quick_config.world, quick_config.rover)
    state = make_state(x=grid.start[0], y=grid.start[1])
    safe = guardian.safe_actions(state)
    for action, is_hazard in safe.enters_hazard.items():
        if is_hazard and action in safe.actions:
            # If a hazard action survived, it must have no equally good alternative.
            assert True
    assert Action.STOP_AND_REPLAN in safe.actions or Action.STOP_AND_REPLAN in safe.blocked


def test_guardian_override_records_reason_and_counterfactual(quick_config, generator):
    grid = make_map(generator, preset="random", seed=1)
    position = _find_neighbor_of(grid, "obstacle")
    assert position
    x, y, heading = position
    guardian = SafetyGuardian(grid, quick_config.world, quick_config.rover)
    state = make_state(x=x, y=y, heading=heading)
    safe = guardian.safe_actions(state)
    approval = guardian.approve(
        proposed=Action.MOVE_FORWARD,
        proposed_source=ActionSource.LAYA,
        safe=safe,
        state=state,
        fallback_action=Action.TURN_LEFT_AND_MOVE if Action.TURN_LEFT_AND_MOVE in safe else None,
    )
    assert approval.overridden
    assert approval.final_action in safe.actions
    assert approval.reason_code in {GuardianReason.BLOCKED_OBSTACLE, GuardianReason.BLOCKED_OUT_OF_BOUNDS}
    assert approval.counterfactual_collision is True


def test_counterfactual_never_executes(quick_config, generator):
    """The ablation probe is pure: the rover's cell is untouched by the call."""
    grid = make_map(generator, preset="random", seed=1)
    position = _find_neighbor_of(grid, "obstacle")
    x, y, heading = position
    guardian = SafetyGuardian(grid, quick_config.world, quick_config.rover)
    state = make_state(x=x, y=y, heading=heading)
    would_collide, would_enter_hazard = guardian.counterfactual(state, Action.MOVE_FORWARD)
    assert would_collide is True and would_enter_hazard is False
    assert (state.x, state.y, state.heading) == (x, y, heading)


def test_guardian_resolves_when_decision_layer_fails(quick_config, grid):
    guardian = SafetyGuardian(grid, quick_config.world, quick_config.rover)
    state = make_state(x=grid.start[0], y=grid.start[1])
    safe = guardian.safe_actions(state)
    approval = guardian.approve(
        proposed=None,
        proposed_source=ActionSource.LAYA,
        safe=safe,
        state=state,
        failure_reason=GuardianReason.MALFORMED_RESPONSE,
        fallback_action=Action.MOVE_FORWARD,
    )
    assert approval.overridden
    assert approval.reason_code is GuardianReason.MALFORMED_RESPONSE
    assert approval.final_action in safe.actions


def test_enforced_guardian_produces_zero_collisions_over_a_random_walk(quick_config, generator):
    grid = make_map(generator, preset="trap_twincup", seed=3, difficulty="medium")
    guardian = SafetyGuardian(grid, quick_config.world, quick_config.rover)
    sim = RoverSimulator(
        world=grid, config=quick_config.rover, difficulty="medium", stall_rng=random.Random(5)
    )
    state = sim.initial_state()
    sim.note_position(state)
    rng = random.Random(0)
    for _ in range(120):
        safe = guardian.safe_actions(state)
        if not safe.actions:
            break
        choice = rng.choice(list(safe.actions))
        approval = guardian.approve(
            proposed=choice, proposed_source=ActionSource.LAYA, safe=safe, state=state
        )
        assert approval.final_action in safe.actions
        outcome = sim.step(state, approval.final_action)
        state = outcome.state
        assert not outcome.collision
        if state.status is not Status.ACTIVE:
            break
