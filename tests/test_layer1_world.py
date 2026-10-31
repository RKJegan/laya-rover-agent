"""Layer 1 - world generation, determinism, RNG separation, traps (FR-1)."""

from __future__ import annotations

import random

import pytest

from conftest import make_map
from laya_rover_agent.types import CellType
from laya_rover_agent.world import (
    MapSpec,
    PRESET_SCHEDULE,
    WorldGenerator,
    ascii_map,
    preset_for_index,
    route_exists,
    stream_seed,
    trap_wall,
)


@pytest.mark.parametrize("preset", ["random", "trap_u", "trap_canyon", "trap_twincup"])
@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
def test_maps_are_deterministic(generator, preset, difficulty):
    first = generator.generate(MapSpec(seed=3, difficulty=difficulty, preset=preset))
    second = generator.generate(MapSpec(seed=3, difficulty=difficulty, preset=preset))
    assert ascii_map(first) == ascii_map(second)
    assert first.start == second.start
    assert first.goal == second.goal


def test_different_seeds_produce_different_maps(generator):
    a = ascii_map(generator.generate(MapSpec(seed=0, difficulty="easy", preset="random")))
    b = ascii_map(generator.generate(MapSpec(seed=1, difficulty="easy", preset="random")))
    assert a != b


@pytest.mark.parametrize("preset", ["random", "trap_u", "trap_canyon", "trap_twincup"])
def test_minimum_viable_route_exists(generator, preset):
    grid = make_map(generator, preset=preset, seed=2, difficulty="hard")
    assert route_exists(grid), f"no traversable route on {preset}"


def test_start_and_goal_cells_are_marked(generator):
    grid = make_map(generator, preset="random", seed=5)
    assert grid.cell(*grid.start) is CellType.START
    assert grid.cell(*grid.goal) is CellType.GOAL


def test_difficulty_presets_change_clutter(generator):
    counts = {}
    for difficulty in ("easy", "medium", "hard"):
        grid = make_map(generator, preset="random", seed=7, difficulty=difficulty)
        rows = "".join("".join(row) for row in grid.rows())
        counts[difficulty] = rows.count(CellType.OBSTACLE.value)
    assert counts["easy"] < counts["medium"] < counts["hard"]


def test_map_rng_stream_is_independent_of_mock_confidence_stream(generator):
    """Sweeping the confidence threshold must not alter terrain (FR-1 RNG separation)."""
    spec = MapSpec(seed=11, difficulty="medium", preset="trap_u")
    baseline = ascii_map(generator.generate(spec))

    # Draw from the mock-confidence stream; the map must be unaffected.
    rng = random.Random(stream_seed(spec.seed * 1_000_003 + 42, "mock_laya_confidence"))
    for _ in range(50):
        rng.gauss(0.85, 0.05)

    assert ascii_map(generator.generate(spec)) == baseline


def test_trap_wall_is_concave_with_a_single_mouth():
    glyph = trap_wall("trap_u", 5, 4, "north")
    assert glyph[0].count("#") == 0, "mouth row must be open"
    inner = glyph[1:-1]
    assert all(row.startswith("#") and row.endswith("#") for row in inner)
    assert glyph[-1].count("#") == 5, "back wall must be closed"


def test_trap_pocket_is_visible_within_sensor_range(quick_config, generator):
    """The enclosing wall must be closer than sensor_range, else the trap is unfair."""
    grid = make_map(generator, preset="trap_u", seed=1, difficulty="easy")
    assert grid.trap_clusters, "trap preset must record its cluster"
    for cluster in grid.trap_clusters:
        interior_depth = cluster["h"] - 1
        assert interior_depth <= quick_config.world.sensor_range


def test_trap_cluster_glyph_is_small_enough_to_route_around(generator):
    grid = make_map(generator, preset="trap_u", seed=4, difficulty="easy")
    cluster = grid.trap_clusters[0]
    assert cluster["w"] <= 8 and cluster["h"] <= 8


def test_preset_schedule_interleaves_all_four_scenarios():
    schedule = [preset_for_index("medium", index, 3) for index in range(8)]
    assert set(schedule) == set(PRESET_SCHEDULE)
    assert schedule[:3] == ["trap_u", "trap_canyon", "trap_twincup"]
    assert schedule[3] == "random"


def test_ascii_map_renders_start_goal_and_obstacles(generator):
    text = ascii_map(make_map(generator, preset="trap_u", seed=1))
    assert "S" in text and "G" in text and "#" in text
    assert len(text.splitlines()) > 0
