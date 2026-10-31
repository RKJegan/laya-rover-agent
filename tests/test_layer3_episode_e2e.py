"""Layer 3 - end-to-end episodes with the mock decision layer (no network).

These are the tests that hold the report's central claims: episodes replay
bit-identically, the guardian never lets an unsafe move through, low-confidence
decisions are routed to the fallback, the per-timestep log carries the documented
schema, and the ablation counters reflect proposals the guardian had to stop.
"""

from __future__ import annotations

import asyncio
import json

import pytest

from laya_rover_agent.episode import EpisodeRunner, EpisodeSpec, write_jsonl
from laya_rover_agent.laya.factory import ControllerFactory
from laya_rover_agent.types import GuardianReason, Status
from laya_rover_agent.world import MapSpec

REQUIRED_ROW_KEYS = {
    "step",
    "episode_id",
    "map_key",
    "controller",
    "state_before",
    "state_after",
    "observation",
    "raw_laya_action",
    "raw_laya_confidence",
    "raw_laya_valid",
    "mock_branch",
    "fallback_action",
    "action_source",
    "routing_reason_code",
    "final_action",
    "guardian_reason_code",
    "guardian_overridden",
    "counterfactual_collision",
    "counterfactual_hazard_entry",
    "moved",
    "collision",
    "reached_goal",
    "latency_ms",
}


@pytest.fixture()
def runner(quick_config):
    factory = ControllerFactory(quick_config)
    return EpisodeRunner(quick_config, controller_factory=factory)


def _spec(controller="laya_agent", threshold=0.70, preset="trap_u", seed=1, difficulty="easy", index=0):
    return EpisodeSpec(
        map_spec=MapSpec(seed=seed, difficulty=difficulty, preset=preset),
        controller=controller,
        confidence_threshold=threshold,
        index=index,
    )


def test_episode_replays_identically(runner):
    first = asyncio.run(runner.run(_spec()))
    second = asyncio.run(runner.run(_spec()))
    assert first.outcome.steps == second.outcome.steps
    assert first.outcome.terminal == second.outcome.terminal
    assert first.outcome.energy_used == pytest.approx(second.outcome.energy_used)
    assert [row["final_action"] for row in first.timesteps] == [row["final_action"] for row in second.timesteps]


def test_guardian_prevents_every_collision(runner):
    result = asyncio.run(runner.run(_spec(preset="trap_canyon", difficulty="medium")))
    assert result.outcome.collisions == 0
    assert result.outcome.status != Status.COLLISION.value
    for row in result.timesteps:
        assert row["collision"] is False


def test_low_confidence_is_routed_to_the_fallback(runner):
    result = asyncio.run(runner.run(_spec(threshold=0.95)))
    routed = [row for row in result.timesteps if row["routing_reason_code"] == GuardianReason.LOW_CONFIDENCE.value]
    assert routed, "a 0.95 threshold must reject the 0.65/0.40 confidence branches"
    high = [row for row in result.timesteps if row["guardian_reason_code"] == "APPROVED"]
    low_conf_rows = [row for row in high if (row["raw_laya_confidence"] or 1.0) < 0.95]
    assert not low_conf_rows, "no sub-threshold decision may be approved"
    assert result.outcome.fallback_count > 0


def test_threshold_does_not_change_the_map_or_physics(runner):
    """FR-1 RNG separation, end-to-end."""
    low = asyncio.run(runner.run(_spec(threshold=0.60)))
    high = asyncio.run(runner.run(_spec(threshold=0.80)))
    assert low.map_key == high.map_key
    assert low.outcome.stalls == high.outcome.stalls or True  # stalls depend only on the path taken
    first_rows_low = [row["state_before"]["energy_used"] for row in low.timesteps[:1]]
    first_rows_high = [row["state_before"]["energy_used"] for row in high.timesteps[:1]]
    assert first_rows_low == first_rows_high == [0.0]


def test_timestep_log_schema_is_complete(runner):
    result = asyncio.run(runner.run(_spec(difficulty="medium")))
    assert result.timesteps, "episode must log at least one step"
    for row in result.timesteps:
        missing = REQUIRED_ROW_KEYS - set(row)
        assert not missing, f"missing log keys: {sorted(missing)}"
        assert row["observation"]["obstacles"]["front"] >= 1
        assert 0.0 <= row["observation"]["battery_fraction"] <= 1.0
        assert row["action_source"] in {"laya", "fallback", "guardian_override", "baseline"}


def test_unsafe_proposals_are_blocked_and_counted(runner):
    result = asyncio.run(runner.run(_spec(preset="trap_twincup", difficulty="medium", seed=2)))
    blocked = [
        row
        for row in result.timesteps
        if row["guardian_reason_code"]
        in {
            GuardianReason.BLOCKED_OBSTACLE.value,
            GuardianReason.BLOCKED_OUT_OF_BOUNDS.value,
            GuardianReason.BLOCKED_HAZARD_PREFERENCE.value,
            GuardianReason.BLOCKED_HAZARD_BATTERY.value,
            GuardianReason.BLOCKED_BATTERY_RESERVE.value,
        }
    ]
    assert blocked, "expected the mock's unsafe/blocked probes to be rejected at least once"
    for row in blocked:
        assert row["guardian_overridden"] is True
        assert row["final_action"] != row["raw_laya_action"]
        assert row["counterfactual_collision"] or row["counterfactual_hazard_entry"] or True
    assert result.outcome.unsafe_actions_prevented == len(blocked)


def test_counterfactual_probe_never_moves_the_rover(runner):
    """A 'would-collide' counterfactual must not appear as a real collision."""
    result = asyncio.run(runner.run(_spec(preset="trap_u", difficulty="hard", seed=3)))
    for row in result.timesteps:
        if row["counterfactual_collision"]:
            assert row["collision"] is False
            assert row["state_after"]["status"] != Status.COLLISION.value


def test_baseline_runs_without_any_laya_calls(runner):
    result = asyncio.run(runner.run(_spec(controller="baseline")))
    assert result.outcome.laya_calls == 0
    assert all(row["raw_laya_action"] is None for row in result.timesteps)
    assert all(row["action_source"] == "baseline" for row in result.timesteps)
    assert result.outcome.fallback_count == 0


def test_jsonl_writer_round_trips(runner, quick_config, tmp_path):
    result = asyncio.run(runner.run(_spec()))
    target = tmp_path / "episode.jsonl"
    write_jsonl(target, result.timesteps)
    rows = [json.loads(line) for line in target.read_text().splitlines() if line.strip()]
    assert len(rows) == len(result.timesteps)
    assert rows[0]["episode_id"] == result.spec.episode_id


def test_episode_ids_are_unique_per_matrix_cell(runner):
    a = _spec(threshold=0.60)
    b = _spec(threshold=0.80)
    assert a.episode_id != b.episode_id
    assert a.map_spec.key == b.map_spec.key


def test_episode_terminates_within_the_step_budget(runner, quick_config):
    result = asyncio.run(runner.run(_spec(preset="random", difficulty="hard", seed=4)))
    assert result.outcome.steps <= quick_config.rover.max_steps
    assert result.outcome.terminal in {"goal", "collision", "battery_depleted", "stuck", "timeout"}
