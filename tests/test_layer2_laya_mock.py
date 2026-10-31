"""Layer 2 - the deterministic mock decision layer (FR-6)."""

from __future__ import annotations

import collections
import json

import pytest

from conftest import make_map
from laya_rover_agent.controllers.base import ControllerContext
from laya_rover_agent.controllers.fallback import FallbackController, FallbackPolicy
from laya_rover_agent.dynamics import RoverSimulator
from laya_rover_agent.laya.mock import MockLayaAdapter
from laya_rover_agent.perception import Perception
from laya_rover_agent.safety import SafetyGuardian
from laya_rover_agent.types import Action, ActionSource, Status


def _step_proposals(quick_config, generator, preset="trap_u", seed=1, steps=120, rng_seed=0, episode_seed=0):
    """Collect bundles along a seeded random walk over the guardian-approved actions.

    Each bundle is ``(observation, safe, step, context, fallback, episode_seed)``.
    The mock's randomness is a pure function of ``(episode_seed, step)`` - exactly
    as FR-6 requires - so a statistically meaningful sample must vary the episode
    seed, not just re-run the same step indices.
    """
    grid = make_map(generator, preset=preset, seed=seed, difficulty="medium")
    guardian = SafetyGuardian(grid, quick_config.world, quick_config.rover)
    perception = Perception(quick_config.world, quick_config.rover)
    fallback = FallbackController(FallbackPolicy(**quick_config.controllers.fallback.model_dump()))
    import random as _random

    rng = _random.Random(rng_seed)
    sim = RoverSimulator(world=grid, config=quick_config.rover, difficulty="medium", stall_rng=_random.Random(3))
    state = sim.initial_state()
    sim.note_position(state)
    history: list[tuple[int, int]] = [(state.x, state.y)]

    out = []
    for step in range(steps):
        safe = guardian.safe_actions(state)
        if not safe.actions:
            break
        observation = perception.observe(grid, state, safe.actions, None)
        context = ControllerContext(current_position=(state.x, state.y), recent_positions=tuple(history[-12:]))
        out.append((observation, safe, step, context, fallback, episode_seed))
        action = rng.choice([a for a in safe.actions if a is not Action.STOP_AND_REPLAN] or list(safe.actions))
        outcome = sim.step(state, action)
        state = outcome.state
        history.append((state.x, state.y))
        if state.status is not Status.ACTIVE:
            break
    return out


def _propose(adapter, bundle):
    """Ask a pre-built adapter for its proposal on this bundle."""
    import asyncio

    observation, safe, step, context, _, _ = bundle
    from laya_rover_agent.types import RoverState

    state = RoverState(
        x=context.current_position[0],
        y=context.current_position[1],
        heading=observation.heading,
        battery=observation.battery_fraction,
        energy_used=observation.energy_used,
        steps=observation.steps,
        status=Status.ACTIVE,
        consecutive_stops=observation.recent_stops,
    )
    return asyncio.run(adapter.propose(observation, state, safe, step, context))


def _propose_at(profile, bundle):
    """Fresh adapter per bundle so the RNG stream varies with the episode seed."""
    return _propose(MockLayaAdapter(profile=profile, episode_seed=bundle[5]), bundle)


@pytest.fixture()
def bundles(quick_config, generator):
    """~150 local situations drawn from several maps and random walks."""
    collected = []
    for index, preset in enumerate(("trap_u", "trap_canyon", "trap_twincup", "random")):
        for walk in range(2):
            collected.extend(
                _step_proposals(
                    quick_config,
                    generator,
                    preset=preset,
                    seed=index,
                    steps=60,
                    rng_seed=index * 10 + walk,
                    episode_seed=index * 100 + walk,
                )
            )
    assert len(collected) >= 250
    return collected


def test_branch_mix_matches_the_profile(quick_config, bundles):
    profile = quick_config.agent.mock_laya_profile
    branches = collections.Counter()
    for bundle in bundles:
        proposal = _propose_at(profile, bundle)
        branches[proposal.diagnostics.get("mock_branch")] += 1
    total = sum(branches.values())
    assert total > 100, "need a decent sample for the mix assertion"
    # 5% of the time the mock probes a blocked action on purpose.
    unsafe = branches["unsafe_probe"]
    assert 0 < unsafe / total < 0.15
    non_unsafe = total - unsafe
    agree = branches["agree_with_fallback"] / non_unsafe
    assert 0.72 < agree < 0.88, f"agree share {agree:.3f} outside {profile.agree_with_fallback_pct} +/- tolerance"
    assert branches["ambiguous_alternative"] / non_unsafe > 0.04
    assert (branches["low_confidence"] + branches["malformed"]) / non_unsafe > 0.04


def test_mock_is_deterministic_per_seed_and_step(quick_config, bundles):
    adapter_a = MockLayaAdapter(profile=quick_config.agent.mock_laya_profile, episode_seed=77)
    adapter_b = MockLayaAdapter(profile=quick_config.agent.mock_laya_profile, episode_seed=77)
    adapter_c = MockLayaAdapter(profile=quick_config.agent.mock_laya_profile, episode_seed=78)
    first = [_propose(adapter_a, bundle) for bundle in bundles[:60]]
    same = [_propose(adapter_b, bundle) for bundle in bundles[:60]]
    other = [_propose(adapter_c, bundle) for bundle in bundles[:60]]

    def fingerprint(proposals):
        return [(p.action.value if p.action else None, p.confidence, p.valid) for p in proposals]

    assert fingerprint(first) == fingerprint(same)
    assert fingerprint(first) != fingerprint(other)


def test_confidences_are_in_range_and_branch_typical(quick_config, bundles):
    profile = quick_config.agent.mock_laya_profile
    by_branch = collections.defaultdict(list)
    import statistics

    for bundle in bundles:
        proposal = _propose_at(profile, bundle)
        if proposal.confidence is None:  # the malformed branch carries no confidence
            assert not proposal.valid
            continue
        assert 0.0 <= proposal.confidence <= 1.0
        by_branch[proposal.diagnostics.get("mock_branch")].append(proposal.confidence)

    assert 0.78 < statistics.mean(by_branch["agree_with_fallback"]) < 0.92
    assert 0.55 < statistics.mean(by_branch["ambiguous_alternative"]) < 0.75
    assert statistics.mean(by_branch["low_confidence"]) < 0.55


def test_unsafe_probe_proposes_a_blocked_action(quick_config, bundles):
    """Force the probe branch (100% share) and check what it proposes."""
    profile = quick_config.agent.mock_laya_profile.model_copy(update={"unsafe_proposal_pct": 1.0})
    checked = 0
    for bundle in bundles:
        safe = bundle[1]
        if not safe.blocked:
            continue
        proposal = _propose_at(profile, bundle)
        assert proposal.diagnostics.get("mock_branch") == "unsafe_probe"
        assert proposal.action is not None
        assert proposal.action not in safe.actions, "the probe must be a blocked action"
        assert proposal.confidence > 0.8, "the probe is deliberately over-confident"
        checked += 1
    assert checked > 5


def test_unsafe_probe_is_rare_under_the_real_profile(quick_config, bundles):
    profile = quick_config.agent.mock_laya_profile
    probes = sum(
        1 for bundle in bundles if _propose_at(profile, bundle).diagnostics.get("mock_branch") == "unsafe_probe"
    )
    assert probes / len(bundles) <= 0.15


def test_malformed_branch_is_invalid_json(quick_config, bundles):
    profile = quick_config.agent.mock_laya_profile
    seen = 0
    for bundle in bundles:
        proposal = _propose_at(profile, bundle)
        if proposal.diagnostics.get("mock_branch") != "malformed":
            continue
        seen += 1
        assert not proposal.valid
        with pytest.raises(json.JSONDecodeError):
            json.loads(proposal.raw_response)
    assert seen > 0, "expected at least one malformed proposal in the sample"


def test_ambiguous_alternative_never_picks_reverse_unless_forced(quick_config, bundles):
    profile = quick_config.agent.mock_laya_profile
    for bundle in bundles:
        proposal = _propose_at(profile, bundle)
        if proposal.diagnostics.get("mock_branch") != "ambiguous_alternative":
            continue
        safe = bundle[1]
        alternatives = [a for a in safe.actions if a is not Action.REVERSE]
        if alternatives:
            assert proposal.action is not Action.REVERSE


def test_mock_reports_its_own_latency(quick_config, bundles):
    adapter = MockLayaAdapter(profile=quick_config.agent.mock_laya_profile, episode_seed=3)
    proposal = _propose(adapter, bundles[0])
    assert proposal.source is ActionSource.LAYA
    assert proposal.latency_ms >= 0.0
    assert proposal.provider_model == "mock_laya"


def test_mock_rng_tag_value_is_frozen():
    """The mock's RNG tag is a hash input - it must never change.

    ``world.stream_seed`` mixes the tag into the episode seed to separate
    streams. Changing it re-rolls every mock episode: same distribution,
    different sequence, which silently invalidates archived artifacts and
    previously published numbers. It is stored as a bare number so the stream
    can never be disturbed by a rename again.
    """
    from laya_rover_agent.laya.mock import MOCK_RNG_TAG_VALUE
    from laya_rover_agent.world import stream_seed

    assert MOCK_RNG_TAG_VALUE == 19687
    # a numeric tag is used directly, and it really does participate in the seed
    assert stream_seed(1234, MOCK_RNG_TAG_VALUE) == 1389905055
    assert stream_seed(1234, MOCK_RNG_TAG_VALUE) != stream_seed(1234, 19688)
    assert stream_seed(1234, "mock_laya_confidence") != stream_seed(1234, MOCK_RNG_TAG_VALUE)
