"""Deterministic ``mock_laya`` adapter (FR-6, Section 15.4).

Fully specified rather than "some deterministic behaviour", because the
confidence-threshold sweep and the fallback logic cannot be evaluated against
an unspecified mock:

* As a deterministic function of ``(episode_seed, step)``:
  * ``agree_with_fallback_pct`` (default 0.80) - selects the same action the
    fallback controller would choose, confidence ~ ``N(0.85, 0.05)``.
  * ``ambiguous_alt_pct`` (default 0.10) - selects a plausible-but-ambiguous
    alternative *safe* action, confidence ~ ``N(0.65, 0.05)``, which straddles
    the confidence thresholds under test.
  * remaining - deliberately unusable: ``malformed_share_of_invalid`` of the
    time an invalid payload, otherwise a low-confidence response
    ~ ``N(0.4, 0.1)``.
* Additionally (documented extension, see README): ``unsafe_proposal_pct``
  (default 0.05) injects a confident proposal of a **guardian-blocked** action
  so the override path and the ablation/safety-case counters have real inputs.
  The guardian rejects it every time; the rover is never moved unsafely.

Every branch is reproducible from the seed, and the mock never shares an RNG
stream with map generation or hazard stalls.
"""

from __future__ import annotations

import json
import random
import time

from ..config import MockLayaProfile
from ..controllers.base import ControllerContext, Proposal
from ..controllers.fallback import FallbackController
from ..safety import SafeActionSet
from ..types import Action, ActionSource, GuardianReason, Observation, RoverState
from ..world import stream_seed
from .prompt import REASON_CODES
#: FROZEN numeric RNG-domain tag for this stream - do NOT change this number.
#:
#: ``world.stream_seed`` mixes a tag into the episode seed to separate streams,
#: so the tag is a *hash input*, not a label. This value is the tag the mock has
#: always used, kept as a bare number so the stream survives renames without
#: re-rolling: changing it would silently alter every mock episode (same
#: distribution, different sequence) and invalidate archived artifacts, prior
#: reports and any external comparison.
MOCK_RNG_TAG_VALUE = 19687

MALFORMED_PAYLOAD = '{"chosen_action": "MOVE_FORWA'  # truncated JSON, on purpose

COLLISION_REASONS = {
    GuardianReason.BLOCKED_OBSTACLE,
    GuardianReason.BLOCKED_OUT_OF_BOUNDS,
    GuardianReason.NOT_IN_SAFE_ACTIONS,
}


class MockLayaAdapter:
    """Offline stand-in for the Laya decision model."""

    name = "mock_laya"

    def __init__(
        self,
        profile: MockLayaProfile,
        episode_seed: int,
        fallback: FallbackController | None = None,
    ) -> None:
        self.profile = profile
        self.episode_seed = episode_seed
        self.fallback = fallback or FallbackController()

    # -- RNG ----------------------------------------------------------------- #

    def _rng(self, step: int) -> random.Random:
        mixed = self.episode_seed * 1_000_003 + step * 7919
        return random.Random(stream_seed(mixed, MOCK_RNG_TAG_VALUE))

    @staticmethod
    def _normal(rng: random.Random, mu: float, sigma: float) -> float:
        return round(min(1.0, max(0.0, rng.gauss(mu, sigma))), 4)

    # -- decision ------------------------------------------------------------ #

    async def propose(
        self,
        observation: Observation,
        state: RoverState,
        safe: SafeActionSet,
        step: int,
        context: ControllerContext | None = None,
    ) -> Proposal:
        started = time.perf_counter()
        rng = self._rng(step)
        branch_roll = rng.random()
        confidence_roll = rng.random()
        probe_roll = rng.random()
        malformed_roll = rng.random()

        fallback_action = self.fallback.choose(observation, safe, context)
        profile = self.profile

        # -- documented extension: confident-but-unsafe probe ---------------- #
        if probe_roll < profile.unsafe_proposal_pct:
            probe_action = self._pick_blocked(safe)
            if probe_action is not None:
                confidence = self._normal(rng, 0.90, 0.03)
                payload = {
                    "chosen_action": probe_action.value,
                    "confidence": confidence,
                    "reason_code": "CLEAR_PATH",
                }
                return Proposal(
                    action=probe_action,
                    source=ActionSource.LAYA,
                    confidence=confidence,
                    valid=True,
                    latency_ms=self._elapsed_ms(started),
                    reason_code="CLEAR_PATH",
                    raw_response=json.dumps(payload),
                    provider_model=self.name,
                    diagnostics={"mock_branch": "unsafe_probe"},
                )

        # -- spec branch 1: agree with fallback ------------------------------ #
        if branch_roll < profile.agree_with_fallback_pct:
            confidence = self._normal(rng, 0.85, 0.05)
            payload = {
                "chosen_action": fallback_action.value,
                "confidence": confidence,
                "reason_code": self._reason_for(fallback_action, observation),
            }
            return Proposal(
                action=fallback_action,
                source=ActionSource.LAYA,
                confidence=confidence,
                valid=True,
                latency_ms=self._elapsed_ms(started),
                reason_code=payload["reason_code"],
                raw_response=json.dumps(payload),
                provider_model=self.name,
                diagnostics={"mock_branch": "agree_with_fallback"},
            )

        # -- spec branch 2: plausible-but-ambiguous alternative -------------- #
        if branch_roll < profile.agree_with_fallback_pct + profile.ambiguous_alt_pct:
            alternative = self._alternative(observation, safe, fallback_action)
            confidence = self._normal(rng, 0.65, 0.05)
            payload = {
                "chosen_action": alternative.value,
                "confidence": confidence,
                "reason_code": self._reason_for(alternative, observation),
            }
            return Proposal(
                action=alternative,
                source=ActionSource.LAYA,
                confidence=confidence,
                valid=True,
                latency_ms=self._elapsed_ms(started),
                reason_code=payload["reason_code"],
                raw_response=json.dumps(payload),
                provider_model=self.name,
                diagnostics={"mock_branch": "ambiguous_alternative"},
            )

        # -- spec branch 3: unusable output ---------------------------------- #
        if malformed_roll < profile.malformed_share_of_invalid:
            return Proposal(
                action=None,
                source=ActionSource.LAYA,
                confidence=None,
                valid=False,
                latency_ms=self._elapsed_ms(started),
                failure_reason=GuardianReason.MALFORMED_RESPONSE,
                raw_response=MALFORMED_PAYLOAD,
                provider_model=self.name,
                diagnostics={"mock_branch": "malformed"},
            )

        confidence = self._normal(rng, 0.40, 0.10)
        payload = {
            "chosen_action": fallback_action.value,
            "confidence": confidence,
            "reason_code": "REPLAN_NEEDED",
        }
        return Proposal(
            action=fallback_action,
            source=ActionSource.LAYA,
            confidence=confidence,
            valid=True,
            latency_ms=self._elapsed_ms(started),
            reason_code="REPLAN_NEEDED",
            raw_response=json.dumps(payload),
            provider_model=self.name,
            diagnostics={"mock_branch": "low_confidence"},
        )

    @staticmethod
    def _elapsed_ms(started: float) -> float:
        """Real measured latency of the mock call (no network, so microseconds).

        Recorded rather than hard-coded to zero so the latency logging path and
        the latency figure are exercised in offline runs too.
        """
        return (time.perf_counter() - started) * 1000.0

    # -- helpers ------------------------------------------------------------- #

    def _pick_blocked(self, safe: SafeActionSet) -> Action | None:
        """A guardian-blocked action - the ablation probe's raw material."""
        for action, reason in safe.blocked.items():
            if reason in COLLISION_REASONS:
                return action
        for action in safe.blocked:
            return action
        return None

    def _alternative(
        self,
        observation: Observation,
        safe: SafeActionSet,
        fallback_action: Action,
    ) -> Action:
        """A safe action that is *not* the fallback's choice, preferring one
        whose next cell is close to the goal (plausible, occasionally better)."""
        from ..safety import SafetyGuardian

        goal_rel = (observation.goal.relative_x, observation.goal.relative_y)
        options = [a for a in safe.actions if a is not fallback_action]
        if not options:
            return fallback_action
        # A *plausible* alternative: never the last-resort reverse unless it is
        # literally the only legal option left.
        non_reverse = [a for a in options if a is not Action.REVERSE]
        pool = non_reverse or options
        return min(
            pool,
            key=lambda a: SafetyGuardian.next_cell_distance(goal_rel, observation.heading, a),
        )

    def _reason_for(self, action: Action, observation: Observation) -> str:
        if action is Action.STOP_AND_REPLAN:
            return "REPLAN_NEEDED"
        if observation.hazards.front and action is not Action.MOVE_FORWARD:
            return "AVOID_HAZARD"
        if observation.obstacles.front <= 1 and action is not Action.MOVE_FORWARD:
            return "AVOID_OBSTACLE"
        if observation.battery_fraction < 0.25:
            return "CONSERVE_ENERGY"
        return "CLEAR_PATH" if REASON_CODES else "CLEAR_PATH"
