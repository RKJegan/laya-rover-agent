#!/usr/bin/env python3
"""Layer 4 - live Laya smoke test. **Manual / opt-in: talks to your laya-serve.**

Laya is a self-hosted decision engine, so "live" here means *your own running
server*, not a remote vendor. This is the only script in the project that makes
real HTTP calls; it is excluded from CI and from the default pytest run.

What it does, in order:

1. reads ``LAYA_API_URL`` / ``LAYA_MODEL`` / ``LAYA_API_KEY`` (``.env`` is loaded
   if present) and applies them over the YAML config;
2. **probes ``/health`` first** - an unreachable or still-cold-loading server is
   reported as such, before any decision is attempted;
3. sends **one** read-only ``choice`` decision request built from a synthetic
   rover observation - no map, no rover, nothing is executed;
4. prints the raw contract, the parsed decision with its native confidence and
   probabilities, the answering checkpoint, the latency, the token usage and the
   truncation flag, then a plain-language verdict with Laya-specific fixes for
   each failure mode;
5. optionally (``--episodes N``) drives the whole control loop against the live
   server, so you can watch the real model steer the rover;
6. proves the key (when one is configured) is redacted from everything it prints.

Start the server first, in another terminal::

    pip install "laya[serve]"
    LAYA_MODELS=english LAYA_HOST=127.0.0.1 laya-serve        # http://127.0.0.1:8000

Then::

    python scripts/smoke_test_laya_api.py                    # health + one decision
    python scripts/smoke_test_laya_api.py --health-only      # pre-flight only
    python scripts/smoke_test_laya_api.py --episodes 2 --preset trap_u
    python scripts/smoke_test_laya_api.py --config configs/laptop_live.yaml

``--config FILE`` layers your YAML over the packaged defaults (the same rule the
CLI uses). Reach for it when the machine is slower than the shipped decision
budget - a CPU-only laptop usually is. Without it every request is cancelled at
``min(request_timeout_s, decision_timeout_ms/1000)`` and the run silently falls
back; ``configs/laptop_live.yaml`` raises both budgets for exactly that case.

No API key is needed by default: a ``laya-serve`` started without ``LAYA_API_KEY``
has no auth at all. Exit codes: 0 = decision received and validated, 1 = the
server answered with an error or with something the adapter refused, 2 = the
server is unreachable or demands a key, 3 = a required credential is missing.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import httpx  # noqa: E402

from laya_rover_agent.config import LayaEnvSettings, apply_env_overrides, load_config  # noqa: E402
from laya_rover_agent.laya.api import LayaApiAdapter  # noqa: E402
from laya_rover_agent.security import SecretRedactor  # noqa: E402
from laya_rover_agent.types import (  # noqa: E402
    GoalObservation,
    HazardObservation,
    Observation,
    ObstacleObservation,
)


def sample_observation() -> Observation:
    """A realistic mid-mission observation: a wall 2 cells ahead, hazard on the left."""
    return Observation(
        goal=GoalObservation(relative_x=9, relative_y=-14, distance=16.626),
        obstacles=ObstacleObservation(front=2, front_left=4, front_right=3, left=6, right=6),
        hazards=HazardObservation(front=False, front_left=True, front_right=False),
        heading="N",
        battery_fraction=0.82,
        energy_used=21.6,
        steps=17,
        previous_action="MOVE_FORWARD",
        recent_stops=0,
        safe_actions=["MOVE_FORWARD", "TURN_LEFT_AND_MOVE", "TURN_RIGHT_AND_MOVE", "REVERSE", "STOP_AND_REPLAN"],
    )


def load_base_config(config_path: Path | None):
    """Packaged defaults + ``--config`` (when given). Same rule as the CLI."""
    return load_config(config_path)


async def call_once(
    settings: LayaEnvSettings, redactor: SecretRedactor, show_raw: bool, config_path: Path | None = None
) -> int:
    config = apply_env_overrides(load_base_config(config_path), settings)
    api_config = config.agent.laya_api
    adapter = LayaApiAdapter(
        config=api_config,
        api_key=settings.api_key,
        decision_timeout_ms=config.agent.decision_timeout_ms,
        redactor=redactor,
    )
    observation = sample_observation()

    print("—" * 72)
    print(f"POST {api_config.endpoint}")
    print(f"model: {api_config.model}   timeout ceiling: {adapter.effective_timeout_s:.2f}s")
    print("—" * 72)

    try:
        result = await adapter.adecide(observation, observation.safe_actions)
    finally:
        await adapter.aclose()

    if show_raw and result.raw_response:
        print("\nraw response (redacted):")
        print(redactor.redact(result.raw_response)[:1200])

    print(f"\nHTTP status      : {result.http_status}")
    print(f"valid            : {result.valid}")
    print(f"answering model  : {result.provider_model}   (the checkpoint that decided)")
    print(f"latency          : {result.latency_ms:.1f} ms")
    print(f"state truncated  : {result.truncated}   (True = the model read a cut-down state)")
    if result.usage:
        tokens = {k: v for k, v in result.usage.items() if "token" in k}
        print(f"token usage      : {tokens}")
        if result.usage.get("truncated_questions"):
            print(f"cut questions    : {result.usage['truncated_questions']}")
    if result.decision is not None:
        print(f"chosen action    : {result.decision.chosen_action.value}")
        print(f"confidence       : {result.decision.confidence:.3f}")
        print(f"reason code      : {result.decision.reason_code}")
        if result.decision.probabilities:
            top = sorted(result.decision.probabilities.items(), key=lambda item: -item[1])[:3]
            print("probabilities    : " + ", ".join(f"{name}={value:.3f}" for name, value in top))
    else:
        print(f"rejected because : {result.reason_code}")
    if result.usage:
        print(f"usage            : {json.dumps(result.usage)}")

    print("\nverdict:")
    if result.valid and result.decision is not None:
        print("  OK - the server answered, and the answer validated against the finite")
        print("       action set. You are ready to run `--episodes N`, then the bounded")
        print("       confirmatory experiment (see docs/live_laya_testing_guide.md).")
        return 0
    print(f"  FAILED - the adapter refused the answer ({result.reason_code}).")
    if result.http_status in (401, 403):
        print("       The server requires auth. Set LAYA_API_KEY to the same value the")
        print("       server was started with (LAYA_API_KEY=... laya-serve), then set")
        print("       agent.laya_api.auth_required: true so a missing key fails closed.")
    elif result.http_status == 422:
        print("       Laya refused the request body (422). Our question schema is")
        print("       malformed - a duplicate/null choice label or a bad score rubric.")
        print("       Nothing the server can recover from; report it as a client bug.")
    elif result.http_status in (429, 503):
        print("       The server is busy or still loading a checkpoint. Give it a moment")
        print("       (the first request after a cold start downloads weights), then retry.")
    elif result.http_status == 404:
        print("       Endpoint not found. A default laya-serve serves `/v1/systemone`;")
        print("       check LAYA_API_URL and any reverse-proxy path prefix.")
    elif result.reason_code is not None and "TIMEOUT" in str(result.reason_code):
        print("       The decision exceeded the budget. On a cold server the first")
        print("       request includes a checkpoint load - warm it up, or raise")
        print("       LAYA_REQUEST_TIMEOUT_S / agent.decision_timeout_ms.")
    elif result.reason_code is not None and "CONNECTION" in str(result.reason_code):
        print("       Could not connect. Is `laya-serve` running, and is LAYA_API_URL")
        print("       pointing at it (default http://127.0.0.1:8000/v1/systemone)?")
    print("       Re-run with --show-raw to inspect the (redacted) response.")
    return 1


async def run_episodes(
    count: int,
    preset: str,
    difficulty: str,
    threshold: float,
    live: bool,
    config_path: Path | None = None,
) -> int:
    """Drive the full control loop against the live model (or offline, for comparison)."""
    from laya_rover_agent.episode import EpisodeRunner, EpisodeSpec
    from laya_rover_agent.laya.factory import ControllerFactory
    from laya_rover_agent.world import MapSpec

    settings = LayaEnvSettings()
    config = apply_env_overrides(load_base_config(config_path), settings)
    if not live:
        config = config.model_copy(
            update={"agent": config.agent.model_copy(update={"mode": "mock_laya"})}
        )
    config = config.model_copy(
        update={"rover": config.rover.model_copy(update={"max_steps": 200})}
    )
    redactor = SecretRedactor.from_env_settings(settings)
    factory = ControllerFactory(config, api_key=settings.api_key, redactor=redactor)
    runner = EpisodeRunner(config, controller_factory=factory)
    try:
        for index in range(count):
            spec = EpisodeSpec(
                map_spec=MapSpec(seed=index, difficulty=difficulty, preset=preset),
                controller="laya_agent",
                confidence_threshold=threshold,
                index=index,
            )
            result = await runner.run(spec)
            outcome = result.outcome
            print(
                f"  episode {index}: {outcome.terminal:17s} steps={outcome.steps:4d} "
                f"energy={outcome.energy_used:7.2f} laya_calls={outcome.laya_calls:4d} "
                f"invalid={outcome.laya_invalid:3d} fallbacks={outcome.fallback_count:3d} "
                f"overrides={outcome.guardian_overrides:3d} "
                f"prevented={outcome.unsafe_actions_prevented:3d}"
            )
            blob = json.dumps({"outcome": str(outcome), "badges": str(result.summary)}, default=str)
            if settings.api_key:
                assert settings.api_key not in blob, "key leaked into a summary!"
    finally:
        await factory.aclose()
    return 0


async def probe_server(settings: LayaEnvSettings, config_path: Path | None = None) -> int:
    """Pre-flight ``GET /health`` so an unreachable server never costs a decision."""
    config = apply_env_overrides(load_base_config(config_path), settings)
    adapter = LayaApiAdapter(
        config=config.agent.laya_api,
        api_key=settings.api_key,
        decision_timeout_ms=config.agent.decision_timeout_ms,
        redactor=SecretRedactor.from_env_settings(settings),
    )
    print("\npre-flight: probing the Laya server ...")
    try:
        health = await adapter.aprobe_health()
    finally:
        await adapter.aclose()

    if health["ok"]:
        payload = health["payload"]
        print(f"  /health OK  {health['url']}")
        if isinstance(payload, dict):
            for key in ("status", "device", "loaded", "preload", "versions"):
                if key in payload:
                    print(f"    {key}: {payload[key]}")
        return 0

    print(f"  /health FAILED  {health['url']}")
    if health.get("status_code") in (401, 403):
        print("    The server requires a bearer token: set LAYA_API_KEY to the value")
        print("    it was started with, and set auth_required: true in the config.")
        return 3
    print("    Nothing was sent to the decision endpoint. Start the server with:")
    print("      pip install \"laya[serve]\" && LAYA_MODELS=english laya-serve")
    print("    or point LAYA_API_URL at the host that runs it.")
    return 2


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Live Laya API smoke test")
    parser.add_argument("--episodes", type=int, default=0, help="also run N full simulated episodes")
    parser.add_argument("--preset", default="trap_u", choices=["random", "trap_u", "trap_canyon", "trap_twincup"])
    parser.add_argument("--difficulty", default="medium", choices=["easy", "medium", "hard"])
    parser.add_argument("--threshold", type=float, default=0.70)
    parser.add_argument("--config", type=Path, default=None,
                        help="extra YAML layered over the packaged defaults (e.g. configs/laptop_live.yaml)")
    parser.add_argument("--show-raw", action="store_true", help="print the redacted raw response")
    parser.add_argument("--offline", action="store_true", help="rehearse the same flow with mock_laya (no server)")
    parser.add_argument("--health-only", action="store_true", help="pre-flight the server and stop")
    args = parser.parse_args(argv)

    try:
        from dotenv import load_dotenv

        load_dotenv(PROJECT_ROOT / ".env")
    except ImportError:  # pragma: no cover - python-dotenv is a hard dependency
        pass

    settings = LayaEnvSettings()
    redactor = SecretRedactor.from_env_settings(settings)

    if args.offline:
        print("offline rehearsal: no network calls, using the deterministic mock decision layer")
        return asyncio.run(
            run_episodes(
                args.episodes or 2, args.preset, args.difficulty, args.threshold, live=False, config_path=args.config
            )
        )

    if settings.api_key:
        print(f"key loaded: {redactor.redact(settings.api_key)}")
    else:
        print("no LAYA_API_KEY: normal for a laya-serve started without one.")

    # Pre-flight: never spend a decision on a server we cannot reach.
    status = asyncio.run(probe_server(settings, args.config))
    if status != 0 or args.health_only:
        return status

    status = asyncio.run(call_once(settings, redactor, args.show_raw, args.config))
    if status != 0 or args.episodes <= 0:
        return status
    print(f"\nrunning {args.episodes} live episode(s) on {args.difficulty}/{args.preset} ...")
    return asyncio.run(
        run_episodes(
            args.episodes, args.preset, args.difficulty, args.threshold, live=True, config_path=args.config
        )
    )


if __name__ == "__main__":
    raise SystemExit(main())
