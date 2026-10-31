"""Layer 2 - live Laya adapter against a local fake server (FR-6 / Section 15).

No network is touched: ``httpx.MockTransport`` stands in for the real endpoint,
so the request *contract* (endpoint, auth header, body shape, response parsing,
error handling, redaction) is tested exhaustively and offline.
"""

from __future__ import annotations

import json

import httpx
import pytest

from conftest import make_map
from laya_rover_agent.laya.api import LayaApiAdapter
from laya_rover_agent.laya.parsing import parse_response
from laya_rover_agent.perception import Perception
from laya_rover_agent.safety import SafetyGuardian
from laya_rover_agent.security import SecretRedactor
from laya_rover_agent.types import Action, GuardianReason, RoverState, Status

API_KEY = "jv_live_test_key_1234567890"


def _observation(quick_config, generator):
    grid = make_map(generator, preset="trap_u", seed=1)
    guardian = SafetyGuardian(grid, quick_config.world, quick_config.rover)
    state = RoverState(
        x=grid.start[0], y=grid.start[1], heading="E", battery=1.0,
        energy_used=0.0, steps=0, status=Status.ACTIVE, consecutive_stops=0,
    )
    safe = guardian.safe_actions(state)
    observation = Perception(quick_config.world, quick_config.rover).observe(grid, state, safe.actions, None)
    return observation, safe.actions


def _adapter(quick_config, handler, *, api_key: str | None = API_KEY, api_config=None, **kwargs):
    return LayaApiAdapter(
        config=api_config or quick_config.agent.laya_api,
        api_key=api_key,
        decision_timeout_ms=quick_config.agent.decision_timeout_ms,
        redactor=SecretRedactor([api_key]),
        transport=httpx.MockTransport(handler),
        **kwargs,
    )


def _ok_payload(action: str, confidence: float = 0.83) -> dict:
    return {
        "model": "english",
        "answers": {
            "next_action": {
                "type": "choice",
                "choice": action,
                "confidence": confidence,
                "probabilities": {action: confidence, "MOVE_FORWARD": 1 - confidence},
            }
        },
        "usage": {"input_tokens": 321, "cost_usd": 0.0004, "credits_remaining_usd": 12.5},
    }


# --------------------------------------------------------------------------- #
# Request contract
# --------------------------------------------------------------------------- #


def test_request_matches_the_laya_contract(quick_config, generator):
    observation, safe_actions = _observation(quick_config, generator)
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["headers"] = dict(request.headers)
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, json=_ok_payload("MOVE_FORWARD"))

    import asyncio

    result = asyncio.run(_adapter(quick_config, handler).adecide(observation, [a.value for a in safe_actions]))

    assert captured["url"] == quick_config.agent.laya_api.endpoint
    assert captured["headers"]["authorization"] == f"Bearer {API_KEY}"
    body = captured["body"]
    assert body["model"] == quick_config.agent.laya_api.model
    assert isinstance(body["state"], (dict, str, list))
    question = body["questions"]["next_action"]
    assert question["type"] == "choice"
    assert set(question["criteria"]).issubset({a.value for a in Action})
    assert 2 <= len(question["criteria"]) <= 255
    assert result.valid
    assert result.decision.chosen_action is Action.MOVE_FORWARD
    assert result.decision.confidence == pytest.approx(0.83)
    assert result.usage["input_tokens"] == 321
    assert result.provider_model == "english"


def test_state_is_numeric_and_self_describing(quick_config, generator):
    observation, _ = _observation(quick_config, generator)
    adapter = _adapter(quick_config, lambda request: httpx.Response(200, json=_ok_payload("MOVE_FORWARD")))
    body = adapter.build_body(observation)
    state = body["state"]
    assert "coordinate_frame" in json.dumps(state)
    assert "null" not in json.dumps(state["observations"]) if "observations" in state else True


def test_no_key_is_normal_for_a_local_laya_server(quick_config, generator):
    """A default ``laya-serve`` has no auth at all: no key must still decide.

    This is the deliberate difference from a hosted, key-authenticated API,
    where an absent key was an immediate AUTH_ERROR. Laya runs on the rover's own host,
    so "no credentials" is a valid deployment - and the request must simply
    carry no Authorization header.
    """
    observation, safe_actions = _observation(quick_config, generator)
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["headers"] = dict(request.headers)
        return httpx.Response(200, json=_ok_payload("MOVE_FORWARD"))

    import asyncio

    result = asyncio.run(
        _adapter(quick_config, handler, api_key=None).adecide(observation, [a.value for a in safe_actions])
    )
    assert result.valid and result.decision.chosen_action is Action.MOVE_FORWARD
    assert "authorization" not in captured["headers"]


def test_key_is_sent_when_the_server_is_authenticated(quick_config, generator):
    """A secured laya-serve gets the bearer token; auth stays optional, not ignored."""
    from laya_rover_agent.config import LayaApiConfig

    observation, safe_actions = _observation(quick_config, generator)
    secured = LayaApiConfig(**{**quick_config.agent.laya_api.model_dump(), "auth_required": True})
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["headers"] = dict(request.headers)
        return httpx.Response(200, json=_ok_payload("MOVE_FORWARD"))

    import asyncio

    result = asyncio.run(
        _adapter(quick_config, handler, api_config=secured).adecide(observation, [a.value for a in safe_actions])
    )
    assert result.valid, result.reason_code
    assert captured["headers"]["authorization"] == f"Bearer {API_KEY}"


def test_required_auth_without_a_key_fails_closed_before_calling_out(quick_config, generator):
    """With ``auth_required`` the missing key is caught locally - no wasted round trip."""
    from laya_rover_agent.config import LayaApiConfig

    observation, safe_actions = _observation(quick_config, generator)
    secured = LayaApiConfig(**{**quick_config.agent.laya_api.model_dump(), "auth_required": True})
    called = False

    def handler(request: httpx.Request) -> httpx.Response:  # pragma: no cover - must not run
        nonlocal called
        called = True
        return httpx.Response(200, json=_ok_payload("MOVE_FORWARD"))

    import asyncio

    result = asyncio.run(
        _adapter(quick_config, handler, api_key=None, api_config=secured).adecide(
            observation, [a.value for a in safe_actions]
        )
    )
    assert not result.valid
    assert result.reason_code is GuardianReason.AUTH_ERROR
    assert not called


# --------------------------------------------------------------------------- #
# Response tolerance + validation
# --------------------------------------------------------------------------- #


def test_flat_chat_style_envelope_is_still_accepted():
    """Spec 15.3 shape A - kept as the documented fallback shape."""
    outcome = parse_response({"chosen_action": "TURN_LEFT_AND_MOVE", "confidence": 0.66, "reason_code": "CLEAR_PATH"})
    assert outcome.decision.chosen_action is Action.TURN_LEFT_AND_MOVE
    assert outcome.decision.confidence == pytest.approx(0.66)


def test_nested_chat_style_envelope_is_still_accepted():
    """Spec 15.3 shape B - the payload wrapped in a provider envelope."""
    payload = {
        "id": "chatcmpl-1",
        "choices": [{"message": {"content": json.dumps({"chosen_action": "REVERSE", "confidence": 0.55})}}],
    }
    outcome = parse_response(payload)
    assert outcome.decision.chosen_action is Action.REVERSE


def test_fenced_json_is_tolerated():
    outcome = parse_response({"output": "```json\n{\"chosen_action\": \"MOVE_FORWARD\", \"confidence\": 0.9}\n```"})
    assert outcome.decision.chosen_action is Action.MOVE_FORWARD


def test_unknown_action_is_rejected():
    outcome = parse_response({"chosen_action": "LAUNCH_THRUSTERS", "confidence": 0.99})
    assert outcome.decision is None
    assert "not in the finite action set" in (outcome.error or "")


def test_out_of_range_confidence_is_rejected():
    outcome = parse_response({"chosen_action": "MOVE_FORWARD", "confidence": 1.7})
    assert outcome.decision is None
    assert outcome.error


def test_free_text_is_never_executed(quick_config, generator):
    """A response with no machine-readable action must fall back, never guess."""
    observation, safe_actions = _observation(quick_config, generator)
    adapter = _adapter(
        quick_config,
        lambda request: httpx.Response(200, json={"answers": {"next_action": {"type": "choice", "choice": None}}}),
    )
    import asyncio

    result = asyncio.run(adapter.adecide(observation, [a.value for a in safe_actions]))
    assert not result.valid


# --------------------------------------------------------------------------- #
# Errors, retries, latency, redaction
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("status", "expected"),
    [
        (401, GuardianReason.AUTH_ERROR),
        (403, GuardianReason.AUTH_ERROR),
        (402, GuardianReason.NO_CREDITS),
        (429, GuardianReason.RATE_LIMITED),
        (500, GuardianReason.API_ERROR),
        (502, GuardianReason.API_ERROR),
    ],
)
def test_http_errors_are_classified_and_never_raise(quick_config, generator, status, expected):
    observation, safe_actions = _observation(quick_config, generator)
    adapter = _adapter(quick_config, lambda request: httpx.Response(status, json={"detail": "nope"}))
    import asyncio

    result = asyncio.run(adapter.adecide(observation, [a.value for a in safe_actions]))
    assert not result.valid
    assert result.reason_code is expected
    assert result.http_status == status


def test_502_is_retried_then_succeeds(quick_config, generator):
    observation, safe_actions = _observation(quick_config, generator)
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(502, json={"detail": "upstream"})
        return httpx.Response(200, json=_ok_payload("TURN_RIGHT_AND_MOVE"))

    import asyncio

    adapter = _adapter(
        quick_config,
        handler,
        api_config=quick_config.agent.laya_api.model_copy(update={"max_retries": 2}),
    )
    result = asyncio.run(adapter.adecide(observation, [a.value for a in safe_actions]))
    assert calls["n"] == 2
    assert result.valid and result.decision.chosen_action is Action.TURN_RIGHT_AND_MOVE


def test_connection_error_falls_back(quick_config, generator):
    observation, safe_actions = _observation(quick_config, generator)

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("boom", request=request)

    import asyncio

    result = asyncio.run(_adapter(quick_config, handler).adecide(observation, [a.value for a in safe_actions]))
    assert not result.valid
    assert result.reason_code is GuardianReason.CONNECTION_ERROR


def test_timeout_is_bounded_by_the_decision_timeout(quick_config, generator):
    observation, safe_actions = _observation(quick_config, generator)

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("slow", request=request)

    import asyncio

    adapter = _adapter(quick_config, handler)
    result = asyncio.run(adapter.adecide(observation, [a.value for a in safe_actions]))
    assert not result.valid
    assert result.reason_code is GuardianReason.DECISION_TIMEOUT
    assert adapter.effective_timeout_s <= quick_config.agent.decision_timeout_ms / 1000.0


def test_api_key_never_leaks_into_the_result(quick_config, generator):
    observation, safe_actions = _observation(quick_config, generator)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={"detail": f"bad key {API_KEY}"})

    import asyncio

    result = asyncio.run(_adapter(quick_config, handler).adecide(observation, [a.value for a in safe_actions]))
    blob = json.dumps(
        {
            "raw_response": result.raw_response,
            "raw_request": result.raw_request,
            "reason": str(result.reason_code),
        },
        default=str,
    )
    assert API_KEY not in blob
    assert "jv_live_" not in blob


def test_malformed_json_body_is_reported_as_malformed(quick_config, generator):
    observation, safe_actions = _observation(quick_config, generator)
    adapter = _adapter(quick_config, lambda request: httpx.Response(200, content=b"{not json"))
    import asyncio

    result = asyncio.run(adapter.adecide(observation, [a.value for a in safe_actions]))
    assert not result.valid
    assert result.reason_code is GuardianReason.MALFORMED_RESPONSE


# --------------------------------------------------------------------------- #
# Laya-specific contract: status codes, busy handling, usage, truncation
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("status", "expected"),
    [
        (401, GuardianReason.AUTH_ERROR),
        (403, GuardianReason.AUTH_ERROR),
        # Laya refuses a structurally invalid question body with 422. Retrying
        # cannot fix it, and it must never move the rover.
        (422, GuardianReason.SCHEMA_VIOLATION),
        # Saturated server: 503 carries Retry-After, 429 is throttling.
        (503, GuardianReason.RATE_LIMITED),
        (429, GuardianReason.RATE_LIMITED),
        (500, GuardianReason.API_ERROR),
    ],
)
def test_laya_status_codes_are_classified(quick_config, generator, status, expected):
    observation, safe_actions = _observation(quick_config, generator)
    adapter = _adapter(quick_config, lambda request: httpx.Response(status, text="nope"))

    import asyncio

    result = asyncio.run(adapter.adecide(observation, [a.value for a in safe_actions]))
    assert not result.valid
    assert result.reason_code is expected
    assert result.http_status == status


def test_busy_server_is_retried_after_retry_after(quick_config, generator):
    """A 503 with a short ``Retry-After`` is ridden out inside the decision budget."""
    observation, safe_actions = _observation(quick_config, generator)
    attempts = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        attempts["n"] += 1
        if attempts["n"] == 1:
            return httpx.Response(503, headers={"Retry-After": "0"}, text="busy")
        return httpx.Response(200, json=_ok_payload("TURN_RIGHT_AND_MOVE"))

    import asyncio

    config = quick_config.agent.laya_api.model_copy(update={"max_retries": 1})
    result = asyncio.run(
        _adapter(quick_config, handler, api_config=config).adecide(observation, [a.value for a in safe_actions])
    )
    assert attempts["n"] == 2
    assert result.valid and result.decision.chosen_action is Action.TURN_RIGHT_AND_MOVE


def test_retry_after_beyond_the_deadline_is_not_waited_out(quick_config, generator):
    """A rover may not stand still past its decision budget: fall back instead."""
    observation, safe_actions = _observation(quick_config, generator)
    attempts = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        attempts["n"] += 1
        return httpx.Response(503, headers={"Retry-After": "30"}, text="busy")

    import asyncio

    config = quick_config.agent.laya_api.model_copy(update={"max_retries": 3})
    result = asyncio.run(
        _adapter(quick_config, handler, api_config=config).adecide(observation, [a.value for a in safe_actions])
    )
    assert attempts["n"] == 1, "must not sleep past the decision deadline"
    assert not result.valid and result.reason_code is GuardianReason.RATE_LIMITED


def test_answer_confidence_is_used_when_confidence_is_absent():
    """Laya attaches ``answer_confidence`` (= max(p)) to every question type."""
    outcome = parse_response(
        {
            "model": "english",
            "answers": {"next_action": {"type": "choice", "choice": "REVERSE", "answer_confidence": 0.71}},
        }
    )
    assert outcome.ok
    assert outcome.decision.chosen_action is Action.REVERSE
    assert outcome.decision.confidence == pytest.approx(0.71)


def test_missing_confidence_on_a_typed_answer_fails_closed():
    """A typed choice answer with no confidence must NOT be trusted by default.

    The old permissive default was 1.0 - i.e. maximum confidence for a provider
    that said nothing. On the typed path that is now a schema violation.
    """
    outcome = parse_response({"model": "english", "answers": {"next_action": {"type": "choice", "choice": "MOVE_FORWARD"}}})
    assert not outcome.ok
    assert "confidence" in (outcome.error or "")


def test_legacy_envelope_may_still_omit_confidence():
    """Spec 15.3 permits an action-only reply on the legacy chat/direct shape."""
    outcome = parse_response({"chosen_action": "MOVE_FORWARD"})
    assert outcome.ok and outcome.decision.confidence == pytest.approx(1.0)


def test_usage_block_keeps_non_numeric_fields(quick_config, generator):
    """``truncated`` (bool) and ``truncated_questions`` (list) must survive."""
    observation, safe_actions = _observation(quick_config, generator)
    payload = _ok_payload("MOVE_FORWARD")
    payload["usage"] = {
        "input_tokens": 321, "output_tokens": 0, "state_tokens": 300,
        "state_tokens_dropped": 21, "truncated": True,
        "truncated_questions": ["next_action"],
    }
    adapter = _adapter(quick_config, lambda request: httpx.Response(200, json=payload))

    import asyncio

    result = asyncio.run(adapter.adecide(observation, [a.value for a in safe_actions]))
    assert result.valid
    assert result.usage["input_tokens"] == 321.0
    assert result.usage["truncated"] is True
    assert result.usage["truncated_questions"] == ["next_action"]
    # and the decision-relevant part is lifted out for the safety record
    assert result.truncated is True


def test_untruncated_decision_reports_false(quick_config, generator):
    observation, safe_actions = _observation(quick_config, generator)
    adapter = _adapter(quick_config, lambda request: httpx.Response(200, json=_ok_payload("MOVE_FORWARD")))

    import asyncio

    result = asyncio.run(adapter.adecide(observation, [a.value for a in safe_actions]))
    assert result.truncated is False


def test_routing_names_the_answering_checkpoint_when_model_is_auto(quick_config, generator):
    """With ``model: auto`` Laya's ``routing`` block is the only record of which
    checkpoint decided; it must reach ``provider_model`` for the run metadata."""
    observation, safe_actions = _observation(quick_config, generator)
    payload = _ok_payload("MOVE_FORWARD")
    payload.pop("model")
    payload["routing"] = {"model": "multilingual", "repo": "convaiinnovations/laya/multilingual", "reason": "script"}

    auto = quick_config.agent.laya_api.model_copy(update={"model": "auto"})
    adapter = _adapter(quick_config, lambda request: httpx.Response(200, json=payload), api_config=auto)

    import asyncio

    result = asyncio.run(adapter.adecide(observation, [a.value for a in safe_actions]))
    assert result.valid
    assert result.provider_model == "multilingual"


def test_request_uses_the_laya_endpoint_and_model(quick_config, generator):
    observation, safe_actions = _observation(quick_config, generator)
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, json=_ok_payload("MOVE_FORWARD"))

    import asyncio

    asyncio.run(_adapter(quick_config, handler).adecide(observation, [a.value for a in safe_actions]))
    assert captured["url"] == "http://127.0.0.1:8000/v1/systemone"
    assert captured["body"]["model"] == "english"


# --------------------------------------------------------------------------- #
# Deployment pre-flight: the health probe must never raise
# --------------------------------------------------------------------------- #


def test_health_probe_reports_a_live_server(quick_config):
    adapter = _adapter(
        quick_config,
        lambda request: httpx.Response(200, json={"status": "ok", "device": "cpu", "loaded": ["english"]}),
    )

    import asyncio

    health = asyncio.run(adapter.aprobe_health())
    assert health["ok"] is True
    assert health["url"].endswith("/health")
    assert health["payload"]["device"] == "cpu"


def test_health_probe_reports_an_unreachable_server_without_raising(quick_config):
    def explode(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    adapter = _adapter(quick_config, explode)

    import asyncio

    health = asyncio.run(adapter.aprobe_health())
    assert health["ok"] is False
    assert "error" in health


def test_health_probe_never_leaks_the_key(quick_config):
    adapter = _adapter(
        quick_config,
        lambda request: httpx.Response(503, text=f"upstream rejected Bearer {API_KEY}"),
    )

    import asyncio

    health = asyncio.run(adapter.aprobe_health())
    assert health["ok"] is False
    assert API_KEY not in json.dumps(health)


# --------------------------------------------------------------------------- #
# Environment precedence (LAYA_* beats the YAML)
# --------------------------------------------------------------------------- #


def test_env_overrides_reach_the_effective_config(monkeypatch, quick_config):
    """`.env` is the natural place to point at your own laya-serve host.

    A value that is recorded in run_metadata.json but ignored by the adapter
    would make the metadata lie about which model answered, so this is asserted
    rather than assumed.
    """
    from laya_rover_agent.config import LayaEnvSettings, apply_env_overrides

    monkeypatch.setenv("LAYA_API_URL", "http://gpu-box.local:9000/v1/systemone")
    monkeypatch.setenv("LAYA_MODEL", "auto")
    monkeypatch.setenv("LAYA_REQUEST_TIMEOUT_S", "7.5")
    monkeypatch.setenv("LAYA_MAX_RETRIES", "2")
    monkeypatch.setenv("LAYA_AUTH_REQUIRED", "true")

    env = LayaEnvSettings(_env_file=None)
    effective = apply_env_overrides(quick_config, env)

    assert effective.agent.laya_api.endpoint == "http://gpu-box.local:9000/v1/systemone"
    assert effective.agent.laya_api.model == "auto"
    assert effective.agent.laya_api.request_timeout_s == pytest.approx(7.5)
    assert effective.agent.laya_api.max_retries == 2
    assert effective.agent.laya_api.auth_required is True
    # the YAML is untouched - overrides are a derived config, not a mutation
    assert quick_config.agent.laya_api.endpoint != "http://gpu-box.local:9000/v1/systemone"


def test_env_does_not_leak_the_key_into_the_config(monkeypatch, quick_config):
    """The key must never enter the hashed config, so it can never enter a log."""
    from laya_rover_agent.config import LayaEnvSettings, apply_env_overrides

    monkeypatch.setenv("LAYA_API_KEY", API_KEY)
    env = LayaEnvSettings(_env_file=None)
    effective = apply_env_overrides(quick_config, env)
    assert API_KEY not in effective.model_dump_json()
    assert env.api_key == API_KEY  # the adapter still gets it, separately


def test_hardware_env_vars_keep_the_rover_prefix(monkeypatch):
    """Laya Rover is the product; its bridge variables keep their own prefix."""
    from laya_rover_agent.config import LayaEnvSettings

    monkeypatch.setenv("ROVER_BACKEND", "serial")
    monkeypatch.setenv("ROVER_SERIAL_PORT", "/dev/ttyACM0")
    monkeypatch.setenv("LAYA_MODEL", "multilingual")
    env = LayaEnvSettings(_env_file=None)
    assert env.rover_backend == "serial"
    assert env.serial_port == "/dev/ttyACM0"
    assert env.model == "multilingual"
