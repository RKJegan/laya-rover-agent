"""Layer 2 - secret redaction (never log a ROVER_* key, including in tracebacks)."""

from __future__ import annotations

import json

import pytest

from laya_rover_agent.security import REDACTED, SecretRedactor

KEY = "jv_live_9f3c0a71b2e448d5test"


def test_exact_secret_is_scrubbed():
    redactor = SecretRedactor([KEY])
    assert KEY not in redactor.redact(f"Authorization: Bearer {KEY}")


@pytest.mark.parametrize(
    "text",
    [
        '{"headers": {"Authorization": "Bearer jv_live_abcdef1234567890"}}',
        "api_key=jv_live_abcdef1234567890",
        "x-api-key: jv_live_abcdef1234567890",
        "token sk-proj-abcdefghijklmnopqrstuvwxyz",
    ],
)
def test_known_patterns_are_scrubbed_without_being_registered(text):
    redactor = SecretRedactor([])
    assert "jv_live_" not in redactor.redact(text)
    assert "sk-proj-" not in redactor.redact(text)
    assert REDACTED in redactor.redact(text)


def test_redact_object_walks_nested_structures():
    redactor = SecretRedactor([KEY])
    payload = {"outer": [{"Authorization": f"Bearer {KEY}"}, f"key={KEY}"], "keep": 3}
    scrubbed = redactor.redact_obj(payload)
    assert KEY not in json.dumps(scrubbed)
    assert scrubbed["keep"] == 3


def test_secret_env_keys_are_blanked_whatever_their_value():
    redactor = SecretRedactor([])
    scrubbed = redactor.redact_obj({"LAYA_API_KEY": "anything-at-all", "other": "visible"})
    assert scrubbed["LAYA_API_KEY"] == REDACTED
    assert scrubbed["other"] == "visible"


def test_traceback_redaction_keeps_the_type_and_message():
    def boom():
        raise RuntimeError(f"request failed with key {KEY}")

    try:
        boom()
    except RuntimeError as exc:
        rendered = SecretRedactor([KEY]).redact_exception(exc)

    assert KEY not in rendered
    assert "RuntimeError" in rendered
    assert "request failed with key" in rendered


def test_chained_exceptions_are_redacted():
    redactor = SecretRedactor([KEY])
    try:
        try:
            raise ValueError(f"inner {KEY}")
        except ValueError as inner:
            raise RuntimeError(f"outer {KEY}") from inner
    except RuntimeError as exc:
        rendered = redactor.redact_exception(exc)

    assert KEY not in rendered
    assert "inner" in rendered and "outer" in rendered


def test_short_values_are_ignored_to_avoid_mangling_output():
    redactor = SecretRedactor(["ab"])
    assert redactor.redact("absolutely fine") == "absolutely fine"
    assert not redactor.has_secrets
