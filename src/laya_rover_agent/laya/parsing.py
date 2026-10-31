"""Tolerant response parsing for Laya-shaped payloads.

Handles, in order:

1. the **live Laya shape** -
   ``{"model": ..., "answers": {"next_action": {"choice": ..., "confidence": ...}}}``
   (Laya also attaches ``answer_confidence`` = ``max(p)`` to every answer type,
   which is used when ``confidence`` is absent)
2. the **direct shape** - ``{"chosen_action": ..., "confidence": ..., "reason_code": ...}``
3. **one-level unwrapping** of ``output`` / ``result`` / ``data`` / ``response``
4. **chat-style** ``choices[0].message.content`` where the content is a
   JSON-encoded *string*
5. fenced / prose-wrapped JSON (`````json ... `````)

Anything else classifies as ``MALFORMED_RESPONSE`` and routes to the fallback
controller. Nothing here raises out to the control loop.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from ..types import Action, DecisionResult

_JSON_FENCE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)
_JSON_OBJECT = re.compile(r"\{.*\}", re.DOTALL)

UNWRAP_KEYS: tuple[str, ...] = ("output", "result", "data", "response")


@dataclass
class ParseOutcome:
    """Result of a parse attempt."""

    decision: DecisionResult | None
    error: str | None = None
    provider_model: str | None = None
    usage: dict[str, Any] | None = None
    shape: str = "unknown"
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.decision is not None


class LayaParseError(Exception):
    """Internal signal for a parse failure (never escapes the adapter)."""


# --------------------------------------------------------------------------- #
# Public entry point
# --------------------------------------------------------------------------- #


def parse_response(payload: Any) -> ParseOutcome:
    """Parse a decoded JSON payload (dict) or a raw string into a decision."""
    if isinstance(payload, str):
        decoded = _loads(payload)
        if decoded is None:
            return ParseOutcome(decision=None, error="body is not valid JSON")
        payload = decoded

    if not isinstance(payload, dict):
        return ParseOutcome(decision=None, error=f"unexpected payload type {type(payload).__name__}")

    model = payload.get("model") if isinstance(payload.get("model"), str) else None
    if model is None:
        # Laya routes per request and explains the choice in ``routing``; when the
        # request said ``model: auto`` this is the only place the answering
        # checkpoint is named. Recorded so a run can prove which model decided.
        routing = payload.get("routing")
        if isinstance(routing, dict) and isinstance(routing.get("model"), str):
            model = routing["model"]
    usage = payload.get("usage") if isinstance(payload.get("usage"), dict) else None

    # -- 1. live Laya answers map --------------------------------------- #
    answers = payload.get("answers")
    answers_error: str | None = None
    if isinstance(answers, dict):
        outcome = _from_answers(answers, model, usage)
        if outcome.ok:
            return outcome
        # Remember the most specific diagnostic: an answers map that failed
        # validation names exactly what was wrong with the decision, where the
        # later fallback shapes can only say "missing chosen_action field".
        answers_error = outcome.error

    # -- 2. direct shape ---------------------------------------------------- #
    outcome = _from_flat_dict(payload, model, usage, shape="direct")
    if outcome.ok:
        return outcome

    # -- 3. one-level unwrap ------------------------------------------------ #
    for key in UNWRAP_KEYS:
        inner = payload.get(key)
        if isinstance(inner, str):
            decoded = _loads(inner)
            if isinstance(decoded, dict):
                inner = decoded
        if isinstance(inner, dict):
            nested = _from_flat_dict(inner, model, usage, shape=f"wrapped:{key}")
            if nested.ok:
                return nested
            if isinstance(inner.get("answers"), dict):
                nested_answers = _from_answers(inner["answers"], model, usage)
                if nested_answers.ok:
                    return nested_answers

    # -- 4. chat-style choices ---------------------------------------------- #
    choices = payload.get("choices")
    if isinstance(choices, list) and choices:
        first = choices[0]
        content: Any = None
        if isinstance(first, dict):
            message = first.get("message")
            if isinstance(message, dict):
                content = message.get("content")
            content = content if content is not None else first.get("text")
        if isinstance(content, str):
            nested = parse_response(content)
            if nested.ok:
                nested.shape = "chat:choices"
                return nested
            return ParseOutcome(
                decision=None,
                error=f"chat content unparseable: {nested.error}",
                provider_model=model,
                usage=usage,
            )

    return ParseOutcome(
        decision=None,
        error=answers_error or outcome.error or "no recognised decision fields present",
        provider_model=model,
        usage=usage,
    )


# --------------------------------------------------------------------------- #
# Internals
# --------------------------------------------------------------------------- #


def _from_answers(answers: dict[str, Any], model: str | None, usage: dict[str, Any] | None) -> ParseOutcome:
    """Live Laya contract: answers[<question id>] = {type, choice, confidence}."""
    for key in ("next_action", "action", "chosen_action", "decision"):
        entry = answers.get(key)
        if isinstance(entry, dict) and "choice" in entry:
            probabilities = entry.get("probabilities")
            # Laya always returns ``confidence`` for a choice answer and adds
            # ``answer_confidence`` (= max(p)) on every question type; prefer the
            # former, fall back to the latter, and fail closed when both are
            # absent rather than inventing a number.
            confidence = entry.get("confidence")
            if confidence is None:
                confidence = entry.get("answer_confidence")
            decision = _validate(
                action_raw=entry.get("choice"),
                confidence=confidence,
                reason_raw=_reason_from_answers(answers),
                probabilities=probabilities if isinstance(probabilities, dict) else None,
                require_confidence=True,
            )
            if isinstance(decision, DecisionResult):
                return ParseOutcome(decision=decision, provider_model=model, usage=usage, shape="laya")
            return ParseOutcome(
                decision=None,
                error=decision,
                provider_model=model,
                usage=usage,
                shape="laya",
            )
    return ParseOutcome(decision=None, error="answers map without a usable choice entry", provider_model=model, usage=usage)


def _reason_from_answers(answers: dict[str, Any]) -> str | None:
    for key in ("reason_code", "reason"):
        entry = answers.get(key)
        if isinstance(entry, dict) and isinstance(entry.get("choice"), str):
            return entry["choice"]
        if isinstance(entry, str):
            return entry
    return None


def _from_flat_dict(
    payload: dict[str, Any],
    model: str | None,
    usage: dict[str, Any] | None,
    shape: str,
) -> ParseOutcome:
    action_raw = payload.get("chosen_action", payload.get("action"))
    if action_raw is None:
        return ParseOutcome(decision=None, error="missing chosen_action field")
    decision = _validate(
        action_raw=action_raw,
        confidence=payload.get("confidence"),
        reason_raw=payload.get("reason_code"),
        probabilities=None,
        require_confidence=False,  # legacy envelope: action-only replies allowed
    )
    if isinstance(decision, DecisionResult):
        return ParseOutcome(decision=decision, provider_model=model, usage=usage, shape=shape)
    return ParseOutcome(decision=None, error=decision, provider_model=model, usage=usage, shape=shape)


def _validate(
    action_raw: Any,
    confidence: Any,
    reason_raw: Any,
    probabilities: dict[str, Any] | None,
    require_confidence: bool = False,
) -> DecisionResult | str:
    """Validate one decision. ``require_confidence`` is the fail-closed switch.

    On the typed Laya path a decision without a numeric confidence is rejected
    outright: defaulting to 1.0 would mean *maximum trust* for a provider that
    said nothing, which is the opposite of what FR-6 requires. The legacy
    chat/direct shapes keep the permissive default because spec Section 15.3
    permits a provider that returns only an action.
    """
    if not isinstance(action_raw, str):
        return f"chosen_action must be a string, got {type(action_raw).__name__}"
    candidate = action_raw.strip().upper()
    try:
        action = Action(candidate)
    except ValueError:
        return f"unknown action {action_raw!r} (not in the finite action set)"
    if confidence is None and require_confidence:
        return "confidence is missing on a typed choice answer (no answer_confidence fallback)"
    try:
        confidence_value = float(confidence) if confidence is not None else 1.0
    except (TypeError, ValueError):
        return f"confidence is not numeric: {confidence!r}"
    if not 0.0 <= confidence_value <= 1.0:
        return f"confidence {confidence_value} outside [0.0, 1.0]"
    probs: dict[str, float] | None = None
    if probabilities:
        try:
            probs = {str(k): float(v) for k, v in probabilities.items()}
        except (TypeError, ValueError):
            probs = None
    return DecisionResult(
        chosen_action=action,
        confidence=confidence_value,
        reason_code=str(reason_raw) if isinstance(reason_raw, str) else None,
        probabilities=probs,
    )


def _loads(text: str) -> Any:
    """JSON parse with fenced/prose extraction fallbacks."""
    text = text.strip()
    if not text:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    fenced = _JSON_FENCE.search(text)
    if fenced:
        try:
            return json.loads(fenced.group(1))
        except json.JSONDecodeError:
            pass
    loose = _JSON_OBJECT.search(text)
    if loose:
        try:
            return json.loads(loose.group(0))
        except json.JSONDecodeError:
            pass
    return None
