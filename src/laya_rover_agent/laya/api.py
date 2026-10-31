"""Live Laya adapter: the decision layer over a self-hosted ``laya-serve``.

Contract implemented (verified against Laya 0.3.26's server, ``laya/serve.py``,
which deliberately speaks the same ``/v1/systemone`` wire protocol):

    POST {LAYA_API_URL}                    # default http://127.0.0.1:8000/v1/systemone
    Authorization: Bearer {LAYA_API_KEY}   # ONLY if the server sets LAYA_API_KEY
    Content-Type: application/json
    {"model": "english", "state": {...}, "questions": {"next_action": {...}}}

    -> {"model": "english",
        "answers": {"next_action": {"type": "choice", "choice": "MOVE_FORWARD",
                                    "confidence": 0.93, "answer_confidence": 0.93,
                                    "probabilities": {...},
                                    "action": {"act_probability": 0.93}}},
        "routing": {"model": "english", "repo": ..., "reason": ...},
        "usage": {"input_tokens": 62, "output_tokens": 0, "state_tokens": 55,
                  "state_tokens_dropped": 0, "truncated": false,
                  "truncated_questions": []}}

Laya returns a **native, calibrated** ``confidence`` for ``choice``/``score``
answers (no synthetic derivation is required, unlike a text-generating model),
and ``answer_confidence`` (= ``max(p)``) on every question type.

Service error codes handled (all fail *closed* into the fallback controller):
401/403 ``AUTH_ERROR``, 402 ``NO_CREDITS`` (dormant for a local server, kept
for a billed proxy), 422 ``SCHEMA_VIOLATION`` (the server refused our question
body - a client bug, never retried), 429 ``RATE_LIMITED``, 503 ``RATE_LIMITED``
with ``Retry-After`` honoured inside the decision budget, 5xx ``API_ERROR``.

Auth is **optional**: a default ``laya-serve`` has no bearer check at all. Set
``auth_required`` when the server does, so a missing key fails closed without a
wasted round-trip per step; the key is sent whenever one is configured.

The legacy chat envelope from spec Section 15.3 stays available via
``request_style="chat"`` and shares the tolerant parser.

Hard guarantees:

* credentials come from :class:`LayaEnvSettings` only - never hard-coded, never
  logged, redacted in *all* error paths including tracebacks;
* :meth:`decide` / :meth:`adecide` **never raise** into the control loop - any
  failure returns an invalid :class:`DecisionCall`, and an API/model error is
  never permission to move;
* the wall-clock budget is a hard ceiling: ``min(request_timeout_s,
  decision_timeout_ms / 1000)`` including retries;
* latency, HTTP status, redacted/truncated raw response, provider model,
  token usage and the truncation flag are recorded for every call.
"""

from __future__ import annotations

import asyncio
import json
import time
from typing import Any, Callable

import httpx

from ..config import LayaApiConfig
from ..security import SecretRedactor
from ..types import Action, DecisionCall, Observation, GuardianReason
from .parsing import parse_response
from .prompt import build_chat_request, build_laya_request

RAW_RESPONSE_LIMIT = 2000
#: HTTP statuses worth another attempt inside the decision budget. 422 is
#: deliberately absent: Laya uses it to refuse a malformed question body, which
#: retrying cannot fix. 503 is Laya's "server busy" answer and carries
#: ``Retry-After``.
RETRYABLE_STATUS = {408, 425, 429, 500, 502, 503, 504}
#: Statuses that mean "the server is saturated right now" - honoured with a
#: ``Retry-After`` delay when the remaining budget allows it.
BUSY_STATUS = {429, 503}


class LayaApiAdapter:
    """Sync + async client for the live Laya decision API."""

    name = "laya_api"

    def __init__(
        self,
        config: LayaApiConfig,
        api_key: str | None,
        decision_timeout_ms: float,
        redactor: SecretRedactor | None = None,
        request_style: str = "typed",
        include_reason_question: bool = True,
        time_source: Callable[[], float] = time.perf_counter,
        transport: httpx.AsyncBaseTransport | httpx.BaseTransport | None = None,
    ) -> None:
        self.config = config
        self.api_key = api_key
        self.decision_timeout_ms = decision_timeout_ms
        self.redactor = redactor or SecretRedactor([api_key])
        self.request_style = request_style
        self.include_reason_question = include_reason_question
        self._time = time_source
        self._transport = transport
        self._async_client: httpx.AsyncClient | None = None
        self._semaphore = asyncio.Semaphore(config.max_concurrent_requests)

        #: hard ceiling for a single decision, including retries
        self.effective_timeout_s = min(
            config.request_timeout_s,
            max(0.05, decision_timeout_ms / 1000.0),
        )

    # ------------------------------------------------------------------ #
    # Request construction
    # ------------------------------------------------------------------ #

    def build_body(self, observation: Observation) -> dict[str, Any]:
        # "chat" is the legacy spec-15.3 envelope; anything else builds the
        # typed-question body that Laya's /v1/systemone expects.
        if self.request_style == "chat":
            return build_chat_request(observation, self.config.model)
        return build_laya_request(observation, self.config.model, include_reason=self.include_reason_question)

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            value = f"{self.config.auth_scheme} {self.api_key}".strip()
            headers[self.config.auth_header] = value
        return headers

    def _client_kwargs(self) -> dict[str, Any]:
        kwargs: dict[str, Any] = {"timeout": self.effective_timeout_s}
        if self._transport is not None:
            kwargs["transport"] = self._transport
        return kwargs

    # ------------------------------------------------------------------ #
    # Async path (default for the experiment runner)
    # ------------------------------------------------------------------ #

    async def adecide(
        self,
        observation: Observation,
        safe_actions: list[str],  # noqa: ARG002 - kept for interface symmetry
    ) -> DecisionCall:
        # Laya on localhost is unauthenticated by default; only an explicit
        # ``auth_required`` deployment turns a missing key into an immediate
        # fail-closed AUTH_ERROR instead of a request the server would reject.
        if self.config.auth_required and not self.api_key:
            return DecisionCall(
                valid=False,
                reason_code=GuardianReason.AUTH_ERROR,
                raw_response="auth_required is set but no LAYA_API_KEY is configured",
            )
        async with self._semaphore:
            return await self._adecide_unlocked(observation)

    async def _adecide_unlocked(self, observation: Observation) -> DecisionCall:
        if self._async_client is None:
            self._async_client = httpx.AsyncClient(**self._client_kwargs())
        body = self.build_body(observation)
        request_text = self._safe_json(body)
        deadline = self._time() + self.effective_timeout_s
        attempts = self.config.max_retries + 1
        last: DecisionCall | None = None

        for attempt in range(attempts):
            started = self._time()
            response: httpx.Response | None = None
            try:
                remaining = max(0.05, deadline - self._time())
                response = await self._async_client.post(
                    self.config.endpoint,
                    headers=self._headers(),
                    json=body,
                    timeout=min(self.effective_timeout_s, remaining),
                )
            except httpx.TimeoutException as exc:
                last = self._failure(GuardianReason.DECISION_TIMEOUT, started, request_text, exc)
            except httpx.HTTPError as exc:
                last = self._failure(GuardianReason.CONNECTION_ERROR, started, request_text, exc)
            except Exception as exc:  # noqa: BLE001 - never leak into the control loop
                last = self._failure(GuardianReason.API_ERROR, started, request_text, exc)
            else:
                last = self._interpret(response, started, request_text)
                if last.valid or last.reason_code not in _RETRYABLE_REASONS:
                    return last
                # A saturated Laya server (503/429) states how long to wait. Only
                # wait when the whole delay still fits inside the decision budget -
                # a rover may not sit still past its deadline.
                delay = _retry_after_seconds(response)
                if delay is not None:
                    if delay >= (deadline - self._time()):
                        # The server asked for longer than the rover can stand
                        # still. Retrying immediately would only draw another
                        # 503, so stop here and let the guardian route to the
                        # fallback controller.
                        return last
                    await asyncio.sleep(delay)

            # retry only if the hard deadline still allows it
            if attempt + 1 >= attempts or self._time() >= deadline:
                break

        return last or DecisionCall(valid=False, reason_code=GuardianReason.API_ERROR)

    async def aprobe_health(self, timeout_s: float = 2.0) -> dict[str, Any]:
        """Probe the Laya server's ``GET /health`` before a mission starts.

        Deployment pre-flight, not a decision path: a rover should learn that
        ``laya-serve`` is unreachable (or still cold-loading a checkpoint)
        *before* it moves, while the fallback controller is still a deliber-
        ate choice rather than an emergency. Never raises.

        With ``LAYA_API_KEY`` set on the server, an unauthenticated caller gets
        the liveness-only payload; the key is sent whenever configured.
        """
        base = httpx.URL(self.config.endpoint)
        probe_url = str(base.copy_with(path="/health", query=None, fragment=None))
        headers = self._headers()
        try:
            async with httpx.AsyncClient(
                timeout=timeout_s,
                transport=self._transport,  # type: ignore[arg-type]
            ) as client:
                response = await client.get(probe_url, headers=headers)
            try:
                payload = response.json()
            except Exception:  # noqa: BLE001 - non-JSON body is still information
                payload = {"status": response.text[:200]}
            # The payload is text an arbitrary server chose; a hostile or buggy
            # deployment can echo the Authorization header back. Redact before it
            # reaches a log, a terminal or a report.
            safe_payload = self.redactor.redact_obj(payload if isinstance(payload, dict) else {"raw": payload})
            return {
                "ok": response.status_code == 200,
                "status_code": response.status_code,
                "url": probe_url,
                "payload": safe_payload,
            }
        except Exception as exc:  # noqa: BLE001 - pre-flight must never raise
            return {
                "ok": False,
                "status_code": None,
                "url": probe_url,
                "error": self.redactor.redact_exception(exc),
            }

    async def aclose(self) -> None:
        if self._async_client is not None:
            await self._async_client.aclose()
            self._async_client = None

    # ------------------------------------------------------------------ #
    # Sync path (smoke test, single-threaded tooling)
    # ------------------------------------------------------------------ #

    def decide(self, observation: Observation, safe_actions: list[str] | None = None) -> DecisionCall:
        try:
            return asyncio.run(self.adecide(observation, safe_actions or list(observation.safe_actions)))
        finally:
            # the loop is closed by asyncio.run; drop the cached client with it
            self._async_client = None

    def decide_with_client(self, observation: Observation) -> DecisionCall:
        """Blocking variant on a private ``httpx.Client`` (no event loop needed)."""
        if self.config.auth_required and not self.api_key:
            return DecisionCall(
                valid=False,
                reason_code=GuardianReason.AUTH_ERROR,
                raw_response="auth_required is set but no LAYA_API_KEY is configured",
            )
        body = self.build_body(observation)
        request_text = self._safe_json(body)
        started = self._time()
        try:
            with httpx.Client(**self._client_kwargs()) as client:
                response = client.post(self.config.endpoint, headers=self._headers(), json=body)
        except httpx.TimeoutException as exc:
            return self._failure(GuardianReason.DECISION_TIMEOUT, started, request_text, exc)
        except httpx.HTTPError as exc:
            return self._failure(GuardianReason.CONNECTION_ERROR, started, request_text, exc)
        except Exception as exc:  # noqa: BLE001
            return self._failure(GuardianReason.API_ERROR, started, request_text, exc)
        return self._interpret(response, started, request_text)

    # ------------------------------------------------------------------ #
    # Response handling
    # ------------------------------------------------------------------ #

    def _interpret(self, response: httpx.Response, started: float, request_text: str) -> DecisionCall:
        latency_ms = (self._time() - started) * 1000.0
        status = response.status_code
        raw = self.redactor.redact(response.text or "")[:RAW_RESPONSE_LIMIT]

        if status == 401 or status == 403:
            return DecisionCall(valid=False, reason_code=GuardianReason.AUTH_ERROR, latency_ms=latency_ms,
                             http_status=status, raw_response=raw, raw_request=request_text)
        if status == 402:
            return DecisionCall(valid=False, reason_code=GuardianReason.NO_CREDITS, latency_ms=latency_ms,
                             http_status=status, raw_response=raw, raw_request=request_text)
        if status in BUSY_STATUS:
            # Laya answers a saturated worker with 503 + Retry-After (and 429 when
            # throttled). Both are transient: retry, then fall back.
            return DecisionCall(valid=False, reason_code=GuardianReason.RATE_LIMITED, latency_ms=latency_ms,
                             http_status=status, raw_response=raw, raw_request=request_text)
        if status == 422:
            # Laya refuses a structurally invalid request this way: a null or
            # duplicate choice label, a too-short score rubric, a non-dict
            # questions block, or a body naming a hook field. Our own bug - it
            # cannot be fixed by retrying, and it must never move the rover.
            return DecisionCall(valid=False, reason_code=GuardianReason.SCHEMA_VIOLATION, latency_ms=latency_ms,
                             http_status=status, raw_response=raw, raw_request=request_text)
        if status >= 500:
            return DecisionCall(valid=False, reason_code=GuardianReason.API_ERROR, latency_ms=latency_ms,
                             http_status=status, raw_response=raw, raw_request=request_text)
        if status >= 400:
            return DecisionCall(valid=False, reason_code=GuardianReason.API_ERROR, latency_ms=latency_ms,
                             http_status=status, raw_response=raw, raw_request=request_text)

        try:
            payload = response.json()
        except Exception:  # noqa: BLE001 - non-JSON body
            payload = response.text

        outcome = parse_response(payload)
        if not outcome.ok or outcome.decision is None:
            reason = (
                GuardianReason.SCHEMA_VIOLATION
                if any(token in (outcome.error or "") for token in ("confidence", "unknown action", "chosen_action"))
                else GuardianReason.MALFORMED_RESPONSE
            )
            return DecisionCall(
                valid=False,
                reason_code=reason,
                latency_ms=latency_ms,
                http_status=status,
                raw_response=raw,
                raw_request=request_text,
                provider_model=outcome.provider_model,
                usage=_clean_usage(outcome.usage),
            )

        decision = outcome.decision
        usage = _clean_usage(outcome.usage)
        truncated = bool(usage.get("truncated")) if usage else False
        if decision.chosen_action not in {Action(a) for a in self._allowed_actions_from_request(request_text)}:
            # Defensive: the criteria map should make this impossible upstream.
            return DecisionCall(
                valid=False,
                reason_code=GuardianReason.SCHEMA_VIOLATION,
                latency_ms=latency_ms,
                http_status=status,
                raw_response=raw,
                raw_request=request_text,
                provider_model=outcome.provider_model,
                usage=usage,
            )

        return DecisionCall(
            valid=True,
            decision=decision,
            latency_ms=latency_ms,
            http_status=status,
            raw_response=raw,
            raw_request=request_text,
            # Laya reports the checkpoint that actually answered, which is the
            # fact worth logging when ``model`` is left on ``auto``.
            provider_model=outcome.provider_model or self.config.model,
            usage=usage,
            truncated=truncated,
        )

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    def _failure(
        self,
        reason: GuardianReason,
        started: float,
        request_text: str,
        exc: BaseException,
    ) -> DecisionCall:
        return DecisionCall(
            valid=False,
            reason_code=reason,
            latency_ms=(self._time() - started) * 1000.0,
            raw_response=self.redactor.redact_exception(exc)[:RAW_RESPONSE_LIMIT],
            raw_request=request_text,
        )

    def _safe_json(self, body: dict[str, Any]) -> str:
        return self.redactor.redact(json.dumps(body, sort_keys=True))[:RAW_RESPONSE_LIMIT]

    @staticmethod
    def _allowed_actions_from_request(request_text: str) -> list[str]:
        try:
            body = json.loads(request_text)
        except json.JSONDecodeError:
            return [a.value for a in Action]
        criteria = (
            body.get("questions", {})
            .get("next_action", {})
            .get("criteria", {})
        )
        return list(criteria.keys()) if isinstance(criteria, dict) and criteria else [a.value for a in Action]


_RETRYABLE_REASONS = {
    GuardianReason.API_ERROR,
    GuardianReason.RATE_LIMITED,
    GuardianReason.CONNECTION_ERROR,
    GuardianReason.DECISION_TIMEOUT,
}


def _clean_usage(usage: dict[str, Any] | None) -> dict[str, Any] | None:
    """Keep Laya's usage record intact, coercing only what is numeric.

    Laya's block mixes types: ``input_tokens``/``output_tokens``/``state_tokens``
    are numbers, ``truncated`` is a bool and ``truncated_questions`` is a list of
    question ids. Numeric strings are coerced to floats; everything else is
    preserved (older code dropped it, which would have hidden the truncation
    flag entirely).
    """
    if not usage:
        return None
    cleaned: dict[str, Any] = {}
    for key, value in usage.items():
        name = str(key)
        if isinstance(value, bool):
            cleaned[name] = value
        elif isinstance(value, (int, float)):
            cleaned[name] = float(value)
        elif isinstance(value, list):
            cleaned[name] = [str(item) for item in value]
        elif isinstance(value, str):
            try:
                cleaned[name] = float(value)
            except ValueError:
                cleaned[name] = value
        elif isinstance(value, dict):
            cleaned[name] = dict(value)
    return cleaned or None


def _retry_after_seconds(response: httpx.Response) -> float | None:
    """``Retry-After`` as seconds, for a busy Laya server (503/429)."""
    if response.status_code not in BUSY_STATUS:
        return None
    raw = response.headers.get("Retry-After")
    if not raw:
        return None
    try:
        return max(0.0, float(raw.strip()))
    except ValueError:
        return None  # HTTP-date form: not worth parsing to save one retry
