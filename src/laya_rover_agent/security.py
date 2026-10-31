"""Secret redaction utilities (Security requirement, FR-6 / Section 7).

The API key must never appear in *any* log, including exception messages and
stack traces. Redaction wraps the whole error-handling path, not just the
happy-path logger, and is covered by an explicit test.
"""

from __future__ import annotations

import re
from typing import Any, Iterable

REDACTED = "***REDACTED***"

#: Keys whose values are secrets regardless of context.
SECRET_ENV_KEYS: tuple[str, ...] = ("LAYA_API_KEY",)

#: Generic credential shapes: any bearer token, any ``api_key=``/``apiKey:``
#: assignment, and the common ``sk-`` key prefix. The configured ``LAYA_API_KEY``
#: is additionally redacted by exact value, whatever it looks like - this list is
#: the second layer for credentials the process was never told about.
_SECRET_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"sk-[A-Za-z0-9_\-]{8,}"),
    re.compile(r"(?i)\b(bearer)\s+([A-Za-z0-9._\-]{8,})"),
    re.compile(r"(?i)\b(api[_-]?key\"?\s*[:=]\s*\"?)([A-Za-z0-9._\-]{8,})"),
)


class SecretRedactor:
    """Holds the live secret(s) for a process and scrubs arbitrary text.

    Instances are created once by the CLI/entry point and injected downward;
    library modules never scan the environment themselves.
    """

    def __init__(self, secrets: Iterable[str | None] = ()) -> None:
        self._secrets: tuple[str, ...] = tuple(
            sorted({s for s in secrets if s and len(s) >= 4}, key=len, reverse=True)
        )

    @classmethod
    def from_env_settings(cls, settings: Any) -> "SecretRedactor":
        return cls([getattr(settings, "api_key", None)])

    @property
    def has_secrets(self) -> bool:
        return bool(self._secrets)

    def redact(self, text: str) -> str:
        if not text:
            return text
        out = text
        for secret in self._secrets:
            out = out.replace(secret, REDACTED)
        for pattern in _SECRET_PATTERNS:
            if pattern.groups >= 2:
                # keep the label, drop the credential
                out = pattern.sub(lambda m: f"{m.group(1)} {REDACTED}", out)
            else:
                out = pattern.sub(REDACTED, out)
        return out

    def redact_obj(self, obj: Any) -> Any:
        """Recursively redact strings inside dicts / lists / tuples."""
        if isinstance(obj, str):
            return self.redact(obj)
        if isinstance(obj, dict):
            return {
                k: (REDACTED if str(k).upper() in SECRET_ENV_KEYS else self.redact_obj(v))
                for k, v in obj.items()
            }
        if isinstance(obj, (list, tuple)):
            return [self.redact_obj(v) for v in obj]
        return obj

    def redact_exception(self, exc: BaseException) -> str:
        """Render an exception (message + chain) with every secret removed."""
        parts: list[str] = []
        seen: set[int] = set()
        current: BaseException | None = exc
        while current is not None and id(current) not in seen:
            seen.add(id(current))
            parts.append(f"{type(current).__name__}: {current}")
            current = current.__cause__ or current.__context__
        return self.redact(" <- ".join(parts))
