"""Laya decision layer: adapter protocol, live client, mock, prompt, parsing.

:mod:`~laya_rover_agent.laya.adapter` is the provider-neutral seam
(:class:`DecisionAdapter`) - import from it in new code. The provider-specific
pieces are re-exported here for the CLI, the runner and the tests.
"""

from .adapter import DecisionAdapter
from .api import LayaApiAdapter
from .mock import MockLayaAdapter
from .parsing import parse_response
from .prompt import (
    ACTION_QUESTION_ID,
    REASON_CODES,
    REASON_QUESTION_ID,
    SYSTEM_PROMPT,
    build_laya_request,
)

__all__ = [
    "DecisionAdapter",
    "LayaApiAdapter",
    "MockLayaAdapter",
    "parse_response",
    "ACTION_QUESTION_ID",
    "REASON_QUESTION_ID",
    "REASON_CODES",
    "SYSTEM_PROMPT",
    "build_laya_request",
]
