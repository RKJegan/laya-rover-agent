"""Laya typed-question template + request builders (Section 8, Section 15.3).

Two request styles are supported:

``typed`` (default, and the real contract of a self-hosted Laya server)
    A ``choice`` question whose ``criteria`` keys *are* the allowed actions.
    Because the answer space is declared in the request, the model structurally
    cannot return an undeclared action - there is no JSON to repair and no
    regex to write. This is the strongest possible form of the "typed action
    from a finite action set" requirement in FR-3.

``chat`` (legacy / other providers, per spec Section 15.3)
    ``{model, system, input, response_format}`` with prompt-enforced JSON,
    parsed tolerantly by :mod:`laya_rover_agent.laya.parsing`.
"""

from __future__ import annotations

from ..types import Action, Observation

ACTION_QUESTION_ID = "next_action"
REASON_QUESTION_ID = "reason_code"

REASON_CODES: tuple[str, ...] = (
    "CLEAR_PATH",
    "AVOID_OBSTACLE",
    "AVOID_HAZARD",
    "CONSERVE_ENERGY",
    "REPLAN_NEEDED",
)

#: Section 8 runtime prompt template - used verbatim for the ``chat`` style.
SYSTEM_PROMPT = """\
You are the local navigation decision layer of a planetary rover. You do not
drive the rover; you choose ONE typed action from a finite set, and a
deterministic safety guardian may veto your choice.

Coordinate frame: origin (0,0) is the TOP-LEFT cell of the grid. x increases
EAST, y increases SOUTH. Heading N decreases y, E increases x, S increases y,
W decreases x. front_left / front_right are the headings 90 degrees from the
current heading relative to the direction of travel.

Action semantics:
  MOVE_FORWARD          - one cell in the current heading.
  TURN_LEFT_AND_MOVE    - rotate 90 deg left, then one cell.
  TURN_RIGHT_AND_MOVE   - rotate 90 deg right, then one cell.
  REVERSE               - one cell opposite the current heading.
  STOP_AND_REPLAN       - stay put and consume a small idle cost (capped).

Rules:
  1. Choose EXACTLY ONE action from the provided allowed_actions list.
  2. Never choose an action that enters an OBSTACLE or leaves the grid.
  3. HAZARD_SLOPE cells are traversable but cost ~1.8x energy and may stall the
     wheels, so prefer a safe non-hazard action when it is equally good.
  4. Keep a battery reserve: do not burn energy on manoeuvres that do not
     reduce the distance to the goal.
  5. If you are absorbing repeated STOP_AND_REPLAN actions or oscillating, the
     rover is in a local minimum - pick a decisive action that changes the
     geometry (typically a turn) rather than repeating a failed one.

Output: return ONLY a JSON object, no prose, no markdown fence:
  {"chosen_action": "<one of allowed_actions>",
   "confidence": <float 0.0-1.0>,
   "reason_code": "<CLEAR_PATH|AVOID_OBSTACLE|AVOID_HAZARD|CONSERVE_ENERGY|REPLAN_NEEDED>"}
"""

#: Short, precise criteria descriptions for the Laya ``choice`` question.
ACTION_CRITERIA_DESCRIPTIONS: dict[Action, str] = {
    Action.MOVE_FORWARD: (
        "Advance one cell in the current heading. Choose when the cell ahead is "
        "free of obstacles and doing so reduces the Euclidean distance to the goal."
    ),
    Action.TURN_LEFT_AND_MOVE: (
        "Rotate 90 degrees left then advance one cell. Choose to steer around an "
        "obstacle or out of a local minimum when the left side is open."
    ),
    Action.TURN_RIGHT_AND_MOVE: (
        "Rotate 90 degrees right then advance one cell. Choose to steer around an "
        "obstacle or out of a local minimum when the right side is open."
    ),
    Action.REVERSE: (
        "Move one cell opposite the current heading without turning. Choose to back "
        "out of a dead end when forward and both turns are blocked or unsafe."
    ),
    Action.STOP_AND_REPLAN: (
        "Stay in place and consume a small idle energy cost (capped by the guardian). "
        "Choose only when no other action is safe."
    ),
}


def build_action_criteria(safe_actions: list[str]) -> dict[str, str]:
    """Criteria map for the ``choice`` question - keys must be the exact action
    names so the guardian can validate the answer without translation."""
    criteria: dict[str, str] = {}
    for value in safe_actions:
        try:
            action = Action(value)
        except ValueError:  # pragma: no cover - defensive
            continue
        criteria[value] = ACTION_CRITERIA_DESCRIPTIONS[action]
    return criteria


def build_questions(observation: Observation, include_reason: bool = True) -> dict[str, object]:
    instructions = (
        "Choose the single next action for the planetary rover. Only actions that "
        "the deterministic safety guardian has already approved are listed; every "
        "option is legal. Prefer the action that makes progress toward the goal "
        "while avoiding obstacle collisions, risky slope entries, and wasted energy."
    )
    questions: dict[str, object] = {
        ACTION_QUESTION_ID: {
            "type": "choice",
            "instructions": instructions,
            "criteria": build_action_criteria(observation.safe_actions),
        }
    }
    if include_reason:
        questions[REASON_QUESTION_ID] = {
            "type": "choice",
            "instructions": "Why is this the right next action?",
            "criteria": {
                "CLEAR_PATH": "The path ahead is clear and moving is productive.",
                "AVOID_OBSTACLE": "An obstacle forces this action.",
                "AVOID_HAZARD": "This avoids or minimises exposure to a risky slope.",
                "CONSERVE_ENERGY": "This action saves energy or protects the battery reserve.",
                "REPLAN_NEEDED": "The rover is stuck or oscillating and must change its geometry.",
            },
        }
    return questions


def build_laya_request(
    observation: Observation,
    model: str,
    include_reason: bool = True,
) -> dict[str, object]:
    """Request body for ``POST {LAYA_API_URL}`` (Laya's ``/v1/systemone``).

    Laya resolves ``model`` through the same aliases its SDK does (``english``,
    ``multilingual``, ``typed-decisions``, ``auto``); ``auto`` lets its router
    choose per request, and the response names the checkpoint that answered.
    """
    return {
        "model": model,
        "state": observation.to_decision_state(),
        "questions": build_questions(observation, include_reason=include_reason),
    }


def build_chat_request(observation: Observation, model: str) -> dict[str, object]:
    """Legacy chat-style request body (spec Section 15.3)."""
    return {
        "model": model,
        "system": SYSTEM_PROMPT,
        "input": observation.model_dump(),
        "response_format": "json",
    }
