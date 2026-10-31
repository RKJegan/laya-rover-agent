# Codebase walkthrough — what every file does and how it is wired together

This is the teaching document. It answers three questions, in order:

1. **What is the machine?** — the mental model, what decides what, and the one
   guarantee the whole design exists to protect (Section 1).
2. **What happens in one step?** — a single decision traced from sensor reading to
   rover movement, naming the exact functions involved (Section 2).
3. **What is each file for?** — every file in the project, grouped by role, with
   what it exposes, who calls it, and what to watch out for (Section 3).

Then: how the pieces are wired (4), how a run turns into a report (5), the rules
you must not break and where to make common changes (6), a hands-on tour (7), and a
plain-language glossary (8).

Read Section 1 and 2 and you will understand the system. Section 3 is a reference
you will come back to.

---

## 1. The mental model

### The one idea

**The decision layer proposes; a small, dumb, certain safety layer disposes.**

The model — Laya, or the offline stand-in — is allowed to have an opinion about
which of five actions to take next. It is *never* allowed to move the rover. Between
its opinion and the wheels sits the **safety guardian**, which

- decides beforehand which actions are even legal from here (`safe_actions`),
- checks afterwards that the chosen action is still one of them,
- and if anything at all is wrong — the model was slow, unreachable, confused, or
  returned something malformed — substitutes a conservative action chosen by an
  ordinary rule-based controller.

The important consequence: **a failure of the model can cost success, but it cannot
cost safety.** Every path that is not an approved, in-range, on-time, legal action
ends in the same place, and that place is safe. Nothing in the codebase is allowed
to break that rule, and the test suite checks it from several directions.

### The four layers

The project is organised into four layers, and you will see the word "layer" in the
test filenames. Each layer only knows about the one below it.

```
  Layer 4  LIVE            the real Laya model on a local server, or a real rover
  ------------------------------------------------------------------------------
  Layer 3  PRODUCT         the six commands, the mission loop, the experiment
                           engine, the artifacts, the report
  ------------------------------------------------------------------------------
  Layer 2  DECISION        Laya over HTTP (and the offline stand-in), the request
                           and reply format, redaction, failure classification
  ------------------------------------------------------------------------------
  Layer 1  DOMAIN          world, perception, safety guardian, dynamics,
                           controllers  — pure, deterministic, no I/O
  ------------------------------------------------------------------------------
```

Layers 1–3 run offline in about 7 seconds and are what the 185 tests cover. Layer 4
is exercised by a script (`scripts/smoke_test_laya_api.py`) because it needs a real
server and, for the rover case, real hardware.

### Who decides what

| Question | Answered by | File |
|---|---|---|
| What does the world look like? | `WorldGenerator` | `world.py` |
| What can the rover see right now? | `Perception` | `perception.py` |
| What is the rover *allowed* to do? | `SafetyGuardian.safe_actions()` | `safety.py` |
| What would a careful, boring controller do? | `FallbackController` | `controllers/fallback.py` |
| What would a *naive* controller do? (the comparison) | `BaselineController` | `controllers/baseline.py` |
| What does the model think? | the Laya layer | `laya/api.py`, `laya/mock.py` |
| Is that allowed? | `SafetyGuardian.approve()` | `safety.py` |
| What physically happens? | `RoverSimulator.step()` | `dynamics.py` |
| What gets written down? | `EpisodeRunner`, `ExperimentRunner` | `episode.py`, `experiment/` |

### The three controllers you will see in every table

| Name | What it is | Why it exists |
|---|---|---|
| `laya_agent` | The model's choice, guarded | The thing being evaluated |
| `baseline` | "Always move towards the goal, break ties by turning" — no history, no caution | A deliberately simple comparison point. It walks into the U-shaped traps, which is the point |
| fallback / `FallbackPolicy` | A careful rule-based controller with short-term memory, dead-end avoidance and hazard aversion | The net the model falls into, and the thing the model's answers are calibrated against |

The fallback is *not* a weaker baseline — it is a different design. The baseline is
greedy and memoryless; the fallback avoids visible pockets, refuses to revisit cells
it just left, and escapes cycles. That difference is what makes "the model beat the
baseline" an interesting result rather than a foregone one.

### The two decision modes

| Mode | What answers | When you use it |
|---|---|---|
| `mock_laya` (default) | `MockLayaAdapter` — a deterministic stand-in that behaves *statistically* like the real model: agrees with the fallback 80% of the time, is ambiguous 10%, unusable 10%, and deliberately proposes an unsafe action 5% of the time so the guardian's rejection path is exercised | Sweeps, tests, anything that must be reproducible or offline |
| `laya_api` | `LayaApiAdapter` — a real HTTP call to a self-hosted `laya-serve` | Confirming the real model behaves like the stand-in, on a few seeds |

### The two things never allowed to happen

1. **The rover never moves because of a model error.** An error produces a
   *fallback* decision, and the step is logged with a machine-readable reason code.
2. **The model never chooses outside `safe_actions`.** A proposal outside that set
   is refused by the guardian (`NOT_IN_SAFE_ACTIONS`), even if it looks harmless.

---

## 2. One step, end to end

This is the heart of the codebase. One iteration of the loop in
`EpisodeRunner.run()` (`episode.py`, starting around line 156). Follow the numbers.

```
 ┌───────────────────────────────────────────────────────────────────────────┐
 │  state: where the rover is, which way it faces, how much battery it has    │
 └───────────────────────────────────────────────────────────────────────────┘
        │
  (1)   │  safe = guardian.safe_actions(state)
        │      → the guardian asks: for each of the 5 actions, would this move
        │        hit an obstacle, leave the map, drain the battery reserve, or
        │        exceed the stop cap?  Result: a SafeActionSet, e.g.
        │        {MOVE_FORWARD, TURN_LEFT_AND_MOVE, TURN_RIGHT_AND_MOVE}
        │
        │      if the set is EMPTY → the episode ends immediately as `stuck`
        │      (reason NO_LEGAL_ACTION); no controller is asked anything
        ▼
  (2)   │  observation = perception.observe(world, state, safe.actions, previous_action)
        │      → five distance beams (front, front_left, front_right, left, right),
        │        each measured up to sensor_range, and reported as sensor_range + 1
        │        ("at least this far") when nothing is in the way
        │      → three hazard flags, goal direction + distance, battery, step count
        │      → the legal actions, as plain strings
        ▼
  (3)   │  context = ControllerContext(position, recent_positions, previous_action, ...)
        │      → the extra facts a *local* policy needs that are not "sensors":
        │        where it has just been, and what it did last
        ▼
  (4)   │  fallback_action = fallback.choose(observation, safe, context)
        │      → the careful controller's answer, computed on EVERY step, whether
        │        or not the model succeeds. It is the yardstick and the net.
        ▼
  (5)   │  proposal = await controller.propose(observation, state, safe, step, context)
        │      → in mock mode: the stand-in rolls a seeded die and returns one of
        │        agree / ambiguous / low-confidence / malformed / unsafe (80/10/10/10,
        │        with 5% of steps flipping to the unsafe probe)
        │      → in live mode: an HTTP POST to Laya with typed questions and a
        │        criteria list built from safe_actions, then parse + validate
        │      → a Proposal: {valid, action, confidence, source, latency_ms,
        │        failure_reason, usage, raw_response, provider_model, diagnostics}
        ▼
  (6)   │  approval, routed_source, routing_reason =
        │        _route_and_approve(spec, proposal, safe, state, guardian, fallback_action)
        │
        │      the routing rules, in order (episode.py, ~line 348):
        │        invalid or no action        → FALLBACK   (reason = the failure class)
        │        latency > decision_timeout  → FALLBACK   (DECISION_TIMEOUT)
        │        confidence < threshold      → FALLBACK   (LOW_CONFIDENCE)
        │        otherwise                   → guardian.approve(...)
        │             approved    → source LAYA
        │             refused     → source GUARDIAN_OVERRIDE, fallback action used
        ▼
  (7)   │  outcome = sim.step(state, approval.final_action)
        │      → moves (or doesn't, for STOP_AND_REPLAN), spends energy, possibly
        │        stalls on a hazard slope, updates battery, decides the new status
        │        (active / goal / collision / battery_depleted / stuck / timeout)
        ▼
  (8)   │  counters += collisions, overrides, prevented-unsafe-actions, stalls, …
        │  one row is appended to `timesteps` with every field above, and - if the
        │  decision was valid - one row to the calibration record
        ▼
        └── loop while status is `active` and steps < max_steps
```

### What route each situation takes

| Situation | `action_source` in the log | Reason code |
|---|---|---|
| Model answered, in range, approved | `laya` | `APPROVED` |
| Model answered, but the action breaks a rule | `guardian_override` | e.g. `BLOCKED_OBSTACLE`, `NOT_IN_SAFE_ACTIONS` |
| Model answered below the confidence threshold | `fallback` | `LOW_CONFIDENCE` |
| Model answered too slowly | `fallback` | `DECISION_TIMEOUT` |
| Model returned junk / wrong schema | `fallback` | `MALFORMED_RESPONSE`, `SCHEMA_VIOLATION` |
| Server unreachable / timed out / busy / refused | `fallback` | `CONNECTION_ERROR`, `DECISION_TIMEOUT`, `RATE_LIMITED`, `API_ERROR` |
| Baseline controller arm | `baseline` (or `guardian_override`) | — |

Notice what every non-`laya` row has in common: the action executed is still one of
`safe_actions`, and it is the one a deterministic policy picked. **Fail closed** is
not an aspiration here; it is the only branch that exists.

### Why the loop reads the way it does

- **The guardian runs first, not last.** `safe_actions()` is computed before the
  observation is built, and the observation *carries it*. So the model's choices and
  the guardian's rules cannot drift apart: the model is literally shown only the
  legal moves, and the guardian refuses anything outside that same list.
- **The fallback runs every step, not only on failure.** It costs microseconds, and
  it means "what a careful controller would have done" is available for every
  comparison: the calibration table, the ablation counters, and the paired statistics
  all use it.
- **Unknown output is a *schema violation*, not a default.** A reply missing its
  confidence is not treated as "confidence 1.0". It is rejected, and the step falls
  back. (This is one of the deliberate deviations recorded in `MIGRATION.md`: the old
  behaviour defaulted to 1.0, which is fail-open and was removed.)
- **The counterfactual is computed without acting.** When the guardian refuses a
  move onto a hazard or into an obstacle, it also asks the simulator "would this have
  collided?" — `RoverSimulator.would_enter()` — and records the answer. That is where
  the safety-case numbers come from; nothing is ever actually tried.

---

## 3. File-by-file map

Every file in the project, grouped by role. For each: **what it is**, **what it
exposes**, **who uses it**, and **the thing to be careful about**.

### 3.1 The domain layer — `src/laya_rover_agent/`

These five modules are pure: no network, no files, no clock (except where injected),
no randomness except through a passed-in generator. That is why the tests can run
them thousands of times per second and expect identical answers.

#### `types.py` — the vocabulary (311 lines)

The single place where shared data shapes are defined, so nothing else has to import
from a sibling to talk.

| Name | What it is |
|---|---|
| `Action` | **The five actions.** `MOVE_FORWARD`, `TURN_LEFT_AND_MOVE`, `TURN_RIGHT_AND_MOVE`, `REVERSE`, `STOP_AND_REPLAN`. Nothing else is a valid action, anywhere |
| `CellType` | `FREE`, `OBSTACLE`, `HAZARD_SLOPE`, `START`, `GOAL` — what a map square is |
| `Status` | How an episode ended: `ACTIVE`, `GOAL`, `COLLISION`, `BATTERY_DEPLETED`, `STUCK`, `TIMEOUT` |
| `ActionSource` | Who decided, for the log: `laya`, `fallback`, `guardian_override`, `baseline` |
| `GuardianReason` | **Every** machine-readable reason code, from `APPROVED` to `BLOCKED_HAZARD_BATTERY`, `LOW_CONFIDENCE`, `CONNECTION_ERROR`, … |
| `RoverState` | position, heading, battery, energy used, steps, consecutive stops, status |
| `Observation` | what a controller sees: goal, obstacles, hazards, heading, battery, steps, previous action, `safe_actions` |
| `GridMap` | the world: cells, start, goal, difficulty, preset, key |
| `DecisionResult` | a validated decision: `chosen_action`, `confidence`, `reason_code` |
| `DecisionCall` | the outcome envelope of one attempt: `valid`, `reason_code`, `latency_ms`, `http_status`, `usage`, `raw_response`, `provider_model`, `truncated` |
| helpers | `turn_left`, `turn_right`, `heading_delta`, `euclidean` |

**Watch out:** `ActionSource` and `GuardianReason` are written into every CSV and
every log row. Adding a member is safe; *renaming* one changes the artifacts and the
report labels.

#### `world.py` — map generation and the RNG contract (470 lines)

| Name | What it does |
|---|---|
| `MapSpec` | the identity of one map: `{seed, difficulty, preset}` plus a stable `key` string |
| `WorldGenerator.generate(spec)` | builds the `GridMap` |
| `build_map_specs(...)` | the experiment matrix: for each difficulty, `seeds` random maps + the trap presets |
| `stream_seed(seed, stream)` | **the determinism contract.** Derives an independent RNG seed per purpose, so the map, the stalls and the mock's rolls never interfere |
| `trap_wall`, `_pocket_glyph`, `_orient` | the trap shapes: `trap_u`, `trap_canyon`, `trap_twincup`, rotated to face the rover's approach |
| `route_exists`, `_astar_or_bfs` | guarantees a route exists from start to goal before a map is accepted |
| `ascii_map(grid)` | the picture you see from `laya-rover map` |

**Three facts worth remembering:**

1. `stream_seed(seed, "map_generation")` returns the seed itself, so map identity is
   obvious. Every other stream mixes in a tag.
2. The mock's stream tag is the frozen **number** `19687` (`MOCK_RNG_TAG_VALUE`,
   defined in `laya/mock.py`). It used to be a *string*; renaming a string changes
   every random draw. Never change this number casually — it invalidates archived
   results. A test pins it.
3. Start and goal are guaranteed to be far apart (default: at least 60% of the map
   diagonal) and on free cells.

#### `perception.py` — the forward window (172 lines)

Fixed, honest, local: no global map, no path planning, just beams.

| Name | What it does |
|---|---|
| `RAY_ORDER` | the five beams, in canonical order: `front`, `front_left`, `front_right`, `left`, `right` |
| `HAZARD_BEAMS` | hazards are only reported on the three forward beams |
| `Perception.observe(...)` | builds the `Observation` |
| `Perception.ray_distances(...)` | the raw distances |

The sentinel: a beam with nothing within `sensor_range` reports **`sensor_range + 1`**
(7 by default). That is a deliberate "at least this far" value, not a bug — it keeps
the numbers pinned to the sensor's real resolution and lets a controller tell
"nothing there" from "something at the edge of range".

#### `safety.py` — the safety guardian (297 lines)

The most important file in the project. It does three jobs.

| Name | What it does |
|---|---|
| `SafetyGuardian.safe_actions(state)` | **pre-filter**: the legal action set. Blocks obstacles, off-map moves, battery-reserve violations, hazard entries that are avoidable, and stops beyond the stop cap |
| `SafetyGuardian.approve(proposed, proposed_source, safe, state, failure_reason, fallback_action)` | **the ruling**. Returns an `Approval` with the final action, whether it overrode, and why |
| `SafetyGuardian.counterfactual(state, action)` | what *would* have happened — collision? hazard entry? — computed without acting |
| `SafeActionSet`, `Approval` | the two small dataclasses it returns |

**Hazard semantics (worth reading twice).** A hazard slope is *traversable but
risky*, not a wall. It is blocked only when entering it would be clearly worse:

- the projected battery after the move would fall below `battery_reserve × 2`, or
- an *equally good* safe non-hazard action exists (its next cell is no further from
  the goal than the hazard cell).

So the rover may cross a hazard when it must, and will avoid one when it can — a
documented policy, not an accident of implementation.

**The ablation switch** (`guardian.mode: log_only_ablation`) does **not** turn the
guardian off. It records what the model proposed even when the guardian overrode it,
so you can count what *would* have happened. The rover still never enters a hazard.
The name is about bookkeeping, not behaviour.

#### `dynamics.py` — physics and consequences (318 lines)

| Name | What it does |
|---|---|
| `RoverSimulator.step(state, action)` | the only way the rover moves. Returns `StepOutcome` |
| `RoverSimulator.initial_state()` | a fresh rover at the map's start |
| `RoverSimulator.projected_cost(state, action)` | energy the action would cost, used by the guardian's budget checks |
| `RoverSimulator.termination_status(state)` | `goal` / `battery_depleted` / `stuck` / `collision`, or `None` to keep going |
| `RoverSimulator.would_enter(state, action)` | the counterfactual probe used by the guardian |
| `action_target(x, y, heading, action)` | which cell an action points at |
| `next_heading(heading, action)` | how the two turn actions change facing (`REVERSE` keeps facing and moves backwards; `STOP_AND_REPLAN` does not move) |

**The numbers you will see in tables:** energy costs are `MOVE_FORWARD 1.0`,
`TURN_LEFT_AND_MOVE 1.3`, `TURN_RIGHT_AND_MOVE 1.3`, `REVERSE 1.5`,
`STOP_AND_REPLAN 0.1`, multiplied by `1.8` when the rover is on a hazard slope.
Capacity is `120.0` per full battery (the "energy_used ~88.7" you see in summaries is
in those units). Hazards can also *stall* the rover with probability 0.10/0.15/0.20
by difficulty — a stall costs energy and a step but is not a collision.

### 3.2 The decision layer — `controllers/` and `laya/`

#### `controllers/base.py` — the two small contracts (74 lines)

| Name | What it is |
|---|---|
| `ControllerContext` | what a local policy may know beyond sensors: current position, recent positions, previous action, the sensor sentinel |
| `Proposal` | what any controller returns: `valid`, `action`, `confidence`, `source`, `latency_ms`, `failure_reason`, `usage`, `raw_response`, `provider_model`, `truncated`, `diagnostics` |
| `Controller` | the protocol every controller satisfies: `propose(observation, state, safe, step, context)` |

#### `controllers/fallback.py` — the careful net (236 lines)

The rule-based controller the rover falls back on, and the yardstick the model's
confidence is measured against. It is local and greedy-with-memory: no planning.

Its policy knobs (all in `configs/*.yaml` under `controllers.fallback`) are the
interesting reading: `dead_end_avoidance` + `dead_end_clearance_threshold` (a wall
within N cells ahead reads as a shallow pocket), `pocket_penalty`, `hazard_aversion_margin`,
`backtrack_penalty` (never immediately undo the last move), `memory_penalty` +
`memory_cells` (avoid re-entering cells just visited), and the escape rules
(`escape_window`, `escape_max_distinct`, `escape_multiplier`) that break a cycle by
hitting visited cells three times harder.

#### `controllers/baseline.py` — the honest comparison (75 lines)

Greedy: pick the action that most reduces distance to the goal, break ties by a fixed
order. No memory, no caution, no hazard awareness beyond the guardian's veto. Kept
simple on purpose — it is the "what does a straightforward approach achieve?" line in
every table.

#### `controllers/laya_agent.py` — the model's arm (67 lines)

A thin adapter: `LayaAgentController.propose(...)` forwards to whichever decision
layer is configured. It exists so the episode loop has exactly one controller
interface, whether the answer comes from a local stand-in or an HTTP server.

#### `laya/adapter.py` — the wire-level seam (72 lines)

`DecisionAdapter` is the protocol a *remote decision service* implements:
`decide(observation, safe_actions)` and `aprobe_health()`. `LayaApiAdapter` fills it.

Note the asymmetry, documented in the module: `MockLayaAdapter` does **not** implement
this protocol, because its whole definition is "what would the fallback have chosen,
and how do I disagree with it" — that needs the rover state and the safe set, which a
wire-level signature cannot carry. It implements the controller-facing `propose()`
seam instead. Both are correctness-checked after the fact by the guardian, so the
difference cannot make either of them unsafe.

#### `laya/prompt.py` — what we send (159 lines)

| Name | What it does |
|---|---|
| `ACTION_QUESTION_ID`, `REASON_QUESTION_ID` | the two typed questions the agent asks: `next_action` and `reason_code` |
| `REASON_CODES` | the vocabulary offered to the model for its *self-reported* reason (advisory only — the authoritative reason code is always the guardian's) |
| `ACTION_CRITERIA_DESCRIPTIONS` | the human-readable description attached to each action label |
| `build_action_criteria(safe_actions)` | builds the choice list **from the guardian's safe set**, so the model is only ever offered legal actions |
| `build_questions(observation, include_reason)` | the typed-question block |
| `build_laya_request(...)` | the full request body |
| `build_chat_request(...)` | the legacy free-text envelope, kept for comparison only |

#### `laya/api.py` — talking to Laya (455 lines)

The only file in the project that makes a network call.

| Name | What it does |
|---|---|
| `LayaApiAdapter.adecide/decide` | one decision, with a concurrency limit and a hard time budget |
| `LayaApiAdapter.aprobe_health(timeout_s)` | `GET /health`, used for the pre-flight |
| `LayaApiAdapter._interpret(response, ...)` | turns an HTTP reply into a `DecisionCall` — the error table below |
| `LayaApiAdapter._allowed_actions_from_request(...)` | re-reads the allowed actions from the request we sent, and refuses an answer outside them |
| `_clean_usage`, `_retry_after_seconds` | tidy the usage block; honour the server's "wait N seconds" hint |

**How every possible reply is classified** (this table *is* the safety behaviour):

| Reply | Reason code | Retried? |
|---|---|---|
| 200 with a valid typed answer | — (valid) | — |
| 200 with junk / missing fields | `MALFORMED_RESPONSE` or `SCHEMA_VIOLATION` | no |
| 200 with an action outside the safe set | `SCHEMA_VIOLATION` | no |
| 401 / 403 | `AUTH_ERROR` | no |
| 402 | `NO_CREDITS` | no (dormant: Laya does not bill — kept because a proxy in front of it might) |
| 429 / 503 | `RATE_LIMITED` | **yes**, once, and only if the server's `Retry-After` fits inside the decision budget |
| 422 | `SCHEMA_VIOLATION` (our request was malformed — our bug) | no |
| 4xx | `API_ERROR` | no |
| 5xx | `API_ERROR` | yes |
| timeout | `DECISION_TIMEOUT` | yes, if budget allows |
| connection failure | `CONNECTION_ERROR` | yes, if budget allows |

Two details that matter: the adapter refuses to sit still beyond its deadline (a
`Retry-After` longer than the remaining budget ends the attempt rather than causing a
second wait), and a missing key is only an error when the *configuration says the
server requires one* (`auth_required`) — a default `laya-serve` is unauthenticated.

#### `laya/parsing.py` — what we accept (276 lines)

`parse_response(payload)` handles both the modern typed envelope (`answers` keyed by
question id, each with `type`, `choice`, `confidence`, `answer_confidence`) and the
flat legacy shape, and returns a `ParseOutcome`. Validation refuses: unknown action
labels, confidence outside `[0, 1]`, and a missing confidence (that is a schema
violation, not an implied 1.0). `LayaParseError` carries the message that becomes the
`raw_response` evidence in the log.

#### `laya/mock.py` — the deterministic stand-in (249 lines)

| Name | What it does |
|---|---|
| `MOCK_RNG_TAG_VALUE = 19687` | the frozen RNG tag (see `world.py` above) |
| `MockLayaAdapter.propose(...)` | rolls a seeded die per (episode, step) and takes one of five branches |

| Branch | Share | What it returns |
|---|---|---|
| `agree_with_fallback` | 80% | exactly what the fallback would do, with a high confidence |
| `ambiguous_alternative` | 10% | a different but legal action, with a middling confidence |
| `low_confidence` | — | a legal action with a confidence below the threshold |
| `malformed` | 20% of the "unusable" share | structurally invalid output (the fail-closed path) |
| `unsafe_probe` | 5% (overrides the others) | an action outside the safe set, to exercise the guardian |

The branch is recorded in every log row as `mock_branch`, which makes the mock's
behaviour auditable rather than mysterious.

#### `laya/factory.py` — one place that builds controllers (102 lines)

`ControllerFactory(config, api_key, redactor, request_style)` → a callable that turns
an `EpisodeSpec` into a ready controller, reusing a single HTTP client across
episodes. `make_single_controller(...)` is the same thing for one-off runs. If you
want to add a controller arm, this is the file you touch.

### 3.3 The loop and the plumbing

#### `episode.py` — one episode, start to finish (502 lines)

The only place the whole loop exists. Also home to:

- `EpisodeSpec` — the identity of one row of the experiment matrix
  (`easy-s000-trap_u-laya_agent-t70` style ids), and `EpisodeResult`;
- the wall-clock guard, separate from `max_steps`: a 120-second episode budget that
  ends the episode as `timeout` rather than letting it hang;
- `_summary_row(...)` — turns a finished episode into the one-row-per-episode record
  that becomes `episode_summaries.csv`;
- `write_jsonl(...)` — the per-step log format.

#### `config.py` — every setting, one source of truth (335 lines)

| Name | What it is |
|---|---|
| `WorldConfig`, `RoverConfig`, `MockLayaProfile`, `LayaApiConfig`, `AgentConfig`, `FallbackPolicyConfig`, `GuardianConfig`, `ExperimentConfig` | the settings tree, all validated by pydantic (ranges enforced: a negative sensor range is a startup error, not a weird run) |
| `load_config(user_yaml, overrides)` | layered loading |
| `LayaEnvSettings` | environment variables, each with an explicit name: `LAYA_API_KEY`, `LAYA_API_URL`, `LAYA_MODEL`, `LAYA_REQUEST_TIMEOUT_S`, `LAYA_MAX_RETRIES`, `LAYA_AUTH_REQUIRED`, plus the rover bridge's `ROVER_BACKEND`, `ROVER_SERIAL_PORT`, `ROVER_ROS2_NS` |
| `apply_env_overrides(config, env)` | folds environment values into the resolved config, so the run metadata and the behaviour can never disagree again |

**Precedence, weakest to strongest:** packaged defaults
(`src/laya_rover_agent/default_config.yaml`) → your `--config` YAML → command-line
flags → environment variables. The API key is deliberately *excluded* from the config
object and therefore from the config hash — it never appears in `run_metadata.json`.

#### `cli.py` — the six commands (384 lines)

`laya-rover {experiment|episode|hardware|map|plan|estimate}`. It is the only place
that reads `.env` (at process start) and the only place that prints. Everything
downstream receives configuration by injection, which is why the library is testable
without a terminal.

Exit codes are meaningful so scripts and CI can react: `0` success, `1` episode or
mission did not reach the goal, `2` configuration/credential/backend problem.

#### `__init__.py` / `__main__.py` — package identity (49 + 10 lines)

The package version, the product banner, and the one line that makes
`python -m laya_rover_agent` work.

### 3.4 The experiment engine — `experiment/`

#### `experiment/runner.py` — turn a config into a folder of results (372 lines)

| Name | What it does |
|---|---|
| `ExperimentRunner.build_specs()` | the full matrix: each map × each controller × each confidence threshold |
| `ExperimentRunner.run()` | runs every episode (in parallel if `jobs > 1`; each episode is self-seeded, so results are identical either way), then assembles all artifacts |
| `RunArtifacts` | the returned list of paths: `output_dir`, `episode_summaries`, `aggregate_metrics`, `paired_stats`, `calibration`, `ablation`, `reason_codes`, `metadata`, `report`, `figures`, `episode_count` |
| `estimate_api_calls(config)` | how many model calls a live run would make — used by `laya-rover estimate` and printed before any live run |
| `summarise_examples(examples)` | picks the example paths drawn in the figures |

Parallelism is safe because an episode's randomness comes only from its own seed
(`stream_seed`), never from shared state — that is the no-global-state rule earning
its keep.

#### `experiment/analytics.py` — the statistics (292 lines)

| Name | What it does |
|---|---|
| `aggregate_metrics(summaries)` | the headline table (rates and means per controller × difficulty × threshold) |
| `paired_comparison(summaries, "laya_agent", "baseline")` | the fair comparison: for each map, the difference between the two controllers on the *same* map, with a 95% percentile bootstrap CI (10,000 resamples, fixed seed `20261003`) and a Wilcoxon p-value when there are at least 6 pairs. Metrics: `success_rate`, `energy_successful`, `steps_successful`, `collisions`, `hazard_entries`, `fallback_rate` |
| `calibration_table(records)` | confidence bins vs how often the decision agreed with the careful controller |
| `calibration_ece(table)` | one number summarising the calibration gap (sample-weighted) |
| `ablation_summary(summaries, threshold)` | the safety case: unsafe actions prevented, collisions prevented, hazard entries prevented, real collisions |
| `reason_code_totals(counts)` | how often each guardian reason fired |
| `bootstrap_ci(values)` | the CI helper |

**Why paired and not "mean of A vs mean of B":** pairing removes map difficulty from
the comparison. If the Laya agent succeeded on the same map where the baseline failed,
the difference is attributable to the controller, not to lucky maps.

**What "calibration" means here, operationally:** for every step where the model
returned a valid decision, the *outcome* is whether it agreed with the safe
conservative controller for that same observation, and whether the guardian approved
it. Well-calibrated means: of the steps where the model said "0.8", about 80% agreed.
Mean confidence minus agreement rate is the gap.

#### `experiment/metadata.py` — the honesty file (115 lines)

`collect_metadata(...)` records the resolved configuration, a SHA-256 of its pretty
JSON (`config_hash_sha256`), the package versions of the dependencies, the git commit
and whether the tree was dirty, timings, and the artifact inventory.
`write_metadata(...)` saves it as `run_metadata.json`.

A run without its metadata is not a reproducible run — which is why the hash is
printed in the guides: two people with the same hash ran the same experiment.

### 3.5 Reporting — `reporting/`

#### `reporting/report.py` — the written report (413 lines)

`build_report(...)` writes `outputs/report.md`: an executive summary, the methods, the
headline tables, the safety case, calibration, the live-API section (only when the
mode is live), limitations, and a pointer to the artifacts. The credential/auth rows
and the five honest limitations live here too — read section 9 of the generated
report before quoting any number from it.

#### `reporting/charts.py` — the six figures (289 lines)

One function per figure, each taking a DataFrame and a path: `success_rate_by_difficulty`,
`energy_by_controller`, `path_examples` (draws real trajectories on their maps),
`latency_distribution`, `guardian_ablation_safety_case`, `confidence_calibration`.
Pure drawing: no statistics are computed here, so if a number looks wrong, the bug is
in `analytics.py`, not in a chart.

### 3.6 The rover connector layer — `hardware/`

This is what makes the project "rover-ready": the same mission loop drives a
simulated rover, a ROS 2 rover, or a serial rover, and the decision layer does not
know or care which.

#### `hardware/base.py` — the one interface every rover must satisfy (91 lines)

| Name | What it is |
|---|---|
| `SensorFrame` | what a rover reports after a moment: position, heading, battery, the five beam distances |
| `ExecutionReport` | what happened when an action was executed: whether it moved, collided, stalled, how much energy it cost |
| `RoverBackend` | the protocol: `connect()`, `read_sensors()`, `execute(action)`, `emergency_stop()`, `close()` |
| `BackendUnavailableError` | raised when the transport is not there (no ROS 2 installed, no serial port) |
| `frame_to_state(frame, previous)` | converts a sensor frame into the `RoverState` the rest of the code uses |

#### `hardware/sensor_grid.py` — the local map the rover keeps (141 lines)

`SensorGridFacade` maintains the small grid of cells the rover has actually seen,
built only from sensor frames, and marks each beam's hit. It is what lets the
guardian (`grid.is_obstacle`, `is_hazard`, `cell`) work on real hardware exactly as it
does in simulation — without pretending the rover knows the whole map.

#### `hardware/bridge.py` — the mission loop for real rovers (350 lines)

`RoverBridge.run_mission_async(...)`: read sensors → build observation → ask the
controller → guardian ruling → execute → log; with a wall-clock budget, a dry-run
switch, JSONL logging, and `MissionSummary` (steps, distance, energy, actions,
stopped reason). `_observation_from_frame(...)` is where a sensor frame becomes the
same `Observation` the simulator produces, which is the trick that makes simulation
and hardware comparable.

**`--dry-run` means: decide, log, guard, but never execute.** Use it first, always.

#### `hardware/sim_backend.py` — the virtual rover (130 lines)

The test rover: wraps the `WorldGenerator` + `RoverSimulator` in the backend
interface, so the whole hardware path can be exercised with no hardware. This is the
default backend and the one every guide starts with.

#### `hardware/ros2_backend.py` — ROS 2 transport (246 lines)

Subscribes to odometry, battery and five range topics under a namespace (default
`/laya_rover`), publishes the chosen action as a string on `cmd_action`, and converts
yaw to the internal heading. Imports are deferred so a machine without ROS 2 can still
use everything else.

#### `hardware/serial_backend.py` — serial transport (258 lines)

A line protocol: `parse_telemetry(line, sensor_range)` reads a rover's status line,
`parse_ack(line)` reads the acknowledgement of a command, `heading_to_yaw(...)` for
output. Deliberately defensive — a garbled line raises `TelemetryParseError` and the
mission stops rather than guessing.

### 3.7 Operations, safety and configuration files

| File | What it is |
|---|---|
| `security.py` | `SecretRedactor`: removes a configured key by exact value, then scans for generic credential shapes (`sk-…`, `Bearer …`). Used on every raw response, every log row and every traceback. `redact_exception(exc)` keeps the exception type and message but never the secret |
| `scripts/smoke_test_laya_api.py` | the **Layer 4** tool: health probe before anything else, then one real decision, then N real episodes. `--offline` rehearses the same flow against the stand-in. Distinct exit codes: 0 validated / 1 server error / 2 unreachable / 3 key demanded |
| `scripts/check_no_secrets.py` | scans files (default: the whole project) for key-shaped strings. Run it on `outputs/` before sharing a run |
| `configs/default.yaml` | the full experiment: 50×50 maps, 30 seeds × 3 difficulties, thresholds 0.60/0.70/0.80, 360 episodes |
| `configs/quick.yaml` | the same shape, small: fewer seeds and smaller maps, for fast iteration |
| `src/laya_rover_agent/default_config.yaml` | the packaged defaults that `load_config` starts from — keep it in step with `configs/default.yaml` |
| `.env.example` | every environment variable, documented, no secrets. Copy to `.env` only if you need to change something |
| `pyproject.toml` | package metadata, dependencies, the `laya-rover` command, and the pytest settings (test paths, the `live` marker) |
| `.vscode/launch.json` | ten run/debug configurations (tests, CLI runs, simulator mission, live Laya) |
| `.vscode/tasks.json` | ten one-click tasks (install, test, coverage, each run, health pre-flight, secrets scan) |
| `.vscode/settings.json` | pytest on, `src` on the analysis path, `.env` wired, results folders excluded from search |
| `README.md` | what the project is and its headline results |
| `MIGRATION.md` | how the project moved to Laya, every behavioural difference, and the honest limitations |
| `CHANGELOG.md` | version history, newest first |

### 3.8 The tests — `tests/`

The suite is the specification written twice. 185 tests, offline, ~7 seconds.

| File | Tests | The promise it protects |
|---|---|---|
| `conftest.py` | — | shared fixtures (`quick_config`, `grid`, `trap_grid`, `guardian`, `simulator`, `fallback`), and it puts `src/` on the path so pytest works without installing |
| `test_layer1_world.py` | 25 | the same seed always gives the same map; a route exists; trap presets really do trap; density settings change maps as documented |
| `test_layer1_dynamics.py` | 10 | energy arithmetic, stalls, hazard multiplier, every termination status |
| `test_layer1_safety_perception.py` | 13 | the five beams and the sentinel, every block reason, the counterfactual probe |
| `test_layer1_controllers.py` | 5 | the fallback policy's escape and memory behaviour; the baseline is greedy |
| `test_layer2_laya_mock.py` | 9 | the stand-in's branch shares and seeding, and the frozen RNG tag |
| `test_layer2_laya_api_contract.py` | 43 | the whole HTTP contract: request shape, every status code, `Retry-After`, truncated state, malformed replies, dead servers |
| `test_layer2_security.py` | 10 | keys are scrubbed from strings, structures and tracebacks |
| `test_layer3_cli_and_hardware.py` | 21 | all six commands, their exit codes, and the three backends |
| `test_layer3_episode_e2e.py` | 11 | whole episodes, including bit-identical replays and the fail-closed end-to-end case |
| `test_layer3_experiment_artifacts.py` | 29 | a full experiment run produces every artifact with the documented columns and sane values |
| `test_no_legacy_backend.py` | 9 | **the project stays Laya-only**: no file mentions the retired provider, no such module loads, the defaults point at a Laya server, and only Laya decision layers exist |

If you change behaviour on purpose, a failing test tells you exactly which promise you
are breaking. If you change behaviour by accident, the same test tells you why it was
a promise.

---

## 4. How everything is wired

### Dependency direction

Dependencies point one way. Nothing in a lower layer imports something from a higher
one — that is what keeps the domain testable and the layers honest.

```
   cli.py ───────────────► episode.py ─────────► controllers/ ──► laya/
      │                        │                      │              │
      │                        │                      └── safety.py ─┤
      ├──► experiment/runner.py┤                                     │
      │        │              └── perception.py ◄───────────────────┘
      │        ├── analytics.py        │
      │        ├── metadata.py         └── world.py ──► dynamics.py ──► types.py
      │        └── reporting/  (report.py, charts.py)
      │
      └──► hardware/ (bridge.py ──► base.py ──► sim / ros2 / serial backends)

   security.py and config.py are used by everyone; they import nothing internal.
```

### The rules that make it hold together

| Rule | Where you see it | Why |
|---|---|---|
| **One decision layer at a time** | `agent.mode` is `mock_laya` or `laya_api`; `controllers/laya_agent.py` does not know which | Swapping the model must not touch the loop |
| **The guardian owns the boundary** | `safe_actions()` before, `approve()` after | Two independent chances to stop a bad move |
| **No global mutable state** | every component receives its config, its RNG and its clock by injection | Parallel episodes and bit-identical replays |
| **Randomness is streamed, not shared** | `stream_seed(seed, stream)` for maps, stalls and the mock | Reruns reproduce exactly; a rename cannot silently re-roll results |
| **Configuration is data** | `config.py` + YAML + env, hashed into the metadata | A run can be reproduced from its metadata alone |
| **Secrets are never logged** | `security.py` applied to every raw response and traceback | Logs and artifacts are safe to share |
| **Failure is a first-class outcome** | reason codes for every failure, not exceptions | A post-mortem can count what went wrong and how often |

### The fail-closed chain, in one picture

```
  model error  /  timeout  /  malformed  /  out of range  /  unknown action
        │
        ├─ classified by  laya/api.py  (or  laya/mock.py  for the stand-in)
        ├─ routed by      episode.py::_route_and_approve
        ├─ decided by     controllers/fallback.py::choose
        ├─ checked by     safety.py::approve  (the fallback action is in the safe set)
        └─ executed by    dynamics.py::step  (or hardware/bridge.py::run_mission_async)
```

There is no branch in that chain that moves the rover on an unvalidated model answer.

### Config precedence, with the key excluded

```
packaged defaults (src/laya_rover_agent/default_config.yaml)
        ▼
your YAML (--config configs/quick.yaml)
        ▼
command-line flags (--mode, --seeds, --threshold, …)
        ▼
environment variables (.env → LAYA_* / ROVER_*)
        ▼
resolved config ──► SHA-256 ──► config_hash_sha256 in run_metadata.json
                     (the API key is never part of this)
```

---

## 5. From a run to a report

What `laya-rover experiment` actually does, in order:

```
 1. cli.py            parse flags, load .env, apply precedence, print what it will do
 2. runner.py         build the matrix: seeds x difficulties x presets x controllers x thresholds
 3. (per episode)     episode.py: generate the map, run the loop, produce one summary row
                      + a full JSONL log of every step
 4. episode_summaries.csv      one row per episode
 5. analytics.py      aggregate_metrics.csv  (headline rates and means)
                      paired_stats.csv      (Laya vs baseline per map, 95% CI)
                      calibration.csv       (confidence vs agreement, plus the gap)
                      ablation.csv          (unsafe actions and collisions prevented)
                      reason_codes.csv      (how often each guardian reason fired)
 6. metadata.py       run_metadata.json      (resolved config + hash + versions + timings)
 7. charts.py         figures/*.png          (6 charts)
 8. report.py         report.md              (everything, written out in prose and tables)
 9. cli.py            print where it all landed
```

Design consequences worth knowing:

- **Episode logs are the source of truth.** Every table is derived from them, so a
  disputed number can always be traced back to the exact step that produced it.
- **The report is generated, never hand-written.** If you change the analysis, rerun
  the experiment and the report changes with it.
- **Metadata is written even for a run that produced nothing useful** — a failed run
  that leaves a `run_metadata.json` is a reproducible failure.

---

## 6. Rules you must not break, and where to change things

### The invariants

| Invariant | Why | Checked by |
|---|---|---|
| The executed action is always in `safe_actions` | the whole safety story | `test_layer1_safety_perception.py`, `test_layer3_episode_e2e.py` |
| A model failure never moves the rover on that answer | fail-closed | `test_layer3_episode_e2e.py` (dead-server case) |
| Only the five `Action` members ever exist | no free-text, no raw actuator values | `test_layer1_dynamics.py` |
| A missing confidence is a schema violation, never a default | fail-open was removed for a reason | `test_layer2_laya_api_contract.py` |
| The same seed gives the same map, episode and figures | reproducibility | `test_layer1_world.py`, `test_layer3_episode_e2e.py` |
| The mock's RNG tag is the frozen number `19687` | renaming re-rolls history | `test_layer2_laya_mock.py` |
| Secrets never reach a log, artifact or traceback | shareable evidence | `test_layer2_security.py` |
| The project stays Laya-only | one decision backend, one story | `test_no_legacy_backend.py` |

### If you want to…

| Goal | Touch these files |
|---|---|
| Add a new action | `types.py` (`Action`), `dynamics.py` (`action_target`, `next_heading`, energy table), `safety.py` (`safe_actions`, `approve`), `controllers/baseline.py`, `configs/*.yaml` energy costs, then the tests |
| Add a new reason code | `types.py` (`GuardianReason`), the place that detects it, `reporting/report.py` if it deserves prose |
| Add a new metric | `episode.py` (`_summary_row`), `experiment/analytics.py` (aggregates/paired), `reporting/report.py`, `reporting/charts.py` if it deserves a chart |
| Add a new map preset | `world.py` (`_pocket_glyph`, `build_map_specs`), the CLI choices in `cli.py`, `configs/*.yaml` if it is configurable |
| Add a new controller arm | `controllers/`, `laya/factory.py`, `experiment/runner.py` (the controllers list) |
| Add a new rover transport | `hardware/base.py` (implement the protocol), `hardware/__init__.py::make_backend`, a new module, `docs/HARDWARE_DEPLOYMENT.md` |
| Change which model Laya uses | `configs/*.yaml` → `agent.laya_api.model` (`english`, `multilingual`, `typed-decisions`, or `auto`), or `LAYA_MODEL` in `.env` |
| Change the safety policy | `safety.py` and the `guardian`/`controllers.fallback` blocks in the configs — and expect the safety-case numbers in `ablation.csv` to move |
| Run the real model | nothing in the code: start `laya-serve`, then `--mode laya_api` (see the run guide) |

---

## 7. A guided tour (five exercises)

Work through these in VS Code; each builds on the last. Total time about 30 minutes.

**1. See the world.** `laya-rover map --seed 0 --preset trap_u`, then the same with
`trap_canyon`. Open `world.py` and find `_pocket_glyph` — match the ASCII picture to
the glyph that produced it.

**2. Watch one decision.** Set a breakpoint in `episode.py` on the
`safe = guardian.safe_actions(state)` line, then F5 →
`CLI: episode trace (easy / trap_u / seed 0)`. When it stops, look at `safe.actions`.
Step to `controller.propose(...)` and inspect the returned `proposal`: is it valid?
what confidence? Then step into `_route_and_approve` and watch which branch is taken.
*This is the whole system in one keypress.*

**3. Watch the net catch something.** Run the episode again with
`--controller laya_agent` and the JSONL log on, then open
`outputs/vscode_episode.jsonl` and find rows where `action_source` is `fallback` or
`guardian_override`. Read `guardian_reason_code` and `mock_branch` on those rows —
that is the stand-in deliberately misbehaving and the guardian refusing it.

**4. Prove fail-closed yourself.** Point the agent at a server that does not exist:

```bash
LAYA_API_URL=http://127.0.0.1:9/v1/systemone laya-rover episode \
    --mode laya_api --seed 0 --difficulty easy --preset trap_u \
    --controller laya_agent --log-jsonl /tmp/dead.jsonl
```

Every row will be `fallback` / `CONNECTION_ERROR`, the episode still completes, exit
code 0, and no collision happens. Then read `laya/api.py::_interpret` and find the
line that produced that classification.

**5. Change something and see the system react.** In `configs/quick.yaml`, set
`agent.confidence_threshold: 0.95` and rerun the quick experiment
(`--output-dir outputs_high`). Open the new `aggregate_metrics.csv` next to the old
one: with a stricter threshold, more steps fall back, so the Laya arm should look
more like the fallback controller. Then reset it. *This is what the confidence knob
is for — and why the report always states the threshold it used.*

---

## 8. Glossary

| Term | Plain meaning |
|---|---|
| **Decision layer** | The thing that answers "which action next?" — Laya in live mode, a deterministic stand-in in mock mode |
| **Proposal** | One answer from the decision layer, with its confidence and any failure attached. Not yet a command |
| **Guardian / safety guardian** | The small, deterministic, rule-based component that decides what is legal, and has the final say on every move |
| **`safe_actions`** | The set of actions the rover is allowed to take from the current state. Both the model and the guardian work from the same set |
| **Fallback controller** | The careful rule-based controller used (a) as the answer whenever the model cannot be trusted, and (b) as the yardstick for calibration |
| **Baseline controller** | The deliberately naive comparison controller: greedy, memoryless |
| **Fail closed** | On any doubt, do the safe thing rather than the model's thing. Here: route to the fallback |
| **Ablation** | Deliberately removing or shadowing a mechanism to measure it. This project's ablation logs what the guardian *would* have stopped, without stopping the safety rules themselves |
| **Counterfactual** | What would have happened — computed by asking the simulator, never by trying it |
| **Calibration** | Does stated confidence match reality? If the model says 0.8, does it agree with the careful controller about 80% of the time? |
| **ECE** | Expected calibration error: one number for how far confidence is from reality |
| **Paired comparison** | Comparing two controllers on the *same* maps, so map difficulty cancels out |
| **Bootstrap CI** | Resampling the observed differences 10,000 times to get a 95% range for the true average difference. If that range excludes zero, the difference is unlikely to be luck |
| **Reason code** | A short machine-readable label (`BLOCKED_OBSTACLE`, `LOW_CONFIDENCE`, …) explaining every guardian ruling |
| **Mock / stand-in** | A fake decision layer that behaves statistically like the real one, so the whole pipeline can run offline and reproducibly |
| **Checkpoint** | A trained model file Laya loads (`english`, `multilingual`, `typed-decisions`) |
| **`laya-serve`** | Laya's own small HTTP server. Runs on your machine, separate from the agent, at `http://127.0.0.1:8000` by default |
| **Truncated state** | The request was longer than the model could read, so it saw only part of it. Recorded per step (`decision_truncated`) because a decision made on partial evidence is a different claim |
| **Sentinel value** | The `sensor_range + 1` reading meaning "nothing within range" — an honest "at least this far", not a measurement |
| **Episode** | One map, one controller, start to finish |
| **Run / sweep** | The full matrix of episodes, and everything derived from it |
| **Stream seed** | A seed derived for one specific purpose, so different random processes cannot interfere |

---

## Where to go next

| If you want to… | Read |
|---|---|
| run it and see results | `docs/VSCODE_RUN_AND_TEST_GUIDE.md` |
| understand the tests in depth | `docs/VSCODE_TESTING.md` |
| run the real model | `docs/INSTALL_LAYA_ON_YOUR_LAPTOP.md`, then `docs/live_laya_testing_guide.md` |
| connect a rover | `docs/HARDWARE_DEPLOYMENT.md` |
| know what changed and why | `MIGRATION.md`, `CHANGELOG.md` |
