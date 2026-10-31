# Laya Rover Agent

![Python](https://img.shields.io/badge/python-3.11%2B-blue)
![Tests](https://img.shields.io/badge/tests-185%20passing-brightgreen)
![License](https://img.shields.io/badge/license-MIT-lightgrey)
![Mode](https://img.shields.io/badge/offline--first-yes-success)

A safety-first, confidence-aware autonomous rover-navigation agent for simulated
planetary terrain, in which **Laya** acts as a bounded, confidence-aware *local*
decision layer - and a deterministic safety guardian sits between the model and
the rover at all times. The decision layer can never directly command the
rover: every proposed action passes through a deterministic guardian that
pre-filters illegal moves, re-checks the model's answer, and falls back to a
conservative controller on low confidence, invalid output, or a dead server.

Laya is a self-hosted, non-autoregressive decision engine: typed `choice` /
`score` / `noul` decisions over a state, in a single forward pass, on your own
hardware. Running locally suits a rover, where the link may be slow or absent -
and it means the decision layer needs **no API key** unless you deliberately put
an authentication layer in front of it.

The same decision stack that runs in the experiment suite also drives a rover:
one transport interface, three backends (`sim` / ROS 2 / serial). No code change
is needed to move from the virtual test simulator to hardware.

* **Simulation, not flight software.** No flight-readiness claims, no raw
  actuator control by a model, no global path planning by Laya, no free-text
  decisions. See [Limitations](#limitations). 
* **Runs fully local and offline.** `mock_laya` mode is a deterministic stand-in
  for the decision layer, so the entire pipeline, report and tests run with zero
  network access.
* **Live mode is one flag away** (`--mode laya_api`) and documented step by step in
  [`docs/live_laya_testing_guide.md`](docs/live_laya_testing_guide.md).

---

## Quickstart (3 commands, no API key)

```bash
cd laya-rover-agent
pip install -e ".[dev]"

# 1. see a deterministic map with its engineered trap cluster
laya-rover map --preset trap_u --seed 0

# 2. one episode, step by step (decision layer, guardian verdict, latency)
laya-rover episode --seed 0 --difficulty easy --preset trap_u --controller laya_agent

# 3. the full experiment sweep (30 seeds x 3 difficulties x 3 thresholds),
#    artifacts + six figures + report into outputs/
laya-rover experiment --config configs/default.yaml --mode mock_laya
```

Then open `outputs/report.md`. `configs/quick.yaml` runs a 10x smaller sweep in
seconds if you just want to see the pipeline work.

Live Laya (against your own server):

```bash
pip install "laya[serve]"
LAYA_MODELS=english LAYA_HOST=127.0.0.1 laya-serve     # terminal 1 - the model

python scripts/smoke_test_laya_api.py                  # terminal 2 - health probe + one real decision
python scripts/smoke_test_laya_api.py --episodes 2     # the real model driving the full loop
laya-rover experiment --mode laya_api --seeds 5 --difficulties medium
```

Step-by-step: [`docs/INSTALL_LAYA_ON_YOUR_LAPTOP.md`](docs/INSTALL_LAYA_ON_YOUR_LAPTOP.md)
to download and connect the model, then [`docs/live_laya_testing_guide.md`](docs/live_laya_testing_guide.md).

---

## Easy setup guide (two environments, by design)

The model server and the agent are deliberately kept in **two separate virtual
environments** — the server needs PyTorch (large, GPU-capable), the agent does
not. They only ever talk over `http://127.0.0.1:8000`, so neither needs to know
where the other lives.

```
your-workspace/
  laya-server/
    .venv-server/        <- PyTorch + Laya, runs laya-serve
  laya-rover-agent/
    .venv-agent/         <- this project, no PyTorch, runs the CLI + tests
```

**Terminal 1 — the model server**

```bash
mkdir laya-server && cd laya-server
python3.11 -m venv .venv-server
source .venv-server/bin/activate        # Windows: .venv-server\Scripts\Activate.ps1

# CPU is the default and needs nothing extra. For an NVIDIA GPU, install the
# CUDA build of PyTorch *before* laya, or pip will silently grab CPU-only:
#   pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install "laya[serve]"

LAYA_MODELS=english LAYA_HOST=127.0.0.1 LAYA_DEVICE=cuda LAYA_PRELOAD=1 laya-serve
# -> leave this terminal running for the rest of the session
```

**Terminal 2 — the agent (this repo)**

```bash
cd laya-rover-agent
python3.11 -m venv .venv-agent
source .venv-agent/bin/activate         # Windows: .venv-agent\Scripts\Activate.ps1
pip install -e ".[dev]"
cp .env.example .env

python -m pytest -q                                       # -> 185 passed, offline
python scripts/smoke_test_laya_api.py --show-raw           # -> one real decision, from the server above
```

If a decision call times out on a first cold GPU request, use
`--config configs/laptop_live.yaml` (raises the decision budget to 8s) on any
`laya-rover` or `smoke_test_laya_api.py` command. Full walkthrough, including
GPU troubleshooting: [`docs/INSTALL_LAYA_ON_YOUR_LAPTOP.md`](docs/INSTALL_LAYA_ON_YOUR_LAPTOP.md).

---

## What the agent actually does

```
                    +--------------------------------------------------+
                    |            per timestep (local only)             |
                    +--------------------------------------------------+
  world / sensors -> perception (5-beam forward window, sentinel) 
                   -> safety guardian PRE-FILTER (legal actions only)
                   -> decision layer  :  Laya service (choice question) |  mock
                   -> safety guardian APPROVAL (reject/override + reason code)
                   -> fallback controller  (low confidence / invalid / timeout)
                   -> rover simulator  or  rover backend (sim | ros2 | serial)
                   -> JSONL log (observation, raw answer, verdict, latency)
```

* **Perception** never exposes the map: five beams (`front`, `front_left`,
  `front_right`, `left`, `right`) capped at `sensor_range`, plus Euclidean goal
  distance and the sentinel `sensor_range + 1` for "nothing detected".
* **Laya** answers exactly one typed `choice` question whose criteria are the five
  actions - an answer outside that set is a schema violation and cannot execute.
* **The guardian is real, not decorative**: obstacle/out-of-bounds/battery/hazard
  rules, the stop cap, and a **counterfactual (ablation) probe** that records what
  *would* have happened without executing it. That is where the safety-case
  numbers come from.
* **The fallback** is a deterministic, conservative, goal-distance-minimising
  controller (`FR-7`) that is *algorithmically distinct* from the comparison
  baseline: hazard-averse, pocket-averse, with short-term visited-cell memory.
* **The baseline** is deliberately naive: greedy, memoryless, hazard-blind, fixed
  tie-break order - the oscillation-prone heuristic the spec asks to beat.

---

## CLI

| Command | Purpose |
|---|---|
| `laya-rover experiment` | run the controller matrix, write all artifacts + report |
| `laya-rover episode` | one episode with a step-by-step table and optional JSONL |
| `laya-rover hardware` | drive a rover through the connector layer (`sim` / `ros2` / `serial`) |
| `laya-rover map` | print a deterministic ASCII map for a seed/preset |
| `laya-rover plan` | show the experiment matrix that would run |
| `laya-rover estimate` | expected live-API call volume before you spend credits |

Useful flags: `--config`, `--mode mock_laya|laya_api`, `--seeds`, `--difficulties`,
`--confidence-threshold 0.6,0.7,0.8`, `--guardian-mode log_only_ablation`,
`--jobs`, `--run-label`, `--max-episodes`.

---

## Project layout

```
configs/
  default.yaml            full sweep (50x50 maps, 30 seeds, 3 thresholds)
  quick.yaml              30x30 / few seeds - seconds, offline
src/laya_rover_agent/
  types.py                pinned conventions, enums, RoverState/Observation/GridMap
  config.py               pydantic models, load_config(), config hashing
  world.py                deterministic maps, difficulty presets, concave trap clusters
  perception.py           FR-4 five-beam forward window
  dynamics.py             motion, energy, hazard stalls, loop detection, termination
  safety.py               guardian: pre-filter, approval, reason codes, counterfactual
  controllers/            base (ControllerContext), fallback, baseline, laya_agent
  laya/                    DecisionAdapter protocol, prompt, parser, deterministic mock,
                           live client for a self-hosted laya-serve, factory
  episode.py              the control loop + per-timestep JSONL schema
  experiment/             matrix runner, analytics (bootstrap CIs, calibration), metadata
  reporting/              six figures + report.md
  hardware/               RoverBackend protocol, sim/ros2/serial backends, RoverBridge
  cli.py, __main__.py     argparse + rich CLI
tests/                    Layer 1-3 offline suite (no network, ~7 s)
scripts/smoke_test_laya_api.py   Layer 4 - manual live smoke test
scripts/check_no_secrets.py     scans logs/artifacts for leaked key material
docs/                     Laya install guide, run-and-test guide (VS Code), codebase
                          walkthrough, live Laya testing guide, hardware deployment
                          guide, test guide
CHANGELOG.md, MIGRATION.md   release notes and the migration record
.vscode/                  committed workspace config: Test Explorer, 10 debug targets, 10 tasks
outputs/                  generated: logs, CSVs, figures, metadata, report.md
```

---

## Confidence: native, not synthesised

The calibration analysis (FR-10.5) is only meaningful if the confidence number
means something, so this project is explicit about where it comes from:

* Laya returns a **native calibrated confidence** for `choice` answers - trained
  with reinforcement learning against strictly proper scoring rules, not derived
  from token log-probabilities and not sampled N times for self-consistency.
  Nothing in this repository fabricates or rescales it.
* Every answer also carries `answer_confidence` (`max(p)`), which Laya defines
  identically across `choice`, `score` and `noul`; the adapter uses it only as a
  fallback when `confidence` is absent.
* A typed answer that arrives **without** any confidence is rejected as a schema
  violation. Earlier code defaulted a missing confidence to `1.0` - maximum trust
  for a provider that said nothing. That path is now fail-closed.
* Calibration is still measured here as *agreement with the fallback controller's
  choice*, a proxy for correctness rather than ground truth. `calibration.csv`
  reports mean confidence per bin against that proxy.

## Running Laya on the rover

* **Cost profile is memory and time, not money.** A few hundred MB of resident
  weights per checkpoint, roughly 0.13 s per decision on CPU.
* **Warm it up before a mission.** The first decision after a cold start may
  include loading (and on first use, downloading) a checkpoint. Probe `/health`
  first - `scripts/smoke_test_laya_api.py --health-only` is exactly that
  pre-flight, and exits non-zero rather than letting a mission start blind.
* **A dead server is a controlled landing, not a crash.** Every failure mode
  (unreachable, timeout, 401, 422, 429, 503, malformed reply, out-of-range
  confidence) becomes `valid=False` and routes to the deterministic fallback
  controller through the guardian. Nothing about an API or model error is ever
  permission to move.
* **Recorded per step:** the answering checkpoint (`provider_model`), token
  usage, and `decision_truncated` - Laya reports when it did not read the whole
  state, and a decision on partial evidence is a safety-relevant fact rather
  than a detail to be averaged away.

## Reproducibility

Every episode is a pure function of
`(seed, difficulty, preset, controller, threshold, resolved config)`:

* independent RNG streams for map generation, hazard stalls and mock confidence,
  so sweeping the confidence threshold cannot perturb terrain or physics;
* the resolved config is SHA-256 hashed into `outputs/run_metadata.json`;
* `tests/test_layer3_episode_e2e.py` asserts bit-identical replays;
* the paired analysis always compares controllers on the *same* `map_key`.

---

## Testing

```bash
python -m pytest -m "not live" -q      # Layers 1-3, offline, no secrets
```

| Layer | Scope | Representative tests |
|---|---|---|
| 1 | world, kinematics, guardian, perception | determinism, sentinel = `sensor_range + 1`, zero real collisions in a 120-step random walk under the guardian, pocket maps |
| 2 | decision layer contract | mock branch mix (80/10/10 + 5% unsafe probe), live adapter against a mock transport (endpoint, optional auth, body shape, 401/403/402/422/429/503/5xx, `Retry-After`, retries, timeout, redaction), `answer_confidence` fallback, fail-closed on a missing confidence, truncation capture, `/health` pre-flight, `LAYA_*` precedence, key never leaks |
| 3 | end-to-end | episode replay, threshold -> fallback routing, JSONL schema, all six figures, all CSVs, `run_metadata.json`, CLI, hardware bridge + serial parser |
| 4 | live Laya API | `scripts/smoke_test_laya_api.py` - **manual only**, excluded from CI; needs a running `laya-serve` (a key only if the server was started with one) |

New to the project? Start with
[`docs/VSCODE_RUN_AND_TEST_GUIDE.md`](docs/VSCODE_RUN_AND_TEST_GUIDE.md) - install,
run the tests, run the agent, read the results - and then
[`docs/CODEBASE_WALKTHROUGH.md`](docs/CODEBASE_WALKTHROUGH.md), which explains what
every file does and how one decision flows through the whole system.
[`docs/VSCODE_TESTING.md`](docs/VSCODE_TESTING.md) goes deeper on the test suite:
the Test Explorer, debugging an episode with breakpoints in the guardian and the
scorer, and testing against the live Laya API without ever committing a key.

---

## Results (mock decision layer, 30 seeds x 3 difficulties x 3 thresholds)

`outputs/report.md` is the source of truth; the headline from the shipped run:

| | baseline | Laya agent |
|---|---|---|
| success, **easy** | 0.33 | **1.00** |
| success, **medium** | 0.03 | **0.43** |
| success, **hard** | 0.00 | **0.07** |
| real collisions | 0 | **0** |
| mean energy per successful episode (easy) | 91.7 | **88.9** |

Paired deltas (laya_agent − baseline) with 95% bootstrap CIs: easy **+0.63**
(+0.47 … +0.80), medium **+0.40** (+0.20 … +0.60) - both excluding zero; hard
+0.07 (0.00 … +0.17), i.e. *not* significant. The guardian prevented 156 unsafe
or illegal actions across the sweep while the rover itself recorded zero
collisions.

Read these numbers as **pipeline validation**, not as evidence about the real
model: in `mock_laya` mode the decision layer is a deterministic stand-in. Live
evidence requires the `laya_api` run described in the guide.

---

## Documented deviations from the literal spec

Each of these is a deliberate, recorded engineering decision. The rationale lives
in the docstring at the point of implementation.

| # | Spec text | What the code does | Why |
|---|---|---|---|
| 1 | Stuck rule: "net displacement toward the goal over the last `stuck_window` positions is ~0" | Also requires that the rover discovered **no new cells** in that window (plus a two-cell escape hatch) | The literal rule also fires on a wide but legitimate detour around a big obstacle cluster; that would end healthy episodes. The spec's own trap presets need a policy that can backtrack |
| 2 | FR-4: "fixed forward-facing local window" of five rays | `front_left`/`front_right` are **parallel offset beams** (the corridor ahead), `left`/`right` are perpendicular | The alternative - making the diagonals duplicate the side rays - gave a controller no view of the corridor at all, and the mission became impossible whenever a wall stood ahead |
| 3 | FR-1: "concave trap obstacle clusters" | Pockets are designed **3 cells deep and visible within `sensor_range`**, placed on the start-goal line, with a slightly longer bypass | A pocket deeper than the sensor window is unwinnable for *every* local policy, so it measures nothing. Visible-avoidable pockets make the greedy-vs-conservative contrast decision-relevant; trap maps are 3 of every 4 seeds for the same reason |
| 4 | Baseline: greedy with tie-break `front > left > right > reverse` | `REVERSE` is used only when no forward/turn action is legal | As an equal-rank candidate it produced in-place front/reverse bounces at pocket mouths, which made the baseline fail for the wrong reason. It stays memoryless, hazard-blind and oscillation-prone |
| 5 | FR-6: "the adapter must return a validated decision" | A typed `choice` answer carrying **no** confidence is rejected (`SCHEMA_VIOLATION`) instead of defaulting to `1.0`; the legacy spec-15.3 chat envelope may still omit it | Defaulting a missing confidence to 1.0 grants a provider maximum trust for saying nothing. Tolerance is kept only where the spec explicitly allows an action-only reply |
| 6 | `unsafe_actions_prevented` | Boxed-in states (no legal action at all) are reported as `NO_LEGAL_ACTION` and are **not** counted as "prevented"; idle states are also loop-detected | Otherwise an episode that is stuck in a pocket for 100 idle steps inflates the safety case with actions nobody proposed |

## Limitations

* Simulation only: virtual terrain and sensing do not model dust, lighting,
  wheel slip, radiation, thermal limits or comms loss.
* Local API latency on a workstation is not onboard flight hardware latency.
* Model confidence is treated as a calibrated signal to be *verified* per
  deployment - `calibration.csv` is how; never as a guarantee.
* The guardian reduces risk in simulation and does not constitute formal
  verification or flight readiness.
* Mock-mode results validate the pipeline, not the model. Only `laya_api` runs
  measure the real model.
* The hardware layer publishes abstract actions to an existing rover controller;
  it never drives actuators directly.

## Acknowledgments

This project's decision layer is powered by [**Laya**](https://github.com/NandhaKishorM/laya),
a self-hosted, non-autoregressive decision engine created by [Nandha Kishor M](https://github.com/NandhaKishorM).
Laya is licensed under [Apache License 2.0](https://github.com/NandhaKishorM/laya/blob/main/LICENSE).

Laya Rover Agent is an independent, community-built application that *uses* Laya as its
decision engine. It is not an official project of, affiliated with, or endorsed by the
Laya project or its maintainer.