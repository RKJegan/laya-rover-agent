# Live Laya testing guide

A step-by-step path from "nothing installed" to "the real Laya model is steering
the rover", written so you can follow it top to bottom on your own machine.
Everything here is **manual and opt-in**: no CI job and no default `pytest` run
touches a live server.

> **What "live" means here.** Laya is a *self-hosted* decision engine, not a
> hosted API with an account. You run `laya-serve` on your own hardware; the
> agent talks to it over HTTP on localhost (or your LAN). There is no vendor, no
> billing and - by default - **no API key**.

---

## 0. What you need

| Requirement | Notes |
|---|---|
| Python 3.10+ | Laya's own floor; this project needs 3.11+ |
| Disk / RAM | the English checkpoint is ~808 MB to download; one resident checkpoint measures ~1.15 GB of RAM upstream, so allow ~2 GB free |
| Time | The very first prediction downloads weights from Hugging Face. Later runs use the local cache |
| GPU | Optional. CPU works and is what the numbers below assume |

Install the two halves **in separate environments or at least separately** - the
server needs PyTorch, the agent does not. For a step-by-step version of this section,
with per-OS commands, the deliberate weight pre-download and laptop tuning, see
[`INSTALL_LAYA_ON_YOUR_LAPTOP.md`](INSTALL_LAYA_ON_YOUR_LAPTOP.md):

```bash
# ---- the model server (heavy: torch + transformers) ----
pip install "laya[serve]"

# ---- the agent (light) ----
cd /path/to/laya-rover-agent
pip install -e ".[dev]"
```

---

## 1. Start the model server

In its own terminal, leave it running:

```bash
LAYA_MODELS=english LAYA_HOST=127.0.0.1 LAYA_DEVICE=cpu LAYA_PRELOAD=1 \
LAYA_THREADS=4 LAYA_IDLE_UNLOAD_SECONDS=600 laya-serve
#   ^ preload only the English checkpoint   ^ bind localhost   ^ CPU inference
#                  ^ cap CPU threads at the physical core count   ^ free the RAM when idle
```

| Variable | Why it matters here |
|---|---|
| `LAYA_MODELS` | Which checkpoints to preload. Preloading the one you will use removes the cold-load cost from your first decision |
| `LAYA_HOST` / `LAYA_PORT` | Bind address and port (defaults `0.0.0.0` / `8000`). Use `127.0.0.1` unless another machine must reach it |
| `LAYA_DEVICE` | `cpu`, `cuda`, `mps` - a preference, not a guarantee: Laya falls back to CPU rather than failing |
| `LAYA_API_KEY` | **Optional.** If set, the server requires `Authorization: Bearer <key>` |
| `LAYA_AUTO_TASK` | Leave `0`. Automatic workflow routing would swap the checkpoint per request, which weakens reproducibility |
| `LAYA_THREADS` | Cap on CPU threads. Set it at the **physical** core count - oversubscribing the logical count is a large slowdown on a laptop |
| `LAYA_IDLE_UNLOAD_SECONDS` | Unload a checkpoint after this many idle seconds, returning ~1.15 GB of RAM; the next request reloads it from cache |
| `LAYA_MAX_LOADED` | How many checkpoints stay resident (default 2). `1` is the laptop setting |

**Slow laptop?** The decision budget is `min(request_timeout_s, decision_timeout_ms/1000)`
- shipped at 1.2 s, sized for a GPU or a warm fast CPU. On a slower machine every step is
cancelled before the model finishes and silently falls back with `DECISION_TIMEOUT`. Layer
`configs/laptop_live.yaml` (8 s) onto any live command to fix it:

```bash
python -m laya_rover_agent episode --config configs/laptop_live.yaml --mode laya_api ...
python scripts/smoke_test_laya_api.py --config configs/laptop_live.yaml --show-raw
```
| `LAYA_MAX_TOKEN_BUDGET` | Server-side ceiling on per-request `max_len`/`head_max_len` |

`laya-serve` serves `POST /v1/systemone` (the endpoint this agent uses) plus
`GET /health`, `GET /models` and `POST /v1/systemone/batch`.

---

## 2. Point the agent at it

```bash
cp .env.example .env
```

The defaults already match a local server. Check them:

```bash
LAYA_API_URL=http://127.0.0.1:8000/v1/systemone
LAYA_MODEL=english
LAYA_REQUEST_TIMEOUT_S=5.0
LAYA_MAX_RETRIES=1
```

`LAYA_MODEL` accepts `english`, `multilingual`, `typed-decisions` or `auto`.
**Keep `english` pinned for experiments**: with `auto`, Laya's router chooses a
checkpoint per request, and two different checkpoints are two different models.
If you do use `auto`, the answering checkpoint is still recorded on every step
(`provider_model`), so a mixed run is detectable rather than invisible.

Environment values win over the YAML; the effective values are written into
`run_metadata.json`, so the report and the adapter can never disagree about
which endpoint was used.

---

## 3. Pre-flight, then one real decision

```bash
python scripts/smoke_test_laya_api.py --health-only   # is the server up?
python scripts/smoke_test_laya_api.py --show-raw      # one real decision
```

The script probes `/health` **first**, so an unreachable or still-loading server
is reported as such instead of being mistaken for a model failure. Then it sends
a single `choice` decision built from a synthetic rover observation - no map, no
rover, nothing moves - and prints:

```
HTTP status      : 200
valid            : True
answering model  : english   (the checkpoint that decided)
latency          : 128.4 ms
state truncated  : False   (True = the model read a cut-down state)
token usage      : {'input_tokens': 341.0, 'output_tokens': 0.0, 'state_tokens': 331.0}
chosen action    : MOVE_FORWARD
confidence       : 0.93
```

Exit codes: `0` decision received and validated · `1` server answered with an
error or something the adapter refused · `2` server unreachable · `3` server
demands a key you did not supply.

---

## 4. Watch the real model drive the loop

```bash
python scripts/smoke_test_laya_api.py --episodes 2 --preset trap_u --difficulty medium
```

Each line is a full episode:

```
episode 0: goal  steps=  74 energy=  87.4 laya_calls=  74 invalid=  1 fallbacks=  9 overrides=  3 prevented=  3
```

| Field | Meaning |
|---|---|
| `laya_calls` | decisions requested from the server (= one per moving step) |
| `invalid` | replies the adapter refused (malformed, out-of-set action, missing confidence) → fallback |
| `fallbacks` | steps routed to the deterministic fallback controller (low confidence, invalid, or error) |
| `overrides` / `prevented` | actions the safety guardian blocked or replaced, with the counterfactual recorded |

Try `--preset trap_twincup` and `--controller baseline` for contrast: the
baseline is greedy and hazard-blind by design.

---

## 5. A bounded live experiment

Laya is local, so the constraint is wall-clock time, not money: roughly 0.13 s
per decision on CPU. Use the confirmatory subset rather than the full sweep -
30 seeds × 3 difficulties × 3 thresholds is tens of thousands of decisions and
measures the same pipeline on a slower clock.

```bash
python -m laya_rover_agent estimate --config configs/default.yaml   # how many decisions?
python -m laya_rover_agent experiment --mode laya_api \
       --seeds 5 --difficulties medium --confidence-threshold 0.70 --run-label live-confirmatory
```

Artifacts land in `outputs/` exactly as in mock mode: per-step JSONL, CSVs, six
figures, `run_metadata.json` and `report.md`. What changes is that the decision
column now reflects the real model.

Careful with batch effects: the `laya_api` run uses a different clock and
different hardware than the `mock_laya` run. Compare *within* a mode (paired
agent-vs-baseline, which is what the experiment does), never across modes.

---

## 6. Read the calibration

`outputs/calibration.csv` is the FR-10.5 artifact:

| Column | Meaning |
|---|---|
| `confidence_bin` | bucket of reported confidence |
| `mean_confidence` | mean confidence inside the bin |
| `agreement_rate` | how often the decision matched the fallback controller's choice |
| `calibration_gap` | `mean_confidence - agreement_rate`; **positive = overconfident** |

Laya's confidence is natively calibrated (trained against strictly proper
scoring rules), so this is a genuine check rather than a formality - but
"agreement with the fallback" is a *proxy* for correctness, not ground truth.
Read a wide gap as "this threshold needs tuning for this deployment", not as a
verdict on the model.

---

## 7. Threshold sweep

```bash
python -m laya_rover_agent experiment --mode laya_api --seeds 3 \
       --difficulties medium --confidence-threshold 0.60,0.70,0.80 --run-label live-sweep
```

A higher threshold sends more steps to the fallback (safer, less model
influence); a lower one gives the model more authority. The interesting output
is where `unsafe_actions_prevented` and success rate cross over. Decision latency
should stay well inside `decision_timeout_ms` (1200 ms); if p95 approaches it,
you are paying for a cold checkpoint or a saturated server.

---

## 8. Safety checklist before you trust a run

- [ ] `--health-only` was green **before** the mission started.
- [ ] `run_metadata.json` names the endpoint and model you think it does.
- [ ] The logs contain **0** `collision` outcomes and 0 real collisions.
- [ ] `invalid` replies routed to the fallback as designed (never executed as-is).
- [ ] `provider_model` is the same on every step (a mixed run means `auto` routed).
- [ ] `decision_truncated` is `false`, or you understand why it is not.
- [ ] `python scripts/check_no_secrets.py outputs` prints `clean`.
- [ ] You accept the documented limits: simulation only; confidence treated as a
      calibrated signal to verify, never a guarantee; no flight-readiness claim.

---

## 9. When something goes wrong

| Symptom | Cause | Fix |
|---|---|---|
| `--health-only` exits 2 | server not running, or wrong `LAYA_API_URL` | start `laya-serve`; check host/port |
| exits 3, or 401/403 in a run | server was started with `LAYA_API_KEY` | put the same value in `.env`, set `auth_required: true` so a missing key fails closed |
| `RATE_LIMITED` (429/503) | server busy or a checkpoint is loading | preload with `LAYA_MODELS`, raise `LAYA_REQUEST_TIMEOUT_S`, or lower concurrency (`max_concurrent_requests`) |
| `SCHEMA_VIOLATION` (422) | our question body is malformed (duplicate/null choice label, short rubric) | client bug - the guardian already blocked it; report it rather than retrying |
| `DECISION_TIMEOUT` | first call after a cold start | warm up with a single decision, or raise the timeout for the first call |
| `MALFORMED_RESPONSE` repeatedly | wrong endpoint or a proxy rewriting the body | `--show-raw` to see the redacted payload; check for a reverse-proxy path prefix |
| Very slow decisions | CPU inference with a large checkpoint | pin `LAYA_MODELS=english`, set `LAYA_THREADS` to the physical core count, or use a GPU |
| `truncated: true` | the state exceeded the checkpoint's window | shorten the observation, or raise `max_len`/`head_max_len` on the request |

---

## 10. Resource hygiene

* One `laya-serve` for all your runs; restarting it re-pays the load cost.
* Preload exactly the checkpoint you use (`LAYA_MODELS=english`).
* Keep `LAYA_MAX_CONCURRENT` modest (the default 16 is fine for one agent).
* Stop the server when you are done - it holds hundreds of MB resident.
* On a rover, decide *before* the mission whether Laya runs onboard (then RAM is
  the constraint) or on the ground station (then the link is). Either way the
  fallback controller is what keeps the rover moving when the model is gone -
  that path is exercised by the offline tests, not only by this guide.
