# Installing Laya on your laptop — the complete download-and-connect guide

**The question this answers:** *"How do I get Laya onto my laptop and connect it to this
project so I can test the real model?"*

**The short answer:** there are **two downloads**, not one, and people usually miss the
second:

| # | What | Size | How |
|---|---|---|---|
| 1 | **The Laya program** (the Python package) | ~2–3 GB with its dependencies (PyTorch) | `pip install "laya[serve]"` |
| 2 | **The model weights** (the checkpoint) | **~808 MB** for the English model | Happens automatically on first use, or download it on purpose (Section 4) |

Nothing needs an account, a payment method, or a key. Both downloads come from the
public PyPI and Hugging Face, and Laya is Apache-2.0 licensed.

**Before you start — will it run on my laptop?**

| Check | Minimum | Comfortable |
|---|---|---|
| Python | 3.10 (this project needs 3.11+, so 3.11 or 3.12 is the sweet spot) | 3.12 |
| Free disk | ~4 GB | ~6 GB |
| Free RAM | ~1.5 GB (one checkpoint resident is measured at ~1.15 GB) | ~3 GB |
| GPU | not needed — CPU-only works | any CUDA/MPS GPU just makes it faster |
| Internet | needed once, for both downloads | — |
| Speed on CPU | a decision takes roughly 0.2–0.5 s on a modern laptop CPU | under 0.2 s on a GPU |

If your laptop is tight on space or RAM, that is fine too — you can keep using the
offline `mock_laya` mode for everything (the whole test suite, the 360-episode sweep,
all figures) and only install Laya for a short confirmation run. Sections 9 and 10 show
how to keep the footprint small and how to remove it afterwards.

---

## 0. Decide where Laya lives

You have two options, and the first is the one I recommend:

**Option A (recommended): a separate environment just for the model server.**

```
laya-rover-agent/.venv          ← the agent (light: no torch)
~/laya-server/.venv             ← Laya + PyTorch (heavy) + the weights cache
```

Why: PyTorch is giant, Laya updates independently of this project, and if the model
server ever breaks you can rebuild it without touching the agent's environment. The
two talk over HTTP, so they don't need to share anything.

**Option B: one environment for both.** Simpler to remember, one fewer terminal to
manage, but the agent's environment becomes multi-gigabyte and its `pip freeze` no
longer tells you anything about the agent. It works fine — just be aware.

The rest of this guide assumes **Option A**. Section 8 shows the small differences for
Option B.

> Commands are given for **macOS/Linux** first and **Windows PowerShell** second.
> Run them in order — each one verifies itself before you move on.

---

## 1. Check your Python

macOS / Linux
```bash
python3 --version      # need 3.10 or newer
```

Windows PowerShell
```powershell
py --version
```

If that fails or shows something older than 3.10, install Python from
[python.org](https://www.python.org/downloads/) (on Windows, tick **"Add python.exe to
PATH"** during setup) and try again.

## 2. Create the Laya environment

```bash
# macOS / Linux
mkdir -p ~/laya-server && cd ~/laya-server
python3 -m venv .venv
```

```powershell
# Windows PowerShell
mkdir $HOME\laya-server; cd $HOME\laya-server
py -m venv .venv
```

(On Debian/Ubuntu, if `venv` complains that `ensurepip` is unavailable, run
`sudo apt install python3-venv` first.)

## 3. Install the Laya program

```bash
# macOS / Linux
.venv/bin/python -m pip install "laya[serve]"
```

```powershell
# Windows PowerShell
.\.venv\Scripts\python.exe -m pip install "laya[serve]"
```

Three details in that one command:

- **`laya`** is the core package: the model runtime (via PyTorch and Hugging Face's
  `transformers`).
- **`[serve]`** adds the HTTP server — a few small web packages. **This is what you
  need**, because this project talks to Laya over HTTP. The plain `laya` package has no
  server.
- **Nothing is downloaded by this step except code.** No weights yet.

Expect this to take a few minutes on a normal connection: PyTorch alone is hundreds of
megabytes (more on Windows/macOS than Linux). If your laptop has a GPU and you want the
GPU build, install the matching PyTorch wheel from
[pytorch.org](https://pytorch.org/get-started/locally/) *before* the line above — but
CPU is perfectly fine for testing.

**Check it worked** (this prints a version number and does **not** download the model):

```bash
# macOS / Linux
.venv/bin/python -I -c "import laya; print(laya.__version__)"
```
```powershell
# Windows PowerShell
.\.venv\Scripts\python.exe -I -c "import laya; print(laya.__version__)"
```

If you see `ModuleNotFoundError: No module named 'laya'`, you are almost certainly
running a different Python than the one you installed into — use the exact paths above.

## 4. Download the model weights

This is the part that makes the download "Laya" as most people mean it. The weights
live on Hugging Face (`convaiinnovations/laya`), and there are three checkpoints:

| Checkpoint | What it is | Download |
|---|---|---|
| **`english`** | 421M-parameter English model. **This is the one this project uses** | ~808 MB |
| `multilingual` | 322M parameters, 100+ languages, faster | ~647 MB |
| `typed-decisions` | 421M parameters, tuned for its own benchmark workflows | ~808 MB |

**You only ever need the English one.** This project asks for it by name
(`LAYA_MODEL=english` in `.env`, or the `model` field in `configs/*.yaml`), which also
keeps runs reproducible — the same checkpoint answers every step.

You have two ways to get it:

**Option 1 — let the server fetch it on first start (simplest).** Skip to Section 5;
the first start downloads the English checkpoint automatically and takes several
minutes. You will see progress in the log.

**Option 2 — download it deliberately first (recommended on a slow connection).** This
way the big download happens while you are not staring at a server that looks stuck:

```bash
# macOS / Linux
.venv/bin/python -c "import laya; laya.load('convaiinnovations/laya'); print('english checkpoint downloaded')"
```
```powershell
# Windows PowerShell
.\.venv\Scripts\python.exe -c "import laya; laya.load('convaiinnovations/laya'); print('english checkpoint downloaded')"
```

**Where it lands:** Hugging Face's cache, by default
`~/.cache/huggingface/hub` (macOS/Linux) or
`C:\Users\<you>\.cache\huggingface\hub` (Windows). Once it is there, Laya never
downloads it again, and you can run with no internet connection at all.

Want the weights somewhere else — a different drive, an external SSD, or a folder you
back up? Set the cache location *before* downloading:

```bash
# macOS / Linux
export HF_HUB_CACHE=$HOME/laya-server/hf-cache
```
```powershell
# Windows PowerShell
$env:HF_HUB_CACHE = "$HOME\laya-server\hf-cache"
```

## 5. Start the model server

Open a terminal you can leave running (in VS Code: **Terminal → New Terminal**, or use
a split terminal — one for the server, one for the agent). Then:

```bash
# macOS / Linux
cd ~/laya-server
LAYA_MODELS=english LAYA_HOST=127.0.0.1 LAYA_DEVICE=cpu LAYA_PRELOAD=1 .venv/bin/laya-serve
```
```powershell
# Windows PowerShell
cd $HOME\laya-server
$env:LAYA_MODELS="english"; $env:LAYA_HOST="127.0.0.1"; $env:LAYA_DEVICE="cpu"; $env:LAYA_PRELOAD=1
.\.venv\Scripts\laya-serve.exe
```

What each part is doing — these are the settings that matter on a laptop:

| Setting | Value | Why |
|---|---|---|
| `LAYA_MODELS` | `english` | Preload **only** the checkpoint this project uses. The default is all three, which would pull ~2.4 GB and hold more RAM than you need |
| `LAYA_HOST` | `127.0.0.1` | Accept connections from this laptop only. The default (`0.0.0.0`) would expose the model to your whole local network |
| `LAYA_DEVICE` | `cpu` (or `cuda` / `mps` / `xpu`) | It is a preference, not a demand: if the GPU is unavailable, Laya quietly uses the CPU. Check yours first: `python scripts/check_laya_device.py` (run it in the environment that has torch) prints which device is usable and the exact command to start with |
| `LAYA_PRELOAD` | `1` | Load the checkpoint at startup, so the first decision isn't the one that waits |

**Using a GPU.** `LAYA_DEVICE=cuda` is worth it when the machine has an NVIDIA GPU the
installed PyTorch can see - about 33 ms per question on a T4 against roughly 130-140 ms
on CPU. The usual reason a GPU sits unused is a **CPU-only PyTorch build**;
`scripts/check_laya_device.py` detects exactly that and prints the fix. AMD GPUs are not
usable on Windows; Intel via `LAYA_DEVICE=xpu` with a matching torch build.

**Optional, but kind to a laptop:**

| Setting | Example | Why |
|---|---|---|
| `LAYA_THREADS` | `4` | Caps CPU threads. Set it to your **physical** core count (not hyperthreads) — oversubscribing is a large slowdown |
| `LAYA_IDLE_UNLOAD_SECONDS` | `600` | Unloads the checkpoint after 10 idle minutes, giving ~1.2 GB of RAM back. The next request reloads it (a few seconds from cache) |
| `LAYA_PORT` | `8010` | Only if something else already uses 8000 — remember to match it in the agent's `.env` |

**Leave this terminal running.** The server is a separate program from the agent; if
you close it, the agent will simply fall back to its safe controller (which is itself a
useful test — see Section 7).

**What a healthy start looks like:** a line or two of logging, then something like
`Uvicorn running on http://127.0.0.1:8000`. The very first start (if you skipped
Section 4) shows download progress first and can take several minutes; later starts
take a few seconds.

## 6. Connect this project to it

In your **agent** terminal (the `laya-rover-agent` folder, its own `.venv` activated):

**Step 1 — the pre-flight check.** This is the tool that tells you whether the two
halves are talking, before any experiment:

```bash
python scripts/smoke_test_laya_api.py --health-only
```

- Server up → prints the health payload including the model's device, exit code 0.
- Nothing listening → a clear "could not connect" message, exit code 2. Go back to
  Section 5.

**Step 2 — one real decision.** This sends an actual observation to the real model and
validates the reply:

```bash
python scripts/smoke_test_laya_api.py --episodes 1 --show-raw
```

`--show-raw` prints the raw reply (with anything key-shaped redacted). Exit codes:
`0` a valid decision came back · `1` the server answered with an error · `2`
unreachable · `3` the server demanded a key and none was set.

**Step 3 — run the agent against the real model.** Nothing needs configuring: the
defaults already point at `http://127.0.0.1:8000/v1/systemone` with
`model=english`. Just ask for live mode:

```bash
# one episode, watched step by step
python -m laya_rover_agent episode --mode laya_api --seed 0 --difficulty easy \
       --preset trap_u --controller laya_agent --log-jsonl outputs/live_trace.jsonl

# a small experiment, live
python -m laya_rover_agent experiment --mode laya_api --seeds 2 --difficulties easy \
       --run-label live-check --output-dir outputs_live
```

In VS Code, the same things are ready-made: task **`check: laya-serve pre-flight
(/health)`**, and the launch configurations **`Live Laya: smoke test, health probe + 1
real decision`** and **`Live Laya: 2 live episodes`** (F5).

**Only if your laptop or server needs different settings**, create a `.env` file in the
project root (copy `.env.example`) and set:

```ini
LAYA_API_URL=http://127.0.0.1:8000/v1/systemone
LAYA_MODEL=english
LAYA_REQUEST_TIMEOUT_S=5.0
```

`.env` is git-ignored, so nothing there can be committed by accident.

## 7. What "it works" looks like — and how to prove it

In the live trace, every row shows where the decision came from:

```
--log-jsonl outputs/live_trace.jsonl   →   "action_source": "laya"       ← the model decided
                                           "action_source": "fallback"   ← it couldn't be trusted, the safe controller drove
```

A healthy first episode has mostly `laya` rows, with `fallback` rows carrying a reason
code like `LOW_CONFIDENCE`. Also visible per row: `provider_model` (which checkpoint
answered), `decision_truncated` (whether the model had to read a shortened state), and
`raw_laya_confidence`.

**A deliberate failure worth doing once.** Stop the server (Ctrl-C in its terminal) and
rerun the episode above. Every row becomes `"action_source": "fallback"` with
`"guardian_reason_code": "CONNECTION_ERROR"`, the episode still finishes, exit code 0,
and nothing unsafe happens. That is the project's central guarantee, live, in front of
you: **the rover never moves on a model error.**

## 8. If you chose Option B (one environment for everything)

Skip Section 2 and install Laya into the agent's own environment:

```bash
cd /path/to/laya-rover-agent
source .venv/bin/activate                # Windows: .venv\Scripts\activate
pip install "laya[serve]"                # heavy: adds torch
laya-serve                               # then run it from the same terminal
```

Everything else is identical. The only real cost is size, and that `pip install -e
".[dev]"` and Laya now share a dependency tree, so an aggressive Laya upgrade could in
principle disturb the agent's packages. If that happens: recreate the environment and
reinstall the agent alone.

## 9. Live testing without melting your laptop

Live mode is for **confirming** the real model behaves like the offline stand-in — a
handful of seeds. It is not for the statistical sweep.

- **Ask first:** `python -m laya_rover_agent estimate` prints exactly how many model
  calls a configuration would make. A single episode can be up to 500 steps; 360
  episodes in live mode would be tens of thousands of decisions.
- **Start tiny:** `--seeds 2 --difficulties easy` finishes in a couple of minutes on
  CPU and tells you what you need to know.
- **Keep the two halves in separate terminals**, so you can watch the server log while
  the agent runs.
- **Let the model answer only once per step.** It already does — one decision per
  rover step, then the guardian rules on it.
- **When done, stop the server** (Ctrl-C). Nothing is left running in the background.
- **If every step falls back with `DECISION_TIMEOUT`:** the laptop answers slower than
  the shipped 1.2-second decision budget. Layer `configs/laptop_live.yaml` onto any live
  command (`--config configs/laptop_live.yaml`); it raises the budget to 8 s. Tune it back
  down once you have seen `latency_mean_ms` in an episode summary.
- **Want it quieter?** `LAYA_IDLE_UNLOAD_SECONDS=600` frees ~1.2 GB between runs,
  `LAYA_THREADS=<physical cores>` stops the CPU spinning, and `LAYA_PORT=8010` moves it
  off a busy port.

## 10. Removing it again / reclaiming space

```bash
# uninstall the program (removes the environment and PyTorch with it)
rm -rf ~/laya-server/.venv

# delete the downloaded weights (they re-download on demand if you come back)
rm -rf ~/.cache/huggingface/hub/models--convaiinnovations--laya
```
```powershell
Remove-Item -Recurse -Force $HOME\laya-server\.venv
Remove-Item -Recurse -Force "$HOME\.cache\huggingface\hub\models--convaiinnovations--laya"
```

The agent keeps working exactly as before in `mock_laya` mode — the offline suite, the
full sweep and all artifacts never needed Laya in the first place.

## 11. Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `ModuleNotFoundError: No module named 'laya'` | ran a different Python than installed into | use the full paths in Sections 3–5, or activate the environment first |
| `laya-serve: command not found` | installed plain `laya`, not `laya[serve]` | `pip install "laya[serve]"` |
| First start seems to hang for minutes | it is downloading ~808 MB of weights | wait (or Ctrl-C and use Option 2 in Section 4, which shows progress) |
| `Address already in use` on port 8000 | something else owns the port | start Laya with `LAYA_PORT=8010`, and put the same port in `LAYA_API_URL` |
| `could not connect` from the smoke test (exit 2) | server not running, on another port, or bound elsewhere | check the server terminal; confirm `LAYA_HOST=127.0.0.1` and that `LAYA_API_URL` matches `LAYA_PORT` |
| First decision is slow (several seconds) | checkpoint loading into memory | later decisions are fast; keep `LAYA_PRELOAD=1` |
| Every step says `fallback` / `DECISION_TIMEOUT` | your CPU needs longer per decision than the shipped 1200 ms budget | use `--config configs/laptop_live.yaml` (raises the budget to 8 s); still too tight? raise `decision_timeout_ms` further |
| `503` with a "busy" body | more concurrent requests than the server accepts | the agent retries once and then falls back; reduce `--jobs`, raise `LAYA_MAX_CONCURRENT` |
| `401` / exit code 3 from the smoke test | the server was started with `LAYA_API_KEY` | put the same value in `.env` (`LAYA_API_KEY=…`) and set `LAYA_AUTH_REQUIRED=true`, or restart the server without a key |
| `422` from the server | the request it received was malformed — a bug on our side, not yours | capture the `raw_response` from the trace and report it; the step still fails closed |
| Laptop fans, everything sluggish | CPU oversubscribed or too many threads | `LAYA_THREADS=<physical cores>`; close other heavy apps; consider `LAYA_DEVICE=mps` on Apple Silicon |
| Out of memory | more than one checkpoint resident | `LAYA_MODELS=english` and `LAYA_MAX_LOADED=1` |
| Chatty log lines during a run | normal server logging | `LAYA_LOG_LEVEL=warning` |
| Windows: crash on model construction, Python 3.14 | a bug fixed in Laya 0.3.7 | `pip install -U laya` (or use Python 3.12) |

## 12. The whole thing, on one screen

```bash
# ---------- once ----------
mkdir -p ~/laya-server && cd ~/laya-server
python3 -m venv .venv
.venv/bin/python -m pip install "laya[serve]"
.venv/bin/python -c "import laya; print('laya', laya.__version__)"
.venv/bin/python -c "import laya; laya.load('convaiinnovations/laya'); print('weights cached')"

# ---------- each session ----------
# terminal 1: the model server
cd ~/laya-server
LAYA_MODELS=english LAYA_HOST=127.0.0.1 LAYA_DEVICE=cpu LAYA_PRELOAD=1 \
LAYA_THREADS=4 LAYA_IDLE_UNLOAD_SECONDS=600 .venv/bin/laya-serve

# terminal 2: the agent
cd /path/to/laya-rover-agent
source .venv/bin/activate
python scripts/smoke_test_laya_api.py --health-only          # is the server up?
python scripts/smoke_test_laya_api.py --episodes 1 --show-raw # one real decision
python -m laya_rover_agent episode --mode laya_api --seed 0 --difficulty easy \
       --preset trap_u --controller laya_agent
```

## 13. Where to go next

| Document | For |
|---|---|
| `docs/VSCODE_RUN_AND_TEST_GUIDE.md` | running everything from VS Code, including the live configurations |
| `docs/live_laya_testing_guide.md` | the live flow in depth: bounded runs, what the artifacts record, operational notes |
| `docs/CODEBASE_WALKTHROUGH.md` | what happens to a decision inside the agent, start to finish |
| Laya's own docs — [nandhakishorm.github.io/laya](https://nandhakishorm.github.io/laya/) and the [repository](https://github.com/NandhaKishorM/laya) | the model itself: checkpoints, hooks, fine-tuning, benchmarks |
