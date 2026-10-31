# Testing this project in VS Code

Everything below works from a **fresh clone**. The repo ships a committed
`.vscode/` folder, so the Test Explorer, the debugger and the task runner are
already wired up - you do not have to configure anything by hand.

| File | What it gives you |
|---|---|
| `.vscode/settings.json` | pytest is the test framework, `src/` is on the analysis path, `.env` is the env file, `outputs/` is excluded from the file watcher |
| `.vscode/launch.json` | 10 ready-made debug targets: tests, CLI episodes, experiments, hardware-sim mission, live Laya smoke test |
| `.vscode/tasks.json` | 9 tasks in `Terminal → Run Task…`: install, test, coverage, runs, secret scan |
| `.vscode/extensions.json` | recommends the Python, Pylance and Debugpy extensions |

---

## 1. One-time setup (2 minutes)

1. **Open the folder** that contains `pyproject.toml`:

   ```bash
   code /path/to/laya-rover-agent          # or: File → Open Folder…
   ```

   If VS Code offers to install the recommended extensions, accept.

2. **Create and select a virtual environment.** Open a terminal in VS Code
   (`Ctrl+` `` ` ``) and run:

   ```bash
   # macOS / Linux
   python3 -m venv .venv

   # Windows (PowerShell)
   py -3 -m venv .venv
   ```

   Then `Ctrl+Shift+P` → **Python: Select Interpreter** → pick the one that ends
   in `.venv`. The status bar at the bottom-left should now show
   `Python 3.x.x ('.venv': venv)`.

   > If `.venv` does not appear in the list: `Ctrl+Shift+P` →
   > **Developer: Reload Window**, then try again. You can also type the path
   > directly (`Enter interpreter path…` → `./.venv/bin/python` on macOS/Linux,
   > `.\.venv\Scripts\python.exe` on Windows).

3. **Install the project and the test dependencies** in that interpreter:

   ```bash
   python -m pip install --upgrade pip
   python -m pip install -e ".[dev]"
   ```

   Or without the terminal: `Ctrl+Shift+P` → **Tasks: Run Task** →
   `setup: install editable + dev extras`.

   `-e` (editable) matters: it is what makes `import laya_rover_agent` work while
   you edit the sources. The tests also work without it, because
   `tests/conftest.py` puts `src/` on `sys.path` itself.

4. Optional, only if you want the serial backend: `python -m pip install pyserial`.

---

## 2. Run the tests (Test Explorer)

Click the **flask icon** in the left activity bar (or `Ctrl+Shift+P` →
**Testing: Focus on Test Explorer View**). You should see a tree of the
**185 tests** grouped by file (counts as of v3.0.0):

```
tests/
  test_layer1_world.py                   25 tests
  test_layer1_dynamics.py                10 tests
  test_layer1_safety_perception.py       13 tests
  test_layer1_controllers.py              5 tests
  test_layer2_laya_mock.py                9 tests
  test_layer2_laya_api_contract.py       43 tests
  test_layer2_security.py                10 tests
  test_layer3_episode_e2e.py             11 tests
  test_layer3_experiment_artifacts.py    29 tests
  test_layer3_cli_and_hardware.py        21 tests
  test_no_legacy_backend.py               9 tests   (the Laya-only guard)
```

What to do with it:

| Goal | How |
|---|---|
| run everything | the **▶ Run Tests** button at the top of the view (~10 s) |
| run one file | ▶ next to the file node |
| run one test | ▶ next to the test node |
| debug a test | the 🐞 **Debug Test** icon - breakpoints work, including inside `src/` |
| see a failure | click the red test - the message, the diff and the captured stdout appear inline |
| re-run only what failed | the ↻ **Rerun Failed Tests** button |

Nothing in this suite touches the network or needs an API key: Layer 2 exercises
the live Laya adapter against `httpx.MockTransport`, so the request contract,
retries, timeouts and error codes are all tested offline. (The `live` marker in
`pyproject.toml` exists for a future opt-in test; today the live path is the
script in section 4, so Test Explorer can never spend credits.)

Command line equivalent, if you prefer the terminal:

```bash
python -m pytest -q                     # everything
python -m pytest tests/test_layer1_dynamics.py -q
python -m pytest -q -k "trap or stuck"  # by name
```

---

## 3. Debug the agent step by step

This is where VS Code earns its keep: the whole decision pipeline is ordinary
Python, so you can stop anywhere.

Open the **Run and Debug** panel (`Ctrl+Shift+D`) and pick a configuration:

| Configuration | What it runs | Good for |
|---|---|---|
| `Tests: all offline (no network)` | the full suite under the debugger | chasing a failing invariant |
| `Tests: current file` | the file open in the editor | focused debugging |
| `Tests: only tests whose name matches the selection` | pytest `-k "<text you selected>"` | very fast narrowing |
| `CLI: episode trace (easy / trap_u / seed 0)` | `laya-rover episode …` with the step table | seeing a whole episode |
| `CLI: experiment (quick config, 4 seeds)` | a 48-episode sweep | pipeline bugs |
| `CLI: full sweep (30 seeds × 3 difficulties × 3 thresholds)` | 360 episodes | reproducing the shipped report |
| `Hardware: virtual simulator mission (no key needed)` | the connector layer + `RoverBridge` | validating the rover path offline |
| `Live Laya: smoke test, 1 real decision` | health probe + 1 decision against your `laya-serve` | connectivity check |
| `Live Laya: 2 live episodes` | the real model driving the full loop | end-to-end live run |
| `Live Laya: offline rehearsal (no network, no key)` | the live flow with `mock_laya` | rehearsing with no server |

Press `F5`, or click ▶. Debugging the episode trace is the most instructive
route - set breakpoints at:

* `src/laya_rover_agent/episode.py` → inside the `while` loop: see `state`,
  `observation`, `safe.actions`, `approval`;
* `src/laya_rover_agent/safety.py` → `approve()`: watch a proposal get rejected
  and a reason code appear;
* `src/laya_rover_agent/controllers/fallback.py` → `choose()`: watch the score
  terms (`distance`, hazard margin, pocket penalty, memory penalty) decide;
* `src/laya_rover_agent/dynamics.py` → `step()`: watch energy, stalls and status.

Useful while stopped: hover any variable, **Debug Console** to evaluate
expressions (`len(result.timesteps)`, `approval.reason_code`), **Watch** to pin
`state.battery` or `outcome.terminal`, and **Call Stack** to see the exact route
from the CLI to the guardian.

To inspect one specific episode, edit the args of the `CLI: episode trace`
configuration in `launch.json` (`--seed`, `--difficulty`, `--preset`,
`--controller`) - the presets are `random`, `trap_u`, `trap_canyon`,
`trap_twincup`.

---

## 4. Test against live Laya from VS Code

> Talks to a server you start yourself. Nothing here runs automatically.

Laya is self-hosted, so step zero is starting it in a terminal:

```bash
pip install "laya[serve]"
LAYA_MODELS=english LAYA_HOST=127.0.0.1 laya-serve     # leave running
```

1. Create the env file once:

   ```bash
   cp .env.example .env      # Windows: copy .env.example .env
   ```

   The defaults already point at a local server. **No key is needed** unless
   you started `laya-serve` with `LAYA_API_KEY` set. `.env` is **git-ignored** -
   verify with `git status` that it never appears.

2. Run the config **`Live Laya: offline rehearsal (no network, no key)`** first.
   It exercises the identical code path with the mock decision layer, so you
   know the plumbing works before touching the server.

3. Run the task **`check: laya-serve pre-flight (/health)`** (or the config
   **`Live Laya: smoke test, health probe + 1 real decision`**,
   `envFile: ${workspaceFolder}/.env`). The Debug Console prints the endpoint,
   the parsed action, confidence, probabilities, latency, the answering
   checkpoint and the truncation flag. A non-zero exit names the cause: `2`
   server unreachable, `3` server wants a key, `1` the reply was refused
   (401/403 auth, 422 malformed request, 429/503 busy, timeout).

4. Run **`Live Laya: 2 live episodes`**, or check the volume first with the task
   `run: estimate live decision volume` (one decision per moving timestep).

5. For a real (bounded) live experiment, use the terminal or this task pattern:

   ```bash
   python -m laya_rover_agent estimate --config configs/default.yaml
   python -m laya_rover_agent experiment --mode laya_api --seeds 5 \
          --difficulties medium --confidence-threshold 0.70 --run-label live-confirmatory
   ```

   Full step-by-step live procedure: [`docs/live_laya_testing_guide.md`](live_laya_testing_guide.md).

6. After any live session, run the task **`check: no secret material in outputs`**
   (or `python scripts/check_no_secrets.py outputs`). It must print `clean`.

---

## 5. Look at the results without leaving the editor

After a run (`run: experiment (quick)` or the full sweep), open:

| File | In VS Code |
|---|---|
| `outputs/report.md` | `Ctrl+Shift+V` for the rendered Markdown (images, tables, all 10 sections) |
| `outputs/figures/*.png` | click the file - images preview inline |
| `outputs/episode_summaries.csv`, `aggregate_metrics.csv`, `paired_stats.csv` | the CSV/table previewer (Rainbow CSV or Data Wrangler extension gives sorting/filtering) |
| `outputs/episode_logs/*.jsonl` | one JSON object per line: observation → raw answer → guardian verdict → latency |
| `outputs/run_metadata.json` | config SHA-256, package versions, git hash, timestamps |

Handy keyboard workflows:

* **Find a specific guardian verdict**: `Ctrl+Shift+F`, search
  `"guardian_reason_code": "BLOCKED_OBSTACLE"`, scope the search to `outputs/`.
* **Compare two episodes**: `Ctrl+P`, type `compare outputs/episode_logs/…` and
  pick **Compare Selected** to diff two JSONL logs side by side.
* **Jump from a figure to its source**: click the figure, then `Alt+F12`
  (Peek Definition) on the file name in `report.md`'s figure section.

---

## 6. Tasks (Terminal → Run Task…)

| Task | Command it runs |
|---|---|
| `setup: install editable + dev extras` | `python -m pip install -e ".[dev]"` |
| `test: offline suite (layers 1-3)` | `python -m pytest -q` |
| `test: coverage report` | `pytest --cov=laya_rover_agent …` (needs `pip install pytest-cov`), HTML into `outputs/coverage` |
| `run: episode trace` | `laya-rover episode …` |
| `run: experiment (quick)` / `run: experiment (full sweep)` | `laya-rover experiment …` |
| `run: hardware mission (simulator backend)` | `laya-rover hardware --backend sim …` |
| `run: estimate live API call volume` | `laya-rover estimate …` |
| `check: no secret material in outputs` | `python scripts/check_no_secrets.py` |

All tasks use `${command:python.interpreterPath}`, so they always run in the
interpreter you selected - no `PATH` surprises.

---

## 7. Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| Test Explorer is empty | pytest not installed in the selected interpreter, or the wrong folder is open | `python -m pip install -e ".[dev]"`; make sure the folder containing `pyproject.toml` is the workspace root; `Ctrl+Shift+P` → **Python: Configure Tests** → pytest → `tests` |
| `ModuleNotFoundError: pydantic_settings` | interpreter is not the venv you installed into | **Python: Select Interpreter** → `.venv` |
| `ModuleNotFoundError: laya_rover_agent` | package not installed and tests run from somewhere else | `python -m pip install -e .` (or run pytest from the repo root) |
| `ImportError: partially initialized module 'enum'` | `python` was launched with the working directory inside `src/laya_rover_agent` | run from the repo root, or set `"cwd": "${workspaceFolder}"` in the launch config (already the default here) |
| Live debug config fails with "Cannot read env file" | `.env` does not exist yet | `cp .env.example .env` (the defaults are fine for a local server) |
| Live run exits `2` | `laya-serve` is not running, or `LAYA_API_URL` is wrong | start the server; check host/port |
| Live run exits `3`, or 401/403 mid-run | the server was started with `LAYA_API_KEY` | set the same key in `.env` and `auth_required: true` |
| ROS 2 backend raises `BackendUnavailableError`, `pyserial` too | optional hardware deps are not installed | `pip install pyserial` / source your ROS 2 install; or use `--backend sim` |
| Plots fail to render headlessly | no display | the code uses the `Agg` backend already; if you added interactive plotting, `export MPLBACKEND=Agg` |
| VS Code feels sluggish after a sweep | 360 JSONL files are watched | already handled: `files.watcherExclude` covers `outputs/`; reload the window after very large runs |
| A test hangs on a live key test | there is none | live verification is `scripts/smoke_test_laya_api.py`, never pytest |

---

## 8. CI parity

CI runs exactly what the Test Explorer runs, so green here means green there:

```bash
python -m pip install -e ".[dev]"
python -m pytest -m "not live" -q     # layers 1-3, zero network, no secrets
```

The live layer is manual by design: `python scripts/smoke_test_laya_api.py`
(see `docs/live_laya_testing_guide.md`).

---

## 9. Where this fits

| Document | For |
|---|---|
| [`VSCODE_RUN_AND_TEST_GUIDE.md`](VSCODE_RUN_AND_TEST_GUIDE.md) | first-run setup: install, run the agent, read the results, live Laya, real hardware |
| [`CODEBASE_WALKTHROUGH.md`](CODEBASE_WALKTHROUGH.md) | what every file does and how the pieces are wired |
| this file | the test suite in depth: markers, fixtures, writing new tests |
