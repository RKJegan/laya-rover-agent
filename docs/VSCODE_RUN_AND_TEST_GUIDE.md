# Run and test the agent on your machine with VS Code

**Who this is for:** you, on your own laptop or desktop, using Visual Studio Code.
No prior knowledge of this codebase is assumed.

**On Windows?** Use [`WINDOWS_SETUP_GUIDE.md`](WINDOWS_SETUP_GUIDE.md) instead of Sections 1-3 and 7 - same steps, Windows commands, nothing to guess.

**What you will have at the end:** the project installed, the full test suite green in
VS Code, the agent running a simulated mission, and a folder of results (tables,
figures and a written report) you can open and read. The last two sections are
optional: running against a real Laya model, and connecting a real rover.

**What you need first**

| Requirement | Notes |
|---|---|
| Python **3.11 or newer** | `python3 --version` (macOS/Linux) or `py --version` (Windows) |
| VS Code | any recent version |
| VS Code extensions | `ms-python.python` + `ms-python.debugpy` (Python + debugger), `ms-python.vscode-pylance` recommended — VS Code will offer them from `.vscode/extensions.json` |
| Disk | about 400 MB for the environment and packages |
| Extra installs | **none** for the offline path. The real Laya model is a separate, optional step (Section 7) and is the only part that needs a big download |
| Time | the offline path below is a 5–10 minute setup |

> Everything in Sections 1–6 works **without internet** after the packages are
> installed, and without any API key. The agent's decisions come from a built-in
> mock decision layer that behaves exactly like the real one, minus the model.

---

## 0. The short version

If you have done this before and just want the commands:

```bash
cd laya-rover-agent                 # the project folder
python3 -m venv .venv && source .venv/bin/activate      # Windows: py -m venv .venv ; .venv\Scripts\activate
pip install -e ".[dev]"
python -m pytest -q                 # expect: 185 passed
laya-rover experiment --config configs/quick.yaml --mode mock_laya --seeds 4 --run-label vscode
```

Then open `outputs/report.md`. Everything else in this guide explains *why* each
step is there and how to do the same things from VS Code buttons instead of the
terminal.

---

## 1. Open the project

1. In VS Code: **File → Open Folder…**, pick the `laya-rover-agent` folder.
   (Do not open the parent folder — VS Code must see `.vscode/`, `src/`,
   `tests/` as direct children.)
2. VS Code may ask **"Do you trust the authors of this folder?"** — choose *Yes,
   trust*. The folder contains tasks and debug configurations, which only run
   after you trust it.
3. A notification may appear: **"This workspace has extension recommendations"** →
   *Install*. That installs the Python and debugpy extensions.

You should now see, in the Explorer sidebar:

```
.vscode/        launch, task and settings files for this project
configs/        the two YAML configurations (default = full sweep, quick = fast)
docs/           guides, including this one
scripts/        smoke test and secrets scanner (run from the terminal)
src/            the agent's source code
tests/          185 automated tests
outputs/        results land here (created when you run experiments)
pyproject.toml  package definition: name, dependencies, the `laya-rover` command
```

## 2. Create the Python environment

Do this once.

**From VS Code (no typing commands):**

1. Open the Command Palette: `Ctrl+Shift+P` (Windows/Linux) or `Cmd+Shift+P` (macOS).
2. Type **Python: Create Environment** and press Enter.
3. Choose **Venv**, then choose the interpreter listed (e.g. *Python 3.12*).
4. Tick **`pyproject.toml`** when VS Code asks which dependencies to install, then
   confirm. VS Code creates `.venv/` in the project and installs everything.

**Or from the VS Code terminal** (`Ctrl+`` ` to open it):

macOS / Linux
```bash
python3 -m venv .venv
source .venv/bin/activate
```

Windows (PowerShell)
```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
```

**Then confirm VS Code is using it:** look at the bottom-right of the window (or
press `Ctrl+Shift+P` → *Python: Select Interpreter*). It must say something like
`.venv (Python 3.12.x)`. If it says "system" or another folder's environment, select
`.venv` manually — a wrong interpreter is the number-one cause of confusing errors
later.

## 3. Install the project

**From VS Code:** press `Ctrl+Shift+P` → **Tasks: Run Task** → **`setup: install
editable + dev extras`**.

**Or in the terminal:**

```bash
pip install -e ".[dev]"
```

Three things to understand about this one command:

- **`-e` means "editable".** Python installs a pointer to the `src/` folder instead
  of a copy. Edit a file, rerun a test, and your change is live — no reinstall.
- **`.[dev]`** adds the two test-only packages (`pytest`, `respx`). There is also
  `.[hardware]`, which you only need for a real serial connection to a rover.
- It installs a **command-line program called `laya-rover`**, which is how you run
  the agent from the terminal: `laya-rover --help`.

Check it worked:

```bash
laya-rover --help          # lists: experiment, episode, hardware, map, plan, estimate
```

## 4. Run the tests — your first green run

The suite is the project's contract: **185 tests, no network, about 7 seconds**.

**Four ways to run it in VS Code, pick whichever you like:**

| Way | How |
|---|---|
| Task (fastest) | `Ctrl+Shift+P` → **Tasks: Run Task** → **`test: offline suite (layers 1-3)`** |
| Debug (with breakpoints) | Press **F5**, choose **`Tests: all offline (no network)`** |
| Test Explorer (visual) | Click the flask icon in the left sidebar — every test appears as a checkbox you can run individually |
| Terminal | `python -m pytest -q` |

**What "passing" looks like:**

```
185 passed in 6.31s
```

If you want to see the tests counted one by one in the Test Explorer:

| Test file | Tests | What it protects |
|---|---|---|
| `test_layer1_world.py` | 25 | Maps are deterministic; trap presets really trap; a route exists |
| `test_layer1_dynamics.py` | 10 | Movement, energy costs, stalls, stuck detection |
| `test_layer1_safety_perception.py` | 13 | The guardian's rules and the five sensor beams |
| `test_layer1_controllers.py` | 5 | The fallback policy and the baseline behave as designed |
| `test_layer2_laya_mock.py` | 9 | The mock decision layer is 80/10/10 + 5% unsafe, seeded |
| `test_layer2_laya_api_contract.py` | 43 | The HTTP conversation with the real Laya server |
| `test_layer2_security.py` | 10 | Keys never appear in logs or tracebacks |
| `test_layer3_cli_and_hardware.py` | 21 | The six commands, and the rover connector layer |
| `test_layer3_episode_e2e.py` | 11 | Whole episodes, including bit-identical replays |
| `test_layer3_experiment_artifacts.py` | 29 | The whole experiment run and every output file |
| `test_no_legacy_backend.py` | 9 | The project stays Laya-only, everywhere |

**Run one test while you work on it:** open the file, put the cursor on the test,
click the little ▶ in the gutter next to the function name. Or press F5 with the
**`Tests: current file`** configuration.

> Note on the tests for the real Laya server (the 43 in
> `test_layer2_laya_api_contract.py`): they do **not** call a real server. They
> feed the code recorded HTTP answers — including every error Laya can return —
> and check the agent reacts correctly. That is what makes the suite runnable on
> a plane.

## 5. Run the agent

Everything below runs the **same engine** the tests exercise, but for real, and
writes results to `outputs/`.

### 5a. See a map first (2 seconds)

```bash
laya-rover map --seed 0 --preset trap_u
```

You get an ASCII picture: `#` walls, `~` hazard slopes, `S` start, `G` goal. The
`trap_u` preset is a U-shaped pocket — a trap that a naive "always head towards
the goal" policy walks into. Try `--preset trap_canyon` and `--preset trap_twincup`
too.

### 5b. Watch one episode step by step (recommended first run)

Press **F5** → **`CLI: episode trace (easy / trap_u / seed 0)`**, or:

```bash
laya-rover episode --seed 0 --difficulty easy --preset trap_u --controller laya_agent --max-rows 40
```

You get the map, then a table with one row per decision:

| column | meaning |
|---|---|
| `step` | how many steps have been taken |
| `action` | what the rover actually did |
| `source` | **who decided**: `laya` (Laya's choice was used), `fallback` (Laya was unusable/unsure, so the safe default drove), `guardian_override` (the guardian rejected the proposal), `baseline` |
| `laya_conf` | how sure the decision layer said it was |
| `guardian` | the machine-readable reason for the final ruling, e.g. `APPROVED`, `LOW_CONFIDENCE`, `BLOCKED_OBSTACLE` |
| `moved` / `battery` / `goal_dist` | the consequence |

At the end you get a JSON summary. Add `--log-jsonl outputs/trace.jsonl` to keep the
full per-step record (every field, not just the table columns).

### 5c. Run an experiment (the numbers and the report)

```bash
# quick: 48 episodes, ~4 seconds
laya-rover experiment --config configs/quick.yaml --mode mock_laya --seeds 4 --run-label vscode

# full: 360 episodes, ~15 seconds
laya-rover experiment --config configs/default.yaml --mode mock_laya --run-label full-mock
```

Or use the tasks/launch configs: **`run: experiment (quick)`** and
**`CLI: full sweep (30 seeds x 3 difficulties x 3 thresholds)`**.

An *episode* is one map driven by one controller: the Laya agent tries it, and the
baseline policy tries the same map with the same random draws, so the two can be
compared fairly. The quick run does 4 seeds × 3 difficulties × (3 confidence
thresholds for Laya + 1 baseline) = **48 episodes**; the full sweep does 30 seeds
per difficulty = **360 episodes**.

### 5d. Drive the virtual rover

The `hardware` command runs the *real* mission loop against a rover interface —
sensors in, action out — and ships with a virtual rover so you can try it safely.

```bash
laya-rover hardware --backend sim --preset trap_u --difficulty easy --seed 0 \
                    --controller laya_agent --log outputs/mission.jsonl
```

Or press **F5** → **`Hardware: virtual simulator mission (no key needed)`**.
This is the closest thing to a real rover without owning one: it is the same code
path a ROS 2 or serial rover uses, only the transport is simulated. Add `--dry-run`
to let it decide, log and guard without ever moving.

## 6. Read the results

Every run writes into `outputs/`:

| File | What to look at |
|---|---|
| **`report.md`** | **Start here.** A written report: what was run, the headline table, the safety case, calibration, limitations |
| `run_metadata.json` | Exactly which settings produced this run: versions, the config fingerprint (`config_hash_sha256`), the mode (`mock_laya`), timings |
| `episode_summaries.csv` | One row per episode: success, steps, energy, collisions, guardian overrides, latency |
| `aggregate_metrics.csv` | The headline table: success/collision/stuck rates and averages, per controller × difficulty × threshold |
| `paired_stats.csv` | Laya vs baseline on identical maps: mean difference, 95% confidence interval, whether it excludes zero |
| `calibration.csv` | Does "confidence 0.8" mean "right 8 times out of 10"? Bins, and the gap |
| `ablation.csv` | The safety case: unsafe actions the guardian stopped, and how many would have hit an obstacle |
| `reason_codes.csv` | Every guardian reason code and how often it fired |
| `episode_logs/*.jsonl` | The full per-step record of every episode — the raw truth behind all the tables |
| `figures/*.png` | 6 charts: success by difficulty, energy, example paths, latency, ablation, calibration |

Viewing CSV files in VS Code is much nicer with the **Data Wrangler** or **Rainbow
CSV** extension — install one and click any `.csv` to get a sortable grid. Images
open in place; `report.md` renders with `Ctrl+Shift+V`.

**In VS Code:** press `Ctrl+P` and type `report.md` to jump straight to the
report. Figures: open the `figures` folder, click a `.png`, and use `←`/`→`
(preview mode) to flick through them.

**What a healthy run looks like** (mock mode, threshold 0.70, full sweep):

| controller | easy | medium | hard |
|---|---|---|---|
| `laya_agent` | 1.00 | 0.43 | 0.07 |
| `baseline` | 0.33 | 0.03 | 0.00 |

(success rate; collisions are 0 for both). The paired table shows the easy and
medium gaps with 95% intervals that exclude zero. In mock mode these numbers are
deterministic — you should reproduce them exactly, and the run's
`config_hash_sha256` should be
`4cd1ae4938a2577a7e77f211afe71d6594d5b1553403cba74f6d6fad175e2acb`.

---

## 7. Optional: run against a real Laya model

The offline path above uses a stand-in. To use the real decision model, you run Laya
as a small server **on your own machine** (nothing is sent to the internet) and point
the agent at it.

**Cost warning, up front:** Laya's checkpoints are roughly 1–2 GB and are downloaded
once; the smallest useful one needs about 1.2 GB of RAM at runtime. If you are on a
metered connection or a small laptop, stay in mock mode — everything else is identical.

```bash
# 1. install Laya with its server, in the same environment
pip install "laya[serve]"

# 2. start the server (it downloads the checkpoint on first use)
LAYA_MODELS=english laya-serve
```

> On Windows PowerShell the environment prefix does not work the same way; use
> `$env:LAYA_MODELS="english"; laya-serve` instead. Leave the server running in its
> own terminal — it is a separate program from the agent.

```bash
# 3. pre-flight: is the server up and answering?
python scripts/smoke_test_laya_api.py --health-only

# 4. one real decision, plus the raw reply so you can see exactly what came back
python scripts/smoke_test_laya_api.py --episodes 1 --show-raw

# 5. two real episodes
python scripts/smoke_test_laya_api.py --episodes 2 --preset trap_u --difficulty medium
```

In VS Code: **`check: laya-serve pre-flight (/health)`** task, then F5 →
**`Live Laya: smoke test, health probe + 1 real decision`**.

The smoke script returns a distinct exit code so you can tell what went wrong:
`0` = a decision came back and validated, `1` = the server answered with an error,
`2` = the server could not be reached, `3` = the server demanded a key and none was set.

**What to expect:** first request can take a while (the model loads once); after
that a decision is a few hundred milliseconds on CPU. If the server is busy it
returns "busy" and the agent retries once, then falls back — the rover never moves
on an unanswered request. Nothing needs to be configured: the agent defaults to
`http://127.0.0.1:8000/v1/systemone` and the `english` checkpoint. A `laya-serve`
started without `LAYA_API_KEY` needs no key; if you set one on the server, put the
same value in `.env` (`LAYA_API_KEY=...`) and the agent will send it. `.env` is
git-ignored; `.env.example` documents every setting.

**The full walk-through of the download — including per-OS commands, pre-downloading the
weights, and laptop tuning — is in [`INSTALL_LAYA_ON_YOUR_LAPTOP.md`](INSTALL_LAYA_ON_YOUR_LAPTOP.md).**

**Where this fits:** live runs are for *confirming* that the real model behaves like
the stand-in on a handful of seeds. The statistical sweep (360 episodes) is meant to
be run in mock mode. See `docs/live_laya_testing_guide.md` for the long version.

## 8. Optional: connect a real rover

There are two transports beyond the simulator:

| `--backend` | Use it when | Needs |
|---|---|---|
| `sim` | always start here; virtual rover in a generated world | nothing |
| `ros2` | the rover publishes/subscribes ROS 2 topics | ROS 2 installed + sourced, `rclpy` available |
| `serial` | the rover talks over a USB/UART link | `pip install -e ".[hardware]"`, the port name |

```bash
# ROS 2: topics under /laya_rover (odometry, battery, five range beams in, one action out)
laya-rover hardware --backend ros2 --ros2-namespace /laya_rover --dry-run --max-steps 20

# serial: a rover on /dev/ttyUSB0
ROVER_SERIAL_PORT=/dev/ttyUSB0 laya-rover hardware --backend serial --dry-run --max-steps 20
```

**Safety habit:** run `--dry-run` first every time. The rover then decides, logs and
guards, but nothing moves. When you are ready, drop `--dry-run` — the CLI prints a
warning before it will move anything, and `Ctrl-C` always requests an emergency stop.

The wire formats (exact topic names, message types, the serial line protocol) are in
`docs/HARDWARE_DEPLOYMENT.md`.

## 9. Everyday workflow in VS Code

**Debug a single step of the agent.** F5 → `CLI: episode trace …`, then set a
breakpoint by clicking left of the line number in `episode.py` (the mission loop).
When it stops, hover over variables, and use the Call Stack to see how you got
there. Useful places to pause:

- `src/laya_rover_agent/episode.py` — one loop iteration = one decision
- `src/laya_rover_agent/safety.py` → `approve()` — watch a proposal get rejected
- `src/laya_rover_agent/controllers/fallback.py` → `choose()` — watch the scores
- `src/laya_rover_agent/dynamics.py` → `step()` — watch energy and stalls

**Debug a test.** Put a breakpoint in the test, then F5 → `Tests: current file`.

**Measure coverage.** Task **`test: coverage report`** writes an HTML report to
`outputs/coverage/index.html`; open it in a browser to see which lines the tests
never touch.

**Check you are not leaking secrets.** Task **`check: no secret material in outputs`**
(or `python scripts/check_no_secrets.py outputs`) scans the result files for anything
that looks like a key. Run it before sharing a run with anyone.

**Save your changes.** The Source Control panel (`Ctrl+Shift+G`) shows this project's
changes. Everything from the migration is currently staged but **not committed** —
commit when you are happy with it.

## 10. Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `ModuleNotFoundError: laya_rover_agent` | not installed in the selected interpreter | run the `setup: install editable + dev extras` task; then *Python: Select Interpreter* → `.venv` |
| Tests show as "not discovered" in the Test Explorer | wrong interpreter, or pytest not installed | select `.venv`; rerun the setup task |
| `pytest: command not found` | environment not activated | activate it (`source .venv/bin/activate`, or `.venv\Scripts\activate`) |
| A test that passed yesterday now fails after edits | a real regression, or a config change | run that one file to see the assertion; read the failure message — it names the invariant |
| `Invalid value for '--mode'` | typo | the only modes are `mock_laya` and `laya_api` |
| `401` from the server | server was started with `LAYA_API_KEY` but the agent has no key | put the same value in `.env` as `LAYA_API_KEY=…`, or restart the server without it |
| `503` with "busy" | model still loading, or too many at once | wait, then rerun; the agent retries once by itself |
| First live decision very slow / timeout | the checkpoint is loading | wait for it to finish loading, then rerun. If *every* step says `DECISION_TIMEOUT`, add `--config configs/laptop_live.yaml` (raises the decision budget for a slow CPU) |
| `Address already in use` on port 8000 | another program (or an older `laya-serve`) is using it | stop it, or set `LAYA_PORT=8010` and `LAYA_API_URL=http://127.0.0.1:8010/v1/systemone` in `.env` |
| `laya-serve: command not found` | Laya installed without the server extra | `pip install "laya[serve]"` |
| Serial backend fails to import | `pyserial` missing | `pip install -e ".[hardware]"` |
| ROS 2 backend fails to import | `rclpy` missing / ROS not sourced | `source /opt/ros/<distro>/setup.bash` (Linux) before starting VS Code |
| Menus/tasks greyed out | folder not trusted | *Workspace: Manage Workspace Trust* → trust the folder |
| Windows: `Activate.ps1` blocked | PowerShell execution policy | `Set-ExecutionPolicy -Scope Process RemoteSigned`, then activate again |

## 11. Quick reference

**Commands**

| Command | What it does |
|---|---|
| `laya-rover map --seed 0 --preset trap_u` | print one map |
| `laya-rover episode --seed 0 --preset trap_u` | one episode with a step table |
| `laya-rover experiment --config configs/quick.yaml --mode mock_laya` | the full pipeline + report |
| `laya-rover hardware --backend sim --preset trap_u` | drive the virtual rover |
| `laya-rover plan` | list the experiment matrix without running it |
| `laya-rover estimate` | how many model calls a live run would make |
| `python -m pytest -q` | the whole test suite |
| `python scripts/smoke_test_laya_api.py --health-only` | is my Laya server up? |
| `python scripts/check_no_secrets.py outputs` | scan results for secrets |

**Useful flags:** `--config FILE` (layer another YAML), `--mode mock_laya|laya_api`,
`--threshold 0.7`, `--seeds N`, `--difficulties easy,medium`, `--jobs N` (parallel
episodes), `--output-dir DIR`, `--run-label NAME`, `--dry-run` (hardware only),
`--log-jsonl FILE`.

**VS Code tasks** (Command Palette → *Tasks: Run Task*): `setup: install editable + dev extras`, `test: offline suite (layers 1-3)`, `test: coverage report`,
`run: episode trace`, `run: experiment (quick)`, `run: experiment (full sweep)`,
`run: hardware mission (simulator backend)`, `run: estimate live decision volume`,
`check: laya-serve pre-flight (/health)`, `check: no secret material in outputs`.

**Launch configurations** (F5): `Tests: all offline (no network)`, `Tests: current file`, `Tests: only tests whose name matches the selection`, `CLI: episode trace`,
`CLI: experiment (quick config, 4 seeds)`, `CLI: full sweep`, `Hardware: virtual simulator mission`, `Live Laya: smoke test`, `Live Laya: 2 live episodes`,
`Live Laya: offline rehearsal`.

## 12. Where to read next

| Document | For |
|---|---|
| `docs/CODEBASE_WALKTHROUGH.md` | what every file does and how they are wired together |
| `docs/VSCODE_TESTING.md` | the test suite in detail: markers, fixtures, writing new tests |
| `docs/INSTALL_LAYA_ON_YOUR_LAPTOP.md` | downloading and connecting the real model, step by step |
| `docs/live_laya_testing_guide.md` | the long version of Section 7 |
| `docs/HARDWARE_DEPLOYMENT.md` | the long version of Section 8: topics, serial protocol, deployment checklist |
| `README.md` | what the project is, design rules, the headline tables |
| `MIGRATION.md` / `CHANGELOG.md` | how the project got here, and what changed in each version |
