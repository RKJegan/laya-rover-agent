# Windows setup guide — nothing to live Laya model

**Follow the steps in order. Do not skip any.**

Every command goes in **PowerShell**, opened inside VS Code.
(Open it: **Terminal → New Terminal** in VS Code, or press `` Ctrl+` ``.)

**What you will have at the end:**

- [ ] Tests passing (185)
- [ ] The agent running offline
- [ ] The real Laya model running and connected
- [ ] One live episode finished

**Before you start:**

- Windows 10 or 11
- About 6 GB free disk space
- Internet connection
- 30–40 minutes

---

# PART 1 — Install the project

## Step 1. Install Python 3.12

1. Go to: https://www.python.org/downloads/
2. Click **Download Python 3.12**.
3. Run the installer.
4. **TICK** the box **"Add python.exe to PATH"**. (Easy to miss. Step 2 fails without it.)
5. Click **Install Now**.

## Step 2. Check Python

```
py -3.12 --version
```

**You should see:** `Python 3.12.x`

**If you see an error:** redo Step 1 and tick the PATH box.

## Step 3. Put the project on your laptop

1. Download `laya-rover-agent-source.zip` (it is in this workspace).
2. Right-click the downloaded zip → **Extract All**.
3. Move the extracted `laya-rover-agent` folder to: `C:\dev\`

**You should have:** `C:\dev\laya-rover-agent\`

**Keep the path short like this.** Deep folders cause Windows long-path errors.

**Do not** put it inside OneDrive or Desktop — OneDrive sync breaks the environment.

## Step 4. Open it in VS Code

1. Open VS Code.
2. **File → Open Folder…** → choose `C:\dev\laya-rover-agent`.
3. Click **Yes, I trust the authors** if asked.
4. Click **Install** if it offers recommended extensions.

## Step 5. Create the environment

```
cd C:\dev\laya-rover-agent
py -3.12 -m venv .venv
```

**You should see:** no errors. A `.venv` folder appears in the project.

## Step 6. Activate the environment

```
.\.venv\Scripts\Activate.ps1
```

**You should see:** `(.venv)` at the start of the command line.

**If you see a red "running scripts is disabled" error:**

```
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

Type `Y` and press Enter. Then run the activate command again.

## Step 7. Install the project

```
pip install -e ".[dev]"
```

**Takes:** 1–2 minutes.

**You should see:** `Successfully installed ... pytest ... respx ...`

## Step 8. Run the tests

```
python -m pytest
```

**You should see:** `185 passed`

**If tests fail:** run `pip install -e ".[dev]"` again. Still failing? Send me the error text.

## Step 9. Run the agent offline

```
python -m laya_rover_agent map --seed 0 --preset trap_u
```

**You should see:** an ASCII map with `S`, `G`, `#` symbols.

```
python -m laya_rover_agent episode --seed 0 --difficulty easy --preset trap_u --controller laya_agent
```

**You should see:** a step table, then a JSON summary.

## Step 10. Run the full experiment

```
python -m laya_rover_agent experiment --config configs/default.yaml --mode mock_laya --run-label windows-mock
```

**Takes:** about 15 seconds.

**You should see:** lines ending with `report -> outputs/report.md`

**Open** `outputs\report.md` in VS Code (`Ctrl+Shift+V` to see it nicely).

---

**PART 1 DONE.** The agent works on your laptop.

---

# PART 2 — Install Laya (the real model)

Laya goes in a **second folder**, separate from the project.

## Step 11. Make the Laya environment

```
cd C:\dev
mkdir laya-server
cd laya-server
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**You should see:** `(.venv)` at the start of the line.

## Step 12. Install Laya

```
pip install "laya[serve]"
```

**Takes:** several minutes. This downloads PyTorch (large).

**You should see:** `Successfully installed ... laya ... fastapi ... uvicorn ...`

```
python -I -c "import laya; print(laya.__version__)"
```

**You should see:** a version number.

## Step 13. Download the model (one time, 808 MB)

```
python -c "import laya; laya.load('convaiinnovations/laya'); print('model downloaded')"
```

**Takes:** several minutes. Progress lines are normal.

**You should see:** `model downloaded`

It is saved in `C:\Users\<you>\.cache\huggingface`. You never download it again.

## Step 14. Start the model server

**Open a SECOND terminal:** in VS Code click the **split terminal** button, or press `` Ctrl+Shift+` ``.

In that second terminal:

```
cd C:\dev\laya-server
.\.venv\Scripts\Activate.ps1
$env:LAYA_MODELS="english"; $env:LAYA_HOST="127.0.0.1"; $env:LAYA_DEVICE="cpu"; $env:LAYA_PRELOAD=1; $env:LAYA_THREADS=4
laya-serve
```

**You should see:** `Uvicorn running on http://127.0.0.1:8000`

**LEAVE THIS TERMINAL RUNNING. Do not close it.**

## Step 14b. Use your GPU (optional - makes decisions faster)

**Nothing in this project blocks the GPU.** It is one setting on the server.

**Step 1. Check what your laptop can use.** In the **laya-server** terminal:

```
python ..\laya-rover-agent\scripts\check_laya_device.py
```

Or, from the project terminal:

```
python scripts\check_laya_device.py
```

**Step 2. Do what it says:**

| It prints | Do this |
|---|---|
| `VERDICT: GPU available - start the server with LAYA_DEVICE=cuda` | restart the server with `cuda` (below) |
| `VERDICT: no GPU visible ... CPU-only build (the usual cause)` | install the CUDA build of PyTorch from https://pytorch.org/get-started/locally/ (pick Pip + Windows + your CUDA), then rerun the checker |
| `NOT INSTALLED in this interpreter` | wrong terminal - use the one where you ran `pip install "laya[serve]"` |

**If the checker says CPU-only build** (`torch.version.cuda = None`, like `2.14.1+cpu`) - exact commands, in the laya-server terminal:

1. Does the laptop have an NVIDIA GPU?

```
nvidia-smi
```

(A table with your GPU name = yes. `CUDA UMD Version` at the top right tells you the newest CUDA the driver supports - 13.x here means the cu130 build is the best match.)

2. **Replace the CPU build.** Two traps to avoid:

- `pip install torch==2.14.1 --index-url ...` does **nothing** if the CPU build is already installed: pip treats `2.14.1` and `2.14.1+cpu` as the same version and reports "Requirement already satisfied". Name the CUDA build exactly.
- Use the index that actually has a Windows wheel for torch 2.14.1: **cu130** or **cu126**. `cu128` and `cu129` have no 2.14.1 Windows wheel.

```
pip install torch==2.14.1+cu130 --index-url https://download.pytorch.org/whl/cu130
```

(About a 2 GB download. If `download.pytorch.org` cannot be reached, see the DNS fixes below, then use `+cu126` / `https://download.pytorch.org/whl/cu126` instead - a CUDA 12.6 build runs fine on a newer driver.)

Plan B if pip still resists:

```
pip uninstall -y torch
pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cu130
```

(While torch is uninstalled the server cannot start - if this fails, restore the CPU build with `pip install torch==2.14.1` and try again later.)

**If the download fails with `getaddrinfo failed` / connection errors:**

```
Resolve-DnsName download.pytorch.org | Select-Object -First 3
```

- Prints an IP address -> just retry the install.
- Error -> DNS cannot find the host. Try, in order:

```
ipconfig /flushdns
nslookup download.pytorch.org 1.1.1.1
```

If `nslookup ... 1.1.1.1` works but the plain one does not, your router/ISP DNS is the problem: set the adapter's DNS to `1.1.1.1` and `8.8.8.8` (Settings -> Network & Internet -> your adapter -> Edit DNS), then retry. Otherwise retry later or use a phone hotspot - it is usually transient filtering.

3. Verify (must print `True` and your GPU name):

```
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

Expected: `2.14.1+cu130 True NVIDIA GeForce RTX 4050 Laptop GPU`

**Step 3. Restart the server on the GPU.** Stop it (Ctrl-C), then:

```
cd C:\dev\laya-server
.\.venv\Scripts\Activate.ps1
$env:LAYA_MODELS="english"; $env:LAYA_HOST="127.0.0.1"; $env:LAYA_DEVICE="cuda"; $env:LAYA_PRELOAD=1
laya-serve
```

**Step 4. Confirm.** In the project terminal:

```
python scripts\smoke_test_laya_api.py --health-only
```

The `device:` line must say `cuda`, not `cpu`.

**Two things to know:**

- Setting `cuda` when no GPU is usable **does not break anything**. Laya treats the device as a preference and quietly uses the CPU.
- **Keep `--config configs\laptop_live.yaml` as well.** GPU or not, a cold model load or a busy laptop can still be slow; the bigger budget is what stops a slow answer from becoming a wasted step.

**How much faster?** Laya's own measurements: about 33 ms per question on a T4 GPU, against roughly 130-140 ms on CPU. On your `easy / trap_u / seed 0` episode (73 decisions) that is the difference between about 10 seconds and under 3.

---

## Step 15. Pre-flight check

**Go back to the FIRST terminal** (the project folder):

```
cd C:\dev\laya-rover-agent
.\.venv\Scripts\Activate.ps1
python scripts/smoke_test_laya_api.py --health-only
```

**You should see:** a health result and exit with no error.

**If it says "could not connect":** the server is not running. Redo Step 14.

## Step 16. One real decision

```
python scripts/smoke_test_laya_api.py --episodes 1 --show-raw
```

**You should see:** a real reply from the model.

**First one is slow. That is normal.**

## Step 17. Run a live episode

```
python -m laya_rover_agent episode --mode laya_api --seed 0 --difficulty easy --preset trap_u --controller laya_agent --log-jsonl outputs/live_trace.jsonl
```

**You should see:** a step table, then a JSON summary.

Check who decided:

```
python -c "import json,collections;rows=[json.loads(l) for l in open('outputs/live_trace.jsonl')];print(collections.Counter(r['action_source'] for r in rows))"
```

**You should see:** mostly `'laya'`.

**If you see only `'fallback'`:** see the Troubleshooting table below.

## Step 17b. If you see `DECISION_TIMEOUT` (every row says `fallback`)

**What it means:** the model *is* working, but one decision takes longer on your laptop than the 1.2-second budget allows, so the request is cut off before the model answers.

**The safety system did its job:** the episode still finished, drove to the goal, no collisions. But the model's answers are never used.

**Fix it in 2 steps.**

1. Make the budget bigger:

```
@"
agent:
  decision_timeout_ms: 8000.0
  laya_api:
    request_timeout_s: 12.0
"@ | Out-File -Encoding utf8 configs\laptop_live.yaml
```

(If that errors, the file also ships in the updated project zip - just copy `configs\laptop_live.yaml`.)

2. Re-run with it:

```
python -m laya_rover_agent episode --config configs\laptop_live.yaml --mode laya_api --seed 0 --difficulty easy --preset trap_u --controller laya_agent --log-jsonl outputs\live_trace.jsonl
```

**You should see:** `source` shows `laya`, and `guardian` shows `APPROVED`.

**Still all `fallback`?** Your laptop needs even longer: change `8000.0` to `15000.0` and rerun.

**Want to speed the model up?** In the **server** terminal, stop it (Ctrl-C) and restart it without the `LAYA_THREADS` line, or set it to your physical core count. Close other heavy apps first.

---

## Step 18. Prove the safety rule

1. Go to the server terminal (Step 14). Press **Ctrl-C**.
2. Re-run the Step 17 command.
3. **You should see:** every row becomes `fallback` and `CONNECTION_ERROR`.
4. **The episode still finishes. Nothing crashes. No collision.**

**Restart the server** (rerun Step 14) when you are done.

---

**PART 2 DONE.** The real model is running with your project.

---

# PART 3 — Optional

## Step 19. Save your work

```
cd C:\dev\laya-rover-agent
git add -A
git commit -m "Windows setup complete"
```

(Only works if the project has git history. If it says "not a git repository", skip this step.)

## Step 20. Run everything from VS Code buttons

Press **Ctrl+Shift+P** → type **Tasks: Run Task** → pick one:

- `test: offline suite (layers 1-3)`
- `run: experiment (quick)`
- `check: laya-serve pre-flight (/health)`

Press **F5** to debug instead, and pick:

- `CLI: episode trace (easy / trap_u / seed 0)`
- `Live Laya: 2 live episodes`

## Step 21. Real rover (later)

Practice with no rover first:

```
python -m laya_rover_agent hardware --backend sim --preset trap_u --difficulty easy --seed 0 --dry-run
```

Real serial rover: find your port in **Device Manager → Ports (COM & LPT)**.

```
$env:ROVER_SERIAL_PORT="COM3"
python -m laya_rover_agent hardware --backend serial --dry-run --max-steps 20
```

**Always run with `--dry-run` first.** It decides and logs but never moves.

---

# Troubleshooting

| What you see | What it means | Do this |
|---|---|---|
| `python is not recognized` | PATH box not ticked | redo Step 1, tick "Add python.exe to PATH" |
| `Activate.ps1 cannot be loaded` | PowerShell blocks scripts | Step 6 fix command, type `Y` |
| `No module named 'laya_rover_agent'` | wrong environment active | run `.\.venv\Scripts\Activate.ps1` in the project folder |
| `No module named 'laya'` | wrong environment active | activate the **laya-server** environment |
| `laya-serve is not recognized` | Laya venv not activated, or missing `[serve]` | activate it, then `pip install "laya[serve]"` |
| Server start "hangs" minutes | it is downloading the model | wait (Step 13 can do it visibly instead) |
| `could not connect` | server not running | redo Step 14 |
| Only `'fallback'` in the trace | model not reachable or too slow | check the server terminal; then use `--config configs\laptop_live.yaml` (Step 17b) |
| `401` | server started with a key | add `LAYA_API_KEY=...` to `.env`, or restart server without one |
| Port 8000 in use | another program uses it | `$env:LAYA_PORT="8010"` and set `LAYA_API_URL=http://127.0.0.1:8010/v1/systemone` in `.env` |
| Laptop slow / fans loud | CPU oversubscribed | `$env:LAYA_THREADS="4"` (your physical cores) before starting the server |
| `device: cpu` but my laptop has a GPU | PyTorch is the CPU-only build | run `scripts\check_laya_device.py`, then install the CUDA build (Step 14b) |
| `DECISION_TIMEOUT` even on the GPU | cold load, or the GPU build is not actually being used | confirm `device: cuda` in the health check; raise the budget with `--config configs\laptop_live.yaml` |
| Out of memory | too many models loaded | `$env:LAYA_MODELS="english"; $env:LAYA_MAX_LOADED=1` |
| Long path / file name too long | folder too deep | move the project to `C:\dev\` |
| Everything slow after first run | OneDrive syncing the venv | move the project out of OneDrive |

---

# Windows cheat sheet

**Activate an environment**

| Terminal | Command |
|---|---|
| PowerShell | `.\.venv\Scripts\Activate.ps1` |
| Command Prompt (cmd) | `.venv\Scripts\activate.bat` |
| Back to normal | `deactivate` |

**Set environment variables (PowerShell only)**

| Purpose | Command |
|---|---|
| One session, several values | `$env:LAYA_MODELS="english"; $env:LAYA_DEVICE="cpu"` |
| Serial rover port | `$env:ROVER_SERIAL_PORT="COM3"` |
| See one value back | `$env:LAYA_MODELS` |
| Permanent (all future terminals) | `setx LAYA_MODELS english` (then open a NEW terminal) |

PowerShell uses `$env:NAME="value"`. The `NAME=value` style is Linux — it does not work here.

**Paths**

| What | Where |
|---|---|
| Project | `C:\dev\laya-rover-agent` |
| Laya server | `C:\dev\laya-server` |
| Model weights | `C:\Users\<you>\.cache\huggingface` |
| Results | `C:\dev\laya-rover-agent\outputs` |

---

# When you finish

Open `NEXT_STEPS.md` and continue from **STEP 2**.

**Stuck?** Copy the exact error text and send it to me.
