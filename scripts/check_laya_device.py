#!/usr/bin/env python3
"""Which device will Laya actually use on this machine?

Run this **inside the environment where Laya is installed** (the one that has
PyTorch). It answers the only question that matters before setting
``LAYA_DEVICE``:

* is there a GPU this PyTorch build can see?
* if not, is it because there is no GPU, or because the installed PyTorch is the
  CPU-only build (the usual reason - ``pip install laya`` pulls whichever wheel
  matches the platform, and on Windows that is often CPU-only)?

It changes nothing and downloads nothing: pure inspection.

Usage::

    # in the laya-server environment
    python scripts/check_laya_device.py
    # or without activating:  .venv\\Scripts\\python.exe scripts\\check_laya_device.py

Exit codes: 0 = a GPU is usable (use LAYA_DEVICE=<printed value>), 1 = CPU only,
2 = PyTorch is not installed in this interpreter.
"""

from __future__ import annotations

import platform
import sys


def main() -> int:
    print(f"interpreter : {sys.executable}")
    print(f"python      : {platform.python_version()} on {platform.system()} {platform.machine()}")

    try:
        import torch
    except ImportError:
        print("\ntorch       : NOT INSTALLED in this interpreter")
        print("\nThis is the wrong environment, or Laya is not installed yet. From the")
        print("environment that runs laya-serve:")
        print('    pip install "laya[serve]"')
        print("then run this script again with that environment's python.")
        return 2

    print(f"torch       : {torch.__version__}   (built for CUDA: {torch.version.cuda or 'no'})")

    cuda = bool(torch.cuda.is_available())
    mps_available = bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())
    xpu_available = bool(getattr(torch, "xpu", None) and torch.xpu.is_available())

    print(f"cuda        : {cuda}" + (f"   {torch.cuda.get_device_name(0)}" if cuda else ""))
    print(f"mps (Apple) : {mps_available}")
    if hasattr(torch, "xpu"):
        print(f"xpu (Intel) : {xpu_available}")

    print()
    if cuda:
        print("VERDICT: GPU available - start the server with LAYA_DEVICE=cuda")
        print("  (Windows PowerShell)")
        print('  $env:LAYA_DEVICE="cuda"; $env:LAYA_MODELS="english"; $env:LAYA_HOST="127.0.0.1"; laya-serve')
        print("  (macOS/Linux)")
        print("  LAYA_DEVICE=cuda LAYA_MODELS=english LAYA_HOST=127.0.0.1 laya-serve")
        return 0
    if mps_available:
        print("VERDICT: Apple GPU available - start the server with LAYA_DEVICE=mps")
        print("  LAYA_DEVICE=mps LAYA_MODELS=english LAYA_HOST=127.0.0.1 laya-serve")
        return 0
    if xpu_available:
        print("VERDICT: Intel GPU (XPU) available - start the server with LAYA_DEVICE=xpu")
        return 0

    print("VERDICT: no GPU visible to this PyTorch build - Laya will use the CPU.")
    print("That works, and the agent handles it (raise the decision budget with")
    print("--config configs/laptop_live.yaml). To use a GPU, in order:")
    print("  1. does the machine have one?")
    print("       NVIDIA : nvidia-smi            (prints a table = yes)")
    print("       Apple  : any M-series Mac     (use LAYA_DEVICE=mps instead)")
    print("  2. is this a CPU-only PyTorch build?")
    if platform.system() == "Darwin":
        print("       macOS wheels include MPS support; check step 3.")
    else:
        print(f"       torch.version.cuda = {torch.version.cuda!r} -> "
              f"{'CUDA build, but no usable driver/GPU detected' if torch.version.cuda else 'CPU-only build (this is the usual cause)'}")
        print("       fix: install the CUDA wheel for this machine from")
        print("       https://pytorch.org/get-started/locally/  (select Pip + your OS + CUDA)")
        print("       then re-run: pip install laya   (afterwards, run this script again)")
    print("  3. NVIDIA driver present and current? (nvidia-smi must work)")
    print("Nothing breaks if you set LAYA_DEVICE anyway - Laya treats it as a")
    print("preference and quietly falls back to the CPU.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
