"""``python -m laya_rover_agent`` entry point (same CLI as the ``laya-rover`` script)."""

from __future__ import annotations

import sys

from .cli import main

if __name__ == "__main__":
    sys.exit(main())
