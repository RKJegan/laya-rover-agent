#!/usr/bin/env python3
"""Fail if any secret-shaped material has leaked into the workspace.

Run this before sharing logs, and after any live-API session:

    python scripts/check_no_secrets.py            # scans the whole project
    python scripts/check_no_secrets.py outputs    # scans just the run artifacts

Exit code 0 = clean, 1 = something matched (the offending files are printed).
This is the command-line twin of the redaction unit tests in
``tests/test_layer2_security.py``.
"""

from __future__ import annotations

import argparse
import pathlib
import re
import sys

#: Patterns that must never appear in logs, artifacts or committed files.
PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("long bearer token", re.compile(r"(?i)bearer\s+[A-Za-z0-9._\-]{20,}")),
    ("OpenAI-style key", re.compile(r"sk-(?:proj-)?[A-Za-z0-9]{16,}")),
    ("bearer token", re.compile(r"(?i)authorization\s*[:=]\s*[\"']?bearer\s+[A-Za-z0-9._\-]{12,}")),
    ("api key assignment", re.compile(r"(?i)\bapi[_-]?key\s*[:=]\s*[\"']?[A-Za-z0-9_\-]{16,}")),
)

#: Substrings that mark a match as an obvious placeholder (docs, examples and
#: the redaction tests use these on purpose). Anything without one of these is
#: treated as a real leak.
PLACEHOLDER_HINTS = (
    "xxx", "your", "test", "dummy", "example", "placeholder", "redacted",
    "sample", "fake", "abcdef", "0000", "1234", "…",
)

SKIP_DIRS = {".git", ".venv", "venv", "__pycache__", ".pytest_cache", "node_modules", ".mypy_cache", "coverage"}
TEXT_SUFFIXES = {
    ".json", ".jsonl", ".md", ".txt", ".csv", ".py", ".yaml", ".yml", ".toml", ".cfg", ".ini", ".env", ".log",
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Scan for leaked secrets")
    parser.add_argument("paths", nargs="*", default=["."], help="files or directories to scan (default: .)")
    args = parser.parse_args(argv)

    root = pathlib.Path(".").resolve()
    findings: list[tuple[pathlib.Path, str, str]] = []
    placeholders: list[tuple[pathlib.Path, str, str]] = []
    scanned = 0

    for raw in args.paths:
        base = pathlib.Path(raw)
        candidates = [base] if base.is_file() else sorted(base.rglob("*"))
        for path in candidates:
            if not path.is_file() or any(part in SKIP_DIRS for part in path.parts):
                continue
            if path.suffix.lower() not in TEXT_SUFFIXES and path.name not in {".env", ".env.example"}:
                continue
            try:
                text = path.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
            scanned += 1
            for label, pattern in PATTERNS:
                match = pattern.search(text)
                if not match:
                    continue
                sample = match.group(0)
                if any(hint in sample.lower() for hint in PLACEHOLDER_HINTS):
                    placeholders.append((path, label, sample[:24]))
                else:
                    findings.append((path, label, sample[:24] + "…"))

    print(f"scanned {scanned} text file(s)")
    if placeholders:
        print(f"ignored {len(placeholders)} obvious placeholder(s):")
        for path, label, sample in placeholders:
            print(f"  {path}  [{label}]  {sample}")
    if not findings:
        print("clean: no real secret material found")
        return 0

    print("\nLEAKS FOUND:")
    for path, label, sample in findings:
        print(f"  {path}  [{label}]  {sample}")
    print("\nRemove or rotate the offending material before sharing this workspace.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
