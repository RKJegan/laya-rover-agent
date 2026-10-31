"""``run_metadata.json`` (FR-11): everything needed to defend a reproducibility claim."""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from importlib import metadata as importlib_metadata
from pathlib import Path

from ..config import ProjectConfig

TRACKED_PACKAGES = (
    "httpx",
    "matplotlib",
    "numpy",
    "pandas",
    "pydantic",
    "pydantic-settings",
    "python-dotenv",
    "PyYAML",
    "pytest",
    "respx",
    "rich",
    "scipy",
)


def _package_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for name in TRACKED_PACKAGES:
        try:
            versions[name] = importlib_metadata.version(name)
        except importlib_metadata.PackageNotFoundError:
            continue
    return versions


def _git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:  # noqa: BLE001 - optional metadata
        return None
    return None


def _git_dirty() -> bool | None:
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        if result.returncode == 0:
            return bool(result.stdout.strip())
    except Exception:  # noqa: BLE001
        return None
    return None


def collect_metadata(
    config: ProjectConfig,
    env_summary: dict[str, object] | None = None,
    run_label: str = "default",
    started_at: datetime | None = None,
    extra: dict[str, object] | None = None,
) -> dict[str, object]:
    """Build the metadata payload. Never contains credentials."""
    canonical = config.canonical_json()
    payload: dict[str, object] = {
        "run_label": run_label,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "started_at_utc": (started_at or datetime.now(timezone.utc)).isoformat(),
        "config_hash_sha256": config.config_hash(),
        "config_bytes": len(canonical.encode("utf-8")),
        "resolved_config": json.loads(canonical),
        "resolved_config_sha256_of_pretty_json": hashlib.sha256(
            json.dumps(json.loads(canonical), indent=2, sort_keys=True).encode("utf-8")
        ).hexdigest(),
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor_count": platform.os.cpu_count() if hasattr(platform, "os") else None,
            "packages": _package_versions(),
        },
        "git": {"commit": _git_commit(), "dirty_worktree": _git_dirty()},
        "secrets": env_summary or {"api_key": None},
        "notes": (
            "config_hash_sha256 is computed over the canonical (sorted-key, compact) JSON of the "
            "resolved configuration. Credentials are never part of this file."
        ),
    }
    if extra:
        payload.update(extra)
    return payload


def write_metadata(path: Path, payload: dict[str, object]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")
    return path
