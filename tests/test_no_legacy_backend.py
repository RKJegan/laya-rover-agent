"""Layer 0 - the agent is Laya-only, and this file proves it automatically.

The project was migrated off a different decision backend, and the migration left
its name in paths, environment variables and comments for a while. This module is
the standing guarantee that it cannot creep back:

1. **Source scan** - no shipped file under ``src/``, ``configs/``, ``scripts/`` or
   ``tests/`` mentions the legacy backend name, in any casing.
2. **Import graph** - after importing the whole package, no module in
   ``sys.modules`` carries the legacy name.
3. **Effective runtime configuration** - the resolved config points at a Laya
   endpoint, uses a Laya checkpoint name, and the decision layer is one of the two
   Laya adapters behind the provider-neutral protocol.

The legacy token is assembled from fragments below rather than written literally,
for the obvious reason: a test that asserts a name is absent must not itself be
the thing that contains it.
"""

from __future__ import annotations

import importlib
import pkgutil
import re
import sys
from pathlib import Path

import pytest

from laya_rover_agent.config import ProjectConfig, load_config
from laya_rover_agent.laya.adapter import DecisionAdapter
from laya_rover_agent.laya.api import LayaApiAdapter
from laya_rover_agent.laya.mock import MockLayaAdapter

#: The retired backend's name, assembled so this file stays clean under its own scan.
LEGACY = "j" + "ev"

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCANNED_SUFFIXES = {".py", ".yaml", ".yml", ".toml", ".md", ".txt"}
#: The migration history legitimately names the old backend - it is the record of
#: what changed, not a runtime dependency. Everything else must be clean.
HISTORY_FILES = {"MIGRATION.md", "CHANGELOG.md"}


def _scannable_files() -> list[Path]:
    roots = [PROJECT_ROOT / name for name in ("src", "configs", "scripts", "tests")]
    files: list[Path] = []
    for root in roots:
        for path in root.rglob("*"):
            if not path.is_file() or path.suffix not in SCANNED_SUFFIXES:
                continue
            if path.name in HISTORY_FILES or "__pycache__" in path.parts:
                continue
            files.append(path)
    return sorted(files)


def test_no_shipped_file_mentions_the_legacy_backend():
    """The single guarantee: the codebase is Laya-only."""
    pattern = re.compile(LEGACY, re.IGNORECASE)
    offenders = [
        f"{path.relative_to(PROJECT_ROOT)}:{line_no}"
        for path in _scannable_files()
        for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1)
        if pattern.search(line)
    ]
    assert not offenders, (
        "the legacy decision backend is still referenced (it must not be - the agent is Laya-only):\n  "
        + "\n  ".join(offenders)
    )


def test_no_legacy_module_is_importable_or_loaded():
    """Importing everything must not pull in a module named after the old backend."""
    package = importlib.import_module("laya_rover_agent")
    for module in pkgutil.walk_packages(package.__path__, f"{package.__name__}."):
        importlib.import_module(module.name)  # every module must import cleanly

    loaded = [name for name in sys.modules if LEGACY in name.lower()]
    assert not loaded, f"legacy modules were loaded: {loaded}"


def test_only_laya_decision_layers_are_available():
    """Two Laya decision layers ship, each filling the seam it really implements."""
    from laya_rover_agent.controllers.base import Controller

    # the remote decision service implements the wire-level protocol
    assert isinstance(LayaApiAdapter, type)
    for method in ("adecide", "decide", "aprobe_health"):
        assert hasattr(LayaApiAdapter, method), f"LayaApiAdapter lacks {method}()"

    # the offline stand-in implements the controller-facing seam (it needs the
    # rover state and the safe set, which a wire-level signature cannot carry)
    mock = MockLayaAdapter(profile=load_config().agent.mock_laya_profile, episode_seed=0)
    assert isinstance(mock, Controller)
    assert hasattr(mock, "propose")

    for layer in (LayaApiAdapter, MockLayaAdapter):
        assert "laya" in layer.__name__.lower()
    # and no decision layer answers to the retired name
    assert not any(re.search(LEGACY, name.__name__, re.IGNORECASE) for name in (LayaApiAdapter, MockLayaAdapter, DecisionAdapter))


def test_default_configuration_targets_a_laya_server():
    """The shipped defaults must describe Laya: its endpoint, model and modes."""
    config: ProjectConfig = load_config()
    api = config.agent.laya_api

    assert api.endpoint == "http://127.0.0.1:8000/v1/systemone"
    assert api.model in {"english", "multilingual", "typed-decisions", "auto"}
    assert config.agent.mode in {"mock_laya", "laya_api"}
    assert "laya" in api.endpoint.lower() or api.endpoint.startswith("http://127.0.0.1")
    assert api.auth_required is False  # a default laya-serve has no auth


def test_client_environment_variables_are_laya_namespaced():
    """Credential and connection settings read LAYA_* - never the retired prefix."""
    from laya_rover_agent.config import LayaEnvSettings

    aliases = {
        field.validation_alias
        for field in LayaEnvSettings.model_fields.values()
        if isinstance(field.validation_alias, str)
    }
    assert "LAYA_API_KEY" in aliases and "LAYA_API_URL" in aliases and "LAYA_MODEL" in aliases
    assert not any(LEGACY in alias.lower() for alias in aliases), sorted(aliases)
    # the rover-bridge variables are intentionally provider-free now
    assert {"ROVER_BACKEND", "ROVER_SERIAL_PORT", "ROVER_ROS2_NS"} <= aliases


def test_console_script_and_distribution_are_laya_named():
    """The installed entry point and package metadata carry no legacy name."""
    text = (PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert not re.search(LEGACY, text, re.IGNORECASE)
    assert 'name = "laya-rover-agent"' in text
    assert 'laya-rover = "laya_rover_agent.cli:main"' in text


@pytest.mark.parametrize("module", ["laya_rover_agent.laya.adapter", "laya_rover_agent.laya.api", "laya_rover_agent.laya.mock"])
def test_laya_package_modules_exist(module):
    assert importlib.import_module(module) is not None
