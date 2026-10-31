"""Configuration models + resolution + hashing.

Resolution order (lowest to highest precedence):

1. ``configs/default.yaml`` (packaged defaults)
2. an optional extra YAML file passed on the CLI
3. CLI overrides (e.g. ``--seeds``, ``--confidence-threshold``)

Credentials are supplied *only* via environment (``LAYA_API_KEY`` etc.) through
:class:`LayaEnvSettings`. No secret is ever read inside a library module at
import time, and no secret ever enters the hashed run configuration or a log.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from .types import Action

# --------------------------------------------------------------------------- #
# YAML-backed config models
# --------------------------------------------------------------------------- #


class WorldConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    width: int = Field(default=50, ge=8, le=512)
    height: int = Field(default=50, ge=8, le=512)
    sensor_range: int = Field(default=6, ge=1, le=32)
    obstacle_density: dict[str, float] = Field(default_factory=lambda: {"easy": 0.10, "medium": 0.18, "hard": 0.25})
    hazard_density: dict[str, float] = Field(default_factory=lambda: {"easy": 0.03, "medium": 0.06, "hard": 0.10})
    trap_cluster_presets_per_difficulty: int = Field(default=3, ge=0, le=100)
    min_start_goal_distance_fraction: float = Field(default=0.60, ge=0.0, le=1.0)
    traversable_includes_hazard: bool = True

    @field_validator("obstacle_density", "hazard_density")
    @classmethod
    def _validate_density(cls, value: dict[str, float]) -> dict[str, float]:
        for key, val in value.items():
            if not 0.0 <= val <= 0.6:
                raise ValueError(f"density for {key!r} must be within [0.0, 0.6], got {val}")
        return value


class RoverConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    initial_battery: float = Field(default=1.0, ge=0.0, le=1.0)
    battery_reserve: float = Field(default=0.05, ge=0.0, le=1.0)
    max_steps: int = Field(default=500, ge=1)
    max_consecutive_stops: int = Field(default=4, ge=1)
    stuck_window: int = Field(default=8, ge=2)
    stuck_displacement_epsilon: float = Field(default=0.75, ge=0.0)
    stuck_min_new_cells: int = Field(default=1, ge=0, le=16)
    energy_capacity: float = Field(default=120.0, gt=0.0)
    energy_cost: dict[str, float] = Field(
        default_factory=lambda: {
            Action.MOVE_FORWARD.value: 1.0,
            Action.TURN_LEFT_AND_MOVE.value: 1.3,
            Action.TURN_RIGHT_AND_MOVE.value: 1.3,
            Action.REVERSE.value: 1.5,
            Action.STOP_AND_REPLAN.value: 0.1,
        }
    )
    hazard_energy_multiplier: float = Field(default=1.8, ge=1.0)
    hazard_stall_probability: dict[str, float] = Field(
        default_factory=lambda: {"easy": 0.10, "medium": 0.15, "hard": 0.20}
    )
    episode_wall_clock_timeout_s: float = Field(default=120.0, gt=0.0)

    @field_validator("energy_cost")
    @classmethod
    def _validate_costs(cls, value: dict[str, float]) -> dict[str, float]:
        missing = {a.value for a in Action} - set(value)
        if missing:
            raise ValueError(f"energy_cost missing entries for: {sorted(missing)}")
        if any(v < 0 for v in value.values()):
            raise ValueError("energy costs must be non-negative")
        return value

    def cost_of(self, action: Action, hazard: bool = False) -> float:
        base = self.energy_cost[action.value]
        return base * self.hazard_energy_multiplier if hazard else base


class MockLayaProfile(BaseModel):
    model_config = ConfigDict(extra="forbid")

    agree_with_fallback_pct: float = Field(default=0.80, ge=0.0, le=1.0)
    ambiguous_alt_pct: float = Field(default=0.10, ge=0.0, le=1.0)
    low_confidence_or_invalid_pct: float = Field(default=0.10, ge=0.0, le=1.0)
    unsafe_proposal_pct: float = Field(default=0.05, ge=0.0, le=1.0)
    malformed_share_of_invalid: float = Field(default=0.20, ge=0.0, le=1.0)

    @field_validator("low_confidence_or_invalid_pct")
    @classmethod
    def _check_sum(cls, value: float, info: Any) -> float:
        others = info.data.get("agree_with_fallback_pct", 0.0) + info.data.get("ambiguous_alt_pct", 0.0)
        if others + value > 1.0 + 1e-9:
            raise ValueError("mock_laya_profile branch percentages must sum to <= 1.0")
        return value


class LayaApiConfig(BaseModel):
    """Connection settings for a self-hosted Laya server (``laya-serve``).

    Laya runs on your own machine, so **no key is required by default**: the
    shipped server has no auth unless ``LAYA_API_KEY`` is set in *its*
    environment. ``auth_required`` mirrors that server-side switch - flip it to
    ``True`` when the server demands a bearer token, so a missing key fails
    closed immediately instead of spending a doomed round-trip per step.

    ``model`` is the Laya checkpoint: ``english``, ``multilingual``,
    ``typed-decisions``, or ``auto`` to let Laya's router choose per request.
    """

    model_config = ConfigDict(extra="forbid")

    endpoint: str = "http://127.0.0.1:8000/v1/systemone"
    #: Pinned checkpoint keeps the confirmatory experiment reproducible; the
    #: answering checkpoint is recorded per step regardless (see ``provider_model``).
    model: str = "english"
    #: Local host by default, so the budget is generous compared with a remote API.
    #: A cold checkpoint load can take far longer - warm the server up first
    #: (the smoke test probes ``/health``).
    request_timeout_s: float = Field(default=5.0, gt=0.0)
    #: Laya answers a saturated server with 503 + ``Retry-After``; one retry inside
    #: the decision budget is enough to ride out a busy moment.
    max_retries: int = Field(default=1, ge=0, le=5)
    max_concurrent_requests: int = Field(default=4, ge=1, le=64)
    auth_required: bool = False
    auth_header: str = "Authorization"
    auth_scheme: str = "Bearer"


class AgentConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    mode: Literal["mock_laya", "laya_api"] = "mock_laya"
    confidence_threshold: float = Field(default=0.70, ge=0.0, le=1.0)
    decision_timeout_ms: float = Field(default=1200.0, gt=0.0)
    retry_count: int = Field(default=0, ge=0, le=5)
    mock_laya_profile: MockLayaProfile = Field(default_factory=MockLayaProfile)
    laya_api: LayaApiConfig = Field(default_factory=LayaApiConfig)


class FallbackPolicyConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dead_end_avoidance: bool = True
    dead_end_clearance_threshold: int = Field(default=3, ge=1, le=16)
    hazard_aversion_margin: float = Field(default=0.75, ge=0.0)
    pocket_penalty: float = Field(default=3.0, ge=0.0)
    memory_cells: int = Field(default=12, ge=0, le=64)
    backtrack_penalty: float = Field(default=2.0, ge=0.0)
    escape_window: int = Field(default=6, ge=2, le=32)
    escape_max_distinct: int = Field(default=4, ge=1, le=32)
    escape_multiplier: float = Field(default=3.0, ge=1.0)
    memory_penalty: float = Field(default=2.5, ge=0.0)
    memory_floor_fraction: float = Field(default=0.5, ge=0.0, le=1.0)


class ControllersConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    fallback: FallbackPolicyConfig = Field(default_factory=FallbackPolicyConfig)


class GuardianConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    mode: Literal["enforced", "log_only_ablation"] = "enforced"


class RngStreamsConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    map_generation: str = "independent"
    hazard_stalls: str = "independent"
    mock_laya_confidence: str = "independent"


class ExperimentConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    seeds_per_difficulty: int = Field(default=30, ge=1)
    difficulties: list[str] = Field(default_factory=lambda: ["easy", "medium", "hard"])
    controllers: list[str] = Field(default_factory=lambda: ["laya_agent", "baseline"])
    confidence_thresholds: list[float] = Field(default_factory=lambda: [0.60, 0.70, 0.80])
    laya_api_confirmatory_seeds: int = Field(default=5, ge=1)
    output_dir: str = "outputs"
    jobs: int = Field(default=1, ge=1, le=16)
    rng_streams: RngStreamsConfig = Field(default_factory=RngStreamsConfig)


class ProjectConfig(BaseModel):
    """Fully-resolved, hashable project configuration."""

    model_config = ConfigDict(extra="forbid")

    world: WorldConfig = Field(default_factory=WorldConfig)
    rover: RoverConfig = Field(default_factory=RoverConfig)
    agent: AgentConfig = Field(default_factory=AgentConfig)
    controllers: ControllersConfig = Field(default_factory=ControllersConfig)
    guardian: GuardianConfig = Field(default_factory=GuardianConfig)
    experiment: ExperimentConfig = Field(default_factory=ExperimentConfig)

    # -- resolution --------------------------------------------------------- #

    def canonical_json(self) -> str:
        """Deterministic JSON (sorted keys) used for the config hash.

        Never contains credentials - the env settings live in a separate model.
        """
        return json.dumps(self.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))

    def config_hash(self) -> str:
        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()


# --------------------------------------------------------------------------- #
# Environment-backed secrets (ROVER_* only, loaded by the CLI, never by libs)
# --------------------------------------------------------------------------- #


class LayaEnvSettings(BaseSettings):
    """Laya credentials / transport settings read from environment or ``.env``.

    ``repr=False`` on the key plus :meth:`safe_dump` keep the secret out of
    tracebacks and logs (see ``security.py``).
    """

    #: ``env_prefix=""`` with an explicit alias per field, because the decision
    #: backend (Laya) and the rover hardware bridge (Laya Rover) have different
    #: owner names and must not share a prefix.
    model_config = SettingsConfigDict(env_file=".env", env_prefix="", extra="ignore")

    api_key: str | None = Field(default=None, repr=False, validation_alias="LAYA_API_KEY")
    api_url: str | None = Field(default=None, validation_alias="LAYA_API_URL")
    model: str | None = Field(default=None, validation_alias="LAYA_MODEL")
    request_timeout_s: float | None = Field(default=None, validation_alias="LAYA_REQUEST_TIMEOUT_S")
    max_retries: int | None = Field(default=None, validation_alias="LAYA_MAX_RETRIES")
    auth_required: bool | None = Field(default=None, validation_alias="LAYA_AUTH_REQUIRED")
    #: Rover hardware bridge - Laya Rover is the product name, so these keep ``ROVER_``.
    rover_backend: str | None = Field(default=None, validation_alias="ROVER_BACKEND")
    serial_port: str | None = Field(default=None, validation_alias="ROVER_SERIAL_PORT")
    ros2_ns: str | None = Field(default=None, validation_alias="ROVER_ROS2_NS")

    def safe_dump(self) -> dict[str, object]:
        """JSON-safe summary with the key redacted - safe to log."""
        return {
            "api_key": "***REDACTED***" if self.api_key else None,
            "api_url": self.api_url,
            "model": self.model,
            "request_timeout_s": self.request_timeout_s,
            "max_retries": self.max_retries,
            "auth_required": self.auth_required,
        }


# --------------------------------------------------------------------------- #
# Loader
# --------------------------------------------------------------------------- #

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "default_config.yaml"


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def apply_env_overrides(config: ProjectConfig, env: LayaEnvSettings) -> ProjectConfig:
    """Fold ``LAYA_*`` environment settings into a resolved config.

    Precedence, lowest to highest:
    packaged defaults -> YAML -> CLI overrides -> **environment**.

    This exists because the deployment that matters - a rover pointing at the
    ``laya-serve`` running on its own host, or at a ground station - is expressed
    in ``.env``, and a value that is *recorded* in ``run_metadata.json`` while
    being ignored by the adapter would make the metadata lie about which model
    answered. Credentials are deliberately absent here: the key travels to the
    adapter through :class:`LayaEnvSettings` and never enters the config (so it
    can never enter the config hash or a log).
    """
    api = config.agent.laya_api
    update: dict[str, Any] = {}
    if env.api_url:
        update["endpoint"] = env.api_url
    if env.model:
        update["model"] = env.model
    if env.request_timeout_s is not None:
        update["request_timeout_s"] = env.request_timeout_s
    if env.max_retries is not None:
        update["max_retries"] = env.max_retries
    if env.auth_required is not None:
        update["auth_required"] = env.auth_required
    if not update:
        return config
    return config.model_copy(
        update={"agent": config.agent.model_copy(update={"laya_api": api.model_copy(update=update)})}
    )


def load_config(
    path: str | Path | None = None,
    overrides: dict[str, Any] | None = None,
) -> ProjectConfig:
    """Load and validate configuration from packaged defaults + YAML + overrides."""
    raw: dict[str, Any] = yaml.safe_load(DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))

    if path is not None:
        user_raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        if not isinstance(user_raw, dict):
            raise ValueError(f"config file {path} must contain a YAML mapping")
        raw = _deep_merge(raw, user_raw)

    if overrides:
        raw = _deep_merge(raw, {k: v for k, v in overrides.items() if v is not None})

    return ProjectConfig.model_validate(raw)
