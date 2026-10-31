"""CLI entry points (standard-library ``argparse`` + ``rich`` rendering).

``python-dotenv`` is loaded **here and only here**, at process startup, so no
library module reaches into the environment on its own; everything downstream
receives configuration by injection.

Commands::

    python -m laya_rover_agent.cli experiment --mode mock_laya
    python -m laya_rover_agent.cli episode --seed 0 --preset trap_u --controller laya_agent
    python -m laya_rover_agent.cli map --seed 0 --preset trap_u
    python -m laya_rover_agent.cli plan
    python -m laya_rover_agent.cli estimate
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any, Sequence

from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table

from .config import LayaEnvSettings, ProjectConfig, apply_env_overrides, load_config
from .episode import EpisodeRunner, EpisodeSpec, write_jsonl
from .experiment.runner import ExperimentRunner, estimate_api_calls
from .hardware import RoverBridge, make_backend
from .laya.factory import ControllerFactory
from .security import SecretRedactor
from .types import euclidean
from .world import MapSpec, WorldGenerator, ascii_map, build_map_specs

console = Console()


# --------------------------------------------------------------------------- #
# Argument parsing
# --------------------------------------------------------------------------- #


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="laya-rover",
        description="Laya Rover Agent - safety-first, confidence-aware rover navigation experiments",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--config", "-c", type=Path, default=None, help="extra YAML layered over defaults")

    experiment = sub.add_parser("experiment", parents=[common], help="run the full controller matrix")
    experiment.add_argument("--mode", choices=["mock_laya", "laya_api"], default=None)
    experiment.add_argument("--seeds", type=int, default=None, help="seeds per difficulty")
    experiment.add_argument("--difficulties", default=None, help="comma list, e.g. easy,medium")
    experiment.add_argument("--confidence-threshold", default=None, help="comma list, e.g. 0.60,0.70")
    experiment.add_argument("--output-dir", default=None)
    experiment.add_argument("--jobs", "-j", type=int, default=None, help="episodes evaluated concurrently")
    experiment.add_argument("--guardian-mode", choices=["enforced", "log_only_ablation"], default=None)
    experiment.add_argument("--request-style", choices=["typed", "chat"], default="typed")
    experiment.add_argument("--run-label", default="default")
    experiment.add_argument("--max-episodes", type=int, default=None, help="debug: cap the episode count")
    experiment.add_argument("--quiet", action="store_true")

    single = sub.add_parser("episode", parents=[common], help="run one episode with a step trace")
    single.add_argument("--seed", type=int, default=0)
    single.add_argument("--difficulty", default="medium")
    single.add_argument("--preset", default="random", choices=["random", "trap_u", "trap_canyon", "trap_twincup"])
    single.add_argument("--controller", default="laya_agent", choices=["laya_agent", "baseline"])
    single.add_argument("--mode", choices=["mock_laya", "laya_api"], default=None)
    single.add_argument("--threshold", type=float, default=None)
    single.add_argument("--max-rows", type=int, default=25)
    single.add_argument("--log-jsonl", type=Path, default=None)

    hardware = sub.add_parser(
        "hardware",
        parents=[common],
        help="drive a rover through the connector layer (virtual simulator, ROS 2 or serial)",
    )
    hardware.add_argument("--backend", choices=["sim", "ros2", "serial"], default=None)
    hardware.add_argument("--controller", choices=["laya_agent", "baseline"], default="laya_agent")
    hardware.add_argument("--preset", default="random", choices=["random", "trap_u", "trap_canyon", "trap_twincup"],
                          help="sim backend only")
    hardware.add_argument("--seed", type=int, default=0, help="sim backend only")
    hardware.add_argument("--difficulty", default="medium", choices=["easy", "medium", "hard"], help="sim backend only")
    hardware.add_argument("--mode", choices=["mock_laya", "laya_api"], default=None)
    hardware.add_argument("--api", action="store_true", help="shorthand for --mode laya_api")
    hardware.add_argument("--max-steps", type=int, default=None)
    hardware.add_argument("--threshold", type=float, default=None)
    hardware.add_argument("--wall-clock-budget", type=float, default=600.0)
    hardware.add_argument("--dry-run", action="store_true", help="decide, log and guard, but never execute")
    hardware.add_argument("--log", type=Path, default=None, help="JSONL mission log path")
    hardware.add_argument("--serial-port", default=None, help="serial backend only (default: ROVER_SERIAL_PORT)")
    hardware.add_argument("--ros2-namespace", default=None, help="ros2 backend only (default: /laya_rover)")

    map_cmd = sub.add_parser("map", parents=[common], help="print a deterministic ASCII map")
    map_cmd.add_argument("--seed", type=int, default=0)
    map_cmd.add_argument("--difficulty", default="medium")
    map_cmd.add_argument("--preset", default="random", choices=["random", "trap_u", "trap_canyon", "trap_twincup"])

    sub.add_parser("plan", parents=[common], help="show the experiment matrix").add_argument(
        "--seeds", type=int, default=None
    )
    sub.add_parser("estimate", parents=[common], help="expected live-API call volume").add_argument(
        "--seeds", type=int, default=None
    )
    return parser


def _overrides(args: argparse.Namespace) -> dict[str, Any]:
    overrides: dict[str, Any] = {}
    if getattr(args, "mode", None):
        overrides.setdefault("agent", {})["mode"] = args.mode
    if getattr(args, "seeds", None) is not None:
        overrides.setdefault("experiment", {})["seeds_per_difficulty"] = args.seeds
    if getattr(args, "difficulties", None):
        overrides.setdefault("experiment", {})["difficulties"] = [
            value.strip() for value in args.difficulties.split(",") if value.strip()
        ]
    if getattr(args, "confidence_threshold", None):
        overrides.setdefault("experiment", {})["confidence_thresholds"] = [
            float(value) for value in args.confidence_threshold.split(",") if value.strip()
        ]
    if getattr(args, "output_dir", None):
        overrides.setdefault("experiment", {})["output_dir"] = args.output_dir
    if getattr(args, "jobs", None) is not None:
        overrides.setdefault("experiment", {})["jobs"] = args.jobs
    if getattr(args, "guardian_mode", None):
        overrides.setdefault("guardian", {})["mode"] = args.guardian_mode
    return overrides


# --------------------------------------------------------------------------- #
# Commands
# --------------------------------------------------------------------------- #


def cmd_experiment(args: argparse.Namespace) -> int:
    config = load_config(args.config, _overrides(args))
    env = LayaEnvSettings()
    config = apply_env_overrides(config, env)
    # Laya is self-hosted: no key is needed unless the server was started with one
    # and the config mirrors that with ``auth_required``.
    if config.agent.mode == "laya_api" and config.agent.laya_api.auth_required and not env.api_key:
        console.print("[red]laya_api mode with auth_required, but LAYA_API_KEY is not set.[/red]")
        console.print("Set LAYA_API_KEY in .env to the value the server was started with.")
        return 2

    if config.agent.mode == "laya_api":
        console.print(f"[dim]decision layer: {config.agent.laya_api.endpoint} "
                      f"(model={config.agent.laya_api.model})[/dim]")
        console.print("[dim]start it with: LAYA_MODELS=english laya-serve; "
                      "pre-flight with scripts/smoke_test_laya_api.py --health-only[/dim]")
        estimate = estimate_api_calls(config)
        console.print(
            f"[yellow]live mode[/yellow]: up to {estimate['confirmatory_max_api_calls']} decisions for this "
            f"configuration (bounded by the {args.max_episodes or 'unlimited'} episode cap and "
            f"{config.rover.max_steps} max steps/episode)."
        )

    runner = ExperimentRunner(
        config=config,
        env=env,
        run_label=args.run_label,
        verbose=not args.quiet,
        request_style=args.request_style,
        max_episodes=args.max_episodes,
    )
    artifacts = asyncio.run(runner.run())
    console.print(f"[bold green]report[/bold green]      -> {artifacts.report}")
    console.print(f"[bold green]summaries[/bold green]   -> {artifacts.episode_summaries}")
    console.print(f"[bold green]aggregate[/bold green]   -> {artifacts.aggregate_metrics}")
    console.print(f"[bold green]paired[/bold green]      -> {artifacts.paired_stats}")
    console.print(f"[bold green]ablation[/bold green]    -> {artifacts.ablation}")
    console.print(f"[bold green]calibration[/bold green] -> {artifacts.calibration}")
    console.print(f"[bold green]metadata[/bold green]    -> {artifacts.metadata}")
    return 0


def cmd_hardware(args: argparse.Namespace) -> int:
    """Run a mission through the rover connector layer (sim | ros2 | serial)."""
    import os

    config = load_config(args.config, _overrides(args))
    if args.api:
        config = config.model_copy(update={"agent": config.agent.model_copy(update={"mode": "laya_api"})})
    if args.threshold is not None:
        config = config.model_copy(
            update={"agent": config.agent.model_copy(update={"confidence_threshold": args.threshold})}
        )

    backend_name = args.backend or os.environ.get("ROVER_BACKEND", "sim")
    env = LayaEnvSettings()
    config = apply_env_overrides(config, env)
    redactor = SecretRedactor([env.api_key])
    if config.agent.mode == "laya_api" and config.agent.laya_api.auth_required and not env.api_key:
        console.print("[red]laya_api mode with auth_required, but LAYA_API_KEY is not set.[/red]")
        return 2

    try:
        if backend_name == "sim":
            backend = make_backend("sim", config, seed=args.seed, difficulty=args.difficulty, preset=args.preset)
        elif backend_name == "serial":
            port = args.serial_port or os.environ.get("ROVER_SERIAL_PORT")
            if not port:
                console.print("[red]--serial-port or ROVER_SERIAL_PORT is required for the serial backend.[/red]")
                return 2
            backend = make_backend("serial", config, port=port, sensor_range=config.world.sensor_range)
        else:
            backend = make_backend(
                "ros2",
                config,
                namespace=args.ros2_namespace or os.environ.get("ROVER_ROS2_NS", "/laya_rover"),
                sensor_range_cells=config.world.sensor_range,
            )
    except Exception as exc:  # noqa: BLE001 - backend construction failures are user-facing
        console.print(f"[red]{redactor.redact_exception(exc)}[/red]")
        return 2

    factory = ControllerFactory(config, api_key=env.api_key, redactor=redactor)
    spec = EpisodeSpec(
        map_spec=MapSpec(seed=args.seed, difficulty=args.difficulty, preset=args.preset),
        controller=args.controller,
        confidence_threshold=config.agent.confidence_threshold,
        guardian_mode=config.guardian.mode,
    )
    controller = factory(spec)
    bridge = RoverBridge(config, backend, controller, log_path=args.log)

    console.print(
        f"[bold]mission[/bold] backend={backend_name} controller={args.controller} "
        f"mode={config.agent.mode} dry_run={args.dry_run} max_steps={args.max_steps or config.rover.max_steps}"
        + ("" if backend_name != "sim" else f" map={args.difficulty}/{args.preset}/seed{args.seed}")
    )
    if not args.dry_run and backend_name != "sim":
        console.print("[yellow]real hardware selected: the rover will move. Ctrl-C now to abort.[/yellow]")

    async def _go():
        try:
            return await bridge.run_mission_async(
                max_steps=args.max_steps,
                dry_run=args.dry_run,
                wall_clock_budget_s=args.wall_clock_budget,
            )
        finally:
            await factory.aclose()

    try:
        summary = asyncio.run(_go())
    except KeyboardInterrupt:  # pragma: no cover - interactive only
        console.print("[yellow]interrupted - emergency stop requested[/yellow]")
        return 130

    table = Table(title="mission summary")
    for key, value in summary.to_dict().items():
        if key == "actions":
            continue
        table.add_row(str(key), json.dumps(value, default=str) if isinstance(value, (dict, list)) else str(value))
    console.print(table)
    console.print(f"actions: {json.dumps(summary.actions)}")
    if args.log is not None:
        console.print(f"[bold green]log[/bold green]      -> {args.log}")
        console.print(f"[bold green]summary[/bold green]  -> {args.log.with_suffix('.summary.json')}")
    if summary.stopped_reason not in {"goal"} and not summary.reached_goal:
        console.print(f"[yellow]mission did not reach the goal ({summary.stopped_reason or 'unknown'})[/yellow]")
    return 0 if summary.reached_goal else 1


def cmd_episode(args: argparse.Namespace) -> int:
    config = load_config(args.config, _overrides(args))
    if args.threshold is not None:
        config = config.model_copy(
            update={"agent": config.agent.model_copy(update={"confidence_threshold": args.threshold})}
        )
    env = LayaEnvSettings()
    config = apply_env_overrides(config, env)
    redactor = SecretRedactor([env.api_key])
    factory = ControllerFactory(config, api_key=env.api_key, redactor=redactor)
    runner = EpisodeRunner(config, controller_factory=factory)
    spec = EpisodeSpec(
        map_spec=MapSpec(seed=args.seed, difficulty=args.difficulty, preset=args.preset),
        controller=args.controller,
        confidence_threshold=config.agent.confidence_threshold,
        guardian_mode=config.guardian.mode,
    )

    async def _go():
        try:
            return await runner.run(spec, capture_map=True)
        finally:
            await factory.aclose()

    result = asyncio.run(_go())
    grid = WorldGenerator(config.world).generate(spec.map_spec)
    console.print(ascii_map(grid))

    table = Table(title=f"{spec.episode_id}  (mode={config.agent.mode})")
    for column in ("step", "action", "source", "laya_conf", "guardian", "moved", "battery", "goal_dist"):
        table.add_column(column)
    for row in result.timesteps[: args.max_rows]:
        after = row["state_after"]
        table.add_row(
            str(row["step"]),
            str(row["final_action"]),
            str(row["action_source"]),
            "-" if row["raw_laya_confidence"] is None else f"{row['raw_laya_confidence']:.2f}",
            str(row["guardian_reason_code"]),
            "yes" if row["moved"] else "no",
            f"{after['battery']:.3f}",
            f"{euclidean(after['x'], after['y'], grid.goal[0], grid.goal[1]):.1f}",
        )
    console.print(table)
    console.print(json.dumps(result.summary, indent=2, default=str))
    if args.log_jsonl is not None:
        write_jsonl(args.log_jsonl, result.timesteps)
        console.print(f"wrote {args.log_jsonl}")
    return 0 if result.outcome.success else 1


def cmd_map(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    grid = WorldGenerator(config.world).generate(
        MapSpec(seed=args.seed, difficulty=args.difficulty, preset=args.preset)
    )
    console.print(ascii_map(grid))
    console.print(f"start={grid.start} goal={grid.goal} preset={grid.preset} difficulty={grid.difficulty}")
    return 0


def cmd_plan(args: argparse.Namespace) -> int:
    config = load_config(args.config, _overrides(args))
    specs = build_map_specs(
        seeds_per_difficulty=config.experiment.seeds_per_difficulty,
        difficulties=config.experiment.difficulties,
        traps_per_difficulty=config.world.trap_cluster_presets_per_difficulty,
    )
    table = Table(title=f"Map matrix ({len(specs)} maps)")
    for column in ("difficulty", "seed", "preset", "map_key"):
        table.add_column(column)
    for spec in specs:
        table.add_row(spec.difficulty, str(spec.seed), spec.preset, spec.key)
    console.print(table)
    laya_episodes = len(specs) * len(config.experiment.confidence_thresholds)
    baseline_episodes = len(specs)
    console.print(f"episodes: laya_agent {laya_episodes} + baseline {baseline_episodes} = {laya_episodes + baseline_episodes}")
    return 0


def cmd_estimate(args: argparse.Namespace) -> int:
    config = load_config(args.config, _overrides(args))
    payload = estimate_api_calls(config)
    console.print(json.dumps(payload, indent=2))
    console.print(
        "[dim]Live mode is intended for the confirmatory subset only; full statistical sweeps run in "
        "mock_laya mode (see README).[/dim]"
    )
    return 0


# --------------------------------------------------------------------------- #

COMMANDS = {
    "experiment": cmd_experiment,
    "hardware": cmd_hardware,
    "episode": cmd_episode,
    "map": cmd_map,
    "plan": cmd_plan,
    "estimate": cmd_estimate,
}


def main(argv: Sequence[str] | None = None) -> int:
    load_dotenv(override=False)  # process-startup dotenv load, CLI only
    parser = build_parser()
    args = parser.parse_args(argv)
    return COMMANDS[args.command](args)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
