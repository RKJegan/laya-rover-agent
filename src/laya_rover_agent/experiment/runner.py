"""Experiment runner (FR-9, FR-10, FR-11).

Executes the controller matrix over matched seeded maps, streams artifacts to
disk (per-episode JSONL written and released immediately, so memory stays flat
over hundreds of episodes), then computes aggregates, paired statistics,
calibration and the ablation safety case, and renders the report.
"""

from __future__ import annotations

import asyncio
import json
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from rich.console import Console

from ..config import LayaEnvSettings, ProjectConfig, apply_env_overrides
from ..episode import EpisodeResult, EpisodeRunner, EpisodeSpec, write_jsonl
from ..laya.factory import ControllerFactory
from ..reporting import charts
from ..reporting.report import build_report
from ..security import SecretRedactor
from ..world import MapSpec, build_map_specs
from . import analytics
from .metadata import collect_metadata, write_metadata

console = Console()

LATENCY_SAMPLE_CAP = 4000


@dataclass
class RunArtifacts:
    output_dir: Path
    episode_summaries: Path
    aggregate_metrics: Path
    paired_stats: Path
    calibration: Path
    ablation: Path
    reason_codes: Path
    metadata: Path
    report: Path
    figures: dict[str, Path] = field(default_factory=dict)
    episode_count: int = 0


@dataclass
class ExperimentRunner:
    config: ProjectConfig
    env: LayaEnvSettings = field(default_factory=LayaEnvSettings)
    run_label: str = "default"
    verbose: bool = True
    request_style: str = "typed"
    jsons: bool = True
    max_episodes: int | None = None

    # ------------------------------------------------------------------ #

    def build_specs(self) -> list[EpisodeSpec]:
        config = self.config
        maps = build_map_specs(
            seeds_per_difficulty=config.experiment.seeds_per_difficulty,
            difficulties=config.experiment.difficulties,
            traps_per_difficulty=config.world.trap_cluster_presets_per_difficulty,
        )
        specs: list[EpisodeSpec] = []
        index = 0
        for map_spec in maps:
            for controller in config.experiment.controllers:
                thresholds = (
                    config.experiment.confidence_thresholds
                    if controller == "laya_agent"
                    else [config.agent.confidence_threshold]
                )
                for threshold in thresholds:
                    specs.append(
                        EpisodeSpec(
                            map_spec=map_spec,
                            controller=controller,
                            confidence_threshold=float(threshold),
                            guardian_mode=config.guardian.mode,
                            index=index,
                        )
                    )
                    index += 1
        if self.max_episodes is not None:
            specs = specs[: self.max_episodes]
        return specs

    # ------------------------------------------------------------------ #

    async def run(self) -> RunArtifacts:
        started_at = datetime.now(timezone.utc)
        t0 = time.perf_counter()
        # ``LAYA_*`` environment values win over the YAML, so the endpoint the
        # metadata reports is the endpoint the adapter uses.
        config = apply_env_overrides(self.config, self.env)
        out_dir = Path(config.experiment.output_dir)
        logs_dir = out_dir / "episode_logs"
        figures_dir = out_dir / "figures"
        for directory in (out_dir, logs_dir, figures_dir):
            directory.mkdir(parents=True, exist_ok=True)

        redactor = SecretRedactor([self.env.api_key])
        factory = ControllerFactory(
            config,
            api_key=self.env.api_key,
            redactor=redactor,
            request_style=self.request_style,
        )
        episode_runner = EpisodeRunner(config, controller_factory=factory)
        specs = self.build_specs()

        metadata_path = out_dir / "run_metadata.json"
        write_metadata(
            metadata_path,
            collect_metadata(
                config,
                env_summary={
                    "api_key": "***REDACTED***" if self.env.api_key else None,
                    "api_url": self.env.api_url or config.agent.laya_api.endpoint,
                    "model": self.env.model or config.agent.laya_api.model,
                    "mode": config.agent.mode,
                },
                run_label=self.run_label,
                started_at=started_at,
                extra={
                    "planned_episodes": len(specs),
                    "controllers": config.experiment.controllers,
                    "difficulties": config.experiment.difficulties,
                    "confidence_thresholds": config.experiment.confidence_thresholds,
                    "seeds_per_difficulty": config.experiment.seeds_per_difficulty,
                    "trap_presets_per_difficulty": config.world.trap_cluster_presets_per_difficulty,
                    "status": "running",
                },
            ),
        )

        if self.verbose:
            console.print(
                f"[bold]Laya Rover experiment[/bold] mode={config.agent.mode} "
                f"episodes={len(specs)} jobs={config.experiment.jobs} "
                f"guardian={config.guardian.mode} out={out_dir}"
            )

        summaries: list[dict[str, object]] = []
        calibration_rows: list[dict[str, object]] = []
        reason_counts: dict[str, int] = defaultdict(int)
        latency_samples: dict[str, list[float]] = defaultdict(list)
        examples: dict[str, dict[str, object]] = {}
        selected_examples = {
            f"{difficulty}_s0_trap_u" for difficulty in config.experiment.difficulties
        }

        semaphore = asyncio.Semaphore(max(1, config.experiment.jobs))

        async def one(spec: EpisodeSpec) -> EpisodeResult:
            capture = spec.map_spec.key in selected_examples
            async with semaphore:
                return await episode_runner.run(spec, capture_map=capture)

        tasks = [asyncio.create_task(one(spec)) for spec in specs]
        completed = 0
        try:
            for coro in asyncio.as_completed(tasks):
                result = await coro
                completed += 1
                summaries.append(result.summary)

                # stream the per-timestep log, then release it
                write_jsonl(logs_dir / f"{result.spec.episode_id}.jsonl", result.timesteps)
                for row in result.timesteps:
                    if row["guardian_reason_code"]:
                        reason_counts[str(row["guardian_reason_code"])] += 1
                calibration_rows.extend(result.outcome.confidence_records)
                label = f"{result.spec.controller}"
                sample = latency_samples[label]
                if len(sample) < LATENCY_SAMPLE_CAP:
                    sample.extend(result.outcome.latencies_ms[: LATENCY_SAMPLE_CAP - len(sample)])
                if result.map_cells is not None:
                    key = result.map_key
                    lane = examples.setdefault(
                        key,
                        {
                            "difficulty": result.spec.map_spec.difficulty,
                            "preset": result.spec.map_spec.preset,
                            "cells": result.map_cells,
                            "start": result.map_start,
                            "goal": result.map_goal,
                        },
                    )
                    if result.spec.controller == "baseline":
                        lane["baseline_path"] = result.trajectory
                    else:
                        lane.setdefault("laya_path", result.trajectory)
                result.timesteps = []  # release memory

                if self.verbose and (completed % 10 == 0 or completed == len(specs)):
                    console.print(f"  episodes {completed}/{len(specs)}", highlight=False)
        finally:
            await factory.aclose()

        # ---------------- aggregate + persist ---------------- #
        summaries_df = pd.DataFrame(summaries)
        summaries_path = out_dir / "episode_summaries.csv"
        summaries_df.to_csv(summaries_path, index=False)

        aggregate_df = analytics.aggregate_metrics(summaries_df)
        aggregate_path = out_dir / "aggregate_metrics.csv"
        aggregate_df.to_csv(aggregate_path, index=False)

        paired_df = analytics.paired_comparison(summaries_df)
        paired_path = out_dir / "paired_stats.csv"
        paired_df.to_csv(paired_path, index=False)

        calibration_df = analytics.calibration_table(pd.DataFrame(calibration_rows))
        calibration_path = out_dir / "calibration.csv"
        calibration_df.to_csv(calibration_path, index=False)

        # The headline safety case and its chart are reported at the preferred
        # threshold, so the CSV matches the calibration figure's slice.
        # Slice used for the headline safety case + calibration figure: the swept
        # threshold closest to the configured default (0.70), which keeps those
        # two artifacts comparable even when the sweep omits 0.70.
        swept = list(config.experiment.confidence_thresholds)
        preferred_threshold = (
            min(swept, key=lambda value: abs(value - config.agent.confidence_threshold)) if swept
            else config.agent.confidence_threshold
        )
        ablation_df = analytics.ablation_summary(summaries_df, threshold=preferred_threshold)
        ablation_path = out_dir / "ablation.csv"
        ablation_df.to_csv(ablation_path, index=False)

        reason_df = analytics.reason_code_totals(dict(reason_counts))
        reason_path = out_dir / "reason_codes.csv"
        reason_df.to_csv(reason_path, index=False)

        # ---------------- figures ---------------- #
        figures: dict[str, Path] = {}
        figures["success_rate_by_difficulty"] = charts.success_rate_by_difficulty(
            summaries_df, figures_dir / "success_rate_by_difficulty.png", threshold=preferred_threshold
        )
        figures["energy_by_controller"] = charts.energy_by_controller(
            summaries_df, figures_dir / "energy_by_controller.png"
        )
        figures["path_examples"] = charts.path_examples(
            [examples[key] for key in sorted(examples)], figures_dir / "path_examples.png"
        )
        figures["latency_distribution"] = charts.latency_distribution(
            dict(latency_samples), figures_dir / "latency_distribution.png",
            decision_timeout_ms=config.agent.decision_timeout_ms,
        )
        figures["guardian_ablation_safety_case"] = charts.guardian_ablation_safety_case(
            ablation_df, figures_dir / "guardian_ablation_safety_case.png"
        )
        figures["confidence_calibration"] = charts.confidence_calibration(
            calibration_df, figures_dir / "confidence_calibration.png", threshold=preferred_threshold
        )

        # ---------------- report ---------------- #
        report_path = out_dir / "report.md"
        build_report(
            config=config,
            env=self.env,
            summaries=summaries_df,
            aggregate=aggregate_df,
            paired=paired_df,
            calibration=calibration_df,
            ablation=ablation_df,
            reason_codes=reason_df,
            figures=figures,
            output_dir=out_dir,
            run_label=self.run_label,
            started_at=started_at,
            duration_s=round(time.perf_counter() - t0, 3),
            path=report_path,
        )

        # ---------------- metadata finalisation ---------------- #
        metadata = collect_metadata(
            config,
            env_summary={
                "api_key": "***REDACTED***" if self.env.api_key else None,
                "api_url": self.env.api_url or config.agent.laya_api.endpoint,
                "model": self.env.model or config.agent.laya_api.model,
                "mode": config.agent.mode,
            },
            run_label=self.run_label,
            started_at=started_at,
            extra={
                "status": "completed",
                "finished_at_utc": datetime.now(timezone.utc).isoformat(),
                "duration_s": round(time.perf_counter() - t0, 3),
                "episodes_completed": completed,
                "artifacts": {
                    "episode_summaries": str(summaries_path),
                    "aggregate_metrics": str(aggregate_path),
                    "paired_stats": str(paired_path),
                    "calibration": str(calibration_path),
                    "ablation": str(ablation_path),
                    "reason_codes": str(reason_path),
                    "report": str(report_path),
                    "figures": {k: str(v) for k, v in figures.items()},
                },
            },
        )
        write_metadata(metadata_path, metadata)

        if self.verbose:
            console.print(f"[green]done[/green] {completed} episodes -> {out_dir}")

        return RunArtifacts(
            output_dir=out_dir,
            episode_summaries=summaries_path,
            aggregate_metrics=aggregate_path,
            paired_stats=paired_path,
            calibration=calibration_path,
            ablation=ablation_path,
            reason_codes=reason_path,
            metadata=metadata_path,
            report=report_path,
            figures=figures,
            episode_count=completed,
        )


async def run_experiment(
    config: ProjectConfig,
    env: LayaEnvSettings | None = None,
    run_label: str = "default",
    verbose: bool = True,
    request_style: str = "typed",
    max_episodes: int | None = None,
) -> RunArtifacts:
    runner = ExperimentRunner(
        config=config,
        env=env or LayaEnvSettings(),
        run_label=run_label,
        verbose=verbose,
        request_style=request_style,
        max_episodes=max_episodes,
    )
    return await runner.run()


def estimate_api_calls(config: ProjectConfig) -> dict[str, int]:
    """Expected live-API call count and rough cost, documented in the README."""
    maps = build_map_specs(
        seeds_per_difficulty=config.experiment.seeds_per_difficulty,
        difficulties=config.experiment.difficulties,
        traps_per_difficulty=config.world.trap_cluster_presets_per_difficulty,
    )
    seeds_in_scope = len(config.experiment.difficulties) * config.experiment.seeds_per_difficulty
    worst_case_steps = config.rover.max_steps
    confirmatory_seeds = config.experiment.laya_api_confirmatory_seeds
    return {
        "maps_total": len(maps),
        "seeds_in_scope": seeds_in_scope,
        "mock_mode_episodes": len(maps) * len(config.experiment.confidence_thresholds) * 2,
        "confirmatory_episodes": confirmatory_seeds,
        "confirmatory_max_api_calls": confirmatory_seeds * worst_case_steps,
        "full_sweep_max_api_calls": seeds_in_scope * len(config.experiment.confidence_thresholds) * worst_case_steps,
    }


def summarise_examples(examples: dict[str, dict[str, object]]) -> str:
    return json.dumps({k: sorted(v.keys()) for k, v in examples.items()}, indent=2)
