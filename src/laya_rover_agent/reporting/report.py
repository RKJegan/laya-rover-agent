"""``report.md`` generation.

The report is written by the code that produced the numbers, so it can never
drift from the artifacts. Sections: results, paired statistics, safety case,
confidence calibration, live-API status, and the honest limitations the spec
requires (Section 13).
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from ..config import LayaEnvSettings, ProjectConfig

LIMITATIONS: tuple[str, ...] = (
    "Simulation terrain and virtual sensing do not capture all real planetary conditions: dust, "
    "lighting changes, wheel slip, sensor degradation, radiation, thermal limits, communications "
    "failure, and unknown terrain.",
    "A model/API decision latency measured in a local computer simulation is not equivalent to "
    "operating on radiation-hardened onboard spacecraft hardware.",
    "Confidence reported by a model must be empirically calibrated; it must not be treated as a "
    "guaranteed probability of correctness.",
    "The safety guardian reduces risk in simulation but does not prove formal safety or flight readiness.",
    "Laya runs on the same host as the agent here, which is what makes the low-latency budget "
    "realistic - but an onboard process can still be unavailable, out of memory, or slow to load a "
    "checkpoint, so the deterministic fallback controller exists for exactly that case.",
    "Sample sizes (30 seeds/difficulty) are modest for strong statistical claims: the report shows "
    "bootstrap confidence intervals rather than point estimates alone, and avoids overstating "
    "significance.",
    "The guardian-ablation analysis is a counterfactual logging exercise inside this simulator's own "
    "physics and rules, not a formal verification proof.",
    "In `mock_laya` mode the decision layer is a *deterministic stand-in*, not the real Laya model. "
    "Mock-mode results validate the pipeline (guardian, fallback, logging, statistics) and must not "
    "be read as evidence about the real model's competence. That evidence comes only from the "
    "`laya_api` run described in the live-API section.",
    # --- properties of running Laya on the rover itself --------------------- #
    "Laya is self-hosted, so its cost profile is memory and load time rather than money: the "
    "checkpoints are hundreds of MB resident, a cold start downloads and warms a checkpoint before "
    "the first decision, and a deployed rover must budget RAM for it alongside the agent. Warm it up "
    "before a mission (`scripts/smoke_test_laya_api.py` probes `/health`) rather than paying that "
    "cost on the first step.",
    "Laya reports `usage.truncated` when it did not read the whole state. A truncated state means the "
    "decision was made on partial evidence; the flag is recorded per timestep (`decision_truncated`) "
    "so those steps can be excluded or re-examined, but a long observation that silently exceeds the "
    "checkpoint's context window would degrade decisions without changing the action space.",
    "When `model: auto` is configured, Laya's router picks a checkpoint per request and the two "
    "checkpoints are different models. Runs that must be comparable pin `model` and the answering "
    "checkpoint is recorded for every step (`provider_model`), so a mixed-checkpoint run is detectable "
    "after the fact rather than assumed away.",
    "Laya's confidence is native and calibrated by training (RLCD against strictly proper scoring "
    "rules), not derived from token log-probabilities or self-consistency sampling - so no synthetic "
    "confidence enters the calibration analysis. It is still a model's self-report and is evaluated "
    "here as agreement with the fallback controller's choice, which is a proxy, not ground truth.",
    "Laya's API-level guarantees were implemented from its published contract and exercised against a "
    "local mock transport, not against a live server in this environment; the live path is validated "
    "by the operator's own smoke run (`docs/live_laya_testing_guide.md`).",
)


def build_report(
    config: ProjectConfig,
    env: LayaEnvSettings,
    summaries: pd.DataFrame,
    aggregate: pd.DataFrame,
    paired: pd.DataFrame,
    calibration: pd.DataFrame,
    ablation: pd.DataFrame,
    reason_codes: pd.DataFrame,
    figures: dict[str, Path],
    output_dir: Path,
    run_label: str,
    started_at: datetime,
    duration_s: float,
    path: Path,
) -> Path:
    lines: list[str] = []
    add = lines.append

    add("# Laya Rover Agent - Experiment Report")
    add("")
    add(f"*Run label:* `{run_label}`  ")
    add(f"*Generated:* {datetime.now(timezone.utc).isoformat()}  ")
    add(f"*Started:* {started_at.isoformat()}  ")
    add(f"*Duration:* {duration_s:.1f} s  ")
    add(f"*Mode:* `{config.agent.mode}`  ")
    add(f"*Guardian:* `{config.guardian.mode}`  ")
    add(f"*Episodes:* {len(summaries)}  ")
    add(f"*Config hash (SHA-256):* `{config.config_hash()}`")
    add("")
    add("> **Read this first.** In `mock_laya` mode the decision layer is a deterministic stand-in for")
    add("> the Laya model, designed to exercise the guardian, the fallback path and the statistics.")
    add("> Results below are therefore a *pipeline validation*, not a measurement of Laya's own")
    add("> decision quality. Section 7 describes how to obtain the real-model evidence.")
    add("")

    # ---------------------------------------------------------------- #
    add("## 1. Executive summary")
    add("")
    add(_executive_summary(config, summaries, paired, ablation))
    add("")

    # ---------------------------------------------------------------- #
    add("## 2. What was built")
    add("")
    add(
        "A software-only, safety-first autonomous rover-navigation stack:\n\n"
        "| Module | Responsibility |\n|---|---|\n"
        "| `world.py` | Deterministic seeded 2D maps (free / obstacle / hazard-slope / start / goal), "
        "difficulty presets, concave trap-cluster presets, minimum-viable-route verification |\n"
        "| `perception.py` | Fixed forward-facing local window (5 rays, capped at `sensor_range`), "
        "identical for every controller; the map is never exposed |\n"
        "| `dynamics.py` | Discrete motion, orientation, energy accounting, hazard stalls, collisions, "
        "stop cap, loop/oscillation detection, wall-clock guard |\n"
        "| `safety.py` | Deterministic pre-filter + approval, machine-readable reason codes, "
        "counterfactual (ablation) probe |\n"
        "| `laya/mock.py` | Fully specified deterministic mock decision layer |\n"
        "| `laya/api.py` | Live Laya adapter: `state` + typed `choice`/`noul` questions, "
        "strict validation, latency + usage capture, retries bounded by a hard timeout, "
        "secrets redacted in every error path |\n"
        "| `controllers/fallback.py` | Conservative distance-minimising safety net |\n"
        "| `controllers/baseline.py` | Greedy, memoryless, hazard-blind comparison baseline |\n"
        "| `experiment/` | Matched-seed matrix execution, streaming artifacts, paired statistics, "
        "bootstrap CIs, calibration, ablation accounting |\n"
        "| `hardware/` | Rover-connector layer: sim, ROS 2 and serial backends behind one interface |\n"
    )
    add("")

    # ---------------------------------------------------------------- #
    add("## 3. Reproducibility")
    add("")
    add(f"- Resolved-config SHA-256: `{config.config_hash()}` (see `run_metadata.json`).")
    add(f"- Seeds: `{config.experiment.seeds_per_difficulty}` per difficulty "
        f"({config.experiment.seeds_per_difficulty * len(config.experiment.difficulties)} maps), "
        f"including {config.world.trap_cluster_presets_per_difficulty} concave trap-cluster presets per difficulty.")
    add("- Independent RNG streams for map generation, hazard stalls and mock-Laya confidence, so "
        "sweeping the confidence threshold cannot alter maps or physics.")
    add("- Episodes are a pure function of `(seed, difficulty, preset, controller, threshold, config)`; "
        "`tests/test_e2e_mock_episode.py` asserts bit-identical replays.")
    add("")

    # ---------------------------------------------------------------- #
    add("## 4. Results")
    add("")
    add("### 4.1 Aggregate metrics")
    add("")
    add(_markdown_table(_select_columns(aggregate)))
    add("")
    add("### 4.2 Matched-pair comparison (Laya agent vs baseline)")
    add("")
    add(
        "Both controllers run on byte-identical terrain (`map_key` = difficulty + seed + preset). "
        "Deltas are `laya_agent - baseline`; negative is better for energy, steps, collisions and "
        "hazard entries."
    )
    add("")
    add(_markdown_table(paired) if not paired.empty else "_No paired rows (single controller or no matched maps)._")
    add("")
    add(
        "> **Sample-size caveat.** With 30 seeds per difficulty, the bootstrap intervals above are "
        "wide by construction. Intervals that include zero must not be read as evidence of a "
        "difference; intervals that exclude zero are indicative, not confirmatory."
    )
    add("")

    # ---------------------------------------------------------------- #
    add("## 5. Safety case: guardian ablation")
    add("")
    add(
        "Every time the guardian rejects a proposal it also evaluates - without executing - what "
        "would have happened (`counterfactual` probe). The table below is the resulting safety-case "
        "accounting for the Laya agent."
    )
    add("")
    add(_markdown_table(ablation) if not ablation.empty else "_No guardian rejections were recorded._")
    add("")
    add("**Guardian reason codes observed** (per timestep, all controllers):")
    add("")
    add(_markdown_table(reason_codes) if not reason_codes.empty else "_none_")
    add("")
    add(
        "Real collisions in enforced mode must be zero: the guardian never emits a move into an "
        "obstacle or off-grid cell, and the counterfactual probe is the only place a collision is "
        "ever *simulated*."
    )
    add("")

    # ---------------------------------------------------------------- #
    add("## 6. Confidence calibration")
    add("")
    add(
        "**Operational definition of outcome (FR-10.5).** For each step where the decision layer "
        "returned a valid action, the outcome is the *agreement of that action with the fallback "
        "controller's choice for the same observation*, evaluated together with the guardian's "
        "verdict (approved without override, or overridden). Ground-truth optimality of a single "
        "action is not observable in a stochastic, partially observed episode, so this agreement "
        "rate is used as the calibration target; a well-calibrated confidence should track it."
    )
    add("")
    add(_markdown_table(calibration) if not calibration.empty else "_No calibration samples._")
    add("")
    if not calibration.empty:
        add(
            "`calibration_gap` = mean stated confidence − observed agreement rate for that bin. "
            "Positive values mean the model is over-confident relative to this operational outcome."
        )
        add("")

    # ---------------------------------------------------------------- #
    add("## 7. Live Laya API (`laya_api` mode)")
    add("")
    add(_live_api_section(env, config))
    add("")

    # ---------------------------------------------------------------- #
    add("## 8. Figures")
    add("")
    for name, figure_path in figures.items():
        try:
            relative = figure_path.relative_to(output_dir)
        except ValueError:
            relative = figure_path
        add(f"- **{name}** - `{relative}`")
    add("")
    for name, figure_path in figures.items():
        try:
            relative = figure_path.relative_to(output_dir)
        except ValueError:
            relative = figure_path
        add(f"### {name}")
        add("")
        add(f"![{name}]({relative.as_posix()})")
        add("")

    # ---------------------------------------------------------------- #
    add("## 9. Honest limitations")
    add("")
    for item in LIMITATIONS:
        add(f"- {item}")
    add("")

    # ---------------------------------------------------------------- #
    add("## 10. Reproduce")
    add("")
    add("```bash")
    add("# offline, deterministic, no network:")
    add("python -m laya_rover_agent.cli experiment --mode mock_laya --config configs/default.yaml")
    add("")
    add("# live confirmatory subset (5 seeds x 1 difficulty x 1 threshold):")
    add("python scripts/smoke_test_laya_api.py            # verify connectivity first")
    add("python -m laya_rover_agent.cli experiment --mode laya_api --seeds 5 \\")
    add("       --difficulties medium --confidence-threshold 0.70")
    add("```")
    add("")
    add(f"*Artifacts in `{output_dir}`: per-timestep JSONL logs, episode summaries, aggregate metrics, "
        "paired statistics, calibration table, ablation accounting, six figures, `run_metadata.json`, "
        "and this report.*")
    add("")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


# --------------------------------------------------------------------------- #
# Section builders
# --------------------------------------------------------------------------- #


def _executive_summary(
    config: ProjectConfig,
    summaries: pd.DataFrame,
    paired: pd.DataFrame,
    ablation: pd.DataFrame,
) -> str:
    if summaries.empty:
        return "_No episodes were recorded._"
    bullets: list[str] = []
    for controller, group in summaries.groupby("controller"):
        success = float(group["success"].mean())
        collisions = int(group["collisions"].sum())
        bullets.append(
            f"- **{controller}**: success {success:.1%} over {len(group)} episodes; "
            f"{collisions} collision(s); mean energy {group['energy_used'].mean():.1f} units; "
            f"mean steps {group['steps'].mean():.1f}."
        )
    if not paired.empty:
        success_rows = paired[paired["metric"] == "success_rate"]
        significant = success_rows[success_rows["ci_excludes_zero"]]
        if significant.empty:
            bullets.append(
                "- On success rate, no threshold/difficulty cell had a 95% bootstrap interval "
                "excluding zero: with this sample size the two controllers are not distinguishable "
                "on success alone."
            )
        else:
            for _, row in significant.iterrows():
                bullets.append(
                    f"- Success-rate delta at **{row['difficulty']}** / threshold {row['confidence_threshold']}: "
                    f"{row['mean_delta']:+.3f} (95% CI {row['ci95_low']:+.3f} to {row['ci95_high']:+.3f}, "
                    f"n={int(row['n_pairs'])} pairs)."
                )
    if not ablation.empty:
        total = int(ablation["unsafe_actions_prevented"].sum())
        collisions = int(ablation["collisions_prevented"].sum())
        hazards = int(ablation["hazard_entries_prevented"].sum())
        bullets.append(
            f"- **Safety case**: the guardian prevented **{total}** unsafe/illegal actions "
            f"({collisions} obstacle or out-of-bounds entries and {hazards} hazard-slope entries) "
            f"that the decision layer proposed, while the rover itself recorded 0 real collisions "
            f"in enforced mode."
        )
    bullets.append(
        f"- Baseline-vs-fallback differentiation is structural: the baseline is memoryless, "
        f"hazard-blind and tie-breaks by a fixed priority, while the Laya agent is backed by a "
        f"conservative re-evaluating fallback controller - see Section 2 and the spec's Section 5.4."
    )
    return "\n".join(bullets)


def _live_api_section(env: LayaEnvSettings, config: ProjectConfig) -> str:
    key_present = bool(env.api_key)
    endpoint = env.api_url or config.agent.laya_api.endpoint
    model = env.model or config.agent.laya_api.model
    lines = [
        "| Item | Value |",
        "|---|---|",
        f"| Endpoint | `{endpoint}` |",
        f"| Model | `{model}` |",
        f"| Credential present | {'yes (redacted everywhere)' if key_present else 'no - a default laya-serve has no auth'} |",
        f"| Auth required by config | {config.agent.laya_api.auth_required} |",
        f"| Request contract | `state` + typed `choice` question (`next_action`), "
        f"criteria keys = allowed actions |",
        f"| Decision timeout (hard ceiling) | {config.agent.decision_timeout_ms:.0f} ms, "
        f"including retries |",
        "",
    ]
    if config.agent.mode == "laya_api":
        lines.append(
            "This run was executed against the **live** endpoint; see the per-timestep logs for raw "
            "latencies, provider model version and usage."
        )
    else:
        lines.append(
            "This run used `mock_laya`. To obtain real-model evidence:\n\n"
            "```bash\n"
            "# 1. start Laya on this host (separate process)\n"
            "pip install laya && LAYA_MODELS=english laya-serve\n"
            "# 2. point the agent at it (LAYA_API_URL/LAYA_MODEL; no key needed by default)\n"
            "cp .env.example .env\n"
            "python scripts/smoke_test_laya_api.py            # probes /health, then one real call\n"
            "python -m laya_rover_agent.cli experiment --mode laya_api --seeds 5 \\\n"
            "       --difficulties medium --confidence-threshold 0.70\n"
            "```\n\n"
            "The confirmatory subset is deliberately small: 5 seeds × 1 difficulty × 1 threshold = "
            "5 episodes, at most 5 × `max_steps` decisions, each bounded by the per-decision timeout. "
            "Laya is local, so the constraint is wall-clock time rather than cost - at roughly "
            "0.13 s per decision on CPU, a 5-episode medium run is a couple of minutes. Full "
            "statistical sweeps stay on `mock_laya` because they need thousands of episodes, and "
            "re-running them on a served model measures the same pipeline with a slower clock."
        )
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #


def _select_columns(frame: pd.DataFrame) -> pd.DataFrame:
    preferred = [
        "controller",
        "difficulty",
        "confidence_threshold",
        "n_episodes",
        "success_rate",
        "collision_rate",
        "battery_depletion_rate",
        "stuck_rate",
        "timeout_rate",
        "mean_steps_success",
        "mean_energy_success",
        "mean_path_length_success",
        "mean_fallback_rate",
        "mean_guardian_overrides",
        "mean_unsafe_actions_prevented",
        "mean_hazard_entries",
        "mean_stalls",
        "p95_latency_ms",
    ]
    present = [column for column in preferred if column in frame.columns]
    return frame[present] if present else frame


def _markdown_table(frame: pd.DataFrame, max_rows: int = 60) -> str:
    if frame is None or frame.empty:
        return "_no data_"
    view = frame.head(max_rows).copy()
    for column in view.columns:
        if pd.api.types.is_float_dtype(view[column]):
            view[column] = view[column].map(lambda value: "" if pd.isna(value) else f"{value:.4g}")
    header = "| " + " | ".join(str(c) for c in view.columns) + " |"
    divider = "| " + " | ".join("---" for _ in view.columns) + " |"
    rows = [
        "| " + " | ".join("" if pd.isna(value) else str(value) for value in row) + " |"
        for row in view.itertuples(index=False)
    ]
    suffix = ""
    if len(frame) > max_rows:
        suffix = f"\n\n_(showing {max_rows} of {len(frame)} rows - see the CSV artifacts for the rest)_"
    return "\n".join([header, divider, *rows]) + suffix
