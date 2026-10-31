"""Aggregation, paired statistics, calibration and ablation accounting.

All statistics are computed from the per-episode summary table plus the
per-timestep records, and every function is pure (DataFrame in, DataFrame out)
so results are reproducible and testable.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

BOOTSTRAP_ITERATIONS = 10_000
BOOTSTRAP_SEED = 20261003
CALIBRATION_BINS: tuple[float, ...] = (0.0, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)

SUCCESS_ONLY_METRICS = ("steps", "energy_used", "path_length", "wall_clock_s")


# --------------------------------------------------------------------------- #
# Aggregate metrics
# --------------------------------------------------------------------------- #


def aggregate_metrics(summaries: pd.DataFrame) -> pd.DataFrame:
    """One row per (controller, difficulty, confidence_threshold)."""
    if summaries.empty:
        return pd.DataFrame()

    rows: list[dict[str, object]] = []
    group_keys = ["controller", "difficulty", "confidence_threshold"]
    for key, group in summaries.groupby(group_keys, dropna=False):
        controller, difficulty, threshold = key
        successes = group[group["success"] == 1]
        row: dict[str, object] = {
            "controller": controller,
            "difficulty": difficulty,
            "confidence_threshold": threshold,
            "n_episodes": len(group),
            "success_rate": _rate(group["success"]),
            "collision_rate": _rate(group["collisions"] > 0),
            "battery_depletion_rate": _rate(group["terminal"] == "battery_depleted"),
            "stuck_rate": _rate(group["terminal"] == "stuck"),
            "timeout_rate": _rate(group["terminal"] == "timeout"),
            "mean_steps_success": _mean(successes["steps"]),
            "median_steps_success": _median(successes["steps"]),
            "mean_energy_success": _mean(successes["energy_used"]),
            "mean_path_length_success": _mean(successes["path_length"]),
            "mean_steps_all": _mean(group["steps"]),
            "mean_energy_all": _mean(group["energy_used"]),
            "mean_final_distance": _mean(group["final_distance_to_goal"]),
            "mean_guardian_overrides": _mean(group["guardian_overrides"]),
            "mean_unsafe_actions_prevented": _mean(group["unsafe_actions_prevented"]),
            "mean_collisions_prevented": _mean(group["collisions_prevented"]),
            "mean_hazard_entries_prevented": _mean(group["hazard_entries_prevented"]),
            "mean_fallback_count": _mean(group["fallback_count"]),
            "mean_fallback_rate": _mean(group["fallback_rate"]),
            "mean_laya_invalid": _mean(group["laya_invalid"]),
            "mean_stalls": _mean(group["stalls"]),
            "mean_hazard_entries": _mean(group["hazard_entries"]),
            "mean_stop_actions": _mean(group["stop_actions"]),
            "mean_latency_ms": _mean(group["latency_mean_ms"], include_zeros=False),
            "p50_latency_ms": _mean(group["latency_p50_ms"], include_zeros=False),
            "p95_latency_ms": _mean(group["latency_p95_ms"], include_zeros=False),
            "total_collisions": int(group["collisions"].sum()),
            "total_unsafe_actions_prevented": int(group["unsafe_actions_prevented"].sum()),
            "total_collisions_prevented": int(group["collisions_prevented"].sum()),
            "total_hazard_entries_prevented": int(group["hazard_entries_prevented"].sum()),
            "mean_wall_clock_s": _mean(group["wall_clock_s"]),
        }
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["controller", "difficulty", "confidence_threshold"]).reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Paired comparison (matched maps, bootstrap CIs)
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class PairComparison:
    metric: str
    n_pairs: int
    mean_delta: float
    ci_low: float
    ci_high: float
    p_value: float | None
    direction: str


def bootstrap_ci(
    values: np.ndarray,
    iterations: int = BOOTSTRAP_ITERATIONS,
    alpha: float = 0.05,
    seed: int = BOOTSTRAP_SEED,
) -> tuple[float, float]:
    """95% percentile-bootstrap CI of the mean."""
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return (float("nan"), float("nan"))
    if values.size == 1:
        return (float(values[0]), float(values[0]))
    generator = np.random.default_rng(seed)
    draws = generator.choice(values, size=(iterations, values.size), replace=True)
    means = draws.mean(axis=1)
    return (float(np.percentile(means, 100 * alpha / 2)), float(np.percentile(means, 100 * (1 - alpha / 2))))


def paired_comparison(
    summaries: pd.DataFrame,
    treatment: str = "laya_agent",
    reference: str = "baseline",
) -> pd.DataFrame:
    """Matched-pair deltas per (difficulty, threshold), with bootstrap CIs.

    Pairing key is ``map_key`` (difficulty + seed + preset), so both controllers
    see byte-identical terrain - the requirement of Section FR-9/FR-10.
    """
    rows: list[dict[str, object]] = []
    if summaries.empty:
        return pd.DataFrame()

    thresholds = sorted(
        summaries.loc[summaries["controller"] == treatment, "confidence_threshold"].unique()
    )
    for difficulty in sorted(summaries["difficulty"].unique()):
        ref = summaries[(summaries["controller"] == reference) & (summaries["difficulty"] == difficulty)]
        ref = ref.set_index("map_key")
        for threshold in thresholds:
            treat = summaries[
                (summaries["controller"] == treatment)
                & (summaries["difficulty"] == difficulty)
                & (summaries["confidence_threshold"] == threshold)
            ].set_index("map_key")
            common = sorted(set(ref.index) & set(treat.index))
            if not common:
                continue
            ref_c = ref.loc[common]
            treat_c = treat.loc[common]

            for metric, label in (
                ("success", "success_rate"),
                ("energy_used", "energy_successful"),
                ("steps", "steps_successful"),
                ("collisions", "collisions"),
                ("hazard_entries", "hazard_entries"),
                ("fallback_rate", "fallback_rate"),
            ):
                delta = (treat_c[metric].astype(float) - ref_c[metric].astype(float)).to_numpy()
                low, high = bootstrap_ci(delta)
                rows.append(
                    {
                        "difficulty": difficulty,
                        "confidence_threshold": threshold,
                        "treatment": treatment,
                        "reference": reference,
                        "metric": label,
                        "n_pairs": len(common),
                        "mean_delta": round(float(np.mean(delta)), 5),
                        "ci95_low": round(low, 5),
                        "ci95_high": round(high, 5),
                        "ci_excludes_zero": bool(low > 0 or high < 0),
                        "p_value_wilcoxon": _wilcoxon(delta),
                    }
                )
    return pd.DataFrame(rows)


def _wilcoxon(delta: np.ndarray) -> float | None:
    try:
        from scipy.stats import wilcoxon

        if np.allclose(delta, 0.0) or delta.size < 6:
            return None
        result = wilcoxon(delta, zero_method="wilcox", alternative="two-sided")
        return round(float(result.pvalue), 6)
    except Exception:  # noqa: BLE001 - statistics must never break a run
        return None


# --------------------------------------------------------------------------- #
# Confidence calibration (FR-10.5)
# --------------------------------------------------------------------------- #


def calibration_table(records: pd.DataFrame) -> pd.DataFrame:
    """Confidence bins vs agreement with the fallback controller's choice.

    Operational definition (documented in ``report.md``): for every step where
    Laya returned a valid decision, the *outcome* of that decision is whether it
    agreed with the safe conservative controller's choice for the same
    observation, and whether the guardian approved it without override.
    """
    if records.empty:
        return pd.DataFrame()
    frame = records.copy()
    frame["bin"] = pd.cut(frame["confidence"], bins=list(CALIBRATION_BINS), include_lowest=True, right=True)
    grouped = frame.groupby(["confidence_threshold", "bin"], observed=True)
    table = grouped.agg(
        n_steps=("confidence", "size"),
        mean_confidence=("confidence", "mean"),
        agreement_rate=("agreed_with_fallback", "mean"),
        approval_rate=("approved_without_override", "mean"),
    ).reset_index()
    # ``pd.cut`` yields a categorical column; materialise the interval bounds as
    # plain floats so downstream maths (and matplotlib) behave.
    intervals = list(table["bin"])
    table = table.drop(columns=["bin"])
    table["bin_low"] = [float(interval.left) for interval in intervals]
    table["bin_high"] = [float(interval.right) for interval in intervals]
    table["calibration_gap"] = (table["mean_confidence"] - table["agreement_rate"]).round(4)
    return table


def calibration_ece(table: pd.DataFrame) -> float:
    """Expected calibration error over the confidence bins (sample-weighted)."""
    if table.empty:
        return float("nan")
    weights = table["n_steps"] / table["n_steps"].sum()
    return float((weights * table["calibration_gap"].abs()).sum())


# --------------------------------------------------------------------------- #
# Guardian ablation accounting (Section 5.3)
# --------------------------------------------------------------------------- #


def ablation_summary(summaries: pd.DataFrame, threshold: float | None = None) -> pd.DataFrame:
    """'Unsafe entries prevented by the guardian' - the safety-case numbers."""
    if summaries.empty:
        return pd.DataFrame()
    frame = summaries
    if threshold is not None:
        subset = frame[(frame["controller"] == "laya_agent") & (frame["confidence_threshold"] == threshold)]
    else:
        subset = frame[frame["controller"] == "laya_agent"]
    if subset.empty:
        return pd.DataFrame()
    rows: list[dict[str, object]] = []
    for difficulty, group in subset.groupby("difficulty"):
        rows.append(
            {
                "difficulty": difficulty,
                "episodes": len(group),
                "unsafe_actions_prevented": int(group["unsafe_actions_prevented"].sum()),
                "collisions_prevented": int(group["collisions_prevented"].sum()),
                "hazard_entries_prevented": int(group["hazard_entries_prevented"].sum()),
                "guardian_overrides": int(group["guardian_overrides"].sum()),
                "real_collisions": int(group["collisions"].sum()),
                "prevented_per_episode": round(float(group["unsafe_actions_prevented"].mean()), 3),
            }
        )
    return pd.DataFrame(rows)


def reason_code_totals(reason_counts: dict[str, int]) -> pd.DataFrame:
    if not reason_counts:
        return pd.DataFrame(columns=["reason_code", "count"])
    return (
        pd.DataFrame(sorted(reason_counts.items()), columns=["reason_code", "count"])
        .sort_values("count", ascending=False)
        .reset_index(drop=True)
    )


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #


def _rate(series: pd.Series) -> float:
    if len(series) == 0:
        return float("nan")
    return round(float(np.mean(series.astype(float))), 4)


def _mean(series: pd.Series, include_zeros: bool = True) -> float:
    values = pd.to_numeric(series, errors="coerce").dropna()
    if not include_zeros:
        values = values[values > 0]
    if len(values) == 0:
        return float("nan")
    return round(float(values.mean()), 4)


def _median(series: pd.Series) -> float:
    values = pd.to_numeric(series, errors="coerce").dropna()
    if len(values) == 0:
        return float("nan")
    return round(float(values.median()), 4)
