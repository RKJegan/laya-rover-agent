"""Figure generation (FR-11).

Six figures, all written as PNGs into ``outputs/figures/``:

1. ``success_rate_by_difficulty.png``
2. ``energy_by_controller.png``
3. ``path_examples.png``
4. ``latency_distribution.png``
5. ``guardian_ablation_safety_case.png``
6. ``confidence_calibration.png``

matplotlib only (no seaborn), headless ``Agg`` backend, deterministic styling.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")  # headless: no display required in CI or on a rover SBC

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from ..types import CellType  # noqa: E402

COLORS = {
    "laya_agent": "#1f77b4",
    "baseline": "#d62728",
    "fallback": "#7f7f7f",
    "accent": "#2ca02c",
    "warn": "#ff7f0e",
}
DIFFICULTY_ORDER = ("easy", "medium", "hard")


def _ensure(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _ordered(values: Iterable[str]) -> list[str]:
    known = [d for d in DIFFICULTY_ORDER if d in set(values)]
    extra = sorted(set(values) - set(known))
    return known + extra


# --------------------------------------------------------------------------- #


def success_rate_by_difficulty(summaries: pd.DataFrame, path: Path, threshold: float | None = None) -> Path:
    path = _ensure(path)
    frame = summaries.copy()
    if threshold is not None:
        frame = frame[(frame["controller"] == "baseline") | (frame["confidence_threshold"] == threshold)]

    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    difficulties = _ordered(frame["difficulty"].unique())
    controllers = [c for c in ("baseline", "laya_agent") if c in set(frame["controller"])]
    width = 0.8 / max(1, len(controllers))

    for index, controller in enumerate(controllers):
        means, errors = [], []
        for difficulty in difficulties:
            group = frame[(frame["controller"] == controller) & (frame["difficulty"] == difficulty)]
            if group.empty:
                means.append(0.0)
                errors.append(0.0)
                continue
            values = group.groupby("map_key")["success"].mean().to_numpy(dtype=float)
            means.append(float(values.mean()))
            errors.append(_sem(values))
        positions = np.arange(len(difficulties)) + (index - (len(controllers) - 1) / 2) * width
        ax.bar(positions, means, width, yerr=errors, capsize=3,
               label=controller, color=COLORS.get(controller, "#333333"), alpha=0.9)

    ax.set_xticks(np.arange(len(difficulties)))
    ax.set_xticklabels(difficulties)
    ax.set_ylabel("Goal success rate")
    ax.set_ylim(0, 1.05)
    title = "Goal success rate by difficulty"
    if threshold is not None:
        title += f" (Laya confidence threshold = {threshold:.2f})"
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def energy_by_controller(summaries: pd.DataFrame, path: Path) -> Path:
    path = _ensure(path)
    successful = summaries[summaries["success"] == 1]
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    difficulties = _ordered(summaries["difficulty"].unique())
    controllers = [c for c in ("baseline", "laya_agent") if c in set(summaries["controller"])]

    positions, labels, data, colors = [], [], [], []
    slot = 0
    for difficulty in difficulties:
        for controller in controllers:
            group = successful[
                (successful["difficulty"] == difficulty) & (successful["controller"] == controller)
            ]
            if group.empty:
                slot += 1
                continue
            positions.append(slot)
            data.append(group["energy_used"].to_numpy(dtype=float))
            labels.append(f"{difficulty}\n{controller}")
            colors.append(COLORS.get(controller, "#333333"))
            slot += 1
        slot += 0.4

    if not data:
        ax.text(0.5, 0.5, "no successful episodes", ha="center", va="center")
    else:
        box = ax.boxplot(data, positions=positions, widths=0.6, patch_artist=True, showfliers=True)
        for patch, color in zip(box["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.55)
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("Energy used (units) - successful episodes")
    ax.set_title("Energy cost comparison (successful runs only)")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def path_examples(examples: Sequence[Mapping[str, object]], path: Path) -> Path:
    """Overlay baseline vs Laya trajectories on the same maps."""
    path = _ensure(path)
    if not examples:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(0.5, 0.5, "no example trajectories captured", ha="center", va="center")
        ax.axis("off")
        fig.savefig(path, dpi=160)
        plt.close(fig)
        return path

    count = len(examples)
    fig, axes = plt.subplots(1, count, figsize=(4.4 * count, 4.6), squeeze=False)
    grid_colors = {
        CellType.FREE.value: 0.95,
        CellType.OBSTACLE.value: 0.15,
        CellType.HAZARD_SLOPE.value: 0.65,
        CellType.START.value: 0.9,
        CellType.GOAL.value: 0.85,
    }
    for ax, example in zip(axes[0], examples):
        cells = example["cells"]
        matrix = np.array([[grid_colors.get(c, 0.95) for c in row] for row in cells], dtype=float)
        ax.imshow(matrix, cmap="Greys", vmin=0.0, vmax=1.0, interpolation="nearest")
        start, goal = example["start"], example["goal"]
        ax.scatter([start[0]], [start[1]], marker="o", s=60, facecolors="none", edgecolors=COLORS["accent"], linewidths=2)
        ax.scatter([goal[0]], [goal[1]], marker="*", s=140, c=COLORS["accent"])
        for key, color, label in (
            ("baseline_path", COLORS["baseline"], "baseline"),
            ("laya_path", COLORS["laya_agent"], "laya_agent"),
        ):
            trajectory = example.get(key) or []
            if len(trajectory) > 1:
                ys = [p[1] for p in trajectory]
                xs = [p[0] for p in trajectory]
                ax.plot(xs, ys, "-", color=color, linewidth=1.8, alpha=0.85, label=label)
        ax.set_title(f"{example['difficulty']} / {example['preset']}", fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(fontsize=8, loc="upper right", frameon=False)
    fig.suptitle("Example trajectories: greedy baseline vs Laya agent (same seeded map)", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def latency_distribution(latencies: Mapping[str, Sequence[float]], path: Path, decision_timeout_ms: float = 1200.0) -> Path:
    path = _ensure(path)
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    plotted = False
    for index, (label, values) in enumerate(sorted(latencies.items())):
        values = [v for v in values if v is not None]
        if not values:
            continue
        plotted = True
        ax.hist(values, bins=30, alpha=0.55, label=f"{label} (n={len(values)})",
                color=list(COLORS.values())[index % len(COLORS)])
    if not plotted:
        ax.text(0.5, 0.5, "no latency samples", ha="center", va="center")
    ax.axvline(decision_timeout_ms, color=COLORS["warn"], linestyle="--", linewidth=1.2,
               label=f"decision_timeout ({decision_timeout_ms:.0f} ms)")
    ax.set_xlabel("Decision latency (ms)")
    ax.set_ylabel("Timesteps")
    ax.set_title("Laya decision latency distribution")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def guardian_ablation_safety_case(ablation: pd.DataFrame, path: Path, total_overrides: int = 0) -> Path:
    """The safety-case chart: unsafe entries the guardian prevented."""
    path = _ensure(path)
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    if ablation.empty:
        ax.text(0.5, 0.5, "no ablation data (no rejected proposals recorded)", ha="center", va="center")
        ax.axis("off")
        fig.savefig(path, dpi=160)
        plt.close(fig)
        return path

    difficulties = list(ablation["difficulty"])
    x = np.arange(len(difficulties))
    collisions = ablation["collisions_prevented"].to_numpy(dtype=float)
    hazards = ablation["hazard_entries_prevented"].to_numpy(dtype=float)
    others = (ablation["unsafe_actions_prevented"].to_numpy(dtype=float) - collisions - hazards).clip(min=0)

    ax.bar(x, collisions, 0.55, label="obstacle/out-of-bounds entries prevented", color=COLORS["baseline"])
    ax.bar(x, hazards, 0.55, bottom=collisions, label="hazard-slope entries prevented", color=COLORS["warn"])
    ax.bar(x, others, 0.55, bottom=collisions + hazards, label="other unsafe/illegal actions prevented", color="#8c8c8c")
    for index, total in enumerate(ablation["unsafe_actions_prevented"]):
        ax.text(index, total + max(0.4, 0.02 * max(1.0, total)), str(int(total)), ha="center", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(difficulties)
    ax.set_ylabel("Unsafe actions prevented (count)")
    ax.set_title("Guardian safety case: unsafe entries prevented by the deterministic safety layer")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def confidence_calibration(calibration: pd.DataFrame, path: Path, threshold: float | None = None) -> Path:
    """Reliability diagram: stated confidence vs agreement with the safe controller."""
    path = _ensure(path)
    fig, ax = plt.subplots(figsize=(7.0, 4.8))
    if calibration.empty:
        ax.text(0.5, 0.5, "no calibration samples", ha="center", va="center")
        ax.axis("off")
        fig.savefig(path, dpi=160)
        plt.close(fig)
        return path

    frame = calibration
    if threshold is not None and "confidence_threshold" in frame:
        subset = frame[frame["confidence_threshold"] == threshold]
        if not subset.empty:
            frame = subset

    ax.plot([0, 1], [0, 1], linestyle="--", color="#999999", linewidth=1, label="perfect calibration")
    for index, (key, group) in enumerate(frame.groupby("confidence_threshold")):
        group = group.sort_values("bin_low")
        centers = (group["bin_low"] + group["bin_high"]) / 2.0
        ax.plot(centers, group["agreement_rate"], "o-", linewidth=1.8,
                color=list(COLORS.values())[index % len(COLORS)], label=f"threshold {key}")
        for x, y, n in zip(centers, group["agreement_rate"], group["n_steps"]):
            if n > 0:
                ax.annotate(f"n={int(n)}", (x, y), textcoords="offset points", xytext=(0, 7), fontsize=6, ha="center")
    ax.set_xlabel("Laya stated confidence (bin centre)")
    ax.set_ylabel("Agreement with fallback controller's choice")
    ax.set_title("Confidence calibration: stated confidence vs operational outcome")
    ax.set_xlim(0, 1.02)
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def _sem(values: np.ndarray) -> float:
    if values.size <= 1:
        return 0.0
    return float(values.std(ddof=1) / np.sqrt(values.size))
