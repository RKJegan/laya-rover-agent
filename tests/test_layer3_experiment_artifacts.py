"""Layer 3 - the experiment pipeline produces every required artifact (FR-9/10/11).

A deliberately tiny matrix (2 seeds x 2 difficulties x 2 thresholds) is run
end-to-end, offline, so the deliverables named in the spec are verified rather
than assumed: JSONL episode logs, the three CSVs, run metadata with a config
hash, the six figures and the report with its mandated sections.
"""

from __future__ import annotations

import asyncio
import csv
import json

import pytest

from laya_rover_agent.config import load_config
from laya_rover_agent.experiment.runner import ExperimentRunner

ARTIFACT_FILES = (
    "episode_summaries.csv",
    "aggregate_metrics.csv",
    "paired_stats.csv",
    "calibration.csv",
    "ablation.csv",
    "run_metadata.json",
    "report.md",
)

FIGURE_FILES = (
    "success_rate_by_difficulty.png",
    "energy_by_controller.png",
    "path_examples.png",
    "latency_distribution.png",
    "guardian_ablation_safety_case.png",
    "confidence_calibration.png",
)

REPORT_SECTIONS = (
    "## 1. Executive summary",
    "## 2. What was built",
    "## 3. Reproducibility",
    "## 4. Results",
    "## 5. Safety case: guardian ablation",
    "## 6. Confidence calibration",
    "## 7. Live Laya API",
    "## 9. Honest limitations",
    "## 10. Reproduce",
)


@pytest.fixture(scope="module")
def artifacts(tmp_path_factory, project_root):
    output_dir = tmp_path_factory.mktemp("experiment")
    config = load_config(
        project_root / "configs" / "quick.yaml",
        overrides={
            "experiment": {
                "seeds_per_difficulty": 2,
                "difficulties": ["easy", "medium"],
                "confidence_thresholds": [0.60, 0.80],
                "output_dir": str(output_dir),
                "jobs": 1,
            }
        },
    )
    runner = ExperimentRunner(config, run_label="pytest", verbose=False)
    result = asyncio.run(runner.run())
    return output_dir, result


def test_number_of_episodes_matches_the_matrix(artifacts):
    output_dir, result = artifacts
    # 2 seeds x 2 difficulties x (3 laya thresholds? no - 2 thresholds + 1 baseline)
    with (output_dir / "episode_summaries.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    controllers = {row["controller"] for row in rows}
    assert controllers == {"laya_agent", "baseline"}
    laya = [row for row in rows if row["controller"] == "laya_agent"]
    baseline = [row for row in rows if row["controller"] == "baseline"]
    assert len(laya) == 2 * 2 * 2  # seeds x difficulties x thresholds
    assert len(baseline) == 2 * 2  # baseline is threshold-independent
    assert len(list((output_dir / "episode_logs").glob("*.jsonl"))) == len(rows)


@pytest.mark.parametrize("filename", ARTIFACT_FILES)
def test_required_artifacts_exist(artifacts, filename):
    output_dir, _ = artifacts
    path = output_dir / filename
    assert path.exists() and path.stat().st_size > 0


@pytest.mark.parametrize("filename", FIGURE_FILES)
def test_all_six_figures_render(artifacts, filename):
    output_dir, _ = artifacts
    path = output_dir / "figures" / filename
    assert path.exists()
    assert path.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n", "figure must be a real PNG"
    assert path.stat().st_size > 5_000, "figure looks empty"


@pytest.mark.parametrize("section", REPORT_SECTIONS)
def test_report_contains_each_section(artifacts, section):
    output_dir, _ = artifacts
    assert section in (output_dir / "report.md").read_text(encoding="utf-8")


def test_run_metadata_pins_the_configuration(artifacts):
    output_dir, _ = artifacts
    metadata = json.loads((output_dir / "run_metadata.json").read_text())
    assert len(metadata["config_hash_sha256"]) == 64
    assert metadata["resolved_config"]["experiment"]["difficulties"] == ["easy", "medium"]
    assert metadata["resolved_config"]["agent"]["mode"] == "mock_laya"
    assert metadata["environment"], "environment/version info must be recorded"
    assert metadata["secrets"], "redaction status must be recorded"
    blob = json.dumps(metadata)
    assert "jv_live_" not in blob and "LAYA_API_KEY=" not in blob


def test_paired_statistics_have_bootstrap_intervals(artifacts):
    output_dir, _ = artifacts
    with (output_dir / "paired_stats.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    success = [row for row in rows if row["metric"] == "success_rate"]
    assert success, "paired success-rate comparison is mandatory"
    for row in success:
        assert float(row["ci95_low"]) <= float(row["mean_delta"]) <= float(row["ci95_high"])
        assert int(row["n_pairs"]) == 2  # seeds per difficulty


def test_calibration_table_covers_confidence_bins(artifacts):
    output_dir, _ = artifacts
    with (output_dir / "calibration.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    assert rows, "calibration must be reported"
    assert {"confidence_threshold", "mean_confidence", "agreement_rate", "calibration_gap"} <= set(rows[0])
    for row in rows:
        assert 0.0 <= float(row["agreement_rate"]) <= 1.0


def test_ablation_reports_prevented_unsafe_actions(artifacts):
    output_dir, _ = artifacts
    with (output_dir / "ablation.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    assert rows
    assert {"unsafe_actions_prevented", "collisions_prevented", "real_collisions"} <= set(rows[0])
    assert all(int(row["real_collisions"]) == 0 for row in rows), "enforced mode must record no real collisions"


def test_episode_logs_are_valid_jsonl(artifacts):
    output_dir, _ = artifacts
    files = sorted((output_dir / "episode_logs").glob("*.jsonl"))
    assert files
    for path in files[:3]:
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        assert rows
        assert {"step", "episode_id", "guardian_reason_code", "final_action"} <= set(rows[0])


def test_artifacts_are_reproducible(tmp_path, project_root):
    """Same config, same seeds -> identical summaries (deterministic pipeline)."""
    config = load_config(
        project_root / "configs" / "quick.yaml",
        overrides={
            "experiment": {
                "seeds_per_difficulty": 1,
                "difficulties": ["easy"],
                "confidence_thresholds": [0.70],
                "output_dir": str(tmp_path / "one"),
                "jobs": 1,
            }
        },
    )
    first = asyncio.run(ExperimentRunner(config, run_label="a", verbose=False).run())
    second_config = config.model_copy(
        update={"experiment": config.experiment.model_copy(update={"output_dir": str(tmp_path / "two")})}
    )
    second = asyncio.run(ExperimentRunner(second_config, run_label="b", verbose=False).run())

    def rows(path):
        with path.open() as handle:
            return [
                (row["episode_id"], row["success"], row["steps"], row["energy_used"])
                for row in csv.DictReader(handle)
            ]

    assert rows(first.episode_summaries) == rows(second.episode_summaries)

    # The hash intentionally covers the *whole* resolved config, output dir
    # included; strip that one key and every scientific setting must match.
    def normalised(path):
        resolved = json.loads(path.read_text())["resolved_config"]
        resolved["experiment"].pop("output_dir", None)
        return resolved

    assert normalised(first.metadata) == normalised(second.metadata)
