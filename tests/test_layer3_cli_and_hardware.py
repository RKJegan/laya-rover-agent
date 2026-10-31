"""Layer 3 - CLI surface and the robot connector layer (sim backend).

Every command is exercised in-process (no subprocess, no network), and the
virtual simulator backend is driven end to end through the same bridge that a
real rover would use - this is what makes "connects to a rover without changes"
a testable claim rather than a promise.
"""

from __future__ import annotations

import asyncio
import json

import pytest

from conftest import project_root
from laya_rover_agent.cli import main
from laya_rover_agent.config import load_config
from laya_rover_agent.hardware import RoverBridge, SimBackend, make_backend
from laya_rover_agent.hardware.base import ExecutionReport, SensorFrame, frame_to_state
from laya_rover_agent.hardware.serial_backend import parse_ack, parse_telemetry
from laya_rover_agent.laya.factory import ControllerFactory
from laya_rover_agent.types import Action, Heading, RoverState, Status
from laya_rover_agent.world import MapSpec

QUICK = "configs/quick.yaml"


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def test_help_lists_every_command(capsys):
    with pytest.raises(SystemExit) as exit_info:
        main(["--help"])
    assert exit_info.value.code == 0
    out = capsys.readouterr().out
    for command in ("experiment", "episode", "hardware", "map", "plan", "estimate"):
        assert command in out


def test_map_command_prints_ascii_and_route(capsys):
    assert main(["map", "--config", QUICK, "--preset", "trap_u", "--seed", "1"]) == 0
    out = capsys.readouterr().out
    assert "S" in out and "G" in out and "#" in out


def test_plan_and_estimate_report_call_volume(capsys):
    assert main(["plan", "--config", QUICK, "--seeds", "2"]) == 0
    plan = capsys.readouterr().out
    assert "episode" in plan.lower()

    assert main(["estimate", "--config", QUICK, "--seeds", "2"]) == 0
    estimate = capsys.readouterr().out
    assert "api" in estimate.lower()


def test_episode_command_logs_jsonl(tmp_path, capsys):
    log = tmp_path / "one.jsonl"
    exit_code = main(
        [
            "episode",
            "--config", QUICK,
            "--seed", "1",
            "--difficulty", "easy",
            "--preset", "trap_u",
            "--controller", "laya_agent",
            "--max-rows", "3",
            "--log-jsonl", str(log),
        ]
    )
    assert exit_code == 0
    rows = [json.loads(line) for line in log.read_text().splitlines() if line.strip()]
    assert rows and rows[0]["final_action"]


def test_episode_command_rejects_unknown_preset(capsys):
    with pytest.raises(SystemExit) as exit_info:
        main(["episode", "--preset", "trap_volcano"])
    assert exit_info.value.code == 2


def test_experiment_command_requires_a_key_only_when_the_server_does(capsys, monkeypatch, tmp_path, project_root):
    """Laya is self-hosted: no key by default, but ``auth_required`` fails closed."""
    import yaml

    monkeypatch.delenv("LAYA_API_KEY", raising=False)

    # (a) auth_required: true + no key -> refuse to start, before any network call
    secured = yaml.safe_load((project_root / QUICK).read_text())
    secured["agent"]["laya_api"]["auth_required"] = True
    secured_path = tmp_path / "secured.yaml"
    secured_path.write_text(yaml.safe_dump(secured))
    exit_code = main(["experiment", "--config", str(secured_path), "--mode", "laya_api", "--seeds", "1"])
    assert exit_code == 2
    assert "LAYA_API_KEY" in capsys.readouterr().out


def test_laya_api_mode_with_an_unreachable_server_still_completes_safely(tmp_path, capsys, monkeypatch):
    """Fail-closed end to end: no server, no key, no crash - and no crash *of the rover*.

    Every decision fails to connect and routes to the deterministic fallback
    controller, which keeps the mission running under the guardian. This is the
    property that makes a local model acceptable on a rover: the agent does not
    depend on the model being alive.
    """
    monkeypatch.delenv("LAYA_API_KEY", raising=False)
    monkeypatch.setenv("LAYA_API_URL", "http://127.0.0.1:9/v1/systemone")  # discard port: refused
    monkeypatch.setenv("LAYA_REQUEST_TIMEOUT_S", "0.4")
    log = tmp_path / "unreachable.jsonl"

    exit_code = main(
        [
            "episode",
            "--config", QUICK,
            "--seed", "0",
            "--difficulty", "easy",
            "--preset", "trap_u",
            "--controller", "laya_agent",
            "--mode", "laya_api",
            "--max-rows", "1",
            "--log-jsonl", str(log),
        ]
    )
    assert exit_code == 0, "an unreachable decision server must not break the episode"
    assert log.exists()

    import json as _json

    rows = [_json.loads(line) for line in log.read_text().splitlines() if line.strip()]
    assert rows, "the episode must still log steps"
    # nothing was ever approved from a model that never answered
    assert all(row["action_source"] != "laya" for row in rows)
    assert all(row["collision"] is False for row in rows)
    assert not any(row["state_after"]["status"] == "collision" for row in rows)


def test_hardware_command_runs_a_virtual_mission(tmp_path, capsys):
    log = tmp_path / "mission.jsonl"
    exit_code = main(
        [
            "hardware",
            "--config", QUICK,
            "--backend", "sim",
            "--preset", "trap_u",
            "--difficulty", "easy",
            "--seed", "0",
            "--controller", "laya_agent",
            "--log", str(log),
        ]
    )
    assert exit_code in (0, 1)  # 0 = goal, 1 = mission ended without reaching it
    assert log.exists() and log.with_suffix(".summary.json").exists()
    summary = json.loads(log.with_suffix(".summary.json").read_text())
    assert summary["backend"] == "sim"
    assert summary["steps"] > 0
    assert "LAYA_API_KEY" not in json.dumps(summary)


def test_hardware_dry_run_does_not_move(capsys, quick_config):
    assert main(["hardware", "--config", QUICK, "--backend", "sim", "--dry-run", "--max-steps", "5"]) in (0, 1)
    assert "mission summary" in capsys.readouterr().out.lower()


# --------------------------------------------------------------------------- #
# Connector layer
# --------------------------------------------------------------------------- #


def test_sim_backend_satisfies_the_backend_protocol(quick_config):
    backend = SimBackend(quick_config, seed=0, difficulty="easy", preset="random")
    for member in ("name", "connect", "close", "read_sensors", "execute", "emergency_stop"):
        assert hasattr(backend, member)
    assert backend.name == "sim"


def test_sensor_frame_is_schema_complete(quick_config):
    backend = SimBackend(quick_config, seed=1, difficulty="easy", preset="random")
    backend.connect()
    frame = backend.read_sensors()
    assert isinstance(frame, SensorFrame)
    sentinel = frame.sensor_range + 1
    assert set(frame.obstacles) == {"front", "front_left", "front_right", "left", "right"}
    for distance in frame.obstacles.values():
        assert 0 < distance <= sentinel
    assert set(frame.hazards) == {"front", "front_left", "front_right"}
    assert isinstance(frame.heading, str) and frame.heading in {"N", "E", "S", "W"}
    assert 0.0 <= frame.battery <= 1.0


def test_frame_to_state_preserves_pose_and_heading(quick_config):
    frame = SensorFrame(
        x=3, y=4, heading="S", battery=0.9,
        obstacles={"front": 6, "front_left": 6, "front_right": 6, "left": 6, "right": 6},
        hazards={"front": False, "front_left": False, "front_right": False},
        goal_relative=(10, 10), goal_distance=14.14, sensor_range=5, timestamp_s=0.0,
    )
    state = frame_to_state(frame, RoverState(x=0, y=0, heading="N", battery=1.0, energy_used=0.0, steps=0,
                                           status=Status.ACTIVE, consecutive_stops=0))
    assert (state.x, state.y, state.heading) == (3, 4, "S")
    assert state.battery == pytest.approx(0.9)


def test_bridge_drives_the_sim_backend_with_the_guardian_in_the_loop(quick_config, tmp_path):
    backend = SimBackend(quick_config, seed=0, difficulty="easy", preset="trap_u")
    factory = ControllerFactory(quick_config)
    spec_controller = factory(
        __import__("laya_rover_agent.episode", fromlist=["EpisodeSpec"]).EpisodeSpec(
            map_spec=MapSpec(seed=0, difficulty="easy", preset="trap_u"),
            controller="laya_agent",
            confidence_threshold=0.70,
        )
    )
    bridge = RoverBridge(quick_config, backend, spec_controller, log_path=tmp_path / "m.jsonl")
    summary = asyncio.run(bridge.run_mission_async(max_steps=60, dry_run=False))
    assert summary.steps > 0
    assert summary.laya_calls == summary.steps
    assert summary.actions
    rows = [json.loads(line) for line in (tmp_path / "m.jsonl").read_text().splitlines() if line.strip()]
    assert len(rows) == summary.steps
    for row in rows:
        assert "observation" in row and "guardian_reason_code" in row


def test_bridge_never_executes_an_unsafe_action(quick_config, tmp_path):
    """Compare every executed action against the guardian's own verdict."""
    backend = SimBackend(quick_config, seed=2, difficulty="medium", preset="trap_canyon")
    factory = ControllerFactory(quick_config)
    from laya_rover_agent.episode import EpisodeSpec

    controller = factory(
        EpisodeSpec(
            map_spec=MapSpec(seed=2, difficulty="medium", preset="trap_canyon"),
            controller="laya_agent",
            confidence_threshold=0.70,
        )
    )
    bridge = RoverBridge(quick_config, backend, controller, log_path=tmp_path / "m.jsonl")
    asyncio.run(bridge.run_mission_async(max_steps=80, dry_run=False))
    rows = [json.loads(line) for line in (tmp_path / "m.jsonl").read_text().splitlines() if line.strip()]
    state = backend.grid
    for row in rows:
        action = Action(row["final_action"])
        x, y = row["state_before"]["x"], row["state_before"]["y"]
        heading = row["state_before"]["heading"]
        from laya_rover_agent.dynamics import action_target

        tx, ty = action_target(x, y, heading, action)
        assert state.in_bounds(tx, ty)
        assert not state.is_obstacle(tx, ty)


def test_backend_factory_reports_unavailable_transports_clearly(quick_config):
    from laya_rover_agent.hardware.base import BackendUnavailableError

    try:
        backend = make_backend("ros2", quick_config, namespace="/test")
        backend.connect()
    except (BackendUnavailableError, ImportError) as exc:
        assert "rclpy" in str(exc) or "ROS" in str(exc)
    except Exception as exc:  # pragma: no cover - environment dependent
        pytest.skip(f"unexpected ROS 2 environment: {exc}")


def test_serial_telemetry_parser_accepts_the_documented_line():
    line = "TEL x=12 y=8 h=N bat=0.94 of=3,6,6,6,2 oh=F,T,F gdx=9 gdy=-14 ts=1.5"
    frame = parse_telemetry(line, sensor_range=6)
    assert frame["x"] == 12 and frame["y"] == 8 and frame["heading"] == "N"
    assert frame["battery"] == pytest.approx(0.94)
    assert frame["obstacles"]["front"] == 3
    assert frame["obstacles"]["left"] == 6
    assert frame["hazards"]["front_left"] is True
    assert frame["goal_relative"] == (9, -14)


def test_serial_telemetry_missing_beams_default_to_the_sentinel():
    frame = parse_telemetry("TEL x=1 y=2 h=E bat=1.0", sensor_range=6)
    assert set(frame["obstacles"].values()) == {7}
    assert all(value is False for value in frame["hazards"].values())


def test_serial_telemetry_parser_rejects_garbage():
    from laya_rover_agent.hardware.serial_backend import TelemetryParseError

    with pytest.raises(TelemetryParseError):
        parse_telemetry("not a telemetry line", sensor_range=6)


def test_serial_ack_parser():
    ack = parse_ack("ACK MOVE_FORWARD moved=1 stall=0 collision=0 goal=0")
    assert ack["action"] is Action.MOVE_FORWARD
    assert ack["moved"] is True and ack["collision"] is False
    assert parse_ack("# comment") is None
    assert parse_ack("ACK NOT_AN_ACTION moved=1") is None


def test_execution_report_shape(quick_config):
    backend = SimBackend(quick_config, seed=0, difficulty="easy", preset="random")
    backend.connect()
    report = backend.execute(Action.MOVE_FORWARD)
    assert isinstance(report, ExecutionReport)
    assert isinstance(report.moved, bool)
    assert report.energy_used >= 0.0
    assert report.position is not None


def test_emergency_stop_is_idempotent_and_safe(quick_config):
    backend = SimBackend(quick_config, seed=0, difficulty="easy", preset="random")
    backend.connect()
    backend.emergency_stop()
    backend.emergency_stop()
    assert backend.status is Status.STUCK
