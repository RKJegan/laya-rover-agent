"""Serial (UART/USB) backend for microcontroller-based rovers.

Line protocol (ASCII, newline-terminated, 115200 baud by default)::

    agent -> rover   ACT MOVE_FORWARD
                     ACT TURN_LEFT_AND_MOVE
                     ESTOP
                     PING

    rover -> agent   TEL x=12 y=7 h=N bat=0.87 \
                         of=3,7,2,5,1 oh=F,F,T \
                         gdx=24 gdy=-9 ts=1234.5
                     ACK MOVE_FORWARD moved=1 stall=0 collision=0 goal=0
                     ERR <message>

``of`` is the five obstacle beam distances in the order
``front, front_left, front_right, left, right`` and uses ``sensor_range + 1`` as
the "nothing detected" sentinel. ``oh`` is the three forward hazard flags.
Anything malformed is ignored (the agent keeps its last good frame) - a rover
must never act on a corrupt sentence.

``pyserial`` is optional; :func:`parse_telemetry` is pure and unit-tested
without any hardware.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from ..types import Action, Heading
from .base import BackendUnavailableError, ExecutionReport, SensorFrame

BEAM_ORDER: tuple[str, ...] = ("front", "front_left", "front_right", "left", "right")
HAZARD_ORDER: tuple[str, ...] = ("front", "front_left", "front_right")


class TelemetryParseError(ValueError):
    """Raised for an unusable telemetry sentence (caller ignores it)."""


def parse_telemetry(line: str, sensor_range: int) -> dict[str, Any]:
    """Parse a ``TEL ...`` sentence into a dict. Pure function - no I/O."""
    line = line.strip()
    if not line.startswith("TEL"):
        raise TelemetryParseError(f"not a telemetry sentence: {line!r}")
    fields: dict[str, str] = {}
    for token in line.split()[1:]:
        if "=" not in token:
            continue
        key, value = token.split("=", 1)
        fields[key] = value

    try:
        x = int(fields["x"])
        y = int(fields["y"])
    except (KeyError, ValueError) as exc:
        raise TelemetryParseError(f"missing/invalid position in {line!r}") from exc

    heading = fields.get("h", "N").upper()
    if heading not in {"N", "E", "S", "W"}:
        raise TelemetryParseError(f"invalid heading {heading!r}")

    try:
        battery = float(fields.get("bat", "1.0"))
    except ValueError as exc:
        raise TelemetryParseError(f"invalid battery {fields.get('bat')!r}") from exc

    distances = []
    for raw in fields.get("of", "").split(","):
        try:
            distances.append(int(raw))
        except ValueError:
            distances.append(sensor_range + 1)
    while len(distances) < len(BEAM_ORDER):
        distances.append(sensor_range + 1)

    flags = [value.strip().upper().startswith("T") for value in fields.get("oh", "").split(",")]
    while len(flags) < len(HAZARD_ORDER):
        flags.append(False)

    return {
        "x": x,
        "y": y,
        "heading": heading,
        "battery": max(0.0, min(1.0, battery)),
        "obstacles": dict(zip(BEAM_ORDER, distances[: len(BEAM_ORDER)])),
        "hazards": dict(zip(HAZARD_ORDER, flags[: len(HAZARD_ORDER)])),
        "goal_relative": (_int_or(fields.get("gdx"), 0), _int_or(fields.get("gdy"), 0)),
        "timestamp_s": _float_or(fields.get("ts"), time.time()),
    }


def parse_ack(line: str) -> dict[str, Any] | None:
    """Parse an ``ACK <ACTION> moved=.. stall=.. collision=.. goal=..`` sentence."""
    line = line.strip()
    if not line.startswith("ACK"):
        return None
    tokens = line.split()
    if len(tokens) < 2:
        return None
    try:
        action = Action(tokens[1])
    except ValueError:
        return None
    fields: dict[str, Any] = {"action": action}
    for token in tokens[2:]:
        if "=" not in token:
            continue
        key, value = token.split("=", 1)
        fields[key] = value in {"1", "true", "True", "yes"}
    return fields


class SerialBackend:
    name = "serial"

    def __init__(
        self,
        port: str,
        baudrate: int = 115200,
        sensor_range: int = 6,
        read_timeout_s: float = 1.5,
        ack_timeout_s: float = 2.0,
    ) -> None:
        self.port = port
        self.baudrate = baudrate
        self.sensor_range = sensor_range
        self.read_timeout_s = read_timeout_s
        self.ack_timeout_s = ack_timeout_s
        self._serial: Any = None
        self._last_frame: SensorFrame | None = None

    # -- lifecycle ---------------------------------------------------------- #

    def connect(self) -> None:
        try:
            import serial  # noqa: PLC0415
        except ImportError as exc:  # pragma: no cover - depends on host
            raise BackendUnavailableError(
                "pyserial is not installed. Install the hardware extra: pip install 'laya-rover-agent[hardware]'"
            ) from exc
        self._serial = serial.Serial(self.port, self.baudrate, timeout=self.read_timeout_s)
        time.sleep(1.0)  # many boards reset on DTR
        self._serial.reset_input_buffer()
        self._serial.write(b"PING\n")

    def close(self) -> None:
        if self._serial is not None:
            self._serial.close()
            self._serial = None

    def emergency_stop(self) -> None:
        self._write("ESTOP")

    # -- I/O ---------------------------------------------------------------- #

    def read_sensors(self) -> SensorFrame:
        deadline = time.monotonic() + self.read_timeout_s
        while time.monotonic() < deadline:
            raw = self._serial.readline()
            if not raw:
                continue
            try:
                line = raw.decode("ascii", errors="ignore")
            except Exception:  # noqa: BLE001
                continue
            if not line.startswith("TEL"):
                continue
            try:
                parsed = parse_telemetry(line, self.sensor_range)
            except TelemetryParseError:
                continue
            goal_relative = parsed["goal_relative"]
            frame = SensorFrame(
                x=parsed["x"],
                y=parsed["y"],
                heading=parsed["heading"],
                battery=parsed["battery"],
                obstacles=parsed["obstacles"],
                hazards=parsed["hazards"],
                goal_relative=goal_relative,
                goal_distance=(goal_relative[0] ** 2 + goal_relative[1] ** 2) ** 0.5,
                sensor_range=self.sensor_range,
                timestamp_s=parsed["timestamp_s"],
                raw={"line": line.strip()},
            )
            self._last_frame = frame
            return frame
        if self._last_frame is None:
            raise TimeoutError("no telemetry received from the rover")
        return self._last_frame

    def execute(self, action: Action) -> ExecutionReport:
        started = time.perf_counter()
        self._write(f"ACT {action.value}")
        deadline = time.monotonic() + self.ack_timeout_s
        while time.monotonic() < deadline:
            raw = self._serial.readline()
            if not raw:
                continue
            line = raw.decode("ascii", errors="ignore")
            ack = parse_ack(line)
            if ack is None or ack["action"] is not action:
                continue
            frame = self.read_sensors()
            return ExecutionReport(
                moved=bool(ack.get("moved", action is not Action.STOP_AND_REPLAN)),
                stalled=bool(ack.get("stall", False)),
                collision=bool(ack.get("collision", False)),
                reached_goal=bool(ack.get("goal", False)),
                battery=frame.battery,
                position=(frame.x, frame.y),
                heading=frame.heading,
                latency_ms=(time.perf_counter() - started) * 1000.0,
                raw=ack,
            )
        # No ACK: treat as a failed manoeuvre that consumed energy but did not
        # advance, and let the caller's stall/step accounting handle it.
        frame = self.read_sensors()
        return ExecutionReport(
            moved=False,
            stalled=True,
            battery=frame.battery,
            position=(frame.x, frame.y),
            heading=frame.heading,
            latency_ms=(time.perf_counter() - started) * 1000.0,
            raw={"timeout": True},
        )

    # -- helpers ------------------------------------------------------------ #

    def _write(self, text: str) -> None:
        if self._serial is None:
            raise RuntimeError("connect() must be called before writing to the port")
        self._serial.write((text + "\n").encode("ascii"))


def _int_or(value: str | None, default: int) -> int:
    try:
        return int(value) if value is not None else default
    except ValueError:
        return default


def _float_or(value: str | None, default: float) -> float:
    try:
        return float(value) if value is not None else default
    except ValueError:
        return default


def heading_to_yaw(heading: Heading) -> float:
    """Inverse of the ROS mapping, handy for firmware authors."""
    import math

    return {"E": 0.0, "S": math.pi / 2, "W": math.pi, "N": -math.pi / 2}[heading]
