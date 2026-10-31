# Connecting to a real rover (no code changes)

The simulation is the reference implementation, but the decision stack is not
welded to it. Everything above the transport layer - perception assembly, the
safety guardian, the Laya adapter, the fallback controller, logging - is shared
verbatim between simulation and hardware. Swapping the transport is a CLI flag.

> **Scope.** This layer is a *connector*, not a motor driver. It publishes the
> five abstract actions (`MOVE_FORWARD`, `TURN_LEFT_AND_MOVE`,
> `TURN_RIGHT_AND_MOVE`, `REVERSE`, `STOP_AND_REPLAN`) to an existing rover
> controller. It never emits raw actuator values, and Laya never sees or controls
> anything below that interface (spec Section 4, non-goals).

---

## 1. The one interface you must satisfy

Everything is expressed by `src/laya_rover_agent/hardware/base.py`:

```python
class RoverBackend(Protocol):
    name: str
    def connect(self) -> None: ...
    def close(self) -> None: ...
    def read_sensors(self) -> SensorFrame: ...
    def execute(self, action: Action) -> ExecutionReport: ...
    def emergency_stop(self) -> None: ...
```

`SensorFrame` is what the rover tells the agent about *this instant*:
`x, y, heading, battery, obstacles{front, front_left, front_right, left, right},
hazards{front, front_left, front_right}, goal_relative, goal_distance,
sensor_range, timestamp_s`. Distances use the sentinel `sensor_range + 1` for
"nothing detected" (never null), and the frame's units are grid cells.

`ExecutionReport` is what the rover reports back: `moved, stalled, collision,
reached_goal, energy_used, battery, position, heading, latency_ms`.

If your rover already exposes those two shapes over any transport, the mission
loop runs unchanged. Three backends ship with the project:

| Backend | Use | Requirement |
|---|---|---|
| `sim` | virtual test simulator - the same maps, physics and hazards as the experiments | none (always available) |
| `ros2` | rover already driven by a ROS 2 stack | `rclpy` on the rover/companion computer |
| `serial` | microcontroller-class rover | `pyserial` and a documented telemetry line format |

---

## 2. Rehearse with the virtual simulator first (no hardware)

```bash
# one mission on a trap map, full logging, no model server needed
python -m laya_rover_agent.cli hardware \
    --backend sim --preset trap_u --difficulty medium --seed 0 \
    --controller laya_agent --log outputs/hw_sim.jsonl

# the same mission, but the decision layer is your local laya-serve
python -m laya_rover_agent.cli hardware --backend sim --api --max-steps 150
```

Both write `<log>.jsonl` (per timestep) and `<log>.summary.json` (mission
summary). Use `--dry-run` to compute decisions *without* executing them - the
right way to sanity-check a new rover's sensor frames before it moves.

---

## 3. ROS 2 rover

The ROS 2 backend assumes the usual split: sensors publish, a controller
subscribes to a discrete command topic.

```bash
export ROVER_ROS2_NS=/laya_rover
python -m laya_rover_agent.cli hardware --backend ros2 --max-steps 200 \
    --log outputs/hw_ros2.jsonl
```

Defaults (override in `configs/default.yaml` or via `ROS2_*` env vars):

| Topic | Type | Direction |
|---|---|---|
| `/laya_rover/odom` | `nav_msgs/Odometry` | subscribe - `x`, `y`, yaw → heading |
| `/laya_rover/battery_state` | `sensor_msgs/BatteryState` | subscribe - `percentage` |
| `/laya_rover/scan_window` | `std_msgs/Float32MultiArray` or `sensor_msgs/LaserScan` | subscribe - the five window distances |
| `/laya_rover/goal` | `geometry_msgs/PointStamped` | subscribe - goal in the map frame |
| `/laya_rover/cmd_action` | `std_msgs/String` | publish - the chosen action name |

If your rover does not publish a ready-made five-beam window, either

1. publish it from your own node (recommended - it keeps the contract explicit), or
2. let `SensorGridFacade` reconstruct it from a `LaserScan` (the backend already
   accepts a laser scan and reduces it to the five window beams).

Latched/stale-sensor policy: a frame older than `stale_after_s` (default 1.0 s)
aborts the mission with `emergency_stop()` rather than acting on stale data.

---

## 4. Serial rover

The serial backend expects newline-delimited ASCII at 115200 baud (configurable):

```
# telemetry, rover -> agent, ~10 Hz
TEL x=12 y=8 h=N bat=0.94 of=3,6,6,6,2 oh=F,T,F gdx=9 gdy=-14 ts=1.5

# acknowledgement, rover -> agent
ACK MOVE_FORWARD moved=1 stall=0 collision=0 goal=0
```

| Field | Meaning |
|---|---|
| `x`, `y` | grid cell (integers) |
| `h` | heading, one of `N E S W` |
| `bat` | battery fraction 0..1 |
| `of` | obstacle distances for `front, front_left, front_right, left, right`, in that order |
| `oh` | hazard flags for `front, front_left, front_right` (`T`/`F`) |
| `gdx`, `gdy` | goal offset in cells (optional) |
| `ts` | sensor timestamp (optional; used for staleness checks) |

* Missing `of`/`oh` values default to the sentinel `sensor_range + 1` / `false`.
* The agent writes `ACT <ACTION_NAME>` (and `PING` on connect, `ESTOP` on
  abort) and expects `ACK <ACTION_NAME> ...` within `ack_timeout_s`
  (default 2.0 s); a missing acknowledgement counts as a failed manoeuvre, and
  `ESTOP` is issued on abort.
* `parse_telemetry()` and `parse_ack()` are pure functions - unit-test your
  rover's exact byte stream against them before attaching hardware.

```bash
export ROVER_SERIAL_PORT=/dev/ttyUSB0
python -m laya_rover_agent.cli hardware --backend serial --max-steps 200 \
    --log outputs/hw_serial.jsonl
```

`parse_telemetry()` and `parse_ack()` are pure functions, so you can unit-test
your rover's exact byte stream against them without the hardware attached.

---

## 5. Writing your own backend

```python
from laya_rover_agent.hardware.base import ExecutionReport, SensorFrame
from laya_rover_agent.types import Action

class MyRoverBackend:
    name = "my_rover"

    def connect(self) -> None: ...
    def close(self) -> None: ...
    def emergency_stop(self) -> None: ...

    def read_sensors(self) -> SensorFrame:
        # read your sensors, fill the five beams, return a frame
        ...

    def execute(self, action: Action) -> ExecutionReport:
        # translate the abstract action into your rover's command,
        # block until it completes, and report what happened
        ...
```

Then run it through the same loop:

```python
import asyncio
from laya_rover_agent.config import load_config
from laya_rover_agent.hardware.bridge import RoverBridge
from laya_rover_agent.laya.factory import ControllerFactory

config = load_config("configs/default.yaml")
bridge = RoverBridge(config, MyRoverBackend(), ControllerFactory(config)(...))
summary = asyncio.run(bridge.run_mission_async(max_steps=200, dry_run=False))
```

You have written no agent code: perception, guardian, decision routing, fallback
and logging are shared.

---

## 6. Safety properties of the bridge

* **Guardian in the loop, always.** Every action - model, fallback or override -
  passes `SafetyGuardian.approve()` before `execute()`. An action that would enter
  an obstacle, leave the grid or enter a hazard without battery headroom is
  *replaced*, never sent.
* **Fail-safe stop.** Any exception, stale sensor frame, missing acknowledgement
  or wall-clock overrun triggers `emergency_stop()` in a `finally` block; the
  rover is never left with an in-flight command.
* **Battery reserve.** Below `battery_reserve` (default 5%) only a goal-reaching
  move is permitted; otherwise the mission stops cleanly.
* **Dry-run mode.** `--dry-run` exercises the whole stack including the live Laya
  call while the rover stays still.
* **No secrets on the wire to the rover.** Serial/ROS 2 messages contain only
  the action name and telemetry - any configured credential never leaves the agent process.

## 7. Pre-flight checklist

- [ ] `python scripts/smoke_test_laya_api.py --offline --episodes 2` passes.
- [ ] If `--api` is used: `python scripts/smoke_test_laya_api.py --health-only` was green first
      (the model server is up and warm, and the agent knows where it is).
- [ ] `--backend sim --dry-run` produces sane decisions on your own map size.
- [ ] Real sensor frames validate against `SensorFrame` (units: cells, not metres,
      unless you keep your scale in the config).
- [ ] Emergency stop tested manually, wheels off the ground.
- [ ] Supervisor present, speed limits set, kill switch reachable.
- [ ] You are aware this is a prototype: no flight readiness, no formal
      verification, simulation-grade assumptions about terrain and sensing
      (spec Section 13).
