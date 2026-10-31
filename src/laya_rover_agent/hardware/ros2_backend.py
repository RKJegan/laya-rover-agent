"""ROS 2 backend.

Maps the agent's typed actions onto a ROS 2 rover stack and folds topics back
into a :class:`SensorFrame`. ``rclpy`` is imported lazily so that the package
installs and tests cleanly on machines without ROS 2.

Topic contract (all names configurable via constructor)

===========================  ==================  ==================================
Topic                        Direction           Type / semantics
===========================  ==================  ==================================
``<ns>/odom``                subscribe           ``nav_msgs/Odometry`` - pose + heading
``<ns>/battery_state``       subscribe           ``sensor_msgs/BatteryState`` - percentage
``<ns>/range_front``         subscribe           ``sensor_msgs/Range`` (also ..._fl, _fr, _left, _right)
``<ns>/cmd_action``          publish             ``std_msgs/String`` - one of the five action names
``<ns>/cmd_vel``             publish             ``geometry_msgs/Twist`` - optional skid-steer bridge
``<ns>/emergency_stop``      publish             ``std_msgs/Bool``
===========================  ==================  ==================================

The rover firmware/translator node is responsible for turning ``cmd_action`` or
``cmd_vel`` into wheel commands and for reporting completion; the agent never
writes actuator values directly (Non-Goal in the spec).
"""

from __future__ import annotations

import math
import time
from typing import Any

from ..types import Action, Heading
from .base import BackendUnavailableError, ExecutionReport, SensorFrame

RANGE_TOPICS = {
    "front": "range_front",
    "front_left": "range_front_left",
    "front_right": "range_front_right",
    "left": "range_left",
    "right": "range_right",
}

#: Nominal per-action wheel speeds/durations for the optional Twist bridge.
TWIST_COMMANDS: dict[Action, tuple[float, float]] = {
    Action.MOVE_FORWARD: (0.15, 0.0),
    Action.TURN_LEFT_AND_MOVE: (0.10, 0.6),
    Action.TURN_RIGHT_AND_MOVE: (0.10, -0.6),
    Action.REVERSE: (-0.10, 0.0),
    Action.STOP_AND_REPLAN: (0.0, 0.0),
}


class Ros2Backend:
    name = "ros2"

    def __init__(
        self,
        namespace: str = "/laya_rover",
        resolution_m: float = 0.25,
        sensor_range_cells: int = 6,
        action_duration_s: float = 1.2,
        publish_twist: bool = True,
        node_name: str = "laya_rover_agent",
    ) -> None:
        self.namespace = namespace.rstrip("/")
        self.resolution_m = resolution_m
        self.sensor_range_cells = sensor_range_cells
        self.action_duration_s = action_duration_s
        self.publish_twist = publish_twist
        self.node_name = node_name
        self._node: Any = None
        self._latest: dict[str, Any] = {}
        self._publishers: dict[str, Any] = {}

    # -- lifecycle ---------------------------------------------------------- #

    def connect(self) -> None:
        try:
            import rclpy  # noqa: PLC0415
            from geometry_msgs.msg import Twist  # noqa: PLC0415
            from nav_msgs.msg import Odometry  # noqa: PLC0415
            from rclpy.node import Node  # noqa: PLC0415
            from sensor_msgs.msg import BatteryState, Range  # noqa: PLC0415
            from std_msgs.msg import Bool, String  # noqa: PLC0415
        except ImportError as exc:  # pragma: no cover - depends on host
            raise BackendUnavailableError(
                "rclpy / ROS 2 message packages are not available. Source your ROS 2 "
                "installation (e.g. `source /opt/ros/jazzy/setup.bash`) or use --backend sim."
            ) from exc

        if not rclpy.ok():
            rclpy.init()

        node = Node(self.node_name)
        self._node = node
        node.create_subscription(Odometry, f"{self.namespace}/odom", self._on_odom, 10)
        node.create_subscription(BatteryState, f"{self.namespace}/battery_state", self._on_battery, 10)
        for beam, topic in RANGE_TOPICS.items():
            node.create_subscription(Range, f"{self.namespace}/{topic}", self._range_callback(beam), 10)
        self._publishers["action"] = node.create_publisher(String, f"{self.namespace}/cmd_action", 10)
        self._publishers["estop"] = node.create_publisher(Bool, f"{self.namespace}/emergency_stop", 10)
        if self.publish_twist:
            self._publishers["twist"] = node.create_publisher(Twist, f"{self.namespace}/cmd_vel", 10)

    def close(self) -> None:
        if self._node is not None:
            self._node.destroy_node()
            self._node = None

    def spin_once(self, timeout_s: float = 0.1) -> None:
        import rclpy  # noqa: PLC0415

        rclpy.spin_once(self._node, timeout_sec=timeout_s)

    # -- callbacks ---------------------------------------------------------- #

    def _on_odom(self, message: Any) -> None:
        position = message.pose.pose.position
        orientation = message.pose.pose.orientation
        yaw = math.atan2(
            2.0 * (orientation.w * orientation.z + orientation.x * orientation.y),
            1.0 - 2.0 * (orientation.y**2 + orientation.z**2),
        )
        self._latest["x"] = position.x
        self._latest["y"] = position.y
        self._latest["yaw"] = yaw

    def _on_battery(self, message: Any) -> None:
        self._latest["battery"] = float(getattr(message, "percentage", 0.0)) or 0.0

    def _range_callback(self, beam: str):
        def _callback(message: Any) -> None:
            distance_m = float(getattr(message, "range", float("inf")))
            if not math.isfinite(distance_m) or distance_m <= 0.0:
                distance_cells = self.sensor_range_cells + 1
            else:
                distance_cells = max(1, int(round(distance_m / self.resolution_m)))
                distance_cells = min(distance_cells, self.sensor_range_cells + 1)
            self._latest[f"range_{beam}"] = distance_cells

        return _callback

    # -- I/O ---------------------------------------------------------------- #

    def read_sensors(self) -> SensorFrame:
        if self._node is None:
            raise RuntimeError("connect() must be called before read_sensors()")
        # Give the executor a moment to deliver fresh messages.
        deadline = time.monotonic() + self.action_duration_s * 0.5
        while time.monotonic() < deadline and "range_front" not in self._latest:
            self.spin_once(0.05)

        x_m = float(self._latest.get("x", 0.0))
        y_m = float(self._latest.get("y", 0.0))
        yaw = float(self._latest.get("yaw", 0.0))
        cell_x = int(round(x_m / self.resolution_m))
        cell_y = int(round(y_m / self.resolution_m))
        heading = _yaw_to_heading(yaw)

        obstacles = {
            beam: int(self._latest.get(f"range_{beam}", self.sensor_range_cells + 1))
            for beam in RANGE_TOPICS
        }
        hazards = {beam: False for beam in ("front", "front_left", "front_right")}
        # Optional: a terrain classifier node may publish hazard flags on
        # `<ns>/hazard_flags`; absent that, no cell is treated as hazardous.
        hazards.update(self._latest.get("hazards", {}))

        goal_relative = self._latest.get("goal_relative", (0, 0))
        goal_distance = math.hypot(float(goal_relative[0]), float(goal_relative[1]))
        return SensorFrame(
            x=cell_x,
            y=cell_y,
            heading=heading,
            battery=float(self._latest.get("battery", 1.0)),
            obstacles=obstacles,
            hazards=hazards,
            goal_relative=(int(goal_relative[0]), int(goal_relative[1])),
            goal_distance=goal_distance,
            sensor_range=self.sensor_range_cells,
            timestamp_s=time.time(),
            raw=dict(self._latest),
        )

    def execute(self, action: Action) -> ExecutionReport:
        if self._node is None:
            raise RuntimeError("connect() must be called before execute()")
        from std_msgs.msg import String  # noqa: PLC0415

        started = time.perf_counter()
        message = String()
        message.data = action.value
        self._publishers["action"].publish(message)

        if self.publish_twist and "twist" in self._publishers:
            from geometry_msgs.msg import Twist  # noqa: PLC0415

            linear, angular = TWIST_COMMANDS[action]
            twist = Twist()
            twist.linear.x = float(linear)
            twist.angular.z = float(angular)
            self._publishers["twist"].publish(twist)
            # stop after the commanded motion window
            time.sleep(self.action_duration_s)
            self._publishers["twist"].publish(Twist())

        self.spin_once(0.05)
        frame = self.read_sensors()
        return ExecutionReport(
            moved=action is not Action.STOP_AND_REPLAN,
            stalled=False,
            collision=False,
            reached_goal=frame.goal_distance < 0.5,
            battery=frame.battery,
            position=(frame.x, frame.y),
            heading=frame.heading,
            latency_ms=(time.perf_counter() - started) * 1000.0,
            raw={"action": action.value},
        )

    def emergency_stop(self) -> None:
        if self._node is None:
            return
        from std_msgs.msg import Bool  # noqa: PLC0415

        message = Bool()
        message.data = True
        self._publishers["estop"].publish(message)


def _yaw_to_heading(yaw: float) -> Heading:
    """Map a ROS yaw to the pinned N/E/S/W frame.

    Convention (documented, and the single place it is decided): the odometry
    frame is aligned with the grid, ROS ``+x`` -> east and ROS ``+y`` -> south.
    Therefore yaw 0 -> E, 90 -> S, 180 -> W, 270 -> N. If your rover's odom
    frame is rotated relative to the grid, set ``yaw_offset_deg`` on the
    backend instead of editing this function.
    """
    angle = math.degrees(math.atan2(math.sin(yaw), math.cos(yaw)))
    if -45.0 <= angle < 45.0:
        return "E"
    if 45.0 <= angle < 135.0:
        return "S"
    if -135.0 <= angle < -45.0:
        return "N"
    return "W"
