"""Rover connector layer: one interface, three transports (sim / ROS 2 / serial)."""

from .base import (
    BackendUnavailableError,
    ExecutionReport,
    RoverBackend,
    SensorFrame,
    frame_to_state,
)
from .bridge import MissionSummary, RoverBridge
from .sensor_grid import SensorGridFacade
from .sim_backend import SimBackend

__all__ = [
    "BackendUnavailableError",
    "ExecutionReport",
    "MissionSummary",
    "RoverBackend",
    "RoverBridge",
    "SensorFrame",
    "SensorGridFacade",
    "SimBackend",
    "frame_to_state",
]


def make_backend(name: str, config, **kwargs):
    """Backend factory used by the CLI (``--backend sim|ros2|serial``).

    Imports are deferred so that a machine without ROS 2 or pyserial can still
    import this package and only fails if it actually selects that transport.
    """
    if name == "sim":
        return SimBackend(config, **kwargs)
    if name == "ros2":
        from .ros2_backend import Ros2Backend

        return Ros2Backend(**kwargs)
    if name == "serial":
        from .serial_backend import SerialBackend

        return SerialBackend(**kwargs)
    raise ValueError(f"unknown backend {name!r} (expected sim | ros2 | serial)")
