"""Read recorded joint streams back from MCAP.

recording.py writes follower/leader joint state as protobuf ``RobotState`` on
``/{node}-robot-state`` (position/velocity/torque) when xdof_sdk is available,
and JSON on ``/{node}/{topic}`` otherwise. This reader tries proto first, then
falls back to JSON, and returns the position timeline (which includes the
gripper joint at index 6).

Timestamps are the message ``log_time`` in nanoseconds — the hardware-clock
epoch seconds the recorder stamped (see recording.py), the same clock the
camera timestamp sidecars use.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def _robot_state_cls():
    try:
        from xdof_sdk.proto.robot_state_pb2 import RobotState
        return RobotState
    except Exception:
        return None


def read_positions(path: str | Path, node_name: str) -> tuple[np.ndarray, np.ndarray]:
    """Return (times_s, positions) where positions is (N, DoF).

    DoF is whatever the node published — for a YAM arm that is 7 (6 arm joints +
    gripper at index 6).
    """
    from mcap.reader import make_reader

    RobotState = _robot_state_cls()
    proto_topic = f"/{node_name}-robot-state"
    json_topics = {f"/{node_name}/joint_state", f"/{node_name}/joint_pos"}

    times: list[float] = []
    positions: list[list[float]] = []
    with open(path, "rb") as f:
        for _schema, channel, msg in make_reader(f).iter_messages():
            pos = None
            if RobotState is not None and channel.topic == proto_topic:
                rs = RobotState.FromString(msg.data)
                pos = list(rs.position)
            elif channel.topic in json_topics:
                data = json.loads(msg.data)
                pos = data.get("position") or data.get("joint_pos")
                # The follower publishes the 6 arm joints in joint_pos and the gripper
                # as a SEPARATE gripper_pos field — stitch it on as the 7th DoF (index 6)
                # so the labeler's gripper index is valid (matches live.py bus_feed).
                if pos is not None and len(pos) == 6:
                    grip = data.get("gripper_pos")
                    if grip is not None:
                        g = grip[0] if isinstance(grip, (list, tuple)) else grip
                        pos = list(pos) + [float(g)]
            if pos is None:
                continue
            times.append(msg.log_time / 1e9)
            positions.append([float(x) for x in pos])

    if not times:
        return np.zeros(0), np.zeros((0, 0))
    order = np.argsort(times)
    return np.asarray(times)[order], np.asarray(positions)[order]


def read_jsonl(path: str | Path) -> list[dict]:
    """Read a cockpit_events.jsonl (one JSON object per line)."""
    p = Path(path)
    if not p.exists():
        return []
    out = []
    for line in p.read_text().splitlines():
        line = line.strip()
        if line:
            out.append(json.loads(line))
    return out
