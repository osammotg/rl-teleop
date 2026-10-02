"""Guards that refuse to run dangerous things. Belt and braces, in code.

THE ARM HAS NO BRAKES. It sags whenever nothing commands it — joint 2 was
measured sagging +48.65 deg during a power gap while parked at home. Starting,
stopping or killing a session that contains a ``RobotNode`` is therefore a
physical risk to hardware and to anyone standing near it.

The fault-injection work these guards protect is built entirely on being able to
SIGKILL, SIGSTOP, starve and restart the thing under test at will. That is safe
for a cameras-only config and NOT safe for anything with an arm in it. Prose in
a handoff document cannot enforce that; this can.

Two independent checks, because either one alone has a hole:

  * :func:`config_has_robot_node` reads the YAML. Catches "someone pointed the
    soak runner at the teleop config".
  * :func:`live_session_ports_bound` looks at the ports. Catches "the operator's
    session is already up and this would steal its cameras" — the RealSense pair
    cannot be opened twice, so starting a second owner takes production down.
"""

from __future__ import annotations

import socket
import subprocess
from pathlib import Path

#: The live teleop bus. Soak work uses 5565/5566 and must never bind these.
LIVE_PUB_PORT = 5555
LIVE_SUB_PORT = 5556

#: Ports soak work must never BIND. Binding one collides with the live setup.
RESERVED_PORTS = {
    5555: "live bus (XSUB)",
    5556: "live bus (XPUB)",
    8791: "live_server",
    8792: "control server",
    8793: "episode_server",
    8799: "cockpit",
}

#: Ports whose being bound means A SESSION IS UP AND OWNS THE CAMERAS.
#:
#: Deliberately NOT the same set. 8799 is a bare `python3 -m http.server` serving
#: the cockpit's static files; on this rig it has been up for over a day and it
#: owns no camera, no bus and no arm. Treating it as evidence of a live session
#: made the guard refuse every hardware soak forever — and a guard that always
#: says no is a guard that gets deleted by the next person in a hurry.
SESSION_PORTS = {
    5555: "live bus (XSUB)",
    5556: "live bus (XPUB)",
    8791: "live_server",
    8792: "control server",
}

#: Node types that command actuators. Never in a fault-injection target.
ACTUATING_NODE_TYPES = frozenset({"RobotNode"})


class UnsafeConfig(RuntimeError):
    """Raised when something would put the arm, or the live session, at risk."""


def config_node_types(config_path: str | Path) -> list[str]:
    """Return the ``type`` of every node in a session YAML."""
    import yaml

    path = Path(config_path)
    with path.open(encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    return [str(n.get("type", "")) for n in (cfg.get("nodes") or []) if isinstance(n, dict)]


def config_has_robot_node(config_path: str | Path) -> list[str]:
    """Return the actuating node types present in a config (empty = safe)."""
    return sorted({t for t in config_node_types(config_path) if t in ACTUATING_NODE_TYPES})


def config_touches_hardware(config_path: str | Path) -> list[str]:
    """Return the real devices a config would open (empty = fully synthetic).

    The distinction matters because the two risks are different. Binding a
    reserved port is a collision; opening a camera that a live session already
    owns is theft — the RealSense pair cannot be opened twice, so a second owner
    takes the operator's production session down. A config that opens nothing
    real can be soaked safely no matter what else is running.
    """
    import yaml

    path = Path(config_path)
    with path.open(encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    devices: list[str] = []
    for node in cfg.get("nodes") or []:
        if not isinstance(node, dict):
            continue
        for key in ("device_path", "device_id"):
            value = node.get(key)
            if value:
                devices.append(f"{node.get('name', '?')}:{value}")
    return devices


def port_is_bound(port: int, host: str = "127.0.0.1") -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.3)
        return s.connect_ex((host, port)) == 0


def live_session_ports_bound() -> list[str]:
    """Names of SESSION ports that already have a listener (see SESSION_PORTS)."""
    return [f"{p} ({what})" for p, what in sorted(SESSION_PORTS.items()) if port_is_bound(p)]


def running_sessions() -> list[str]:
    """``pgrep -fa rr-session`` output lines, best effort.

    Informational only — never a reason to touch anything. If a session is
    running, the correct response is to leave it completely alone.
    """
    try:
        out = subprocess.run(
            ["pgrep", "-fa", "rr-session"],
            check=False, capture_output=True, text=True, timeout=5,
        ).stdout
    except Exception:
        return []
    lines = []
    for line in out.splitlines():
        # Skip our own command line and any grep/pgrep that merely mentions it.
        if "pgrep" in line or "safety_guard" in line:
            continue
        lines.append(line.strip())
    return lines


def assert_safe_to_soak(
    config_path: str | Path,
    pub_port: int,
    sub_port: int,
    allow_live_ports: bool = False,
) -> None:
    """Raise :class:`UnsafeConfig` unless this is safe to fault-inject.

    Called before anything is started. Every failure mode is a hard stop with a
    message that says what to do instead — a guard that fails vaguely gets
    disabled by the next person in a hurry.
    """
    problems: list[str] = []

    actuating = config_has_robot_node(config_path)
    if actuating:
        problems.append(
            f"{Path(config_path).name} contains {', '.join(actuating)}. THE ARM HAS NO "
            f"BRAKES: it sags whenever nothing commands it, and this runner exists to "
            f"kill, stop and starve the process it starts. Point it at "
            f"configs/yam/cameras_only_soak.yaml instead."
        )

    if pub_port in RESERVED_PORTS or sub_port in RESERVED_PORTS:
        problems.append(
            f"refusing to bind reserved ports (pub={pub_port}, sub={sub_port}). "
            f"Soak work uses 5565/5566; {LIVE_PUB_PORT}/{LIVE_SUB_PORT} is the live bus."
        )

    # The live-port check only guards against STEALING HARDWARE. A config that
    # opens no real device cannot steal anything, so it stays runnable while the
    # operator's session is up — which is the whole reason
    # cameras_only_soak_fake.yaml exists.
    hardware = config_touches_hardware(config_path)
    if hardware and not allow_live_ports:
        bound = live_session_ports_bound()
        if bound:
            problems.append(
                "something is already listening on: " + ", ".join(bound) + ". A live "
                "session is probably up, and this config opens real devices ("
                + ", ".join(hardware) + "). The RealSense cameras cannot be opened "
                "twice, so starting this would steal the operator's cameras — or fail "
                "trying. Use the hardware-free target instead:\n"
                "        tools/camera_soak.py --config configs/yam/cameras_only_soak_fake.yaml\n"
                "    or run the hermetic tier: pytest tests/sensors/cameras/"
            )
            sessions = running_sessions()
            if sessions:
                problems.append("running sessions (LEAVE THESE ALONE): " + "; ".join(sessions))

    if problems:
        raise UnsafeConfig(
            "refusing to start the camera soak:\n  - " + "\n  - ".join(problems)
        )
