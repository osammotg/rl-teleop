"""Session — top-level orchestrator.

The Session's monitor thread subscribes to every topic on the bus and measures
live publish-Hz per node.  All recording is delegated to the nodes themselves
via start_recording(save_dir) / stop_recording().

  - Nodes own their writers — no MCAP/video writing in the monitor thread.
  - Session.start_episode() creates the episode directory and calls
    host.start_recording(save_dir) for all hosts.
  - Session.end_episode() calls host.stop_recording() for all hosts.
"""

from __future__ import annotations

import datetime
import json
import logging
import signal
import subprocess
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

import zmq

from robots_realtime.runtime.node import Node, ProcessHost
from robots_realtime.runtime.transport.message_bus import (
    DEFAULT_PUB_PORT,
    DEFAULT_SUB_PORT,
    MessageBus,
)
from robots_realtime.runtime.transport.serialization import unpack


_HZ_WINDOW = 30


def _port_is_bound(port: int, host: str = "127.0.0.1") -> bool:
    """True if something is already listening on `port`.

    Used only to make --attach-bus fail loudly. ZMQ's connect() succeeds
    against a port with no listener and simply queues forever, so without this
    check a typo'd port produces a session where every node looks healthy and
    no message is ever delivered.
    """
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.5)
        return s.connect_ex((host, port)) == 0


logger = logging.getLogger(__name__)


def _run_git(args: list[str], cwd: Path, timeout: float = 2.0) -> str | None:
    try:
        proc = subprocess.run(
            ["git", *args],
            cwd=str(cwd),
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=timeout,
        )
    except Exception:
        return None
    if proc.returncode != 0:
        return None
    return proc.stdout.strip()


def _git_dirty(cwd: Path) -> bool | None:
    status = _run_git(["status", "--porcelain"], cwd)
    if status is None:
        return None
    return bool(status)


def _git_metadata() -> dict:
    repo_root_raw = _run_git(["rev-parse", "--show-toplevel"], Path.cwd())
    if not repo_root_raw:
        return {"available": False}

    repo_root = Path(repo_root_raw)
    meta: dict = {
        "available": True,
        "root": str(repo_root),
        "commit": _run_git(["rev-parse", "HEAD"], repo_root),
        "branch": _run_git(["branch", "--show-current"], repo_root),
        "dirty": _git_dirty(repo_root),
        "submodules": [],
    }

    submodule_lines = _run_git(["submodule", "status", "--recursive"], repo_root)
    if not submodule_lines:
        return meta

    for line in submodule_lines.splitlines():
        parts = line.strip().split()
        if len(parts) < 2:
            continue
        state = line[0] if line else " "
        commit = parts[0].lstrip("+-U")
        path = parts[1]
        desc = parts[2].strip("()") if len(parts) >= 3 else ""
        sub_path = repo_root / path
        meta["submodules"].append(
            {
                "path": path,
                "commit": commit,
                "descriptor": desc,
                "state": state,
                "dirty": _git_dirty(sub_path) if sub_path.exists() else None,
            }
        )
    return meta


def _node_descriptor(node) -> dict:
    d: dict = {
        "name": node.name,
        "published_topics": list(getattr(node, "published_topics", [])),
        "subscribed_topics": list(getattr(node, "subscribed_topics", [])),
    }
    # Include sim node config so replay tools can auto-detect scene/task.
    sim_cfg: dict = {}
    if getattr(node, "_scene", None) is not None:
        sim_cfg["scene"] = node._scene
    if getattr(node, "_task", None) is not None:
        sim_cfg["task"] = node._task
    if sim_cfg:
        d["sim_config"] = sim_cfg
    return d


#: A node that has sent nothing for this long is reported at 0 Hz rather than at
#: its last healthy rate. Two seconds is above any legitimate gap — the slowest
#: camera on this rig runs at 15 Hz and every node emits _step_hz at 1 Hz — and
#: it keeps detection inside the 2 s budget the acceptance bar asks for.
_STALE_AFTER_S = 2.0


@dataclass
class NodeStatus:
    name: str
    alive: bool = True
    pub_hz: float = 0.0
    step_hz: float = 0.0
    exitcode: int | None = None
    #: Latest ``<node>/health`` payload for camera nodes, or None for nodes that
    #: do not publish one.
    health: dict | None = None
    _timestamps: dict[str, deque] = field(default_factory=dict, repr=False)
    _last_msg_t: float | None = field(default=None, repr=False)
    _last_step_t: float | None = field(default=None, repr=False)
    _last_health_t: float | None = field(default=None, repr=False)

    @property
    def hz(self) -> float:
        """Backwards-compatible alias for pub_hz."""
        return self.pub_hz

    def record_message(self, topic_suffix: str) -> None:
        now = time.perf_counter()
        self._last_msg_t = now
        buf = self._timestamps.setdefault(topic_suffix, deque(maxlen=_HZ_WINDOW))
        buf.append(now)
        best = max(self._timestamps.values(), key=len)
        if len(best) >= 2:
            span = best[-1] - best[0]
            self.pub_hz = (len(best) - 1) / span if span > 0 else 0.0

    def record_step_hz(self, step_hz: float) -> None:
        self._last_step_t = time.perf_counter()
        self.step_hz = step_hz

    def decay(self, now: float | None = None, stale_after: float = _STALE_AFTER_S) -> None:
        """Drop the rates to zero when nothing has arrived recently.

        WHY THIS EXISTS. ``record_message()`` only ever ran on receipt, so with
        no messages there was no update and the last healthy value stood
        forever. A camera that stopped publishing entirely kept reporting the
        rate it had when it was working — the real incident shows
        ``camera_right ● live 29.5 Hz`` for a node that had put nothing on the
        bus for minutes. The number was not wrong-ish, it was a fossil. Rates
        that cannot fall are not measurements.
        """
        now = time.perf_counter() if now is None else now
        if self.pub_hz and (self._last_msg_t is None or now - self._last_msg_t > stale_after):
            self.pub_hz = 0.0
        if self.step_hz and (self._last_step_t is None or now - self._last_step_t > stale_after):
            self.step_hz = 0.0

    def record_health(self, health: dict) -> None:
        self._last_health_t = time.perf_counter()
        self.health = health

    @property
    def health_age_s(self) -> float | None:
        """Seconds since the last health message, or None if there never was one."""
        if self._last_health_t is None:
            return None
        return time.perf_counter() - self._last_health_t

    @property
    def health_is_stale(self) -> bool:
        """True when this node's health record has itself stopped updating.

        FOUND BY RED, and it is the same bug one level up. SIGSTOP a camera node
        and it publishes NOTHING — including no health. The last health message
        on the bus still said ``ok``, so anything reading the health topic saw a
        healthy camera that had been frozen for as long as you cared to wait. A
        health record that cannot go stale is exactly the fossil that
        ``pub_hz`` used to be; the cure for the disease had the disease.
        """
        age = self.health_age_s
        return age is not None and age > _STALE_AFTER_S

    @property
    def camera_state(self) -> str | None:
        """``ok``/``degraded``/``reopening``/``failed`` for camera nodes.

        Reports ``stale`` when the record itself has stopped arriving — that is
        a different fact from any state the camera last claimed, and collapsing
        the two is what made SIGSTOP invisible.
        """
        if not self.health:
            return None
        if self.health_is_stale:
            return "stale"
        return str(self.health.get("state") or "") or None

    @property
    def is_healthy(self) -> bool:
        """Everything the session knows, combined into one honest verdict."""
        if not self.alive:
            return False
        if self.health is not None:
            if self.health_is_stale:
                return False
            if not self.health.get("healthy", False):
                return False
        return True


class Session:
    """Orchestrates a graph of Nodes.

    Each node runs in its own subprocess via ProcessHost.  Recording is
    delegated to the nodes via start_recording() / stop_recording().

    Args:
        nodes:                List of Node instances to run.
        save_root:            Root directory for episode recordings.
        record_node_names:    Subset of node names to record; defaults to all.
        record_topic:         Full bus topic carrying the boolean record signal
                              (e.g. "gello_left/record").
        auto_record_duration: If set, automatically start recording on
                              session start and stop after this many seconds.
        episode_timeout:      If set, automatically stop recording and pause
                              after this many seconds from episode start.
        instruction:          Optional text instruction for the recorded episode.
        instruction_mappings: Mapping from number keys to episode instructions.
        pub_port:             MessageBus XSUB port.
        sub_port:             MessageBus XPUB port.
    """

    def __init__(
        self,
        nodes: list,
        save_root: str | Path = "recordings",
        record_node_names: list[str] | None = None,
        record_topic: str | None = None,
        auto_record_duration: float | None = None,
        start_paused: bool = False,
        record_on_unpause: bool = False,
        episode_timeout: float | None = None,
        instruction: str | None = None,
        instruction_mappings: dict[str, str] | None = None,
        pub_port: int = DEFAULT_PUB_PORT,
        sub_port: int = DEFAULT_SUB_PORT,
        require_healthy_cameras: bool = False,
    ) -> None:
        self._pub_port = pub_port
        self._sub_port = sub_port
        self._save_root = Path(save_root)
        self._record_topic = record_topic
        self._auto_record_duration = auto_record_duration
        # start_paused: begin with RobotNode commands gated so the arms don't
        # start tracking the policy (or any cmd_topic producer) until the
        # operator explicitly hits space. Recommended for policy configs where
        # the arm could snap to an unexpected pose on startup.
        # record_on_unpause: when the operator unpauses, automatically start an
        # episode if one isn't already running. Useful for policy eval where
        # you want every rollout captured from the instant the policy takes over.
        self._start_paused = bool(start_paused)
        self._record_on_unpause = bool(record_on_unpause)
        self._episode_timeout = episode_timeout
        self._instruction = instruction or ""
        self._instruction_mappings = self._normalize_instruction_mappings(
            instruction_mappings
        )
        self._episode_timeout_timer: threading.Timer | None = None
        self._is_paused: bool = False
        self._session_start_time = time.time()

        # attach_bus: do not spawn a broker, assume one is already running on
        # these ports (see rr_bus_cli). Nodes connect() either way, so this
        # changes nothing about how they talk -- only who owns the ports.
        # Set via attach_to_existing_bus() so it can also be flipped on a
        # session built by a make_session() module.
        self._attach_bus: bool = False

        self._hosts: list[ProcessHost] = []
        all_node_names: list[str] = []
        self._node_descriptors: list[dict] = []

        for item in nodes:
            if isinstance(item, ProcessHost):
                # Accept pre-built hosts (advanced usage)
                self._hosts.append(item)
                all_node_names.extend(item.node_names)
            elif isinstance(item, Node):
                self._hosts.append(ProcessHost(item))
                all_node_names.extend([item.name])
                self._node_descriptors.append(_node_descriptor(item))
            else:
                raise TypeError(f"Unexpected item in nodes list: {type(item)}")

        self._record_node_names: list[str] = record_node_names or all_node_names

        self._bus = MessageBus(pub_port=pub_port, sub_port=sub_port)

        self._status: dict[str, NodeStatus] = {
            name: NodeStatus(name=name) for name in all_node_names
        }

        self._stop_event = threading.Event()
        self._prev_record_signal = False
        self._monitor_thread: threading.Thread | None = None
        self._log_dir: Path | None = None

        # Recording state
        self._is_recording: bool = False
        self._episode_dir: Path | None = None
        self._episode_start_time: float | None = None
        self._recording_lock = threading.Lock()

        # Supervision state (see _supervise_nodes)
        self._last_supervise_t: float = 0.0
        self._last_bus_msg_t: float | None = None
        self._bus_down: bool = False
        self._failed_recording_hosts: list[str] = []
        # Nodes seen unhealthy at any point during the current episode. Kept for
        # the whole episode, not sampled at the end: a camera that dies in the
        # middle and recovers before the operator stops recording would otherwise
        # leave a clean-looking episode with a hole in it.
        self._episode_unhealthy: dict[str, dict] = {}
        self._require_healthy_cameras = bool(require_healthy_cameras)

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def configure_bus_ports(
        self,
        pub_port: int | None = None,
        sub_port: int | None = None,
    ) -> None:
        """Set broker and node ZMQ ports before the session is started."""
        if self._hosts:
            for host in self._hosts:
                if getattr(host, "_proc", None) is not None:
                    raise RuntimeError("Cannot change MessageBus ports after nodes start")

        if pub_port is not None:
            self._pub_port = int(pub_port)
        if sub_port is not None:
            self._sub_port = int(sub_port)

        self._bus = MessageBus(pub_port=self._pub_port, sub_port=self._sub_port)
        for host in self._hosts:
            node = getattr(host, "_node", None)
            if node is not None:
                node._pub_port = self._pub_port
                node._sub_port = self._sub_port

    def attach_to_existing_bus(self, attach: bool = True) -> None:
        """Use a broker someone else already runs instead of spawning one."""
        for host in self._hosts:
            if getattr(host, "_proc", None) is not None:
                raise RuntimeError("Cannot change bus ownership after nodes start")
        self._attach_bus = bool(attach)

    def start(self) -> None:
        import tempfile
        self._log_dir = Path(tempfile.mkdtemp(prefix="rr_logs_"))

        if self._attach_bus:
            # Fail here rather than letting every node connect() to a dead port
            # and sit silent forever -- a connect to nothing does not raise.
            if not _port_is_bound(self._sub_port):
                raise RuntimeError(
                    f"--attach-bus: nothing is listening on port {self._sub_port}.\n"
                    "  Start the broker first:  rr-bus\n"
                    "  Or drop --attach-bus to let this session own the bus."
                )
        else:
            self._bus.start()
        time.sleep(0.1)

        for host in self._hosts:
            log_path = self._log_dir / f"{host.node_name}.log"
            host.start(log_path=log_path)

        # If configured to start paused, broadcast PAUSE to every subprocess
        # BEFORE calling send_start(). Each ProcessHost's control socket is
        # already bound once host.start() returns, so the PAUSE arrives before
        # the node's step loop begins — RobotNode.step()'s first tick will see
        # self._paused=True and skip command_joint_pos.
        if self._start_paused:
            self._is_paused = True
            for host in self._hosts:
                try:
                    host.pause()
                except Exception:
                    pass

        for host in self._hosts:
            host.send_start()

        self._setup_signal_handlers()
        self._monitor_thread = threading.Thread(
            target=self._monitor_loop, daemon=True, name="SessionMonitor"
        )
        self._monitor_thread.start()

        if self._auto_record_duration is not None:
            t = threading.Thread(
                target=self._auto_record_timer,
                args=(self._auto_record_duration,),
                daemon=True,
            )
            t.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._is_recording:
            self.end_episode(save=True)
        threads = [
            threading.Thread(target=host.stop, daemon=True)
            for host in self._hosts
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=8.0)
        # Never stop a bus this session did not start: other sessions and dev
        # tools are still on it.
        if not self._attach_bus:
            self._bus.stop()

    def wait(self) -> None:
        try:
            self._stop_event.wait()
        except KeyboardInterrupt:
            pass
        finally:
            self.stop()

    # ── Recording controls ────────────────────────────────────────────────────

    @property
    def is_recording(self) -> bool:
        return self._is_recording

    def flag_episode(self, tag: str) -> bool:
        """Attach an operator quality flag (e.g. 're_grasp', 'bad', 'slow') to the
        CURRENTLY-recording episode. Written to operator_flags.json in the episode
        dir, so it survives save and is dropped with the dir on discard. No-op when
        not recording (returns False)."""
        with self._recording_lock:
            d = self._episode_dir
        if d is None:
            return False
        p = Path(d) / "operator_flags.json"
        try:
            data = json.loads(p.read_text()) if p.exists() else {"flags": []}
        except Exception:
            data = {"flags": []}
        data.setdefault("flags", []).append({"tag": tag, "t": time.time()})
        try:
            p.write_text(json.dumps(data, indent=2) + "\n")
        except Exception:
            return False
        return True

    @property
    def episode_start_time(self) -> float | None:
        return self._episode_start_time

    @property
    def save_root(self) -> Path:
        return self._save_root

    @property
    def instruction(self) -> str:
        return self._instruction

    @instruction.setter
    def instruction(self, value: str | None) -> None:
        self._instruction = value or ""

    @property
    def instruction_mappings(self) -> dict[str, str]:
        return dict(self._instruction_mappings)

    @staticmethod
    def _normalize_instruction_mappings(
        instruction_mappings: dict[str, str] | None,
    ) -> dict[str, str]:
        if not instruction_mappings:
            return {}

        normalized: dict[str, str] = {}
        for raw_key, raw_instruction in instruction_mappings.items():
            key = str(raw_key)
            if len(key) != 1 or key not in "0123456789":
                raise ValueError(
                    f"Instruction mapping key must be a number key 0-9, got {raw_key!r}"
                )
            instruction = str(raw_instruction or "").strip()
            if not instruction:
                raise ValueError(f"Instruction for key {key!r} must not be empty")
            normalized[key] = instruction
        return normalized

    # ── health-gated recording ────────────────────────────────────────────────
    #
    # THE DECISION, AND WHY.
    #
    # The handoff asks for one of two behaviours when a camera is unhealthy at
    # record time: refuse the episode, or record it with a loud machine-readable
    # degraded marker. The default here is MARK, NOT REFUSE, and
    # `require_healthy_cameras: true` opts into refusing.
    #
    # Reasons, in order of weight:
    #
    #  1. The operator is physically mid-take on a brakeless arm. A refusal is
    #     silent from where they are standing — they press record, the arm is
    #     already moving, and the take is simply gone. Losing a real demo to
    #     protect against a possibly-degraded one is the wrong trade when the
    #     degradation is recorded and greppable.
    #  2. A degraded episode is USEFUL. Three of the four cameras are usually
    #     fine; joint trajectories are unaffected. What was never acceptable was
    #     not knowing — three right-arm episodes were recorded with no wrist
    #     video and nothing anywhere said so.
    #  3. The marker is machine-readable, so `export_lerobot.py` and any future
    #     dataset filter can drop degraded episodes deterministically, which a
    #     human eyeballing a directory listing cannot.
    #
    # What is NOT acceptable, and is now impossible: a quietly incomplete
    # episode. Either every camera was healthy for the whole episode, or
    # session_meta.json says `"degraded": true` and names the cameras.

    def _unhealthy_now(self) -> dict[str, dict]:
        """Nodes currently dead or reporting an unhealthy camera."""
        out: dict[str, dict] = {}
        for st in self._status.values():
            if st.is_healthy:
                continue
            out[st.name] = {
                "alive": st.alive,
                "exitcode": st.exitcode,
                "state": st.camera_state,
                "reason": (st.health or {}).get("reason"),
                "detail": (st.health or {}).get("detail"),
                "t": time.time(),
            }
        return out

    def start_episode(self) -> None:
        unhealthy = self._unhealthy_now()
        if unhealthy and self._require_healthy_cameras:
            logger.error(
                "REFUSING TO START AN EPISODE: %s unhealthy (require_healthy_cameras=true). "
                "Fix the camera or set require_healthy_cameras=false to record a marked, "
                "degraded episode instead.",
                ", ".join(sorted(unhealthy)),
            )
            raise RuntimeError(
                "start_episode refused: unhealthy nodes "
                + ", ".join(f"{k}({v.get('state') or 'dead'})" for k, v in sorted(unhealthy.items()))
            )

        with self._recording_lock:
            if self._is_recording:
                return
            self._episode_unhealthy = dict(unhealthy)
            save_dir = self._make_episode_dir()
            self._episode_dir = Path(save_dir)
            self._episode_start_time = time.time()
            self._is_recording = True

        # Delegate recording to all hosts.
        #
        # A host that fails here records NOTHING for the whole episode, and until
        # 2026-08-10 this was `except Exception: pass` — so the take completed,
        # the TUI stayed green, and the missing camera was only discoverable by
        # listing the episode directory afterwards. Three right-arm episodes were
        # recorded that way with no wrist video and no error anywhere.
        #
        # Still non-fatal: losing one camera is better than aborting a take the
        # operator is physically in the middle of. But it is now LOUD, and the
        # names are collected so the caller can surface them.
        self._failed_recording_hosts = []
        for host in self._hosts:
            # host.node_name, not host._node.name: ProcessHost exposes the name
            # publicly, and Session explicitly accepts pre-built hosts that have
            # no `_node` at all. Reaching for the private attribute made the
            # loudest error in the whole recording path report '<unknown>' —
            # an alert that cannot name the camera it is about is barely an alert.
            name = getattr(host, "node_name", None) or getattr(
                getattr(host, "_node", None), "name", "<unknown>"
            )
            try:
                host.start_recording(save_dir)
            except Exception as exc:
                self._failed_recording_hosts.append(name)
                logger.error(
                    "start_recording FAILED for node %r — this node will record "
                    "nothing for episode %s: %s: %s",
                    name, Path(save_dir).name, type(exc).__name__, exc,
                )
        if self._failed_recording_hosts:
            logger.error(
                "episode %s is recording WITHOUT: %s",
                Path(save_dir).name, ", ".join(self._failed_recording_hosts),
            )
            for name in self._failed_recording_hosts:
                self._episode_unhealthy.setdefault(
                    name, {"state": "writer_open_failed", "reason": "start_recording raised",
                           "alive": True, "t": time.time()},
                )

        if self._episode_unhealthy:
            logger.error(
                "EPISODE %s IS DEGRADED: %s. session_meta.json records this; do not "
                "treat this episode as a clean take.",
                Path(save_dir).name, ", ".join(sorted(self._episode_unhealthy)),
            )
        self._write_degraded_marker(Path(save_dir))

        if self._episode_timeout is not None:
            self._episode_timeout_timer = threading.Timer(
                self._episode_timeout, self._on_episode_timeout
            )
            self._episode_timeout_timer.daemon = True
            self._episode_timeout_timer.start()

    def _on_episode_timeout(self) -> None:
        self.end_episode(save=True)
        self.pause()

    def end_episode(
        self,
        save: bool = True,
        instruction: str | None = None,
    ) -> Path | None:
        if self._episode_timeout_timer is not None:
            self._episode_timeout_timer.cancel()
            self._episode_timeout_timer = None

        with self._recording_lock:
            if not self._is_recording:
                return None
            self._is_recording = False
            episode_dir = self._episode_dir
            self._episode_dir = None
            self._episode_start_time = None

        # Delegate stop_recording to all hosts in parallel. Camera MP4 writers
        # may need to drain encoder queues on close; doing this serially makes
        # episode save latency add up across cameras.
        threads = [
            threading.Thread(target=self._safe_stop_recording, args=(host,), daemon=True)
            for host in self._hosts
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=12.0)

        if not save and episode_dir is not None:
            import shutil
            shutil.rmtree(episode_dir, ignore_errors=True)
            self._episode_unhealthy = {}
            return None

        if episode_dir is not None:
            # Re-stamp: cameras that went bad during the take are folded in by
            # _supervise_nodes, so the final verdict covers the whole episode.
            for name, info in self._unhealthy_now().items():
                self._episode_unhealthy.setdefault(name, info)
            self._write_degraded_marker(episode_dir)
            if self._episode_unhealthy:
                logger.error(
                    "episode %s saved DEGRADED — unhealthy at some point: %s",
                    episode_dir.name, ", ".join(sorted(self._episode_unhealthy)),
                )
        self._episode_unhealthy = {}

        if instruction is not None:
            self._instruction = instruction
            if episode_dir is not None:
                self._write_session_meta(episode_dir, instruction=instruction)

        return episode_dir

    @staticmethod
    def _safe_stop_recording(host) -> None:
        try:
            host.stop_recording()
        except Exception:
            pass

    def toggle_recording(self) -> None:
        if self._is_recording:
            self.end_episode(save=True)
        else:
            self.start_episode()

    # ------------------------------------------------------------------
    # Pause / resume — gates RobotNode command output; other nodes keep
    # running (cameras, agents) so the TUI, viser, and inference stay live.
    # ------------------------------------------------------------------

    @property
    def is_paused(self) -> bool:
        return getattr(self, "_is_paused", False)

    def pause(self) -> None:
        if getattr(self, "_is_paused", False):
            return
        self._is_paused = True
        for host in self._hosts:
            try:
                host.pause()
            except Exception:
                pass

    def resume(self) -> None:
        if not getattr(self, "_is_paused", False):
            return
        self._is_paused = False
        for host in self._hosts:
            try:
                host.resume()
            except Exception:
                pass
        # Optional: prime recording when the operator unpauses. Lets a policy
        # eval config capture every rollout from the instant of handoff.
        if self._record_on_unpause and not self._is_recording:
            try:
                self.start_episode()
            except Exception:
                pass

    def toggle_pause(self) -> None:
        if self.is_paused:
            self.resume()
        else:
            self.pause()

    @property
    def log_dir(self) -> Path | None:
        return self._log_dir

    @property
    def web_endpoints(self) -> list[str]:
        urls = []
        for host in self._hosts:
            urls.extend(host._node.web_endpoints)
        return urls

    # ── Status ────────────────────────────────────────────────────────────────

    def node_statuses(self) -> list[NodeStatus]:
        return list(self._status.values())

    # ── Internal ──────────────────────────────────────────────────────────────

    def _make_episode_dir(self) -> str:
        """Create the episode directory and write session_meta.json."""
        now = datetime.datetime.now()
        uid = uuid.uuid4().hex[:8]
        path = (
            self._save_root
            / now.strftime("%Y%m%d")
            / f"episode_{now.strftime('%H%M%S')}_{uid}"
        )
        path.mkdir(parents=True, exist_ok=True)
        self._write_session_meta(path, instruction=self._instruction)

        return str(path)

    def _write_session_meta(self, path: Path, instruction: str) -> None:
        meta_path = path / "session_meta.json"
        meta = {}
        if meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text())
            except Exception:
                meta = {}
        if not meta:
            meta = {
                "session_start_time": self._session_start_time,
                "episode_start_time": time.time(),
                "episode_dir": str(path),
                "nodes": self._node_descriptors,
                "record_topic": self._record_topic,
                "save_root": str(self._save_root),
                "git": _git_metadata(),
            }
        meta["instruction"] = instruction
        meta["instruction_mappings"] = self._instruction_mappings
        try:
            meta_path.write_text(json.dumps(meta, indent=2, default=str))
        except Exception:
            pass

    def _write_degraded_marker(self, path: Path) -> None:
        """Stamp the episode's session_meta.json with its health verdict.

        Written at start AND at end, so an episode whose recording process is
        SIGKILLed still carries the verdict it had when it began. The keys are
        deliberately flat and boring — ``degraded`` is the one an exporter or a
        shell one-liner will test:

            jq -r 'select(.degraded) | .episode_dir' */*/session_meta.json

        Absence of the key means an episode recorded before this change existed,
        which is NOT the same as a clean episode; that is why ``degraded`` is
        always written, including as ``false``.
        """
        meta_path = path / "session_meta.json"
        try:
            meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
        except Exception:
            meta = {}
        meta["degraded"] = bool(self._episode_unhealthy)
        meta["degraded_nodes"] = sorted(self._episode_unhealthy)
        meta["degraded_detail"] = self._episode_unhealthy
        meta["health_snapshot"] = self.health_snapshot()
        try:
            meta_path.write_text(json.dumps(meta, indent=2, default=str))
        except Exception as exc:
            logger.error("could not write the degraded marker into %s: %s", meta_path, exc)

    def _monitor_loop(self) -> None:
        """Subscribe to all bus topics; measure Hz per node.

        Also watches record_topic for start/stop signals.
        """
        ctx = zmq.Context.instance()
        sock = ctx.socket(zmq.SUB)
        sock.connect(f"tcp://127.0.0.1:{self._sub_port}")
        sock.setsockopt(zmq.SUBSCRIBE, b"")

        node_names = set(self._status)

        while not self._stop_event.is_set():
            while sock.poll(0):
                try:
                    parts = sock.recv_multipart(zmq.NOBLOCK)
                except zmq.Again:
                    break
                if len(parts) < 2:
                    continue

                topic_b, payload_b = parts[0], parts[1]
                topic = topic_b.decode()
                parts = topic.split("/", 1)
                if not parts:
                    continue
                node_name = parts[0]
                topic_suffix = parts[1] if len(parts) > 1 else ""

                if node_name not in node_names:
                    continue

                self._last_bus_msg_t = time.perf_counter()

                # Internal step-rate report from the node process
                if topic_suffix == "_step_hz":
                    try:
                        envelope = unpack(payload_b)
                        self._status[node_name].record_step_hz(
                            float(envelope.get("data", {}).get("step_hz", 0.0))
                        )
                    except Exception:
                        pass
                    continue

                # Camera health. Kept OUT of the pub_hz measurement on purpose:
                # health is published even by a camera that is delivering no
                # frames at all, so counting it as publish traffic would let a
                # dead camera hold a non-zero rate — reintroducing the exact lie
                # this whole change removes.
                if topic_suffix == "health":
                    try:
                        envelope = unpack(payload_b)
                        self._status[node_name].record_health(dict(envelope.get("data", {})))
                    except Exception:
                        pass
                    continue

                # Measure publish Hz
                self._status[node_name].record_message(topic_suffix)

                # Handle record signal from gello (or any configured topic)
                if self._record_topic and topic == self._record_topic:
                    try:
                        envelope = unpack(payload_b)
                        want = bool(envelope.get("data", {}).get("record", False))
                        if want and not self._prev_record_signal:
                            self.start_episode()
                        elif not want and self._prev_record_signal:
                            self.end_episode(save=True)
                        self._prev_record_signal = want
                    except Exception:
                        pass

            self._supervise_nodes()
            time.sleep(0.005)

        sock.close(linger=0)

    # ── supervision ───────────────────────────────────────────────────────────

    def _supervise_nodes(self, period_s: float = 0.25) -> None:
        """Refresh liveness, decay stale rates, and notice a dead bus.

        Runs off the monitor loop, ~4 Hz. Everything it touches used to be
        either a constant or a fossil:

          * ``alive`` was initialised True and never assigned, so the TUI's
            green dot was a literal.
          * ``pub_hz``/``step_hz`` only moved on receipt, so they froze at the
            last healthy value instead of falling to zero.
          * a dead broker produced no symptom at all — every node kept
            publishing into nothing and every display kept its last reading.
        """
        now = time.perf_counter()
        if now - getattr(self, "_last_supervise_t", 0.0) < period_s:
            return
        self._last_supervise_t = now

        for host in self._hosts:
            for name in host.node_names:
                st = self._status.get(name)
                if st is None:
                    continue
                was_alive = st.alive
                try:
                    alive = host.is_alive()
                except Exception:
                    alive = False
                st.alive = alive
                st.exitcode = getattr(host, "exitcode", None)
                if was_alive and not alive:
                    # The single loudest line in the log. A node that dies during
                    # a session used to leave nothing but a defunct process.
                    logger.error(
                        "NODE DEAD: %r exited (exitcode=%s). Its logs are in %s/%s.log — "
                        "nothing it published after this point is real.",
                        name, st.exitcode, self._log_dir, name,
                    )
                if not alive:
                    st.pub_hz = 0.0
                    st.step_hz = 0.0
                else:
                    st.decay(now)

        # An episode is degraded if ANY camera was unhealthy at ANY point during
        # it, not just at the start or the end. A camera that drops out mid-take
        # and recovers before the operator stops recording would otherwise leave
        # a clean-looking episode with a hole in the middle of it.
        if self._is_recording:
            for name, info in self._unhealthy_now().items():
                self._episode_unhealthy.setdefault(name, info)

        # Reap any child that finished without going through stop(). is_alive()
        # above already waitpid()s the hosts we know about; this catches anything
        # multiprocessing spawned that we do not hold a handle to.
        try:
            import multiprocessing as _mp
            _mp.active_children()
        except Exception:
            pass

        # Is the bus itself carrying anything? Nodes publishing into a dead
        # broker look perfect from inside their own process.
        last_bus = getattr(self, "_last_bus_msg_t", None)
        any_alive = any(st.alive for st in self._status.values())
        bus_down = bool(any_alive and last_bus is not None and now - last_bus > _STALE_AFTER_S)
        if bus_down != getattr(self, "_bus_down", False):
            self._bus_down = bus_down
            if bus_down:
                logger.error(
                    "BUS SILENT: no message from any node in %.1fs while %d node(s) are "
                    "alive. The broker on port %d may be gone; nothing on the bus is real.",
                    now - last_bus, sum(1 for st in self._status.values() if st.alive),
                    self._sub_port,
                )
            else:
                logger.info("BUS RECOVERED: messages are arriving again")

    @property
    def bus_down(self) -> bool:
        return bool(getattr(self, "_bus_down", False))

    def unhealthy_nodes(self) -> list[str]:
        """Names of nodes that are dead or reporting an unhealthy camera."""
        return [st.name for st in self._status.values() if not st.is_healthy]

    def health_snapshot(self) -> dict:
        """Everything the session knows about node health, as plain data.

        Consumed by the control server (and through it the cockpit) and written
        into ``session_meta.json`` when an episode records while degraded.
        """
        return {
            "t": time.time(),
            "bus_down": self.bus_down,
            "nodes": {
                st.name: {
                    "alive": st.alive,
                    "exitcode": st.exitcode,
                    "pub_hz": round(st.pub_hz, 2),
                    "step_hz": round(st.step_hz, 2),
                    "healthy": st.is_healthy,
                    "camera_state": st.camera_state,
                    "health_age_s": (
                        None if st.health_age_s is None else round(st.health_age_s, 2)
                    ),
                    "camera": st.health,
                }
                for st in self._status.values()
            },
        }

    def _auto_record_timer(self, duration: float) -> None:
        # Brief warmup so ZMQ sockets connect and nodes start publishing
        time.sleep(0.3)
        self.start_episode()
        time.sleep(duration)
        self.end_episode(save=True)
        self._stop_event.set()

    def _setup_signal_handlers(self) -> None:
        try:
            signal.signal(signal.SIGUSR1, lambda *_: self.toggle_recording())
            signal.signal(signal.SIGUSR2, lambda *_: self.end_episode(save=False))
        except (OSError, ValueError):
            pass
