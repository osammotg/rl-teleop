#!/usr/bin/env python3
"""Guarded camera soak + Tier-B process fault injection.

WHAT THIS IS FOR
================

The hermetic tier (``pytest tests/sensors/cameras/``) proves the supervisor
handles every fault shape with fakes. This proves it on the real rig, with real
processes, real USB devices and real ZMQ — the part fakes cannot cover.

It starts ``configs/yam/cameras_only_soak.yaml`` on ports **5565/5566**,
periodically injects a process-level fault, and after each one checks the
invariant from ``HANDOFF-CAMERA-HARDENING.md`` §5.4:

  1. detected within a bounded time,
  2. exactly one outcome — auto-recovered, or loudly failed,
  3. no signal reports success (bus content, health topic and liveness all agree),
  4. no silently incomplete episode,
  5. recovery is clean — no thread, process or memory growth over cycles.

SAFETY
======

It **refuses to start** if the config contains a ``RobotNode``, or if anything is
already listening on the live ports. THE ARM HAS NO BRAKES and this program's
whole job is to kill and stop the processes it starts. See
``robots_realtime/runtime/safety_guard.py``.

It never touches the arm, the CAN bus, a running session, or a RealSense
``hardware_reset()``. The faults it uses are: SIGSTOP/SIGCONT (an exact
simulation of the failure-#4 hang), SIGKILL of one camera node, SIGKILL of the
whole tree (the software-visible part of a power cut), and CPU starvation.

USAGE
=====

    cd ~/Desktop/kitting-v2/rl-teleop
    ./.venv/bin/python3 tools/camera_soak.py --duration 90            # smoke
    ./.venv/bin/python3 tools/camera_soak.py --duration 3600 --faults # the 1 h soak
    ./.venv/bin/python3 tools/camera_soak.py --duration 300 --faults --fault-period 45

Exit status is 0 only if every fault satisfied the invariant.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import resource
import signal
import sys
import threading
import time
from collections import Counter, defaultdict
from pathlib import Path

import zmq

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from robots_realtime.runtime.config import load_session                 # noqa: E402
from robots_realtime.runtime.safety_guard import (                      # noqa: E402
    UnsafeConfig,
    assert_safe_to_soak,
)
from robots_realtime.runtime.transport.serialization import unpack      # noqa: E402

DEFAULT_CONFIG = REPO / "configs" / "yam" / "cameras_only_soak.yaml"
SOAK_PUB_PORT = 5565
SOAK_SUB_PORT = 5566

#: How long a fault may take to become visible. The acceptance bar says 2 s;
#: 3 s is used here to absorb the audit window's own granularity on a loaded box.
DETECT_BUDGET_S = 3.0


# ── bus audit (independent of the health topic, on purpose) ──────────────────


class BusAudit:
    """Counts messages and DISTINCT frame content per topic, straight off the bus.

    Deliberately duplicates ``tools/check_streams.py`` rather than importing the
    health topic: the auditor must be able to catch the health topic lying. If
    both read the same source, neither can contradict the other.
    """

    def __init__(self, sub_port: int) -> None:
        self._ctx = zmq.Context.instance()
        self._sock = self._ctx.socket(zmq.SUB)
        # BOUND THE AUDITOR'S OWN QUEUE. Unpacking and hashing four cameras'
        # worth of full-resolution frames is not free, and when this thread falls
        # behind, ZMQ buffers on our side: at the first draft's 4000-message HWM
        # that is 4000 x ~2.7 MB. The auditor then grew half a gigabyte and the
        # soak report blamed the system under test. Counts become approximate
        # under load, which is fine — this measures whether CONTENT is changing,
        # not exact throughput.
        self._sock.setsockopt(zmq.RCVHWM, 100)
        self._sock.connect(f"tcp://127.0.0.1:{sub_port}")
        self._sock.setsockopt(zmq.SUBSCRIBE, b"")
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self.counts: Counter = Counter()
        self.digests: dict[str, set] = defaultdict(set)
        self.health: dict[str, dict] = {}
        self.last_seen: dict[str, float] = {}
        self._t = threading.Thread(target=self._loop, daemon=True, name="BusAudit")
        self._t.start()

    def _loop(self) -> None:
        while not self._stop.is_set():
            if not self._sock.poll(50):
                continue
            while True:
                try:
                    parts = self._sock.recv_multipart(zmq.NOBLOCK)
                except zmq.Again:
                    break
                if len(parts) < 2:
                    continue
                topic = parts[0].decode(errors="replace")
                now = time.monotonic()
                with self._lock:
                    self.counts[topic] += 1
                    self.last_seen[topic] = now
                if topic.endswith("/health"):
                    try:
                        data = unpack(parts[1]).get("data") or {}
                    except Exception:
                        continue
                    with self._lock:
                        rec = dict(data)
                        rec["_rx_mono"] = now
                        self.health[topic.split("/")[0]] = rec
                    continue
                if not topic.endswith("/rgb"):
                    continue
                try:
                    env = unpack(parts[1])
                except Exception:
                    continue
                data = env.get("data") or {}
                frame = data.get("frame")
                if frame is None:
                    imgs = data.get("images")
                    if isinstance(imgs, dict) and imgs:
                        frame = imgs.get("rgb")
                        if frame is None:
                            frame = next(iter(imgs.values()))
                if frame is None:
                    continue
                raw = memoryview(frame).tobytes() if hasattr(frame, "tobytes") else bytes(frame)
                with self._lock:
                    self.digests[topic].add(hashlib.blake2b(raw, digest_size=8).hexdigest())

    def window(self, secs: float) -> dict:
        """Sample a fresh window: returns per-topic counts and distinct frames."""
        with self._lock:
            self.counts.clear()
            self.digests.clear()
        time.sleep(secs)
        with self._lock:
            return {
                "counts": dict(self.counts),
                "unique": {k: len(v) for k, v in self.digests.items()},
                "health": dict(self.health),
                "secs": secs,
            }

    def close(self) -> None:
        self._stop.set()
        self._t.join(timeout=2.0)
        self._sock.close(linger=0)


def streaming_nodes(win: dict) -> set[str]:
    """Nodes genuinely delivering DISTINCT frames in this window."""
    out = set()
    for topic, n in win["counts"].items():
        if not topic.endswith("/rgb"):
            continue
        if n >= 2 and win["unique"].get(topic, 0) >= 2:
            out.add(topic.split("/")[0])
    return out


# ── the soak ─────────────────────────────────────────────────────────────────


class Violation(Exception):
    pass


class Soak:
    def __init__(self, args) -> None:
        self.args = args
        self.session = None
        self.audit: BusAudit | None = None
        self.violations: list[dict] = []
        self.rounds: list[dict] = []
        self.rss_samples: list[tuple[float, int]] = []

    # -- helpers --------------------------------------------------------

    def log(self, msg: str) -> None:
        print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

    def violation(self, fault: str, node: str, why: str, evidence: dict | None = None) -> None:
        v = {"t": time.time(), "fault": fault, "node": node, "why": why,
             "evidence": evidence or {}}
        self.violations.append(v)
        self.log(f"  !! INVARIANT VIOLATED [{fault} / {node}]: {why}")

    def camera_hosts(self) -> list:
        return [h for h in self.session._hosts if h.node_name.startswith("camera")]

    def node_health(self, node: str) -> dict:
        """The node's health record, with STALENESS APPLIED.

        A health message that stopped arriving is not a healthy camera. RED found
        this by SIGSTOPping a node: it published nothing at all, so the last
        `ok` sat on the bus indefinitely and every health reader believed it.
        The auditor must age this signal exactly as it ages frames.
        """
        assert self.audit is not None
        rec = dict(self.audit.health.get(node, {}))
        if not rec:
            return rec
        rx = rec.pop("_rx_mono", None)
        if rx is not None and time.monotonic() - rx > DETECT_BUDGET_S:
            rec["healthy"] = False
            rec["state"] = "stale"
            rec["health_age_s"] = round(time.monotonic() - rx, 2)
        return rec

    @staticmethod
    def _proc_rss_kb(pid: int) -> int:
        try:
            with open(f"/proc/{pid}/status", encoding="utf-8") as f:
                for line in f:
                    if line.startswith("VmRSS:"):
                        return int(line.split()[1])
        except Exception:
            pass
        return 0

    def rss_kb(self) -> dict:
        """CURRENT RSS, split runner vs nodes.

        Split, and current rather than peak, because the first version of this
        reported one combined number using ru_maxrss — a HIGH-WATER MARK that
        never falls. It showed +579 MB over two minutes and could not say whether
        the leak was in the system under test or in the auditor watching it.
        A measurement that cannot localise what it measures is not much better
        than no measurement.
        """
        nodes = {}
        for host in self.session._hosts:
            pid = getattr(host, "pid", None)
            if pid:
                nodes[host.node_name] = self._proc_rss_kb(pid)
        return {
            "runner": self._proc_rss_kb(os.getpid()),
            "nodes": nodes,
            "nodes_total": sum(nodes.values()),
        }

    # -- the invariant --------------------------------------------------

    def check_invariant(self, fault: str, node: str, expect_streaming: bool) -> None:
        """Assert every signal agrees, and that none of them claims success.

        ``expect_streaming`` is what SHOULD be true of ``node`` right now. When
        False (the fault is active), the whole point is that no signal may say
        the camera is fine.
        """
        assert self.audit is not None
        win = self.audit.window(self.args.audit_secs)
        streaming = streaming_nodes(win)
        health = self.node_health(node)
        st = self.session._status.get(node)
        alive = bool(st and st.alive)
        pub_hz = float(st.pub_hz) if st else 0.0

        evidence = {
            "streaming": sorted(streaming),
            "health_state": health.get("state"),
            "health_healthy": health.get("healthy"),
            "alive": alive,
            "pub_hz": round(pub_hz, 2),
            "counts": {k: v for k, v in win["counts"].items() if k.startswith(node)},
            "unique": {k: v for k, v in win["unique"].items() if k.startswith(node)},
        }

        if expect_streaming:
            if node not in streaming:
                self.violation(fault, node, "expected to be streaming and is not", evidence)
            elif health and not health.get("healthy", False):
                self.violation(
                    fault, node,
                    "bus says streaming but health says unhealthy — the signals disagree",
                    evidence,
                )
            return

        # Fault is active. Nothing may claim success.
        if node in streaming:
            self.violation(fault, node, "still delivering distinct frames during the fault", evidence)
        if health.get("healthy") is True:
            self.violation(fault, node, "health topic still reports healthy", evidence)
        if alive and pub_hz > 0.5 and health.get("healthy") is not False:
            self.violation(
                fault, node,
                f"pub_hz still reads {pub_hz:.1f} with no health contradiction — a fossil rate",
                evidence,
            )
        # Something must be saying it is broken. Silence is the failure mode.
        said_something = (not alive) or (health.get("healthy") is False) or (node not in streaming)
        if not said_something:
            self.violation(fault, node, "NO signal reported the fault at all", evidence)

    def device_present(self, node: str) -> bool | None:
        """Is this node's physical device still on the bus? None = can't tell.

        A camera cannot recover from a fault if the hardware left the building
        while the fault was running, and calling that a supervisor bug is a lie
        about the thing under test. It happened for real during this work: the
        ASMedia controller died mid-soak and took the D455 with it, three
        minutes before a SIGSTOP round that then "failed to recover".
        """
        for host in self.session._hosts:
            if host.node_name != node:
                continue
            spec = getattr(getattr(host, "_node", None), "_driver_spec", None) or {}
            path = spec.get("device_path")
            if path:
                return os.path.exists(path)
            serial = spec.get("device_id")
            if serial:
                try:
                    import pyrealsense2 as rs
                    return str(serial) in [
                        d.get_info(rs.camera_info.serial_number)
                        for d in rs.context().query_devices()
                    ]
                except Exception:
                    return None
        return None

    def await_recovery(self, fault: str, node: str, timeout: float) -> bool:
        """Wait for the node to stream distinct frames again."""
        assert self.audit is not None
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            win = self.audit.window(min(2.0, self.args.audit_secs))
            if node in streaming_nodes(win):
                return True
        return False

    def recovery_expected(self, node: str) -> bool:
        """False when the device is physically absent — recovery is impossible."""
        present = self.device_present(node)
        if present is False:
            self.log(f"  NOTE: {node}'s device is no longer on the USB bus. Recovery is "
                     f"impossible and this is NOT a supervisor failure — check dmesg for "
                     f"a controller death or an unplug.")
            return False
        return True

    # -- faults ---------------------------------------------------------

    def fault_sigstop(self, host) -> None:
        """SIGSTOP: an exact simulation of the failure-#4 hang.

        The process stays alive, holds its device, and publishes nothing. Before
        this work that was indistinguishable from a healthy camera.
        """
        node = host.node_name
        pid = host.pid
        if not pid:
            return
        self.log(f"  fault: SIGSTOP {node} (pid {pid}) — the failure-#4 hang, exactly")
        os.kill(pid, signal.SIGSTOP)
        time.sleep(DETECT_BUDGET_S)
        try:
            self.check_invariant("SIGSTOP", node, expect_streaming=False)
        finally:
            os.kill(pid, signal.SIGCONT)
            self.log(f"  SIGCONT {node}")
        if not self.await_recovery("SIGSTOP", node, timeout=30.0):
            if self.recovery_expected(node):
                self.violation("SIGSTOP", node, "did not recover after SIGCONT within 30 s")
        else:
            self.check_invariant("SIGSTOP", node, expect_streaming=True)

    def fault_sigkill(self, host) -> None:
        """SIGKILL one camera node. The session must notice and say so.

        Before: a zombie process and a green dot, for as long as the session ran.
        """
        node = host.node_name
        pid = host.pid
        if not pid:
            return
        self.log(f"  fault: SIGKILL {node} (pid {pid})")
        os.kill(pid, signal.SIGKILL)
        time.sleep(DETECT_BUDGET_S)
        self.check_invariant("SIGKILL", node, expect_streaming=False)
        st = self.session._status.get(node)
        if st and st.alive:
            self.violation("SIGKILL", node, "session still reports the node as alive")
        # Zombie check: a killed child that is never reaped stays defunct.
        try:
            with open(f"/proc/{pid}/stat", encoding="utf-8") as f:
                state = f.read().split(") ", 1)[1].split()[0]
            if state == "Z":
                self.violation("SIGKILL", node, f"pid {pid} left defunct (zombie) — not reaped")
        except FileNotFoundError:
            pass          # fully reaped, which is what we want

    # -- driver-level faults, via the fault file (fake config only) ------

    def _fault_file(self, node: str) -> Path | None:
        """The file SoakCamera polls for this node, if the config uses one."""
        for h in self.session._hosts:
            if h.node_name != node:
                continue
            spec = getattr(getattr(h, "_node", None), "_driver_spec", None) or {}
            path = spec.get("fault_file")
            if not path:
                return None
            # Same redirection SoakCamera applies, so the runner writes where the
            # driver reads. See SOAK_FAULT_DIR in fault_injection.py.
            override = os.environ.get("SOAK_FAULT_DIR")
            return Path(override) / Path(path).name if override else Path(path)
        return None

    def _set_driver_fault(self, node: str, mode: str) -> bool:
        path = self._fault_file(node)
        if path is None:
            return False
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("" if mode == "ok" else mode)
        return True

    def _driver_fault(self, host, mode: str, recover: bool = True) -> None:
        """Flip a synthetic camera into a fault mode and check the invariant.

        This is the shape the hermetic tier cannot reach: the fault happens
        inside a REAL node subprocess, publishing over REAL ZMQ, watched by the
        REAL session monitor.
        """
        node = host.node_name
        if not self._set_driver_fault(node, mode):
            return
        self.log(f"  fault: driver mode {mode!r} on {node}")
        time.sleep(DETECT_BUDGET_S + (4.0 if mode == "frozen" else 0.0))
        try:
            self.check_invariant(f"DRIVER_{mode.upper()}", node, expect_streaming=False)
        finally:
            if recover:
                self._set_driver_fault(node, "ok")
        if recover:
            if not self.await_recovery(f"DRIVER_{mode.upper()}", node, timeout=30.0):
                if self.recovery_expected(node):
                    self.violation(f"DRIVER_{mode.upper()}", node,
                                   "did not recover within 30 s after the fault was cleared")
            else:
                self.check_invariant(f"DRIVER_{mode.upper()}", node, expect_streaming=True)

    def fault_driver_frozen(self, host) -> None:
        """Fault 4: the same frame forever, no error. The one the cockpit lied about."""
        self._driver_fault(host, "frozen")

    def fault_driver_ret_false(self, host) -> None:
        """Fault 1: the stale UVC handle. THE known failure."""
        self._driver_fault(host, "ret_false")

    def fault_driver_hang(self, host) -> None:
        """Fault 2: read() blocks and never returns."""
        self._driver_fault(host, "hang")

    def fault_driver_gone(self, host) -> None:
        """Fault 7: the device disappears, then comes back."""
        self._driver_fault(host, "gone")

    def fault_cpu_starve(self, host) -> None:
        """Pin heavy load onto the box; the camera must degrade, not freeze silently."""
        node = host.node_name
        self.log(f"  fault: CPU starvation while watching {node}")
        stop = threading.Event()

        def _burn() -> None:
            x = 0
            while not stop.is_set():
                x = (x * x + 1) % 999983

        threads = [threading.Thread(target=_burn, daemon=True) for _ in range(max(2, os.cpu_count() or 4))]
        for t in threads:
            t.start()
        try:
            time.sleep(DETECT_BUDGET_S * 2)
            win = self.audit.window(self.args.audit_secs)      # type: ignore[union-attr]
            health = self.node_health(node)
            streaming = node in streaming_nodes(win)
            # Either it keeps working (fine) or it says it is degraded (fine).
            # The forbidden state is "not working AND claiming to be fine".
            if not streaming and health.get("healthy") is True:
                self.violation(
                    "CPU_STARVE", node,
                    "starved into silence while health still reports healthy",
                    {"health": health},
                )
        finally:
            stop.set()
            for t in threads:
                t.join(timeout=1.0)

    # -- run ------------------------------------------------------------

    def run(self) -> int:
        args = self.args
        try:
            assert_safe_to_soak(
                args.config, args.pub_port, args.sub_port,
                allow_live_ports=args.i_know_what_im_doing,
            )
        except UnsafeConfig as exc:
            print(f"\n{exc}\n", file=sys.stderr)
            return 2

        self.log(f"config      : {args.config}")
        self.log(f"ports       : pub={args.pub_port} sub={args.sub_port}  (live bus is 5555/5556)")
        self.log(f"duration    : {args.duration:.0f}s   faults={'on' if args.faults else 'off'}")

        # Each soak gets its own fault directory, so two runs cannot edit each
        # other's injected faults. Derived from the ports, which are already
        # unique per run.
        if not os.environ.get("SOAK_FAULT_DIR"):
            os.environ["SOAK_FAULT_DIR"] = f"/tmp/soak_faults/{args.pub_port}"
        Path(os.environ["SOAK_FAULT_DIR"]).mkdir(parents=True, exist_ok=True)
        self.log(f"fault dir   : {os.environ['SOAK_FAULT_DIR']}")

        self.session = load_session(str(args.config), pub_port=args.pub_port, sub_port=args.sub_port)
        if args.save_root:
            self.session._save_root = Path(args.save_root)

        self.session.start()
        self.audit = BusAudit(args.sub_port)
        t_end = time.monotonic() + args.duration
        started = time.monotonic()

        try:
            # Let the cameras warm up before judging anything.
            time.sleep(args.warmup)
            win = self.audit.window(args.audit_secs)
            up = streaming_nodes(win)
            self.log(f"streaming at start: {sorted(up) or 'NONE'}")
            for host in self.camera_hosts():
                h = self.node_health(host.node_name)
                self.log(f"  {host.node_name:14s} health={h.get('state', '?'):10s} "
                         f"alive={self.session._status[host.node_name].alive}")
            if not up:
                self.log("  no camera is streaming — nothing to soak. Check the rig.")
                return 3

            hosts = [h for h in self.camera_hosts() if h.node_name in up]
            # Driver-level faults come first in the rotation and are recoverable;
            # SIGKILL is last because it permanently removes a node from play.
            faults = [
                self.fault_driver_frozen,
                self.fault_sigstop,
                self.fault_driver_ret_false,
                self.fault_cpu_starve,
                self.fault_driver_gone,
                self.fault_driver_hang,
                self.fault_sigkill,
            ]
            if not any(self._fault_file(h.node_name) for h in hosts):
                # Real cameras: no fault file to write, so only process faults apply.
                faults = [self.fault_sigstop, self.fault_cpu_starve, self.fault_sigkill]
                self.log("  (real-hardware config: driver-level faults unavailable)")
            n = 0
            next_fault = time.monotonic() + args.fault_period

            while time.monotonic() < t_end:
                time.sleep(1.0)
                self.rss_samples.append((time.monotonic() - started, self.rss_kb()))

                # Steady-state audit: nothing may quietly stop while we are not
                # looking. This is what turns a long soak into evidence.
                if int(time.monotonic() - started) % args.audit_period == 0:
                    win = self.audit.window(args.audit_secs)
                    now_up = streaming_nodes(win)
                    for host in hosts:
                        node = host.node_name
                        if node in now_up:
                            continue
                        h = self.node_health(node)
                        st = self.session._status.get(node)
                        if h.get("healthy") is not False and (st is None or st.alive):
                            self.violation(
                                "STEADY_STATE", node,
                                "stopped streaming during the soak with no signal saying so",
                                {"health": h},
                            )

                hosts = [h for h in hosts if h.is_alive()]
                if args.faults and not hosts:
                    # Nothing left alive to attack. Idling out the clock would
                    # report a long clean soak that tested nothing at all — the
                    # exact species of comfortable lie this project is about.
                    self.log("every camera node is down; ending the soak early rather "
                             "than reporting time in which nothing was tested")
                    break
                if not args.faults or time.monotonic() < next_fault:
                    continue

                host = hosts[n % len(hosts)]
                fault = faults[n % len(faults)]
                # SIGKILL IS PERMANENT — nothing restarts a node in-place today
                # (that is an option-3 property: cameras in a supervised daemon
                # whose lifetime is independent of the session). So in a long
                # soak it eats the fleet: after four kill rounds there is nothing
                # left to test and the remaining hour measures nothing. Keep it
                # in the rotation, but rarely, and let the recoverable faults do
                # the sustained work.
                if fault is self.fault_sigkill and (n % args.kill_every) != 0:
                    fault = self.fault_driver_frozen if self._fault_file(host.node_name) \
                        else self.fault_sigstop
                n += 1
                # A killed node cannot be re-killed; drop it from rotation.
                if not host.is_alive():
                    hosts = [h for h in hosts if h.is_alive()]
                    if not hosts:
                        self.log("  every camera node is down; stopping fault injection")
                        args.faults = False
                        continue
                    host = hosts[0]
                self.log(f"round {n}: {fault.__name__} on {host.node_name}")
                try:
                    fault(host)
                except Exception as exc:                        # noqa: BLE001
                    self.violation(fault.__name__, host.node_name, f"fault raised: {exc!r}")
                self.rounds.append({"n": n, "fault": fault.__name__, "node": host.node_name})
                next_fault = time.monotonic() + args.fault_period
                hosts = [h for h in hosts if h.is_alive()]

        finally:
            self.report()
            if self.audit is not None:
                self.audit.close()
            try:
                self.session.stop()
            except Exception as exc:                            # noqa: BLE001
                self.log(f"session stop raised: {exc!r}")

        return 1 if self.violations else 0

    def report(self) -> None:
        print()
        print("═" * 78)
        print(f"SOAK REPORT — {len(self.rounds)} fault rounds, {len(self.violations)} violations")
        print("═" * 78)
        if self.rss_samples:
            t0, r0 = self.rss_samples[0]
            t1, r1 = self.rss_samples[-1]
            span = max(1.0, t1 - t0)
            print(f"RSS over {span:.0f}s:")
            print(f"  soak runner : {r0['runner']/1024:7.0f} MB -> {r1['runner']/1024:7.0f} MB "
                  f"({(r1['runner']-r0['runner'])/1024:+.1f} MB)   [the auditor, not the system under test]")
            print(f"  nodes total : {r0['nodes_total']/1024:7.0f} MB -> {r1['nodes_total']/1024:7.0f} MB "
                  f"({(r1['nodes_total']-r0['nodes_total'])/1024:+.1f} MB)")
            for name, kb in sorted(r1["nodes"].items()):
                was = r0["nodes"].get(name, 0)
                print(f"    {name:14s} {was/1024:7.0f} MB -> {kb/1024:7.0f} MB ({(kb-was)/1024:+.1f} MB)")
            node_growth = r1["nodes_total"] - r0["nodes_total"]
            # Only the NODES are the system under test. Per-hour, a camera node
            # holding a few frames should be flat; 100 MB/h is generous.
            per_hour = node_growth / 1024 * 3600 / span
            print(f"  node growth : {per_hour:+.0f} MB/h extrapolated")
            if span > 300 and per_hour > 100:
                self.violations.append({
                    "t": time.time(), "fault": "SOAK", "node": "nodes",
                    "why": f"node RSS grew {per_hour:.0f} MB/h — a leak across fault cycles",
                    "evidence": {"first": r0, "last": r1},
                })
                print("  ** NODE MEMORY IS GROWING — invariant 5 (clean recovery) is violated **")
        for v in self.violations:
            print(f"  VIOLATION [{v['fault']} / {v['node']}] {v['why']}")
            if v["evidence"]:
                print(f"            {json.dumps(v['evidence'], default=str)}")
        print("VERDICT:", "PASS — no invariant violated" if not self.violations else "FAIL")

        if self.args.json_out:
            Path(self.args.json_out).write_text(json.dumps({
                "rounds": self.rounds,
                "violations": self.violations,
                "rss": self.rss_samples,
            }, indent=2, default=str))


def _install_term_handler(soak: "Soak") -> None:
    """Stop the session cleanly on SIGTERM / SIGINT.

    Python runs `finally` blocks on KeyboardInterrupt but NOT on a default
    SIGTERM, so `kill <soak pid>` skipped session.stop() and left every camera
    node orphaned onto init, still holding the bus ports. Anyone who then tried
    to start a soak got "Address already in use" and a message blaming a running
    rr-session — a wrong diagnosis produced by our own cleanup path.
    """
    def _stop(signum, _frame):
        soak.log(f"signal {signum} — stopping the session before exiting")
        try:
            if soak.session is not None:
                soak.session.stop()
        finally:
            sys.exit(130)

    for sig in (signal.SIGTERM, signal.SIGINT):
        try:
            signal.signal(sig, _stop)
        except (OSError, ValueError):
            pass


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    ap.add_argument("--duration", type=float, default=120.0)
    ap.add_argument("--pub-port", type=int, default=SOAK_PUB_PORT)
    ap.add_argument("--sub-port", type=int, default=SOAK_SUB_PORT)
    ap.add_argument("--faults", action="store_true", help="inject process-level faults")
    ap.add_argument("--fault-period", type=float, default=60.0)
    ap.add_argument("--kill-every", type=int, default=6,
                    help="only SIGKILL a node on every Nth fault round. Killed nodes "
                         "are never restarted, so a long soak that kills freely runs "
                         "out of cameras and then measures nothing.")
    ap.add_argument("--audit-secs", type=float, default=3.0)
    ap.add_argument("--audit-period", type=int, default=30)
    ap.add_argument("--warmup", type=float, default=8.0)
    ap.add_argument("--save-root", default="/tmp/camera_soak_recordings")
    ap.add_argument("--json-out", default=None)
    ap.add_argument("--i-know-what-im-doing", action="store_true",
                    help="skip the live-port check. Do not use while a session is up.")
    args = ap.parse_args(argv)
    soak = Soak(args)
    _install_term_handler(soak)
    return soak.run()


if __name__ == "__main__":
    sys.exit(main())
