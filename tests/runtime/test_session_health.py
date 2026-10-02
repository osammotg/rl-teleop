"""Session-level health: liveness that can go red, rates that decay, honest episodes.

Three things in ``session.py`` used to be incapable of reporting a problem:

  * ``NodeStatus.alive`` was initialised True and assigned NOWHERE in the repo,
    so the TUI's green dot was a hardcoded literal. A camera node that died in
    setup() sat defunct inside a live session for eleven minutes, green.
  * ``pub_hz`` / ``step_hz`` only updated on message receipt, so a stopped
    camera reported its last healthy rate forever (``camera_right ● live
    29.5 Hz`` — a fossil, not a measurement).
  * an episode recorded while a camera was down was indistinguishable from a
    clean one.

These tests hold all three shut.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from robots_realtime.runtime.session import NodeStatus, Session


class FakeHost:
    """Minimal ProcessHost stand-in: only what the supervisor touches."""

    def __init__(self, name: str, alive: bool = True) -> None:
        self._name = name
        self.alive = alive
        self.exitcode: int | None = None
        self.started: list[str] = []
        self.stopped = 0

    @property
    def node_names(self) -> list[str]:
        return [self._name]

    @property
    def node_name(self) -> str:
        return self._name

    def is_alive(self) -> bool:
        return self.alive

    def start_recording(self, save_dir: str) -> None:
        self.started.append(save_dir)

    def stop_recording(self) -> str:
        self.stopped += 1
        return ""


def _session(tmp_path: Path, names: list[str], **kw) -> tuple[Session, dict[str, FakeHost]]:
    s = Session([], save_root=tmp_path, **kw)
    hosts = {n: FakeHost(n) for n in names}
    s._hosts = list(hosts.values())          # type: ignore[assignment]
    s._status = {n: NodeStatus(name=n) for n in names}
    s._record_node_names = list(names)
    return s, hosts


# ── rate decay ───────────────────────────────────────────────────────────────


def test_rates_decay_to_zero_when_nothing_arrives() -> None:
    st = NodeStatus(name="camera_right")
    for _ in range(10):
        st.record_message("rgb")
        time.sleep(0.001)
    st.record_step_hz(29.5)
    assert st.pub_hz > 0 and st.step_hz == 29.5

    # Nothing arrives for longer than the staleness window.
    st.decay(now=time.perf_counter() + 5.0)
    assert st.pub_hz == 0.0, "a stopped camera must not keep reporting its old rate"
    assert st.step_hz == 0.0


def test_rates_survive_a_normal_gap() -> None:
    """Decay must not fire on ordinary jitter — a false 0 Hz is its own lie."""
    st = NodeStatus(name="camera_scan")
    for _ in range(10):
        st.record_message("rgb")
    st.record_step_hz(15.0)
    st.decay(now=time.perf_counter() + 0.2)
    assert st.pub_hz > 0
    assert st.step_hz == 15.0


# ── liveness ─────────────────────────────────────────────────────────────────


def test_alive_goes_false_when_the_process_dies(tmp_path: Path) -> None:
    s, hosts = _session(tmp_path, ["camera_top"])
    s._supervise_nodes(period_s=0.0)
    assert s._status["camera_top"].alive is True

    hosts["camera_top"].alive = False
    hosts["camera_top"].exitcode = 1
    s._supervise_nodes(period_s=0.0)

    st = s._status["camera_top"]
    assert st.alive is False, "the TUI dot must be able to turn red"
    assert st.exitcode == 1
    assert st.pub_hz == 0.0 and st.step_hz == 0.0
    assert st.is_healthy is False
    assert "camera_top" in s.unhealthy_nodes()


def test_camera_health_topic_makes_a_live_node_unhealthy(tmp_path: Path) -> None:
    """A node can be alive and still be useless. Both are reported."""
    s, _ = _session(tmp_path, ["camera_right"])
    st = s._status["camera_right"]
    st.health = {"state": "failed", "healthy": False, "reason": "timeout"}
    assert st.alive is True
    assert st.is_healthy is False
    assert st.camera_state == "failed"
    assert s.unhealthy_nodes() == ["camera_right"]


def test_a_health_record_that_stops_arriving_is_not_evidence_of_health(
    tmp_path: Path,
) -> None:
    """RED found this by SIGSTOPping a camera node on the rig.

    A stopped process publishes NOTHING — including no health. The last health
    message still said ``ok``, so every consumer of the health topic reported a
    perfectly healthy camera that had not moved for as long as you cared to
    wait. That is the same fossil-signal bug as the frozen pub_hz, one level up:
    the cure for the disease had caught the disease.

    Health must be aged by whoever reads it, exactly like a frame.
    """
    s, _ = _session(tmp_path, ["camera_scan"])
    st = s._status["camera_scan"]
    st.record_health({"state": "ok", "healthy": True})
    assert st.is_healthy is True
    assert st.camera_state == "ok"

    # ...and now the node stops publishing entirely.
    st._last_health_t = time.perf_counter() - 10.0
    assert st.health_is_stale is True
    assert st.camera_state == "stale", "a frozen health record must not read as ok"
    assert st.is_healthy is False
    assert "camera_scan" in s.unhealthy_nodes()


def test_a_node_that_never_publishes_health_is_not_penalised(tmp_path: Path) -> None:
    """Only camera nodes publish health; everything else must stay healthy."""
    s, _ = _session(tmp_path, ["gello_right"])
    st = s._status["gello_right"]
    assert st.health is None
    assert st.health_is_stale is False
    assert st.is_healthy is True


def test_health_snapshot_is_plain_serializable_data(tmp_path: Path) -> None:
    s, _ = _session(tmp_path, ["camera_top", "camera_left"])
    s._supervise_nodes(period_s=0.0)
    snap = s.health_snapshot()
    json.dumps(snap)                       # must not raise
    assert set(snap["nodes"]) == {"camera_top", "camera_left"}


# ── dead bus ─────────────────────────────────────────────────────────────────


def test_a_silent_bus_is_reported(tmp_path: Path) -> None:
    """Nodes publishing into a dead broker look perfect from inside themselves."""
    s, _ = _session(tmp_path, ["camera_top"])
    s._last_bus_msg_t = time.perf_counter()
    s._supervise_nodes(period_s=0.0)
    assert s.bus_down is False

    s._last_bus_msg_t = time.perf_counter() - 10.0
    s._supervise_nodes(period_s=0.0)
    assert s.bus_down is True


# ── health-gated recording ───────────────────────────────────────────────────


def _meta(episode_dir: Path) -> dict:
    return json.loads((episode_dir / "session_meta.json").read_text())


def test_a_clean_episode_is_marked_not_degraded(tmp_path: Path) -> None:
    s, _ = _session(tmp_path, ["camera_top"])
    s._supervise_nodes(period_s=0.0)
    s.start_episode()
    episode = s._episode_dir
    assert episode is not None
    meta = _meta(episode)
    # ALWAYS written, including as False: a missing key means "recorded before
    # this existed", which is not the same as "clean".
    assert meta["degraded"] is False
    assert meta["degraded_nodes"] == []
    s.end_episode(save=True)


def test_an_episode_with_a_dead_camera_is_marked_degraded(tmp_path: Path) -> None:
    s, hosts = _session(tmp_path, ["camera_top", "camera_right"])
    hosts["camera_right"].alive = False
    s._supervise_nodes(period_s=0.0)

    s.start_episode()
    episode = s._episode_dir
    assert episode is not None
    meta = _meta(episode)
    assert meta["degraded"] is True
    assert meta["degraded_nodes"] == ["camera_right"]
    s.end_episode(save=True)
    assert _meta(episode)["degraded"] is True


def test_a_camera_that_dies_MID_episode_still_marks_it(tmp_path: Path) -> None:
    """The nastiest shape: healthy at the start, healthy again at the end, and a
    hole in the middle. Sampling health only at the edges would miss it."""
    s, hosts = _session(tmp_path, ["camera_top", "camera_left"])
    s._supervise_nodes(period_s=0.0)
    s.start_episode()
    episode = s._episode_dir
    assert episode is not None
    assert _meta(episode)["degraded"] is False

    hosts["camera_left"].alive = False        # dies mid-take
    s._supervise_nodes(period_s=0.0)
    hosts["camera_left"].alive = True         # ...and comes back
    s._supervise_nodes(period_s=0.0)

    s.end_episode(save=True)
    meta = _meta(episode)
    assert meta["degraded"] is True
    assert "camera_left" in meta["degraded_nodes"]


def test_strict_mode_refuses_rather_than_recording(tmp_path: Path) -> None:
    """The opt-in behaviour, for anyone who prefers no take to a marked take."""
    s, hosts = _session(tmp_path, ["camera_top"], require_healthy_cameras=True)
    hosts["camera_top"].alive = False
    s._supervise_nodes(period_s=0.0)

    with pytest.raises(RuntimeError) as ei:
        s.start_episode()
    assert "camera_top" in str(ei.value)
    assert s.is_recording is False


def test_a_writer_that_failed_to_open_makes_the_episode_degraded(tmp_path: Path) -> None:
    """Fault 12: the writer never opened, so that camera records NOTHING.

    Until 2026-08-10 this was ``except Exception: pass`` and three right-arm
    episodes were recorded with no wrist video and no error anywhere.
    """
    s, hosts = _session(tmp_path, ["camera_top", "camera_scan"])
    s._supervise_nodes(period_s=0.0)

    def _boom(save_dir: str) -> None:
        raise OSError("No space left on device")

    hosts["camera_scan"].start_recording = _boom      # type: ignore[method-assign]
    s.start_episode()
    episode = s._episode_dir
    assert episode is not None
    meta = _meta(episode)
    assert meta["degraded"] is True
    assert "camera_scan" in meta["degraded_nodes"]
    s.end_episode(save=True)


def test_a_discarded_episode_clears_the_degraded_state(tmp_path: Path) -> None:
    s, hosts = _session(tmp_path, ["camera_top"])
    hosts["camera_top"].alive = False
    s._supervise_nodes(period_s=0.0)
    s.start_episode()
    s.end_episode(save=False)
    assert s._episode_unhealthy == {}
