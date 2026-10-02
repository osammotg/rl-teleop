"""Control-surface tests — no robot, no cameras, no CAN bus.

The point of this file: prove the cockpit's buttons drive a real Session's
methods, with correct state transitions and correct refusals, WITHOUT any
hardware. A FakeSession implements exactly the surface the TUI key handler
touches, so if these pass, the only thing left untested is the hardware
behind the real Session methods.

Run:  uv run pytest tests/runtime/test_control_server.py -q
"""

from __future__ import annotations

import json
import threading
import time
import urllib.error
import urllib.request

import pytest

from robots_realtime.runtime.control_server import ControlServer


class FakeSession:
    """Mirrors the Session API the TUI drives. Records every call."""

    def __init__(self, mappings: dict | None = None):
        self._recording = False
        self._paused = False
        self._start_t: float | None = None
        self.calls: list[tuple] = []
        self.saved: list[dict] = []
        self.flags: list[str] = []
        self._mappings = mappings if mappings is not None else {"1": "kitting"}
        # set → session actions hang, to exercise the "busy" path.
        # NOTE: spin on is_set(), not Event.wait() — wait() returns *immediately*
        # when the flag is true, which is the opposite of blocking.
        self.block = threading.Event()

    def _maybe_block(self) -> None:
        deadline = time.time() + 5
        while self.block.is_set() and time.time() < deadline:
            time.sleep(0.01)

    # -- properties the status payload reads -------------------------------
    @property
    def is_recording(self) -> bool: return self._recording
    @property
    def is_paused(self) -> bool: return self._paused
    @property
    def episode_start_time(self): return self._start_t
    @property
    def instruction(self) -> str: return "kitting"
    @property
    def instruction_mappings(self) -> dict: return self._mappings
    @property
    def save_root(self) -> str: return "/tmp/fake-recordings"

    # -- the actions the TUI calls -----------------------------------------
    def start_episode(self) -> None:
        self._maybe_block()
        self.calls.append(("start_episode",))
        self._recording = True
        self._start_t = time.time()

    def end_episode(self, save: bool = True, instruction: str | None = None) -> None:
        self._maybe_block()
        self.calls.append(("end_episode", save, instruction))
        if save:
            self.saved.append({"instruction": instruction})
        self._recording = False
        self._start_t = None

    def toggle_recording(self) -> None:
        self.calls.append(("toggle_recording",))
        self._recording = not self._recording

    def toggle_pause(self) -> None:
        self.calls.append(("toggle_pause",))
        self._paused = not self._paused

    def flag_episode(self, tag: str) -> bool:
        self.calls.append(("flag_episode", tag))
        self.flags.append(tag)
        return True


# ── harness ──────────────────────────────────────────────────────────────

@pytest.fixture
def server():
    sess = FakeSession()
    # port 0 → let the OS pick a free one, so the suite never collides with a
    # real session that already holds 8792
    srv = ControlServer(sess, port=0)
    assert srv.start(), "control server failed to bind"
    srv.port = srv.httpd.server_address[1]
    yield srv, sess
    srv.stop()


def _req(srv, path, method="GET", body=None):
    url = f"http://127.0.0.1:{srv.port}{path}"
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, method=method,
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=3) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def _settle(sess, pred, timeout=2.0):
    """Session actions run on a worker thread; wait for the effect."""
    end = time.time() + timeout
    while time.time() < end:
        if pred(sess):
            return True
        time.sleep(0.01)
    return False


# ── tests ────────────────────────────────────────────────────────────────

def test_health_and_status_shape(server):
    srv, _ = server
    code, body = _req(srv, "/health")
    assert code == 200 and body["ok"] is True

    code, body = _req(srv, "/status")
    assert code == 200
    for key in ("recording", "paused", "elapsed_s", "instruction",
                "instruction_mappings", "save_root", "busy"):
        assert key in body, f"/status missing {key}"
    assert body["recording"] is False


def test_start_then_save_drives_the_session(server):
    srv, sess = server
    code, body = _req(srv, "/record/start", "POST")
    assert code == 200 and body["accepted"] is True
    assert _settle(sess, lambda s: s.is_recording), "start_episode never ran"

    # status reflects the recorder, not a client-side guess
    _, st = _req(srv, "/status")
    assert st["recording"] is True and st["elapsed_s"] >= 0

    code, body = _req(srv, "/record/save", "POST")
    assert code == 200 and body["accepted"] is True
    assert _settle(sess, lambda s: not s.is_recording), "end_episode never ran"
    assert sess.saved == [{"instruction": "kitting"}], sess.saved


def test_save_uses_explicit_instruction_when_given(server):
    srv, sess = server
    _req(srv, "/record/start", "POST")
    _settle(sess, lambda s: s.is_recording)
    _req(srv, "/record/save", "POST", {"instruction": "grasp-only"})
    assert _settle(sess, lambda s: bool(s.saved))
    assert sess.saved[-1]["instruction"] == "grasp-only"


def test_discard_does_not_save(server):
    srv, sess = server
    _req(srv, "/record/start", "POST")
    _settle(sess, lambda s: s.is_recording)
    _req(srv, "/record/discard", "POST")
    assert _settle(sess, lambda s: not s.is_recording)
    assert sess.saved == []
    assert ("end_episode", False, None) in sess.calls


def test_refusals_are_explicit_not_silent(server):
    """A button that does nothing must say why — this is what made the old
    cockpit feel broken."""
    srv, sess = server
    code, body = _req(srv, "/record/save", "POST")
    assert code == 200 and body["accepted"] is False and body["reason"] == "not recording"

    _req(srv, "/record/start", "POST")
    _settle(sess, lambda s: s.is_recording)
    code, body = _req(srv, "/record/start", "POST")
    assert body["accepted"] is False and body["reason"] == "already recording"


def test_advance_mirrors_the_right_arrow(server):
    """idle → start a take; recording → save it. Same helper the TUI uses."""
    srv, sess = server
    _req(srv, "/record/advance", "POST")
    assert _settle(sess, lambda s: s.is_recording), "advance did not start"

    _req(srv, "/record/advance", "POST")
    assert _settle(sess, lambda s: not s.is_recording), "advance did not save"
    assert len(sess.saved) == 1


def test_advance_without_mappings_toggles(server):
    """No instruction mappings → the TUI falls back to toggle_recording, and
    the control surface must match that, not diverge."""
    sess = FakeSession(mappings={})
    srv = ControlServer(sess, port=0)
    assert srv.start()
    srv.port = srv.httpd.server_address[1]
    try:
        _req(srv, "/record/advance", "POST")
        assert _settle(sess, lambda s: ("toggle_recording",) in s.calls)
    finally:
        srv.stop()


def test_rerecord_discards_then_restarts(server):
    srv, sess = server
    _req(srv, "/record/start", "POST")
    _settle(sess, lambda s: s.is_recording)
    sess.calls.clear()
    _req(srv, "/record/rerecord", "POST")
    assert _settle(sess, lambda s: ("start_episode",) in s.calls)
    assert ("end_episode", False, None) in sess.calls, "old take was not discarded"
    assert sess.saved == []


def test_pause_toggles_and_reports_immediately(server):
    srv, sess = server
    _, body = _req(srv, "/pause", "POST")
    assert body["paused"] is True and sess.is_paused is True
    _, body = _req(srv, "/pause", "POST")
    assert body["paused"] is False


def test_flag_validates_tag(server):
    srv, sess = server
    code, body = _req(srv, "/flag", "POST", {"tag": "nonsense"})
    assert code == 400 and body["ok"] is False

    code, body = _req(srv, "/flag", "POST", {"tag": "re_grasp"})
    assert code == 200 and body["accepted"] is True
    assert _settle(sess, lambda s: "re_grasp" in s.flags)


def test_busy_is_reported_not_dropped(server):
    """While a session action is in flight the surface must refuse loudly, so
    the cockpit can show 'busy' rather than a dead button."""
    srv, sess = server
    sess.block.set()                       # make start_episode hang
    _req(srv, "/record/start", "POST")
    time.sleep(0.15)                       # let the worker take the lock

    _, body = _req(srv, "/record/advance", "POST")
    assert body["accepted"] is False and body["reason"] == "busy"
    _, st = _req(srv, "/status")
    assert st["busy"] is True

    sess.block.clear()                     # release
    assert _settle(sess, lambda s: s.is_recording, timeout=6)


def test_cors_allows_the_cockpit_origin(server):
    """The cockpit is served from :8799 and calls this on :8792 — without
    CORS every button silently fails in the browser."""
    srv, _ = server
    url = f"http://127.0.0.1:{srv.port}/status"
    with urllib.request.urlopen(url, timeout=3) as r:
        assert r.headers.get("Access-Control-Allow-Origin") == "*"

    req = urllib.request.Request(url, method="OPTIONS")
    with urllib.request.urlopen(req, timeout=3) as r:
        assert r.status == 204
        assert "POST" in r.headers.get("Access-Control-Allow-Methods", "")


def test_unknown_route_404s(server):
    srv, _ = server
    code, _ = _req(srv, "/nope")
    assert code == 404


def test_bind_failure_is_not_fatal():
    """A busy port must degrade to keyboard-only, never crash the session."""
    a = ControlServer(FakeSession(), port=0)
    assert a.start()
    port = a.httpd.server_address[1]
    try:
        b = ControlServer(FakeSession(), port=port)
        assert b.start() is False, "second bind on the same port should fail cleanly"
    finally:
        a.stop()
