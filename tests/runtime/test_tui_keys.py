"""Keyboard dispatch for the recording TUI: arrow keys + quality-flag keys."""
from __future__ import annotations

import json
import os
import pty
import sys
import threading
import time
import tty
from types import SimpleNamespace

import pytest

from robots_realtime.runtime.session import Session
from robots_realtime.runtime.tui import (
    _FLAG_KEYS,
    _advance,
    _default_instruction,
    _read_arrow,
    _read_keys,
    _rerecord,
)


class _Stub:
    def __init__(self, recording: bool, mappings: dict):
        self.is_recording = recording
        self.instruction_mappings = mappings
        self.calls: list = []

    def start_episode(self):
        self.calls.append(("start",))

    def end_episode(self, save=True, instruction=None):
        self.calls.append(("end", save, instruction))

    def toggle_recording(self):
        self.calls.append(("toggle",))


def _wait(stub, n=1, timeout=2.0):
    """_advance/_rerecord dispatch on a daemon thread; wait for it to land."""
    end = time.time() + timeout
    while time.time() < end and len(stub.calls) < n:
        time.sleep(0.01)


def test_default_instruction_prefers_key_1():
    assert _default_instruction(_Stub(False, {"0": "play", "1": "kit", "2": "other"})) == "kit"
    assert _default_instruction(_Stub(False, {"0": "play", "3": "z"})) == "z"
    assert _default_instruction(_Stub(False, {})) is None


def test_right_arrow_idle_starts_episode():
    s = _Stub(recording=False, mappings={"1": "kit"})
    _advance(s, threading.Lock())
    _wait(s)
    assert s.calls == [("start",)]


def test_right_arrow_recording_saves_with_default_instruction():
    s = _Stub(recording=True, mappings={"1": "kit"})
    _advance(s, threading.Lock())
    _wait(s)
    assert s.calls == [("end", True, "kit")]


def test_left_arrow_discards_and_restarts():
    s = _Stub(recording=True, mappings={"1": "kit"})
    _rerecord(s, threading.Lock())
    _wait(s, n=2)
    assert s.calls == [("end", False, None), ("start",)]


def test_flag_keys_mapping():
    assert _FLAG_KEYS == {"g": "re_grasp", "x": "bad", "s": "slow"}


def test_flag_episode_writes_sidecar(tmp_path):
    obj = SimpleNamespace(_recording_lock=threading.Lock(), _episode_dir=tmp_path)
    assert Session.flag_episode(obj, "re_grasp") is True
    assert Session.flag_episode(obj, "slow") is True
    data = json.loads((tmp_path / "operator_flags.json").read_text())
    assert [f["tag"] for f in data["flags"]] == ["re_grasp", "slow"]


def test_flag_episode_noop_when_not_recording():
    obj = SimpleNamespace(_recording_lock=threading.Lock(), _episode_dir=None)
    assert Session.flag_episode(obj, "bad") is False


# ── End-to-end stdin decoding ────────────────────────────────────────────────
# The dispatch tests above call _advance/_rerecord directly, so they passed even
# while _read_keys silently swallowed every arrow key: sys.stdin.read(1) pulled
# the whole ESC[C into Python's text buffer, after which select() on the fd said
# "nothing waiting". These tests drive a REAL pty so the decode path is covered.


@pytest.fixture
def fake_stdin():
    """Replace sys.stdin with a cbreak pty; yields the master fd to type into."""
    master, slave = pty.openpty()
    tty.setcbreak(slave)
    real_stdin, sys.stdin = sys.stdin, os.fdopen(slave, "r")
    try:
        yield master
    finally:
        sys.stdin.close()
        sys.stdin = real_stdin
        os.close(master)


def _drive(fake_stdin, stub, keys: bytes, expect: int) -> None:
    """Run _read_keys against the pty until `expect` session calls land."""
    stop = threading.Event()
    t = threading.Thread(target=_read_keys, args=(stub, stop), daemon=True)
    t.start()
    try:
        os.write(fake_stdin, keys)
        _wait(stub, n=expect)
    finally:
        stop.set()
        t.join(timeout=2.0)


def test_read_keys_decodes_right_arrow_when_idle(fake_stdin):
    s = _Stub(recording=False, mappings={"1": "kit"})
    _drive(fake_stdin, s, b"\x1b[C", expect=1)
    assert s.calls == [("start",)], "RIGHT arrow must start a take"


def test_read_keys_decodes_right_arrow_when_recording(fake_stdin):
    s = _Stub(recording=True, mappings={"1": "kit"})
    _drive(fake_stdin, s, b"\x1b[C", expect=1)
    assert s.calls == [("end", True, "kit")], "RIGHT arrow must SAVE the take"


def test_read_keys_decodes_left_arrow(fake_stdin):
    s = _Stub(recording=True, mappings={"1": "kit"})
    _drive(fake_stdin, s, b"\x1b[D", expect=2)
    assert s.calls == [("end", False, None), ("start",)]


def test_read_keys_decodes_application_mode_arrow(fake_stdin):
    """Some terminals send ESC O C instead of ESC [ C."""
    s = _Stub(recording=False, mappings={"1": "kit"})
    _drive(fake_stdin, s, b"\x1bOC", expect=1)
    assert s.calls == [("start",)]


def test_read_keys_still_handles_plain_letter_keys(fake_stdin):
    s = _Stub(recording=False, mappings={"1": "kit"})
    _drive(fake_stdin, s, b"r", expect=1)
    assert s.calls == [("start",)]


def test_read_keys_ignores_up_down_arrows(fake_stdin):
    s = _Stub(recording=False, mappings={"1": "kit"})
    _drive(fake_stdin, s, b"\x1b[A\x1b[B", expect=1)  # expect never reached
    assert s.calls == []


def test_read_arrow_returns_empty_on_bare_escape(fake_stdin):
    """A lone ESC must not block waiting for a sequence that never arrives."""
    os.write(fake_stdin, b"\x1b")
    time.sleep(0.05)
    fd = sys.stdin.fileno()
    assert os.read(fd, 1) == b"\x1b"
    t0 = time.monotonic()
    assert _read_arrow(fd, timeout=0.05) == ""
    assert time.monotonic() - t0 < 1.0
