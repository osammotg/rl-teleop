from __future__ import annotations

import json
import sys

import pytest

from robots_realtime import rr_session_cli
from robots_realtime.runtime import config
from robots_realtime.runtime.session import Session


class FakeSession:
    def __init__(self) -> None:
        self.events: list[str] = []
        self.save_root = "recordings"

    def start(self) -> None:
        self.events.append("start")

    def wait(self) -> None:
        self.events.append("wait")

    def stop(self) -> None:
        self.events.append("stop")


def test_no_tui_shutdown_stops_session_before_forced_exit(monkeypatch: pytest.MonkeyPatch) -> None:
    session = FakeSession()

    monkeypatch.setattr(sys, "argv", ["rr-session", "session.yaml", "--no-tui"])
    monkeypatch.setattr(rr_session_cli.os.path, "exists", lambda _path: True)
    monkeypatch.setattr(config, "load_session", lambda _path, **_kwargs: session)

    def fake_exit(code: int) -> None:
        session.events.append(f"exit:{code}")
        raise SystemExit(code)

    monkeypatch.setattr(rr_session_cli.os, "_exit", fake_exit)

    with pytest.raises(SystemExit) as exc_info:
        rr_session_cli.main()

    assert exc_info.value.code == 0
    assert session.events == ["start", "wait", "stop", "exit:0"]


def test_session_meta_includes_instruction(tmp_path) -> None:
    session = Session(
        nodes=[],
        save_root=tmp_path,
        instruction="push the box to the right",
    )

    session.start_episode()
    episode_dir = session.end_episode(save=True)

    assert episode_dir is not None
    meta_path = episode_dir / "session_meta.json"
    meta = json.loads(meta_path.read_text())
    assert meta["instruction"] == "push the box to the right"


def test_session_meta_uses_save_time_instruction(tmp_path) -> None:
    session = Session(
        nodes=[],
        save_root=tmp_path,
        instruction_mappings={"1": "push the box to the right"},
    )

    session.start_episode()
    episode_dir = session.end_episode(
        save=True,
        instruction=session.instruction_mappings["1"],
    )

    assert episode_dir is not None
    meta_path = episode_dir / "session_meta.json"
    meta = json.loads(meta_path.read_text())
    assert meta["instruction"] == "push the box to the right"
    assert meta["instruction_mappings"] == {"1": "push the box to the right"}


def test_yaml_instruction_mappings_are_applied(tmp_path) -> None:
    config_path = tmp_path / "session.yaml"
    config_path.write_text(
        """
version: "1"
session:
  save_root: recordings
  instructions:
    1: push the box to the right
    2: pull the box to the left
nodes: []
"""
    )

    session = config.load_session(str(config_path))

    assert session.instruction_mappings == {
        "1": "push the box to the right",
        "2": "pull the box to the left",
    }
