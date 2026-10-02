"""The bridge must stop serving a frame once it stops being evidence.

WHAT THIS PREVENTS. On 2026-08-10 the cockpit showed a wrist panel that looked
like a working camera and was one frame repeated forever:

    /cam/wristR    HTTP 200, 21 KB of valid JPEG, ~15 fps    <- looked healthy
    the bus        NOTHING. not one message.                 <- the actual truth

``CameraBridge`` returned the last envelope it had received, for as long as
anyone asked. Nothing in the picture said how old it was, so an operator could
not tell a live camera from a photograph of one.

Two independent gates now, because either alone has a hole:
  * envelope AGE — catches a publisher that stopped,
  * the camera's own HEALTH record — catches a publisher that is still sending
    at full rate and sending the same picture every time.
"""

from __future__ import annotations

import time

import numpy as np
import pytest

from robots_realtime.labeling.live_server import CameraBridge


class FakeSub:
    """Stand-in for Subscriber: a dict of topic -> envelope, set by the test."""

    def __init__(self) -> None:
        self.latest: dict[str, dict] = {}

    def get_latest(self, topic: str):
        return self.latest.get(topic)

    def close(self) -> None:
        pass


def _bridge(stale_after_s: float = 2.0) -> tuple[CameraBridge, FakeSub]:
    b = CameraBridge.__new__(CameraBridge)
    b._id_to_topic = {"top": "camera_top/rgb", "wristR": "camera_right/rgb"}
    b._health_topics = {"top": "camera_top/health", "wristR": "camera_right/health"}
    sub = FakeSub()
    b._sub = sub
    b._stale_after_s = stale_after_s
    b._started = time.time()
    return b, sub


def _frame_env(age_s: float = 0.0, value: int = 7) -> dict:
    img = np.full((8, 8, 3), value, dtype=np.uint8)
    return {"ts": time.time() - age_s, "src": "camera_top", "data": {"images": {"rgb": img}}}


def _health_env(healthy: bool, state: str = "ok", age_s: float = 0.0, reason: str = "") -> dict:
    return {
        "ts": time.time() - age_s,
        "data": {"state": state, "healthy": healthy, "reason": reason},
    }


def test_a_fresh_frame_is_served() -> None:
    b, sub = _bridge()
    sub.latest["camera_top/rgb"] = _frame_env(age_s=0.0)
    assert b.state("top")["state"] == "ok"
    assert b.frame("top") is not None
    assert b.jpeg("top") is not None


def test_a_stale_frame_is_not_served() -> None:
    """The original bug, in one assertion."""
    b, sub = _bridge(stale_after_s=2.0)
    sub.latest["camera_top/rgb"] = _frame_env(age_s=30.0)
    st = b.state("top")
    assert st["state"] == "stale"
    assert st["age_s"] >= 30.0
    assert b.frame("top") is None, "a 30-second-old frame was served as if it were live"
    assert b.jpeg("top") is None


def test_a_fresh_frame_from_an_unhealthy_camera_is_not_served() -> None:
    """Envelope freshness is NOT enough.

    A frozen camera publishes a brand-new message every 33 ms carrying the same
    picture. The age gate waves every one of them through — which is the very
    failure the bridge exists to stop, arriving by a different road.
    """
    b, sub = _bridge()
    sub.latest["camera_top/rgb"] = _frame_env(age_s=0.0)
    sub.latest["camera_top/health"] = _health_env(healthy=False, state="failed", reason="frozen")

    st = b.state("top")
    assert st["state"] == "unhealthy"
    assert st["camera"]["reason"] == "frozen"
    assert b.frame("top") is None
    assert b.jpeg("top") is None


def test_a_health_record_that_itself_went_stale_is_not_trusted() -> None:
    """SIGSTOP a camera node and it publishes nothing — including no health.

    RED found this on the rig: the last `ok` sat on the bus indefinitely and
    every health reader believed it. A health topic that cannot go stale is the
    same fossil the frame was.
    """
    b, sub = _bridge(stale_after_s=2.0)
    sub.latest["camera_top/rgb"] = _frame_env(age_s=0.0)
    sub.latest["camera_top/health"] = _health_env(healthy=True, state="ok", age_s=30.0)

    health = b.camera_health("top")
    assert health["stale"] is True
    assert health["healthy"] is False
    assert health["state"] == "stale"
    assert b.state("top")["state"] == "unhealthy"
    assert b.frame("top") is None


def test_an_unmapped_panel_is_unmapped_not_blank() -> None:
    """An unmapped id used to fall back to the `default` topic, so three panels
    rendered the top-down camera and looked perfectly healthy. A blank frame
    says 'no source'; a wrong frame asserts a source that isn't there."""
    b, _ = _bridge()
    st = b.state("handgelenk_links")
    assert st["state"] == "unmapped"
    assert b.frame("handgelenk_links") is None


def test_a_mapped_panel_with_no_traffic_says_no_data() -> None:
    b, _ = _bridge()
    assert b.state("wristR")["state"] == "no_data"
    assert b.frame("wristR") is None


def test_all_states_covers_every_mapped_panel() -> None:
    b, sub = _bridge()
    sub.latest["camera_top/rgb"] = _frame_env(age_s=0.0)
    allst = b.all_states()
    assert set(allst["cams"]) == {"top", "wristR"}
    assert allst["cams"]["top"]["state"] == "ok"
    assert allst["cams"]["wristR"]["state"] == "no_data"
    assert allst["stale_after_s"] == 2.0


@pytest.mark.parametrize("age", [0.0, 1.9])
def test_normal_jitter_does_not_blank_a_working_panel(age: float) -> None:
    """A false 'dead' is its own failure — it would send the operator hunting a
    camera that is fine, and teach them to ignore the indicator."""
    b, sub = _bridge(stale_after_s=2.0)
    sub.latest["camera_top/rgb"] = _frame_env(age_s=age)
    assert b.state("top")["state"] == "ok"
    assert b.frame("top") is not None
