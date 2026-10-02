"""Faults NOT in the handoff catalogue — invented while attacking the system.

The catalogue in HANDOFF-CAMERA-HARDENING.md §5.1 is a floor, not a ceiling.
§5.5 asks for newly-invented faults on top of it, and these are the ones that
found real holes. Each docstring records what the code did BEFORE the fix,
because a regression test whose failure mode you cannot picture is a test people
delete.
"""

from __future__ import annotations

import threading
import time

import numpy as np
import pytest

from robots_realtime.sensors.cameras.camera import CameraData, CameraDriver
from robots_realtime.sensors.cameras.supervised_camera import (
    STATE_DEGRADED,
    STATE_FAILED,
    STATE_OK,
    CameraUnavailable,
    SupervisedCamera,
)

SHAPE = (48, 64)


class RingCamera(CameraDriver):
    """A device cycling a SMALL RING of buffers: A, B, A, B, ... forever.

    A realistic driver failure — a stuck DMA ring, or a UVC layer replaying its
    buffer queue. Every consecutive pair differs, so it looks perfectly alive.
    """

    def __init__(self, n: int = 2) -> None:
        self.frames = [np.full((*SHAPE, 3), 10 + i * 37, np.uint8) for i in range(n)]
        for i, f in enumerate(self.frames):
            f[0, 0, 1] = i
        self.i = 0
        self.device_path = "/dev/fake-ring"

    def read(self) -> CameraData:
        time.sleep(0.003)
        f = self.frames[self.i % len(self.frames)]
        self.i += 1
        return CameraData(images={"rgb": f}, timestamp=time.time() * 1000)

    def read_calibration_data_intrinsics(self):
        return {}

    def get_camera_info(self):
        return {}

    def stop(self) -> None:
        pass


class StaticSceneCamera(CameraDriver):
    """A REAL camera pointed at a motionless bench: one scene, sensor noise.

    The control for the ring test. Any stall detector that flags this is worse
    than useless — it would blank a working panel and teach the operator to
    ignore the indicator.
    """

    def __init__(self) -> None:
        rng = np.random.default_rng(7)
        self.base = rng.integers(40, 60, (*SHAPE, 3), dtype=np.uint8)
        self._rng = rng
        self.device_path = "/dev/fake-static"

    def read(self) -> CameraData:
        time.sleep(0.003)
        noise = self._rng.integers(-2, 3, self.base.shape)
        f = np.clip(self.base.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        return CameraData(images={"rgb": f}, timestamp=time.time() * 1000)

    def read_calibration_data_intrinsics(self):
        return {}

    def get_camera_info(self):
        return {}

    def stop(self) -> None:
        pass


def _cam(factory, **kw) -> SupervisedCamera:
    kw.setdefault("read_deadline_s", 0.4)
    kw.setdefault("freeze_timeout_s", 0.6)
    kw.setdefault("reopen_backoff_s", (0.05, 0.1))
    kw.setdefault("target_fps", 200.0)
    kw.setdefault("expected_shape", SHAPE)
    return SupervisedCamera(factory, name="invented", **kw)


def _drive(cam: SupervisedCamera, secs: float) -> tuple[str | None, set[str]]:
    """Read for `secs`; return (first failure reason, every state seen)."""
    t0 = time.monotonic()
    first: str | None = None
    seen: set[str] = set()
    while time.monotonic() - t0 < secs:
        try:
            cam.read()
        except CameraUnavailable as exc:
            if first is None:
                first = exc.reason
        seen.add(cam.state)
        time.sleep(0.003)
    return first, seen


# ── invented fault A — a ring of buffers ─────────────────────────────────────


@pytest.mark.parametrize("ring", [2, 3, 5, 8])
def test_a_camera_cycling_a_ring_of_buffers_is_caught(ring: int) -> None:
    """A freeze with extra steps, and it sailed straight through.

    The freeze check compared each frame only against the one before it, so
    A,B,A,B was "distinct" at every single step. Rings of 2, 3, 5 and 8 all
    reported ``ok`` indefinitely while carrying no new information whatsoever —
    a camera delivering a slideshow of the same two pictures at 30 Hz, with
    every indicator green.
    """
    cam = _cam(lambda: RingCamera(ring))
    try:
        cam.wait_until_open(5.0)
        first, seen = _drive(cam, 6.0)
        assert first == "stalled", f"a ring of {ring} was not detected at all"
        assert cam.state == STATE_FAILED, f"ring of {ring} ended in {cam.state!r}, not failed"
        assert cam.health()["healthy"] is False
    finally:
        cam.stop()


def test_a_motionless_scene_from_a_real_sensor_is_not_flagged() -> None:
    """The control. A false 'dead' is its own failure.

    Sensor noise makes every frame from a real camera distinct even when
    nothing in front of it moves, which is exactly why the stall test can afford
    to be strict.
    """
    cam = _cam(StaticSceneCamera)
    try:
        cam.wait_until_open(5.0)
        first, _ = _drive(cam, 4.0)
        assert first is None, f"a working camera was flagged as {first!r}"
        assert cam.state == STATE_OK
    finally:
        cam.stop()


def test_recovery_from_a_ring_requires_a_varied_window_not_two_good_frames() -> None:
    """Recovery has to be judged over the window too.

    A ring passes 'N consecutive distinct frames' trivially, so with only the
    streak test the camera climbed back to ``ok`` after every reopen and the
    incident counter reset with it. It never converged — the same flap as the
    ring-of-1 case, arriving by a different road.
    """
    holder: dict = {"ring": 2}

    def factory():
        return RingCamera(holder["ring"])

    cam = _cam(factory)
    try:
        cam.wait_until_open(5.0)
        _drive(cam, 5.0)
        assert cam.state == STATE_FAILED
    finally:
        cam.stop()


def test_a_healthy_camera_reaches_ok_quickly_from_cold() -> None:
    """Recovery must not require a full stall window, or startup cries wolf.

    The stall check needs 30 frames to judge a ring of buffers. Making the
    `ok` transition wait for that too left every node in `degraded (warmup)` for
    a second or more after startup — long enough for an auto-record episode,
    which begins 0.3 s after the session does, to be stamped degraded on every
    single clean take. A marker that fires on good episodes is worth less than
    no marker: it teaches the operator to ignore it.
    """
    cam = _cam(StaticSceneCamera)
    try:
        assert cam.wait_until_open(5.0)
        t0 = time.monotonic()
        deadline = t0 + 3.0
        while time.monotonic() < deadline and cam.state != STATE_OK:
            try:
                cam.read()
            except CameraUnavailable:
                pass
        elapsed = time.monotonic() - t0
        assert cam.state == STATE_OK, f"still {cam.state!r} after {elapsed:.2f}s from cold"
        assert elapsed < 1.0, f"took {elapsed:.2f}s to report ok from cold"
    finally:
        cam.stop()


# ── invented fault G — delivering, but nowhere near the configured rate ─────


class SlowCamera(CameraDriver):
    """Perfect frames, far too few of them. Found by starving a node of CPU.

    Every existing check passed: the frames were fresh, distinct, well-formed
    and correctly shaped. Nothing in the health model had an opinion about HOW
    MANY frames a camera owes you, so a camera running at a fraction of its
    configured rate reported ``ok`` indefinitely.
    """

    def __init__(self, period_s: float = 0.2) -> None:
        self.period_s = period_s
        self.n = 0
        self.device_path = "/dev/fake-slow"

    def read(self) -> CameraData:
        time.sleep(self.period_s)
        self.n += 1
        return CameraData(
            images={"rgb": np.full((*SHAPE, 3), self.n % 251, np.uint8)},
            timestamp=time.time() * 1000,
        )

    def read_calibration_data_intrinsics(self):
        return {}

    def get_camera_info(self):
        return {}

    def stop(self) -> None:
        pass


def test_a_camera_delivering_far_below_its_configured_rate_is_degraded() -> None:
    """5 Hz where 30 was configured must not read as healthy."""
    cam = _cam(lambda: SlowCamera(0.2), target_fps=30.0, slow_grace_s=0.5,
               read_deadline_s=2.0)
    try:
        cam.wait_until_open(5.0)
        # Drive reads for a fixed span. NOT "while state == ok": a camera that is
        # slow from cold never reaches ok in the first place, so that loop exits
        # before issuing a single read and the test proves nothing.
        _drive(cam, 6.0)
        h = cam.health()
        assert cam.state != STATE_OK, "a camera at 5 Hz of a configured 30 reported ok"
        assert h["reason"] == "slow"
        assert h["healthy"] is False
        assert 0 < h["fps"] < 15, f"reported fps {h['fps']} is not the delivered rate"
        assert h["target_fps"] == 30.0
    finally:
        cam.stop()


def test_a_camera_at_its_configured_rate_is_not_called_slow() -> None:
    """The control: a false 'slow' would blank a working panel."""
    cam = _cam(StaticSceneCamera, target_fps=30.0, slow_grace_s=0.5)
    try:
        cam.wait_until_open(5.0)
        first, _ = _drive(cam, 4.0)
        assert first is None
        assert cam.state == STATE_OK, f"a healthy camera was marked {cam.state!r}"
    finally:
        cam.stop()


def test_a_slow_camera_recovers_when_the_rate_comes_back() -> None:
    """Slow is degraded, never failed, and never triggers a reopen — the cause
    is load, not the device, and reopening under load makes it worse."""
    holder = {"period": 0.2}

    class Variable(SlowCamera):
        def read(self):
            self.period_s = holder["period"]
            return super().read()

    cam = _cam(Variable, target_fps=30.0, slow_grace_s=0.5, read_deadline_s=2.0)
    try:
        cam.wait_until_open(5.0)
        _drive(cam, 6.0)
        assert cam.state == STATE_DEGRADED, f"expected degraded, got {cam.state!r}"
        assert cam.health()["reason"] == "slow"
        reopens_while_slow = cam.health()["reopens"]

        holder["period"] = 1.0 / 120.0         # the load goes away
        deadline = time.monotonic() + 15.0
        while time.monotonic() < deadline and cam.state != STATE_OK:
            try:
                cam.read()
            except CameraUnavailable:
                pass
        assert cam.state == STATE_OK, "never recovered when the rate came back"
        assert cam.health()["reopens"] == reopens_while_slow, (
            "being slow triggered a reopen — that makes a loaded box worse"
        )
    finally:
        cam.stop()


# ── invented fault H — an absent device silences its own health topic ───────


def test_an_absent_device_is_not_opened_at_all() -> None:
    """Measured on the rig, and the fix for a real degradation.

    Opening an ABSENT RealSense makes five attempts, each enumerating every
    connected device's sensors and stream profiles. That is seconds of C code
    holding the GIL, which freezes every Python thread in the node — including
    the one publishing health. Measured with the top-down camera absent: FOUR
    health messages in twenty-five seconds, against two per second everywhere
    else. Nothing lied (consumers correctly reported `stale`), but "the camera
    that cannot be opened is also the camera whose health stops arriving" is a
    poor trade, and it hammers a controller that may be dying.

    A cheap presence check turns five expensive failures into zero.
    """
    opens = {"n": 0}

    def factory():
        opens["n"] += 1
        raise RuntimeError("No device connected")

    cam = _cam(factory, presence_check=lambda: False, give_up_after=2)
    try:
        assert cam.wait_until_open(5.0)
        _drive(cam, 2.0)
        h = cam.health()
        assert opens["n"] == 0, f"the factory was called {opens['n']}x for an absent device"
        assert h["state"] == STATE_FAILED
        assert h["reason"] == "device_absent"
        assert h["healthy"] is False
    finally:
        cam.stop()


def test_a_device_that_appears_is_opened_and_recovers() -> None:
    """The presence check must not become a way to never try again."""
    present = {"yes": False}

    cam = _cam(StaticSceneCamera, presence_check=lambda: present["yes"], give_up_after=2)
    try:
        cam.wait_until_open(5.0)
        _drive(cam, 1.5)
        assert cam.health()["reason"] == "device_absent"

        present["yes"] = True                      # the camera is plugged back in
        deadline = time.monotonic() + 10.0
        while time.monotonic() < deadline and cam.state != STATE_OK:
            try:
                cam.read()
            except CameraUnavailable:
                pass
        assert cam.state == STATE_OK, "never opened the device once it appeared"
    finally:
        cam.stop()


# ── invented fault B — the wall clock steps backwards under the publisher ────


def test_publisher_throttle_survives_a_backwards_clock() -> None:
    """Fault 13 in the catalogue, which the handoff flagged as suspected-not-proven.

    IT WAS REAL. ``Publisher.publish()`` throttled on ``time.time()`` and kept
    the last send time in a dict. One backwards step — an NTP correction, a
    manual ``date``, a VM resume — put that timestamp in the FUTURE, so every
    later ``now - last`` came out negative, compared as "< min_interval", and the
    topic was throttled off the bus for the entire length of the step. Silently:
    the writer kept recording and every rate display kept its last value.
    """
    from robots_realtime.runtime.transport import publisher as pub_mod

    sent: list[tuple] = []

    class FakeSock:
        def send_multipart(self, parts):
            sent.append(tuple(parts))

        def close(self, linger=0):
            pass

        def setsockopt(self, *a):
            pass

        def connect(self, *a):
            pass

    p = pub_mod.Publisher.__new__(pub_mod.Publisher)
    p._node_name = "camera_top"
    p._writer = None
    p._publish_freq = 30.0
    p._min_interval = 1.0 / 30.0
    p._last_sent = {}
    p._sock = FakeSock()
    p._ctrl_sock = FakeSock()

    assert p.publish("rgb", {"a": 1}) is True
    n_before = len(sent)

    # The wall clock now runs an hour backwards on every single call.
    real_time = time.time
    try:
        state = {"t": real_time()}

        def backwards() -> float:
            state["t"] -= 3600.0
            return state["t"]

        pub_mod.time.time = backwards            # type: ignore[assignment]
        for _ in range(50):
            time.sleep(0.002)
            p.publish("rgb", {"a": 2})
    finally:
        pub_mod.time.time = real_time            # type: ignore[assignment]

    assert len(sent) > n_before, (
        "a backwards wall clock wedged the publish throttle — the topic went "
        "silent while the writer kept recording"
    )


# ── invented fault C — the health topic itself is the liar ───────────────────


def test_the_bus_auditor_does_not_read_the_health_topic() -> None:
    """check_streams.py must stay INDEPENDENT of the thing it audits.

    Its entire value is that it can catch the health topic lying: it counts
    frames whose CONTENT differs, straight off the bus. If it ever starts
    reading ``<node>/health`` to decide whether a camera is well, the two
    signals can only ever agree — including when they are both wrong.
    """
    from pathlib import Path

    src = Path(__file__).resolve().parents[3] / "tools" / "check_streams.py"
    assert src.exists(), f"check_streams.py not found at {src}"
    body = src.read_text(encoding="utf-8")
    code = "\n".join(
        line for line in body.splitlines()
        if not line.lstrip().startswith("#")
    )
    assert "/health" not in code, (
        "check_streams.py now reads the health topic. That destroys the one "
        "property it has: independence from the signal under audit."
    )


# ── invented fault D — a driver that lies about its own timestamps ───────────


class BackwardsTimestampCamera(CameraDriver):
    """Frames are fine; the timestamps run backwards.

    Seen in the wild when a camera's clock is reset mid-stream. It matters
    because CameraNode divides the driver timestamp into the envelope ``ts``,
    and every staleness check downstream compares against that.
    """

    def __init__(self) -> None:
        self.t = time.time() * 1000
        self.n = 0
        self.device_path = "/dev/fake-clock"

    def read(self) -> CameraData:
        time.sleep(0.003)
        self.n += 1
        self.t -= 1000.0
        f = np.full((*SHAPE, 3), self.n % 251, np.uint8)
        return CameraData(images={"rgb": f}, timestamp=self.t)

    def read_calibration_data_intrinsics(self):
        return {}

    def get_camera_info(self):
        return {}

    def stop(self) -> None:
        pass


def test_backwards_driver_timestamps_do_not_wedge_the_supervisor() -> None:
    """The supervisor must pace itself on the monotonic clock, never the driver's.

    A driver whose timestamps run backwards must not be able to make read()
    block, nor make the camera look stale to its own supervisor — the frames
    themselves are perfectly good.
    """
    cam = _cam(BackwardsTimestampCamera)
    try:
        cam.wait_until_open(5.0)
        first, _ = _drive(cam, 3.0)
        assert first is None, f"good frames were rejected as {first!r}"
        assert cam.state == STATE_OK
        assert cam.health()["last_frame_age_s"] < 2.0, (
            "the health record's frame age came from the driver's clock rather "
            "than a monotonic one"
        )
    finally:
        cam.stop()


# ── invented fault E — the factory hands back a driver that is already dead ──


def test_a_factory_returning_a_dead_driver_does_not_loop_forever_claiming_ok() -> None:
    """Open 'succeeds' and every read fails. The camera must converge on failed,
    not oscillate between reopening and a hopeful degraded forever."""

    class DeadOnArrival(CameraDriver):
        device_path = "/dev/fake-doa"

        def read(self):
            raise RuntimeError("device disappeared between open and first read")

        def read_calibration_data_intrinsics(self):
            return {}

        def get_camera_info(self):
            return {}

        def stop(self):
            pass

    cam = _cam(DeadOnArrival, give_up_after=3)
    try:
        cam.wait_until_open(5.0)
        first, seen = _drive(cam, 5.0)
        assert first in ("read_error", "timeout")
        assert cam.health()["healthy"] is False
        assert STATE_OK not in seen, "a camera that never delivered a frame reported ok"
    finally:
        cam.stop()


# ── invented fault F — repeated reopen under a driver that leaks nothing ─────


def test_a_hundred_reopens_do_not_grow_the_thread_count() -> None:
    """Longer than the 20-cycle test, and specifically about the SUPERVISOR.

    Every reopen starts a pump thread and a close thread. If retirement is even
    slightly leaky, an overnight run with a flaky camera exhausts the process —
    and it would pass every functional test on the way there.
    """
    state = {"fail": True}

    class Flaky(CameraDriver):
        device_path = "/dev/fake-flaky"

        def __init__(self):
            self.n = 0

        def read(self):
            time.sleep(0.002)
            self.n += 1
            if state["fail"] and self.n > 2:
                raise RuntimeError("flaked")
            return CameraData(
                images={"rgb": np.full((*SHAPE, 3), self.n % 251, np.uint8)},
                timestamp=time.time() * 1000,
            )

        def read_calibration_data_intrinsics(self):
            return {}

        def get_camera_info(self):
            return {}

        def stop(self):
            pass

    cam = _cam(Flaky, reopen_backoff_s=(0.01,))
    try:
        cam.wait_until_open(5.0)
        baseline = threading.active_count()
        _drive(cam, 8.0)
        reopens = cam.health()["reopens"]
        assert reopens >= 20, f"only {reopens} reopens — the test did not exercise much"
        time.sleep(0.5)
        grown = threading.active_count() - baseline
        assert grown <= 3, f"thread count grew by {grown} over {reopens} reopens"
    finally:
        cam.stop()
