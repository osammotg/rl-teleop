"""Regression tests for SupervisedCamera — one test per fault in the catalogue.

Each test asserts THE INVARIANT from HANDOFF-CAMERA-HARDENING.md §5.4, not just
"the code did something":

  1. the fault is DETECTED within a bounded time (these tests use a 1 s read
     deadline and assert detection well inside 2 s),
  2. there is EXACTLY ONE outcome — auto-recovered, or loudly failed,
  3. NO signal reports success while the camera is broken,
  4. recovery is CLEAN (no thread/handle growth over repeated cycles).

If one of these ever starts failing, the corresponding silent failure is back.
Do not relax a threshold to make one pass — that is the finding, write it down.
"""

from __future__ import annotations

import threading
import time

import numpy as np
import pytest

from robots_realtime.sensors.cameras.fault_injection import (
    MODE_EMPTY,
    MODE_FROZEN,
    MODE_HANG,
    MODE_NONE,
    MODE_OK,
    MODE_RAISE,
    MODE_RESOLUTION_CHANGE,
    MODE_RET_FALSE,
    MODE_SLOW,
    MODE_WRONG_DTYPE,
    MODE_WRONG_SHAPE,
    FaultSpec,
    FaultyDriverFactory,
    fake_identity,
)
from robots_realtime.sensors.cameras.supervised_camera import (
    STATE_FAILED,
    STATE_OK,
    CameraUnavailable,
    SupervisedCamera,
)

# Deliberately tight so the whole suite stays fast, and still 5x slower than the
# 30 Hz frame period the real cameras run at.
DEADLINE = 0.4
FREEZE = 0.5


def _cam(spec: FaultSpec, **kw) -> SupervisedCamera:
    kw.setdefault("read_deadline_s", DEADLINE)
    kw.setdefault("freeze_timeout_s", FREEZE)
    kw.setdefault("reopen_backoff_s", (0.05, 0.1))
    kw.setdefault("target_fps", 200.0)
    kw.setdefault("expected_shape", spec.shape)
    return SupervisedCamera(FaultyDriverFactory(spec), name="cam_test", **kw)


def _read_until(cam: SupervisedCamera, predicate, timeout: float = 5.0):
    """Poll read() until ``predicate(result_or_exception)``; return it.

    Returns the first value satisfying the predicate, or raises AssertionError.
    """
    deadline = time.monotonic() + timeout
    last = None
    while time.monotonic() < deadline:
        try:
            last = cam.read()
        except CameraUnavailable as exc:
            last = exc
        if predicate(last):
            return last
        time.sleep(0.01)
    raise AssertionError(f"predicate never satisfied within {timeout}s; last={last!r}")


def _read_until_ok(cam: SupervisedCamera, timeout: float = 8.0):
    """Read until the camera has EARNED the ok state.

    Not the same as "one read succeeded". A recovering camera has to deliver
    several frames whose content differs before it may call itself healthy —
    one frame proves the handle is open, not that the sensor is producing.
    """
    deadline = time.monotonic() + timeout
    last = None
    while time.monotonic() < deadline:
        try:
            last = cam.read()
        except CameraUnavailable as exc:
            last = exc
        if cam.state == STATE_OK:
            return last
        time.sleep(0.005)
    raise AssertionError(f"camera never reached ok within {timeout}s (state={cam.state!r})")


def _wait_state(cam: SupervisedCamera, state: str, timeout: float = 5.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if cam.state == state:
            return
        try:
            cam.read()
        except CameraUnavailable:
            pass
        time.sleep(0.01)
    raise AssertionError(f"state never became {state!r} (still {cam.state!r}) within {timeout}s")


@pytest.fixture
def spec() -> FaultSpec:
    return FaultSpec(shape=(48, 64))


# ── the happy path, so the failures below mean something ─────────────────────


def test_healthy_camera_reads_and_reports_ok(spec: FaultSpec) -> None:
    cam = _cam(spec)
    try:
        assert cam.wait_until_open(5.0)
        data = _read_until_ok(cam)
        assert data.images["rgb"].shape == (48, 64, 3)
        h = cam.health()
        assert h["state"] == STATE_OK
        assert h["healthy"] is True
        assert h["last_frame_age_s"] < 1.0
    finally:
        cam.stop()


# ── fault 1 — ret=False forever (FAILURE #4, the known one) ──────────────────


def test_fault1_stale_handle_is_bounded_and_loud(spec: FaultSpec) -> None:
    """The original bug: read() spun forever and the node published nothing.

    The whole point is the WALL CLOCK here. Before the fix this call never
    returned at all.
    """
    cam = _cam(spec)
    try:
        cam.wait_until_open(5.0)
        _read_until_ok(cam)

        spec.set(MODE_RET_FALSE)
        t0 = time.monotonic()
        exc = _read_until(cam, lambda r: isinstance(r, CameraUnavailable), timeout=3.0)
        elapsed = time.monotonic() - t0

        assert isinstance(exc, CameraUnavailable)
        assert elapsed < 2.0, f"device loss took {elapsed:.2f}s to surface — budget is 2 s"
    finally:
        cam.stop()


def test_fault1_health_says_unhealthy_and_never_ok(spec: FaultSpec) -> None:
    cam = _cam(spec)
    try:
        cam.wait_until_open(5.0)
        _read_until_ok(cam)
        spec.set(MODE_RET_FALSE)
        _read_until(cam, lambda r: isinstance(r, CameraUnavailable), timeout=3.0)
        h = cam.health()
        # Invariant 3: NO signal reports success.
        assert h["healthy"] is False
        assert h["state"] != STATE_OK
        assert h["consecutive_failures"] >= 1
    finally:
        cam.stop()


# ── fault 2 — read blocks forever ────────────────────────────────────────────


def test_fault2_blocking_read_still_returns_within_the_deadline(spec: FaultSpec) -> None:
    cam = _cam(spec)
    try:
        cam.wait_until_open(5.0)
        _read_until_ok(cam)
        spec.set(MODE_HANG)
        t0 = time.monotonic()
        with pytest.raises(CameraUnavailable) as ei:
            for _ in range(20):
                cam.read()
        elapsed = time.monotonic() - t0
        assert ei.value.reason in ("timeout", "read_error")
        assert elapsed < 2.0, f"a wedged driver took {elapsed:.2f}s to surface"
        assert cam.health()["healthy"] is False
    finally:
        cam.stop()


def test_fault2_wedged_reader_is_counted_not_hidden(spec: FaultSpec) -> None:
    """An abandoned pump thread is reported, because it cannot be killed."""
    cam = _cam(spec)
    try:
        cam.wait_until_open(5.0)
        _read_until_ok(cam)
        spec.set(MODE_HANG)
        for _ in range(6):
            try:
                cam.read()
            except CameraUnavailable:
                pass
        time.sleep(0.3)
        # orphans_total, not orphaned_readers: this fake releases its wedged read
        # when the handle is closed, so the LIVE count drops back to zero within
        # milliseconds. A real wedge that keeps happening would look identical to
        # one that never happened if only the live count were reported.
        assert cam.health()["orphans_total"] >= 1
    finally:
        cam.stop()


# ── fault 3 — read raises (the RealSense shape) ──────────────────────────────


def test_fault3_raising_driver_does_not_kill_the_caller(spec: FaultSpec) -> None:
    cam = _cam(spec)
    try:
        cam.wait_until_open(5.0)
        spec.set(MODE_RAISE)
        exc = _read_until(cam, lambda r: isinstance(r, CameraUnavailable), timeout=3.0)
        # It surfaces as the SAME exception type as every other device-loss
        # shape. One contract for both drivers is the architectural fix.
        assert isinstance(exc, CameraUnavailable)
        assert cam.health()["healthy"] is False
    finally:
        cam.stop()


# ── fault 4 — frozen content, no error at all ────────────────────────────────


def test_fault4_frozen_frames_are_detected(spec: FaultSpec) -> None:
    """The hardest one: every other signal looks perfect."""
    cam = _cam(spec)
    try:
        cam.wait_until_open(5.0)
        _read_until_ok(cam)
        spec.set(MODE_FROZEN)
        exc = _read_until(
            cam,
            lambda r: isinstance(r, CameraUnavailable) and r.reason == "frozen",
            timeout=5.0,
        )
        assert exc.reason == "frozen"
        assert cam.health()["healthy"] is False
    finally:
        cam.stop()


def test_fault4_a_persistent_freeze_converges_to_failed_and_does_not_flap(
    spec: FaultSpec,
) -> None:
    """RED found this on the rig, and it is the nastiest kind of regression:
    the CURE reintroduced the disease.

    Detecting a freeze triggers a reopen. The reopened camera's first frame was
    accepted as proof of health, so the state went straight back to ``ok`` — and
    three seconds later froze again. A permanently frozen camera therefore
    oscillated ok -> reopening -> ok forever, spending MOST of its time
    reporting healthy while publishing one repeated image at a convincing 29 Hz.
    Every individual transition was correct; the loop was the bug.

    A freeze that survives reopening must converge on ``failed`` and stay there.
    """
    cam = _cam(spec)
    try:
        cam.wait_until_open(5.0)
        _read_until_ok(cam)
        spec.set(MODE_FROZEN)
        _wait_state(cam, STATE_FAILED, timeout=15.0)

        # ...and it must STAY failed, not blink back to ok on the next frame.
        seen = set()
        deadline = time.monotonic() + 3.0
        while time.monotonic() < deadline:
            try:
                cam.read()
            except CameraUnavailable:
                pass
            seen.add(cam.state)
            time.sleep(0.02)
        assert seen == {STATE_FAILED}, f"state flapped through {sorted(seen)}"
        assert cam.health()["healthy"] is False
    finally:
        cam.stop()


def test_a_recovering_camera_must_earn_ok_with_changing_content(spec: FaultSpec) -> None:
    """One frame proves the handle opened. It does not prove the sensor works."""
    cam = _cam(spec, min_healthy_frames=4)
    try:
        cam.wait_until_open(5.0)
        _read_until_ok(cam)
        spec.set(MODE_RAISE)
        _read_until(cam, lambda r: isinstance(r, CameraUnavailable), timeout=3.0)

        spec.set(MODE_OK)
        # The very first frame after the reopen must NOT be enough.
        first = _read_until(cam, lambda r: not isinstance(r, Exception), timeout=5.0)
        assert first is not None
        assert cam.state != STATE_OK, "one frame was treated as proof of health"

        _read_until(cam, lambda r: cam.state == STATE_OK, timeout=5.0)
        assert cam.state == STATE_OK
    finally:
        cam.stop()


def test_fault4_a_still_scene_within_the_grace_period_is_not_dropped(spec: FaultSpec) -> None:
    """A motionless bench is legitimate. Freezing must need time, not one frame."""
    cam = _cam(spec, freeze_timeout_s=30.0)
    try:
        cam.wait_until_open(5.0)
        _read_until_ok(cam)
        spec.set(MODE_FROZEN)
        for _ in range(5):
            cam.read()          # must not raise inside the grace period
        assert cam.state == STATE_OK
    finally:
        cam.stop()


# ── fault 5 — malformed frames ───────────────────────────────────────────────


@pytest.mark.parametrize("mode", [MODE_NONE, MODE_EMPTY, MODE_WRONG_SHAPE, MODE_WRONG_DTYPE])
def test_fault5_malformed_frames_are_rejected(spec: FaultSpec, mode: str) -> None:
    cam = _cam(spec)
    try:
        cam.wait_until_open(5.0)
        _read_until_ok(cam)
        spec.set(mode)
        exc = _read_until(cam, lambda r: isinstance(r, CameraUnavailable), timeout=4.0)
        assert exc.reason in ("malformed", "geometry", "timeout", "read_error")
        assert cam.health()["healthy"] is False
    finally:
        cam.stop()


# ── fault 6 — flapping: intermittently over the deadline ─────────────────────


def test_fault6_flapping_does_not_thrash_reopen(spec: FaultSpec) -> None:
    """A slow camera must degrade, not reopen itself to death.

    Reopening on every late frame turns a recoverable hiccup into a permanent
    outage: each reopen costs a device open, and the camera never gets far
    enough to deliver a frame before the next deadline expires.
    """
    spec.slow_s = DEADLINE * 2
    spec.slow_every = 4
    cam = _cam(spec)
    try:
        cam.wait_until_open(5.0)
        _read_until_ok(cam)
        opens_before = spec.opens
        spec.set(MODE_SLOW)
        for _ in range(25):
            try:
                cam.read()
            except CameraUnavailable:
                pass
        reopens = spec.opens - opens_before
        # Some reopening is correct; one per late frame is thrashing.
        assert reopens <= 8, f"flapping caused {reopens} reopens — that is thrashing"
    finally:
        cam.stop()


# ── fault 7 — device disappears, then comes back (THE recovery path) ─────────


def test_fault7_device_returns_and_the_camera_recovers_and_says_so(spec: FaultSpec) -> None:
    cam = _cam(spec)
    try:
        cam.wait_until_open(5.0)
        _read_until_ok(cam)
        reopens_before = cam.health()["reopens"]

        spec.heal_after(0.6)                   # gone now, back in 600 ms
        _read_until(cam, lambda r: isinstance(r, CameraUnavailable), timeout=3.0)
        assert cam.health()["healthy"] is False

        # Invariant 2: exactly one outcome, and here it is AUTO-RECOVERED.
        _read_until_ok(cam, timeout=8.0)
        h = cam.health()
        assert h["state"] == STATE_OK
        assert h["healthy"] is True
        # ...and it SAID so: the recovery is visible in the record, not silent.
        assert h["reopens"] > reopens_before
    finally:
        cam.stop()


# ── fault 8 — silent geometry change ─────────────────────────────────────────


def test_fault8_resolution_change_is_terminal_not_silently_accepted(spec: FaultSpec) -> None:
    """A camera that quietly renegotiates its profile poisons the dataset.

    Reopening cannot fix it, so it must stop pretending it might.
    """
    cam = _cam(spec)
    try:
        cam.wait_until_open(5.0)
        _read_until_ok(cam)
        spec.set(MODE_RESOLUTION_CHANGE)
        _wait_state(cam, STATE_FAILED, timeout=8.0)
        h = cam.health()
        assert h["reason"] == "geometry"
        assert h["terminal"] is True
        assert h["healthy"] is False
    finally:
        cam.stop()


# ── fault 9 — open fails at setup ────────────────────────────────────────────


def test_fault9_open_failure_is_loud_and_keeps_retrying(spec: FaultSpec) -> None:
    """'No device connected' used to kill the node process, invisibly."""
    spec.open_mode = "open_fail"
    cam = _cam(spec, give_up_after=2)
    try:
        assert cam.wait_until_open(5.0)
        _wait_state(cam, STATE_FAILED, timeout=6.0)
        h = cam.health()
        assert h["state"] == STATE_FAILED
        assert h["reason"] == "open_failed"
        assert h["healthy"] is False
        assert spec.open_failures >= 2

        # ...and it is still trying, so a camera that comes back recovers
        # without a session restart (the expensive operation on a brakeless arm).
        spec.open_mode = MODE_OK
        _read_until_ok(cam, timeout=10.0)
        assert cam.state == STATE_OK
    finally:
        cam.stop()


# ── fault 10 — opens fine, never delivers ────────────────────────────────────


def test_fault10_open_succeeds_but_no_frames(spec: FaultSpec) -> None:
    spec.open_mode = "warmup_silent"
    cam = _cam(spec)
    try:
        cam.wait_until_open(5.0)
        t0 = time.monotonic()
        exc = _read_until(cam, lambda r: isinstance(r, CameraUnavailable), timeout=4.0)
        assert time.monotonic() - t0 < 2.0
        assert exc.reason in ("timeout", "read_error")
        assert cam.health()["healthy"] is False
    finally:
        cam.stop()


# ── fault 11 — identity swap (the SN0001 trap) ───────────────────────────────


def test_fault11_a_different_camera_on_the_path_is_terminal(spec: FaultSpec) -> None:
    """Both wrist cameras report USB serial SN0001; the port is the only identity.

    A swap must refuse loudly. Silently accepting it mislabels every frame from
    that point on, and the mislabelling survives into the training set.
    """
    cam = _cam(spec, identity_fn=fake_identity)
    try:
        cam.wait_until_open(5.0)
        _read_until_ok(cam)
        assert cam.health()["identity"] == "camera-A"

        spec.identity = "camera-B"        # a different physical camera
        spec.set(MODE_RAISE)              # force a reopen so the check runs
        _wait_state(cam, STATE_FAILED, timeout=8.0)
        h = cam.health()
        assert h["reason"] == "identity_mismatch"
        assert h["terminal"] is True
    finally:
        cam.stop()


def test_fault11_a_replug_that_changes_devN_is_not_a_false_alarm(spec: FaultSpec) -> None:
    """/dev/videoN changes on every replug. Asserting on it would break recovery."""
    cam = _cam(spec, identity_fn=fake_identity)
    try:
        cam.wait_until_open(5.0)
        _read_until_ok(cam)
        spec.heal_after(0.4)              # a replug: same camera, new device node
        _read_until_ok(cam, timeout=8.0)
        assert cam.state == STATE_OK
        assert cam.health()["terminal"] is False
    finally:
        cam.stop()


# ── invariant 5 — recovery is clean over repeated cycles ─────────────────────


def test_repeated_fault_recover_cycles_do_not_leak_threads(spec: FaultSpec) -> None:
    """Twenty fault/recover cycles must not grow the thread count.

    This is the check that catches a supervisor which reopens correctly but
    leaves a pump behind every time — it would pass every other test in this
    file and take the rig down after an hour.
    """
    cam = _cam(spec)
    try:
        cam.wait_until_open(5.0)
        _read_until_ok(cam)
        baseline = threading.active_count()

        for _ in range(20):
            spec.set(MODE_RAISE)
            _read_until(cam, lambda r: isinstance(r, CameraUnavailable), timeout=3.0)
            spec.set(MODE_OK)
            _read_until_ok(cam, timeout=5.0)

        time.sleep(0.5)
        grown = threading.active_count() - baseline
        assert grown <= 2, f"thread count grew by {grown} over 20 fault/recover cycles"
        assert cam.state == STATE_OK
        assert cam.health()["reopens"] >= 20
    finally:
        cam.stop()


def test_stop_is_clean_and_idempotent(spec: FaultSpec) -> None:
    cam = _cam(spec)
    cam.wait_until_open(5.0)
    _read_until_ok(cam)
    before = threading.active_count()
    cam.stop()
    cam.stop()
    time.sleep(0.3)
    assert threading.active_count() <= before


def test_health_record_is_msgpack_safe(spec: FaultSpec) -> None:
    """Health goes on the bus, so it must survive serialization unchanged."""
    from robots_realtime.runtime.transport.serialization import pack, unpack

    cam = _cam(spec)
    try:
        cam.wait_until_open(5.0)
        _read_until_ok(cam)
        h = cam.health()
        assert unpack(pack(h))["state"] == h["state"]
    finally:
        cam.stop()
