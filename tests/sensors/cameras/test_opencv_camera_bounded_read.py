"""Failure #4, driven through the REAL production OpencvCamera.read().

Only the ``cv2.VideoCapture`` handle is faked. Everything else — the retry loop,
the deadline, the exception, the timestamp arithmetic — is the shipping code.
That distinction is the whole reason the original investigation's reproduction
was believed: faking the driver tests the fake.

Before the fix, ``test_stale_handle_raises_instead_of_spinning_forever`` would
never have completed. It would have sat inside ``read()`` calling ``cap.read()``
several hundred times a second until pytest was killed — which is exactly what
the wrist camera did to the rig, for hours, while the TUI showed 29.5 Hz.
"""

from __future__ import annotations

import time

import numpy as np
import pytest

from robots_realtime.sensors.cameras.fault_injection import (
    MODE_NONE,
    MODE_OK,
    MODE_RAISE,
    MODE_RET_FALSE,
    FaultSpec,
    FaultyCapture,
)
from robots_realtime.sensors.cameras.opencv_camera import (
    OpencvCamera,
    OpencvCameraReadError,
)


def _camera(spec: FaultSpec, read_timeout_s: float = 0.3) -> OpencvCamera:
    """Build an OpencvCamera without touching a device node.

    ``__post_init__`` opens the device, so it is bypassed with
    ``__new__`` + explicit field assignment; every field ``read()`` touches is
    set here, and ``cap`` is the fake.
    """
    cam = OpencvCamera.__new__(OpencvCamera)
    cam.device_path = "/dev/video-fake"
    cam.camera_type = "fake"
    cam.image_transfer_time_offset = 80
    cam.resolution = (spec.shape[1], spec.shape[0])
    cam.fps = 30
    cam.fourcc = None
    cam.v4l2_controls = None
    cam.auto_exposure_enabled = False
    cam.auto_exposure_target = 110.0
    cam.auto_exposure_deadband = 8.0
    cam.auto_exposure_min = 5
    cam.auto_exposure_max = 200
    cam.auto_exposure_speed = 0.25
    cam.auto_exposure_period_s = 0.5
    cam.read_timeout_s = read_timeout_s
    cam.name = "fake"
    cam._last_auto_exposure_s = 0.0
    cam._exposure = None
    cam.cap = FaultyCapture(spec)
    return cam


def test_healthy_read_returns_a_frame() -> None:
    spec = FaultSpec(shape=(48, 64))
    cam = _camera(spec)
    data = cam.read()
    assert data.images["rgb"].shape == (48, 64, 3)
    assert data.timestamp > 0


def test_stale_handle_raises_instead_of_spinning_forever() -> None:
    """THE regression test for failure #4.

    A stale UVC handle answers ``(False, None)`` forever without raising. The
    old loop had no timeout, no log, no raise and no bound, so this call never
    returned and the node stopped publishing entirely — including its heartbeat.
    """
    spec = FaultSpec(shape=(48, 64), mode=MODE_RET_FALSE)
    cam = _camera(spec, read_timeout_s=0.2)

    t0 = time.monotonic()
    with pytest.raises(OpencvCameraReadError) as ei:
        cam.read()
    elapsed = time.monotonic() - t0

    assert 0.15 < elapsed < 1.5, f"read() took {elapsed:.3f}s; the deadline was 0.2s"
    assert "stale" in str(ei.value)
    assert cam.cap.calls > 1, "it must still retry — one dropped USB frame is normal"


def test_a_transient_failure_is_ridden_out_not_escalated() -> None:
    """One bad read must NOT become a device-loss event.

    These webcams drop the occasional frame under bandwidth pressure. Raising on
    the first ``ret=False`` would turn normal USB behaviour into a reopen storm.
    """
    spec = FaultSpec(shape=(48, 64), mode=MODE_RET_FALSE)
    cam = _camera(spec, read_timeout_s=2.0)

    def _heal() -> None:
        time.sleep(0.05)
        spec.set(MODE_OK)

    import threading
    threading.Thread(target=_heal, daemon=True).start()

    data = cam.read()
    assert data.images["rgb"].shape == (48, 64, 3)


def test_ret_true_with_no_buffer_is_a_failure_not_a_crash() -> None:
    """Truncated MJPEG has produced ret=True with no frame on this rig.

    The old code went straight into np.ascontiguousarray(None) and died with a
    TypeError that said nothing about cameras.
    """
    spec = FaultSpec(shape=(48, 64), mode=MODE_NONE)
    cam = _camera(spec, read_timeout_s=0.2)
    with pytest.raises(OpencvCameraReadError) as ei:
        cam.read()
    assert "no frame" in str(ei.value)


def test_a_raising_capture_still_propagates() -> None:
    """A genuine cv2 error must not be swallowed by the retry loop."""
    spec = FaultSpec(shape=(48, 64), mode=MODE_RAISE)
    cam = _camera(spec, read_timeout_s=0.2)
    with pytest.raises(Exception) as ei:
        cam.read()
    assert not isinstance(ei.value, OpencvCameraReadError)


def test_deadline_is_monotonic_so_a_backwards_clock_cannot_extend_it(monkeypatch) -> None:
    """A wall-clock step backwards must not turn the deadline into a hang.

    ``read()`` still stamps frames with time.time() (consumers align that
    against recorded timestamps) but must never *bound* anything with it.
    """
    spec = FaultSpec(shape=(48, 64), mode=MODE_RET_FALSE)
    cam = _camera(spec, read_timeout_s=0.2)

    # Wall clock runs an hour backwards on every call. If the deadline used it,
    # this read would never expire.
    state = {"t": 1_000_000.0}

    def _going_backwards() -> float:
        state["t"] -= 3600.0
        return state["t"]

    monkeypatch.setattr(time, "time", _going_backwards)

    t0 = time.monotonic()
    with pytest.raises(OpencvCameraReadError):
        cam.read()
    assert time.monotonic() - t0 < 1.5
