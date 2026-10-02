from __future__ import annotations

import threading
import time as _time_mod
from typing import Dict

import pytest

from robots_realtime.sensors.cameras.realsense_camera import RealSenseCamera


class FakeCameraInfo:
    serial_number = "serial_number"


class FakeStream:
    color = "color"
    depth = "depth"
    infrared = "infrared"


class FakeFormat:
    rgb8 = "rgb8"
    z16 = "z16"


class FakeProfile:
    def __init__(self, device: "FakeDevice | None" = None) -> None:
        self._device = device

    def get_device(self) -> "FakeDevice | None":
        return self._device


class FakeConfig:
    def enable_device(self, _device_id: str) -> None:
        pass

    def enable_stream(self, *_args) -> None:
        pass


class FakeFrames:
    def __init__(self, state: Dict[str, int | bool]) -> None:
        self._state = state

    def get_color_frame(self):
        return "color" if self._state.get("started") else None

    def get_infrared_frame(self):
        return "infrared" if self._state.get("started") else None


class FakePipeline:
    def __init__(self, state: Dict[str, int | bool], device: "FakeDevice | None" = None) -> None:
        self._state = state
        self._device = device
        self._stopped = False

    def start(self, _cfg: FakeConfig) -> FakeProfile:
        self._state["attempts"] = int(self._state["attempts"]) + 1
        if int(self._state["attempts"]) <= int(self._state["failures"]):
            raise RuntimeError("Device or resource busy")
        self._state["started"] = True
        return FakeProfile(self._device)

    def stop(self) -> None:
        self._stopped = True
        self._state["started"] = False

    def wait_for_frames(self, timeout_ms: int = 5000) -> FakeFrames:
        if not self._state.get("started"):
            raise RuntimeError("Frame didn't arrive within timeout")
        return FakeFrames(self._state)


class FakeDevice:
    def __init__(self, serial: str) -> None:
        self.serial = serial
        self.reset_count = 0

    def get_info(self, info: str) -> str:
        if info == FakeCameraInfo.serial_number:
            return self.serial
        return ""

    def query_sensors(self) -> list:
        # _start_pipeline_once() calls _configure_exposure(), which iterates the
        # device's sensors. The fake had no such method, so these tests died on
        # AttributeError before reaching their own assertions — a pre-existing
        # failure unrelated to the camera-hardening work. No sensors is a valid
        # answer and exercises the "nothing to configure" path.
        return []

    def hardware_reset(self) -> None:
        self.reset_count += 1


class FakeContext:
    def __init__(self, devices: list[FakeDevice]) -> None:
        self._devices = devices

    def query_devices(self) -> list[FakeDevice]:
        return self._devices


class FakeRs:
    camera_info = FakeCameraInfo
    config = FakeConfig
    format = FakeFormat
    stream = FakeStream

    def __init__(self, devices: list[FakeDevice], state: Dict[str, int | bool]) -> None:
        self._devices = devices
        self._state = state

    def context(self) -> FakeContext:
        return FakeContext(self._devices)

    def pipeline(self) -> FakePipeline:
        return FakePipeline(self._state, self._devices[0] if self._devices else None)


def _camera_for_recovery(*, serial: str = "123") -> RealSenseCamera:
    camera = object.__new__(RealSenseCamera)
    camera.device_id = serial
    camera._lock = threading.Lock()
    camera._stopped = False
    camera.intrinsic_data = {}
    return camera


# ------------------------------------------------------------------ #
# Kernel UVC busy behavior — no process killing
# ------------------------------------------------------------------ #


def test_busy_open_retries_without_killing_processes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Busy errors should only retry with backoff, never kill PIDs."""
    device = FakeDevice("123")
    state: Dict[str, int | bool] = {"attempts": 0, "failures": 99, "started": False}
    camera = _camera_for_recovery(serial="123")
    camera._rs = FakeRs([device], state)
    camera._width = 640
    camera._height = 480
    camera._depth_width = 640
    camera._depth_height = 480
    camera._pipeline = None
    camera._profile = None
    camera._align = None
    camera._depth_scale = 0.001
    camera._use_infrared = False
    camera.enable_depth = False
    camera.align_depth = False
    camera.fps = 30

    monkeypatch.setattr(camera, "_device_has_color_sensor", lambda: True)
    # The predicate is called _is_retryable_open_error; _is_busy_error has not
    # existed for some time, so this monkeypatch raised AttributeError and the
    # test never ran. Pre-existing, unrelated to the camera-hardening work.
    monkeypatch.setattr(camera, "_is_retryable_open_error", lambda exc: True)

    monkeypatch.setattr(_time_mod, "sleep", lambda _seconds: None)

    with pytest.raises(RuntimeError, match="busy"):
        camera._open_with_retries(max_retries=3)

    assert state["attempts"] == 3
    assert device.reset_count == 0


def test_busy_open_succeeds_after_retries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Busy errors should succeed once the device becomes available."""
    device = FakeDevice("123")
    state: Dict[str, int | bool] = {"attempts": 0, "failures": 1, "started": False}
    camera = _camera_for_recovery(serial="123")
    camera._rs = FakeRs([device], state)
    camera._width = 640
    camera._height = 480
    camera._depth_width = 640
    camera._depth_height = 480
    camera._pipeline = None
    camera._profile = None
    camera._align = None
    camera._depth_scale = 0.001
    camera._use_infrared = False
    camera.enable_depth = False
    camera.align_depth = False
    camera.fps = 30

    monkeypatch.setattr(camera, "_device_has_color_sensor", lambda: True)

    monkeypatch.setattr(_time_mod, "sleep", lambda _seconds: None)

    camera._open_with_retries(max_retries=3)

    assert state["started"] is True
    assert camera._pipeline is not None
    assert device.reset_count == 0


# ------------------------------------------------------------------ #
# Conservative cleanup — no default hardware_reset
# ------------------------------------------------------------------ #


def test_stop_clears_refs_even_if_pipeline_stop_raises() -> None:
    """stop() must clear internal references even when pipeline.stop() fails."""
    state: Dict[str, int | bool] = {"attempts": 1, "failures": 0, "started": True}
    device = FakeDevice("123")
    camera = _camera_for_recovery(serial="123")
    camera._rs = FakeRs([device], state)
    camera._pipeline = FakePipeline(state, device)
    camera._profile = FakeProfile(device)
    camera._align = "align_obj"
    camera._stopped = False

    def failing_stop():
        raise RuntimeError("pipeline stop failed")

    camera._pipeline.stop = failing_stop  # type: ignore[method-assign]

    camera.stop()

    assert camera._pipeline is None
    assert camera._profile is None
    assert camera._align is None
    assert camera._stopped is True
    assert device.reset_count == 0


def test_stop_does_not_call_hardware_reset_by_default() -> None:
    """stop() must not call hardware_reset() unless explicitly configured."""
    state: Dict[str, int | bool] = {"attempts": 1, "failures": 0, "started": True}
    device = FakeDevice("123")
    camera = _camera_for_recovery(serial="123")
    camera._rs = FakeRs([device], state)
    camera._pipeline = FakePipeline(state, device)
    camera._profile = FakeProfile(device)
    camera._align = None
    camera._stopped = False

    def failing_stop():
        raise RuntimeError("pipeline stop failed")

    camera._pipeline.stop = failing_stop  # type: ignore[method-assign]

    camera.stop()

    assert device.reset_count == 0
    assert camera._stopped is True


def test_stop_is_idempotent() -> None:
    """Calling stop() twice must not crash or double-release."""
    state: Dict[str, int | bool] = {"attempts": 1, "failures": 0, "started": True}
    device = FakeDevice("123")
    camera = _camera_for_recovery(serial="123")
    camera._rs = FakeRs([device], state)
    camera._pipeline = FakePipeline(state, device)
    camera._profile = FakeProfile(device)
    camera._align = None
    camera._stopped = False

    stop_calls = []
    original_stop = camera._pipeline.stop

    def tracking_stop():
        stop_calls.append(1)
        return original_stop()

    camera._pipeline.stop = tracking_stop  # type: ignore[method-assign]

    camera.stop()
    camera.stop()

    assert len(stop_calls) == 1
    assert camera._stopped is True


# ------------------------------------------------------------------ #
# atexit cleanup
# ------------------------------------------------------------------ #


def test_atexit_cleanup_calls_stop() -> None:
    """atexit handler calls stop() without raising."""
    state: Dict[str, int | bool] = {"attempts": 1, "failures": 0, "started": True}
    device = FakeDevice("123")
    camera = _camera_for_recovery(serial="123")
    camera._rs = FakeRs([device], state)
    camera._pipeline = FakePipeline(state, device)
    camera._profile = FakeProfile(device)
    camera._align = None
    camera._stopped = False

    stop_called = []
    original_stop = camera.stop

    def tracking_stop():
        stop_called.append(1)
        return original_stop()

    camera.stop = tracking_stop  # type: ignore[method-assign]

    camera._atexit_cleanup()
    assert len(stop_called) == 1
    assert camera._stopped is True


def test_atexit_cleanup_suppresses_exception() -> None:
    """atexit handler must never raise, even if stop() fails."""
    camera = _camera_for_recovery(serial="123")
    camera._stopped = False

    def raising_stop():
        raise RuntimeError("device gone")

    camera.stop = raising_stop  # type: ignore[method-assign]

    camera._atexit_cleanup()
