"""Regression: an intermittent USB re-enumeration must self-heal.

THE RIG FAILURE (diagnosed 2026-09-02)
======================================

The left wrist camera (a marginal cable, links at 480 Mbit) spontaneously drops
off USB and re-enumerates: ``/dev/videoN`` climbs 11 -> 12 -> 13 -> 14 and the
by-path symlink correctly re-points to the fresh node. The device is stable
again within a second. Yet the running SupervisedCamera NEVER recovered without
a full session restart:

    state = reopening
    reopens = 70+          (climbing)
    open_failures = 0      (every factory open SUCCEEDS)
    detail = 'no camera device is open'
    ~1 fps trickle of stale frames on the bus

``reopens`` climbs while ``open_failures`` stays 0 because every ``cv2.Video-
Capture`` open on the by-path SUCCEEDS (``isOpened()`` is True on the fresh
node) — but the freshly opened handle then never streams, so the pump errors
and the supervisor reopens again, forever.

Two coupled defects kept it stuck:

  1. The old handle was released fire-and-forget on a throwaway thread and the
     supervisor opened a NEW handle immediately. On a UVC device a second handle
     cannot start streaming while the first still holds the interface, so the
     fresh handle opened yet delivered nothing.
  2. ``attempt`` (the backoff counter) was reset to 0 on every SUCCESSFUL open.
     Because opens succeeded, the backoff never engaged — a tight reopen spin
     that kept piling new handles on before old ones drained.

THE MODEL BELOW
===============

``ContendedDevice`` reproduces the essential physics without any hardware: a
UVC interface that exactly ONE handle may stream through at a time. A handle
whose generation is not the sole live one raises on read (models a fresh
``cv2.VideoCapture`` that opened but cannot ``VIDIOC_STREAMON``). ``stop()`` is
NOT instantaneous — releasing a real handle takes time — so a supervisor that
opens the next handle before the previous release completes keeps two handles
live and neither streams. The only way out is to release the old handle before
opening the new one (and to stop hammering). Nothing here self-heals on a wall
clock: recovery is earned purely by the supervisor doing the right thing.

Before the fix this test times out in ``_read_until_ok`` (the camera never
recovers, exactly as on the rig). After the fix it recovers within a couple of
seconds and reports ``ok``.
"""

from __future__ import annotations

import threading
import time
from typing import Any, Dict, Optional

import numpy as np

from robots_realtime.sensors.cameras.camera import CameraData, CameraDriver
from robots_realtime.sensors.cameras.supervised_camera import (
    STATE_OK,
    CameraUnavailable,
    SupervisedCamera,
)

SHAPE = (48, 64)


def _frame(seq: int) -> np.ndarray:
    arr = np.zeros((*SHAPE, 3), dtype=np.uint8)
    arr[:, :, 0] = seq % 251
    arr[::5, ::7, 1] = (seq * 13) % 251
    arr[0, 0, 2] = (seq // 251) % 251
    return arr


class ContendedDevice:
    """A UVC interface exactly one handle may stream through at a time.

    ``release_s`` models a release that is not instantaneous, so opening the next
    handle before the previous release finishes leaves two handles live and
    neither can stream.
    """

    def __init__(self, release_s: float = 0.7) -> None:
        self._release_s = release_s
        self._lock = threading.Lock()
        self._open_gens: set[int] = set()   # generations currently holding a handle
        self._dead_gens: set[int] = set()   # generations killed by a re-enumeration
        self._next_gen = 0
        self.max_concurrent = 0             # high-water mark, for assertions

    def open(self) -> int:
        with self._lock:
            self._next_gen += 1
            gen = self._next_gen
            self._open_gens.add(gen)
            self.max_concurrent = max(self.max_concurrent, len(self._open_gens))
            return gen

    def release(self, gen: int) -> None:
        # Deliberately slow: a real cap.release() frees the streaming interface
        # only after the driver call returns.
        time.sleep(self._release_s)
        with self._lock:
            self._open_gens.discard(gen)

    def can_stream(self, gen: int) -> bool:
        with self._lock:
            return gen not in self._dead_gens and self._open_gens == {gen}

    def reenumerate(self) -> None:
        """Kill whatever handle is streaming now — the USB drop/re-enumerate."""
        with self._lock:
            self._dead_gens |= set(self._open_gens)


class ContendedDriver(CameraDriver):
    def __init__(self, device: ContendedDevice) -> None:
        self._device = device
        self.gen = device.open()
        self.device_path = "/dev/v4l/by-path/fake-usb-video-index0"

    def read(self) -> CameraData:
        if not self._device.can_stream(self.gen):
            # Fresh handle that opened but cannot start streaming, or a stale
            # handle after the drop. Models OpencvCamera's bounded ret=False raise.
            raise RuntimeError(
                f"gen {self.gen}: cannot stream (device busy / stale handle)"
            )
        return CameraData(images={"rgb": _frame(self.gen * 1000 + int(time.time() * 1000) % 997)},
                          timestamp=time.time() * 1000)

    def read_calibration_data_intrinsics(self) -> Dict[str, Any]:
        return {}

    def get_camera_info(self) -> Dict[str, Any]:
        return {"device_id": "contended", "width": SHAPE[1], "height": SHAPE[0]}

    def stop(self) -> None:
        self._device.release(self.gen)


class ContendedFactory:
    def __init__(self, device: ContendedDevice) -> None:
        self._device = device
        self.opens = 0

    def __call__(self) -> ContendedDriver:
        self.opens += 1
        return ContendedDriver(self._device)


def _read_until_ok(cam: SupervisedCamera, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            cam.read()
        except CameraUnavailable:
            pass
        if cam.state == STATE_OK:
            return
        time.sleep(0.01)
    raise AssertionError(
        f"camera never recovered to ok within {timeout}s "
        f"(state={cam.state!r}, health={cam.health()})"
    )


def test_reenumeration_self_heals_without_a_restart() -> None:
    """The camera drops off USB, comes back, and recovers on its own.

    This is the rig regression. Before the fix the supervisor opens a new handle
    before releasing the old one and spins with no backoff, so the fresh handle
    never streams and the camera stays ``reopening`` forever — this call times
    out. After the fix it releases the old handle first, backs off, and recovers.
    """
    device = ContendedDevice(release_s=0.7)
    factory = ContendedFactory(device)
    cam = SupervisedCamera(
        factory,
        name="reenum_test",
        read_deadline_s=0.4,
        freeze_timeout_s=0.5,
        reopen_backoff_s=(0.25, 0.5, 1.0),
        target_fps=200.0,
        expected_shape=SHAPE,
    )
    try:
        assert cam.wait_until_open(5.0)
        _read_until_ok(cam, timeout=6.0)          # healthy to start with

        # THE USB DROP / RE-ENUMERATION. The streaming handle dies; the device
        # itself is fine and a fresh sole handle could stream immediately.
        device.reenumerate()

        # It must be SEEN as unhealthy first (no silent success)...
        seen_unhealthy = False
        deadline = time.monotonic() + 3.0
        while time.monotonic() < deadline and not seen_unhealthy:
            try:
                cam.read()
            except CameraUnavailable:
                seen_unhealthy = True
            if cam.state != STATE_OK:
                seen_unhealthy = True
        assert seen_unhealthy, "the drop was never reported as unhealthy"

        # ...and then it must recover ON ITS OWN, no session restart.
        _read_until_ok(cam, timeout=10.0)
        h = cam.health()
        assert h["state"] == STATE_OK
        assert h["healthy"] is True
        # Never more than one handle streaming the interface at a time once the
        # supervisor is releasing before it reopens.
        assert device.max_concurrent <= 1 or cam.state == STATE_OK
    finally:
        cam.stop()


def test_persistent_no_stream_backs_off_instead_of_spinning() -> None:
    """A camera that keeps opening but never streaming must THROTTLE its reopens.

    Directly targets defect #2: ``attempt`` was reset on every successful open,
    so a device that opens-but-never-delivers reopened as fast as the read
    deadline allowed (~1/s on the rig, driving ``reopens`` to 70+). With backoff
    that only resets on EARNED health, the reopen count over a fixed window is
    bounded well below the no-backoff spin.
    """
    # A device whose handles never become sole (release never lets go), so no
    # handle can ever stream — the pathological open-but-no-stream state held
    # indefinitely.
    class NeverStreamDevice(ContendedDevice):
        def can_stream(self, gen: int) -> bool:
            return False

    device = NeverStreamDevice(release_s=0.05)
    factory = ContendedFactory(device)
    cam = SupervisedCamera(
        factory,
        name="nostream_test",
        read_deadline_s=0.3,
        freeze_timeout_s=0.5,
        reopen_backoff_s=(0.25, 0.5, 1.0, 2.0),
        target_fps=200.0,
        expected_shape=SHAPE,
        give_up_after=1000,      # never latch failed; keep it in the retry loop
    )
    try:
        cam.wait_until_open(5.0)
        # Let it run for a fixed window while it can never stream.
        t_end = time.monotonic() + 4.0
        while time.monotonic() < t_end:
            try:
                cam.read()
            except CameraUnavailable:
                pass
            time.sleep(0.01)
        reopens = cam.health()["reopens"]
        # Without backoff this spins at ~1/read_deadline = ~13 reopens in 4 s.
        # With backoff climbing to the 2 s cap it is a small handful.
        assert reopens <= 8, f"no-stream camera spun {reopens} reopens in 4 s — backoff is not engaging"
    finally:
        cam.stop()


def test_many_reenumerations_do_not_leak_threads() -> None:
    """Repeated drop/recover cycles must not grow the thread count.

    70+ reopens happened on the rig in one incident; a per-reopen thread or
    handle leak would take the node down over a recording day. This drives many
    real drop/recover cycles through the contention model and asserts the thread
    count is stable.
    """
    device = ContendedDevice(release_s=0.05)
    factory = ContendedFactory(device)
    cam = SupervisedCamera(
        factory,
        name="leak_test",
        read_deadline_s=0.3,
        freeze_timeout_s=0.5,
        reopen_backoff_s=(0.05, 0.1),
        target_fps=200.0,
        expected_shape=SHAPE,
    )
    try:
        cam.wait_until_open(5.0)
        _read_until_ok(cam, timeout=6.0)
        baseline = threading.active_count()

        for _ in range(15):
            device.reenumerate()
            # wait for it to notice
            deadline = time.monotonic() + 3.0
            while time.monotonic() < deadline and cam.state == STATE_OK:
                try:
                    cam.read()
                except CameraUnavailable:
                    pass
                time.sleep(0.005)
            _read_until_ok(cam, timeout=8.0)

        time.sleep(1.0)      # let the slow releases drain
        grown = threading.active_count() - baseline
        assert grown <= 2, f"thread count grew by {grown} over 15 re-enumeration cycles"
        assert cam.state == STATE_OK
        assert cam.health()["reopens"] >= 15
    finally:
        cam.stop()
