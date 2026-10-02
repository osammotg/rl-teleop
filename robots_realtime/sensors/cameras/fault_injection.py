"""Hermetic camera fault injection — the RED catalogue, as code.

This is test scaffolding that ships inside the package on purpose: the same
fakes are used by ``tests/sensors/cameras/test_supervised_camera.py`` (the
regression suite), by ``tools/camera_soak.py --faults`` (the long soak), and by
anyone reproducing a failure by hand.  Keeping them importable is what makes
"the operator can re-run the whole thing in one command" true.

NOTHING HERE TOUCHES HARDWARE.  No ``/dev/video*``, no RealSense, no arm, no
CAN.  Every fault is produced by a fake driver or a fake ``cv2.VideoCapture``
handle, which is why this tier is safe to run even while a session owns the real
cameras (Mode B in the handoff).

THE CATALOGUE
=============

The mode names below map one-to-one onto the fault table in
``HANDOFF-CAMERA-HARDENING.md`` §5.1:

===========================  ===  ==============================================
mode                         #    what it simulates
===========================  ===  ==============================================
``ret_false``                1    stale UVC handle: cap.read() -> (False, None)
                                  forever.  THE known failure (#4).
``hang``                     2    read() blocks and never returns
``raise``                    3    read() raises (the RealSense shape)
``frozen``                   4    the same frame forever, no error at all
``none`` / ``empty`` /       5    malformed frames
``wrong_shape`` /
``wrong_dtype``
``slow``                     6    intermittently exceeds the read deadline
``gone``                     7    device disappears, returns after N seconds
``resolution_change``        8    silently negotiates a different geometry
``open_fail``                9    open raises ("No device connected")
``open_hang``                9    open blocks forever
``warmup_silent``            10   opens fine, then never delivers a frame
``identity_swap``            11   a DIFFERENT camera answers on this path
===========================  ===  ==============================================

Faults 12-16 (writer failure, backwards clock, dead broker, HWM drops, publish
raising) are not driver-level and are injected directly in the tests that cover
them.
"""

from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

import numpy as np

from robots_realtime.sensors.cameras.camera import CameraData, CameraDriver

MODE_OK = "ok"
MODE_RET_FALSE = "ret_false"
MODE_HANG = "hang"
MODE_RAISE = "raise"
MODE_FROZEN = "frozen"
MODE_NONE = "none"
MODE_EMPTY = "empty"
MODE_WRONG_SHAPE = "wrong_shape"
MODE_WRONG_DTYPE = "wrong_dtype"
MODE_SLOW = "slow"
MODE_GONE = "gone"
MODE_RESOLUTION_CHANGE = "resolution_change"
MODE_WARMUP_SILENT = "warmup_silent"


@dataclass
class FaultSpec:
    """Shared, mutable fault state.

    One instance is handed to the factory and to every driver it produces, so a
    test can flip a camera from healthy to broken and back WITHOUT rebuilding
    anything — which is what makes "device disappears, then returns" testable as
    one continuous run rather than as two separate ones.
    """

    mode: str = MODE_OK
    #: Faults for the OPEN path, applied by :class:`FaultyDriverFactory`.
    open_mode: str = MODE_OK
    #: How long a ``slow`` read takes.
    slow_s: float = 2.0
    #: Only every Nth read is slow when mode is ``slow`` (0 = every read).
    slow_every: int = 3
    #: Geometry handed out; ``resolution_change`` switches to ``alt_shape``.
    shape: Tuple[int, int] = (480, 640)
    alt_shape: Tuple[int, int] = (240, 320)
    #: Identity reported by :func:`fake_identity`; change it to swap cameras.
    identity: str = "camera-A"
    #: Wall-clock (monotonic) time after which ``gone`` heals itself.
    heal_at: Optional[float] = None
    #: Counters, for assertions.
    reads: int = 0
    opens: int = 0
    open_failures: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    _seq: int = 0

    def set(self, mode: str) -> None:
        with self._lock:
            self.mode = mode

    def heal_after(self, seconds: float) -> None:
        """Enter ``gone`` now and return to ``ok`` after ``seconds``."""
        with self._lock:
            self.mode = MODE_GONE
            self.heal_at = time.monotonic() + seconds

    def current_mode(self) -> str:
        with self._lock:
            if self.mode == MODE_GONE and self.heal_at is not None and time.monotonic() >= self.heal_at:
                self.mode = MODE_OK
                self.heal_at = None
            return self.mode


class FaultyDriverError(RuntimeError):
    """What an injected driver raises, standing in for a real driver error."""


def _frame(shape: Tuple[int, int], seq: int) -> np.ndarray:
    """A distinct frame per sequence number.

    Distinctness matters: the freeze detector compares consecutive frames, so a
    fake that returns a constant array would look frozen to every test rather
    than only to the freeze test.
    """
    h, w = shape
    arr = np.zeros((h, w, 3), dtype=np.uint8)
    arr[:, :, 0] = seq % 251
    arr[0, 0, 1] = (seq // 251) % 251
    # A little structure so the strided fingerprint has something to bite on.
    arr[::7, ::5, 2] = (seq * 13) % 251
    return arr


class FaultyDriver(CameraDriver):
    """A ``CameraDriver`` that fails on demand, in every shape the rig produces."""

    def __init__(self, spec: FaultSpec, name: str = "fake") -> None:
        self._spec = spec
        self.name = name
        self.device_path = f"/dev/fake/{name}"
        self._stopped = threading.Event()
        self._frozen_frame: Optional[np.ndarray] = None
        self._n = 0
        self._warmup_silent = spec.open_mode == MODE_WARMUP_SILENT

    # -- CameraDriver ----------------------------------------------------

    def read(self) -> CameraData:
        mode = self._spec.current_mode()
        with self._spec._lock:
            self._spec.reads += 1
            self._spec._seq += 1
            seq = self._spec._seq
        self._n += 1

        if self._warmup_silent:
            # Opened cleanly, then never delivers. Blocks until stopped so the
            # supervisor's deadline is the only thing that can end it.
            self._stopped.wait()
            raise FaultyDriverError("stopped")

        if mode == MODE_HANG:
            # The other hang shape: blocks forever. Waits on the stop event so
            # the test can still tear the fake down at the end.
            self._stopped.wait()
            raise FaultyDriverError("stopped while hung")

        if mode in (MODE_RAISE, MODE_GONE):
            raise FaultyDriverError(f"{self.name}: device not connected ({mode})")

        if mode == MODE_RET_FALSE:
            # The real OpencvCamera turns a persistent ret=False into a bounded
            # raise. Modelling it as a raise here keeps this fake honest about
            # what the production driver now does; test_opencv_camera.py drives
            # the actual (False, None) loop through the real read().
            raise FaultyDriverError(f"{self.name}: cap.read() returned ret=False (stale handle)")

        if mode == MODE_SLOW and (self._spec.slow_every <= 0 or self._n % self._spec.slow_every == 0):
            self._stopped.wait(self._spec.slow_s)

        if mode == MODE_NONE:
            return CameraData(images={"rgb": None}, timestamp=time.time() * 1000)  # type: ignore[dict-item]
        if mode == MODE_EMPTY:
            return CameraData(images={"rgb": np.zeros((0, 0, 3), np.uint8)}, timestamp=time.time() * 1000)
        if mode == MODE_WRONG_SHAPE:
            return CameraData(images={"rgb": np.zeros((480, 640), np.uint8)}, timestamp=time.time() * 1000)
        if mode == MODE_WRONG_DTYPE:
            return CameraData(
                images={"rgb": np.zeros((*self._spec.shape, 3), np.float32)},
                timestamp=time.time() * 1000,
            )
        if mode == MODE_RESOLUTION_CHANGE:
            return CameraData(
                images={"rgb": _frame(self._spec.alt_shape, seq)}, timestamp=time.time() * 1000
            )
        if mode == MODE_FROZEN:
            if self._frozen_frame is None:
                self._frozen_frame = _frame(self._spec.shape, seq)
            return CameraData(images={"rgb": self._frozen_frame}, timestamp=time.time() * 1000)

        return CameraData(images={"rgb": _frame(self._spec.shape, seq)}, timestamp=time.time() * 1000)

    def read_calibration_data_intrinsics(self) -> Dict[str, Any]:
        return {}

    def get_camera_info(self) -> Dict[str, Any]:
        return {"device_id": self.name, "width": self._spec.shape[1], "height": self._spec.shape[0]}

    def stop(self) -> None:
        self._stopped.set()


class FaultyDriverFactory:
    """Zero-arg callable that builds :class:`FaultyDriver`, failing on demand."""

    def __init__(self, spec: FaultSpec, name: str = "fake") -> None:
        self._spec = spec
        self._name = name

    def __call__(self) -> FaultyDriver:
        with self._spec._lock:
            self._spec.opens += 1
            open_mode = self._spec.open_mode
        if open_mode == "open_fail":
            with self._spec._lock:
                self._spec.open_failures += 1
            raise FaultyDriverError(f"{self._name}: No device connected")
        if open_mode == "open_hang":
            time.sleep(3600)
        return FaultyDriver(self._spec, self._name)


class SoakCamera(CameraDriver):
    """A YAML-constructible synthetic camera, for hardware-free rig soaks.

    WHY IT EXISTS. Tier B (real processes, SIGSTOP/SIGKILL, port squatting,
    broker death, cold start) needs real subprocesses but does NOT need real
    devices. Separating those two is what makes the process-level tier runnable
    while the operator's session owns all four physical cameras — Mode B in the
    handoff, where opening a RealSense a second time would steal it.

    Faults are driven through a FILE rather than through this object, because
    the driver lives in a node subprocess that the soak runner cannot reach
    in-process. Writing a mode name into ``fault_file`` flips the camera; an
    absent or empty file means healthy. That is also how the operator reproduces
    a fault by hand:

        echo frozen  > /tmp/soak_faults/camera_top
        echo ret_false > /tmp/soak_faults/camera_left
        : > /tmp/soak_faults/camera_top      # healthy again
    """

    def __init__(
        self,
        name: str = "soak",
        resolution: tuple[int, int] | list[int] = (640, 480),
        fps: float = 30.0,
        fault_file: Optional[str] = None,
        **_ignored: Any,
    ) -> None:
        self.name = name
        w, h = int(resolution[0]), int(resolution[1])
        self.resolution = (w, h)
        self._shape = (h, w)
        self.fps = float(fps) or 30.0
        self.device_path = f"soak://{name}"
        # SOAK_FAULT_DIR redirects the fault files without touching the config.
        #
        # Two soaks running at once — a long one and a round runner, say — both
        # read the fault_file path baked into the YAML, so each was silently
        # clearing the other's injected fault mid-check. Neither reported a
        # violation, and neither result meant anything: an experiment whose
        # independent variable is being edited by another process is not an
        # experiment. Each run now gets its own directory.
        if fault_file:
            override = os.environ.get("SOAK_FAULT_DIR")
            if override:
                fault_file = os.path.join(override, os.path.basename(fault_file))
        self.fault_file = fault_file
        self._seq = 0
        self._frozen: Optional[np.ndarray] = None
        self._next_t = time.monotonic()
        self._stopped = threading.Event()

    def _mode(self) -> str:
        if not self.fault_file:
            return MODE_OK
        try:
            with open(self.fault_file, encoding="utf-8") as f:
                return f.read().strip() or MODE_OK
        except FileNotFoundError:
            return MODE_OK
        except Exception:
            return MODE_OK

    def read(self) -> CameraData:
        # Pace like a real camera so the node's loop rate is realistic.
        self._next_t += 1.0 / self.fps
        delay = self._next_t - time.monotonic()
        if delay > 0:
            self._stopped.wait(delay)
        else:
            self._next_t = time.monotonic()

        mode = self._mode()
        self._seq += 1

        if mode == MODE_HANG:
            self._stopped.wait()
            raise FaultyDriverError("stopped while hung")
        if mode in (MODE_RAISE, MODE_GONE):
            raise FaultyDriverError(f"{self.name}: device not connected ({mode})")
        if mode == MODE_RET_FALSE:
            raise FaultyDriverError(f"{self.name}: cap.read() returned ret=False (stale handle)")
        if mode == MODE_FROZEN:
            if self._frozen is None:
                self._frozen = _frame(self._shape, self._seq)
            return CameraData(images={"rgb": self._frozen}, timestamp=time.time() * 1000)
        if mode == MODE_RESOLUTION_CHANGE:
            h, w = self._shape
            return CameraData(
                images={"rgb": _frame((h // 2, w // 2), self._seq)}, timestamp=time.time() * 1000
            )
        if mode == MODE_NONE:
            return CameraData(images={"rgb": None}, timestamp=time.time() * 1000)  # type: ignore[dict-item]

        self._frozen = None
        return CameraData(images={"rgb": _frame(self._shape, self._seq)}, timestamp=time.time() * 1000)

    def read_calibration_data_intrinsics(self) -> Dict[str, Any]:
        return {}

    def get_camera_info(self) -> Dict[str, Any]:
        return {
            "device_id": self.name,
            "width": self.resolution[0],
            "height": self.resolution[1],
            "fps": self.fps,
            "synthetic": True,
        }

    def stop(self) -> None:
        self._stopped.set()


def fake_identity(driver: CameraDriver) -> str:
    """Identity function for the fakes — reads the shared spec, so a test can
    swap the physical camera behind the path by assigning ``spec.identity``."""
    spec = getattr(driver, "_spec", None)
    return "" if spec is None else str(spec.identity)


# ── fake cv2.VideoCapture, for driving the REAL OpencvCamera.read() ───────────


class FaultyCapture:
    """Stand-in for ``cv2.VideoCapture`` so the PRODUCTION read() is under test.

    ``CAMERA-RELIABILITY-FINDINGS.md`` §1.1's reproduction works this way for a
    reason: faking the driver tests the fake, faking only the capture handle
    tests the shipping code. Assign one over ``OpencvCamera.cap`` and the real
    retry loop, the real deadline and the real exception are all exercised.
    """

    def __init__(self, spec: FaultSpec) -> None:
        self._spec = spec
        self.calls = 0
        self.released = False
        self._frozen: Optional[np.ndarray] = None

    def isOpened(self) -> bool:                                  # noqa: N802 (cv2 API)
        return self._spec.open_mode != "open_fail"

    def read(self):
        self.calls += 1
        mode = self._spec.current_mode()
        with self._spec._lock:
            self._spec._seq += 1
            seq = self._spec._seq
        if mode == MODE_RET_FALSE:
            return False, None
        if mode == MODE_RAISE:
            raise FaultyDriverError("capture exploded")
        if mode == MODE_NONE:
            return True, None
        if mode == MODE_FROZEN:
            if self._frozen is None:
                self._frozen = _frame(self._spec.shape, seq)
            return True, self._frozen
        return True, _frame(self._spec.shape, seq)

    def set(self, *_args, **_kwargs):
        return True

    def get(self, *_args, **_kwargs):
        return 0.0

    def release(self) -> None:
        self.released = True
