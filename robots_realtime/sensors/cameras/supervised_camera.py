"""SupervisedCamera — the liveness contract for any CameraDriver.

WHY THIS EXISTS
===============

Two camera drivers on this rig fail in opposite ways for the same physical
event, and both are invisible to the operator:

  * ``OpencvCamera.read()`` used to contain an unbounded, silent retry loop.  A
    UVC handle that has gone stale returns ``(False, None)`` forever without
    raising, so the loop never exited, ``Node.step()`` never returned,
    ``Node._tick()`` never ran, and the node published NOTHING — not even its
    ``_step_hz`` heartbeat — while staying alive and leaving a valid mp4 on
    disk.  That is failure #4 in ``CAMERA-RELIABILITY-FINDINGS.md``.

  * ``RealSenseCamera.read()`` raises, ``CameraNode.step()`` re-raised, and the
    node process died — loud only inside ``/tmp/rr_logs_*/``, invisible
    everywhere else.

One cause, two behaviours, both silent.  The architectural defect is that there
was no shared contract about what a camera must do when it loses its device.
This class is that contract, and it lives in a *driver wrapper* rather than in
``CameraNode`` on purpose: the same object works unchanged whether cameras run
inside the session (today) or inside a standalone daemon on ``rr-bus`` (later).

THE CONTRACT
============

1. **No read blocks forever.**  ``read()`` returns within ``read_deadline_s`` or
   raises :class:`CameraUnavailable`.  A read that exceeds its deadline is a
   device-loss event, not a slow frame.
2. **Exactly one outcome per fault**: the camera auto-recovers, or it reports
   ``failed`` loudly.  Never a third thing, and never silence.
3. **Health is observable**, via :meth:`health`, which ``CameraNode`` publishes
   on ``<node>/health``.

HOW THE DEADLINE IS ENFORCED
============================

``cv2.VideoCapture.read()`` is a blocking C call that cannot be interrupted from
Python.  Neither can ``rs.pipeline.wait_for_frames()``.  So the deadline cannot
be implemented by "call read() with a timeout" — it has to be implemented by
calling read() somewhere we are allowed to walk away from.

A dedicated *pump* thread owns the driver and does nothing but read frames into
a one-slot mailbox.  ``read()`` waits on that mailbox with a deadline and
returns whatever is there.  If the pump is wedged inside the driver, ``read()``
still returns on time, the supervisor thread opens a NEW driver, and the wedged
thread is abandoned.

Abandoning a thread is not free and this class does not pretend otherwise: the
orphan still holds the old device handle until the driver call it is stuck in
eventually returns.  Orphans are COUNTED and reported in the health record as
``orphaned_readers``; past ``max_orphans`` the camera goes ``failed`` rather
than quietly accumulating threads.  A leak that is reported is a very different
thing from a leak that is not, and Python offers no way to kill a thread stuck
in a C call.  Note the known real-world failure (#4) does not hang at all — it
spins returning ``(False, None)`` — so the orphan path is the rarer one.

WHAT COUNTS AS A FAILURE
========================

- the pump raised (RealSense path, and any driver that reports honestly)
- no fresh frame within ``read_deadline_s`` (the hang, and the silent-retry spin)
- the frame is None / empty / not (H,W,3) / not uint8  (malformed)
- the frame is bit-identical to the previous one for ``freeze_timeout_s``
  (frozen content — the failure the cockpit lied about, and the only one that
  looks perfect from every other angle)
- the frames are DISTINCT but repeating: a device cycling a small ring of
  buffers, A,B,A,B forever, which passes any consecutive-frame comparison and
  carries no new information. Judged over a window, not a pair.
- frames arrive, correctly formed, at a small fraction of the configured rate
  (``degraded (slow)``) — a camera nobody should record from, which every other
  check called healthy
- the frame geometry is not the configured geometry (silent profile change)
- the device identity changed across a reopen (the two wrist cameras share USB
  serial ``SN0001``, so the port path is their only identity)

Two of those — the ring, and the rate — were not in the original fault catalogue.
They were invented while attacking the finished code, and both were real holes.
The lesson is in the shape they share with the bug this class was written for:
every one of these failures produced output that looked, to any check narrow
enough, exactly like success.

RECOVERY POLICY
===============

``give_up_after`` consecutive failed OPEN attempts moves the state to ``failed``
and logs at ERROR.  It does **not** stop the supervisor: retries continue at the
capped backoff, because a camera that comes back after five minutes must
recover without a session restart (a session restart on a brakeless arm is the
expensive operation on this rig).  ``failed`` therefore means "loudly broken,
still trying", and the transition back to ``ok`` is logged too.

Two failures are **terminal** — the supervisor stops retrying, because retrying
cannot fix them and continuing to retry would look like progress:

  * ``identity_mismatch`` — a different camera answered on this path.
  * ``geometry`` persisting past ``geometry_give_up`` reopens — the device
    negotiated a profile that is not the one configured.

Both are configuration/hardware facts, and on this rig both have already caused
a silent data-quality failure.
"""

from __future__ import annotations

import hashlib
import logging
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np

from robots_realtime.sensors.cameras.camera import CameraData, CameraDriver

logger = logging.getLogger(__name__)


# ── states ────────────────────────────────────────────────────────────────────

STATE_OK = "ok"
STATE_DEGRADED = "degraded"
STATE_REOPENING = "reopening"
STATE_FAILED = "failed"

_ALL_STATES = (STATE_OK, STATE_DEGRADED, STATE_REOPENING, STATE_FAILED)

#: States a consumer must treat as "do not trust this camera".
UNHEALTHY_STATES = (STATE_DEGRADED, STATE_REOPENING, STATE_FAILED)


class CameraUnavailable(RuntimeError):
    """Raised by :meth:`SupervisedCamera.read` when no good frame is available.

    Carries ``reason`` (a short machine-readable slug) and ``state``.  It is a
    *normal, expected* control-flow signal — ``CameraNode`` catches it, skips the
    frame publish, and keeps ticking so the node's heartbeat stays alive and the
    health topic keeps flowing.  Anything else propagating out of read() is a
    real bug and must still kill the node.
    """

    def __init__(self, reason: str, state: str = STATE_DEGRADED, detail: str = "") -> None:
        super().__init__(f"{reason}: {detail}" if detail else reason)
        self.reason = reason
        self.state = state
        self.detail = detail


# ── health record ─────────────────────────────────────────────────────────────


@dataclass
class CameraHealth:
    """One camera's health, as published on ``<node>/health``.

    Every field is either a number, a string or a bool so it survives msgpack
    without a custom encoder.
    """

    name: str
    state: str = STATE_REOPENING
    last_frame_age_s: float = float("inf")
    consecutive_failures: int = 0
    device_path: str = ""
    opened_at: Optional[float] = None
    reopens: int = 0
    reason: str = ""
    detail: str = ""
    frames: int = 0
    open_failures: int = 0
    #: Reader threads wedged inside the driver RIGHT NOW.
    orphaned_readers: int = 0
    #: Reader threads that have EVER been abandoned. Never decremented — a
    #: driver that unwedges when the handle is released clears the live count
    #: within milliseconds, which would make a real, repeating wedge look like it
    #: never happened. The cumulative number is the one worth alarming on.
    orphans_total: int = 0
    identity: str = ""
    #: Frames per second actually DELIVERED, and the rate the config asked for.
    fps: float = 0.0
    target_fps: float = 0.0
    terminal: bool = False
    since: float = 0.0          # wall-clock time the current state was entered

    def as_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "state": self.state,
            "last_frame_age_s": (
                None if self.last_frame_age_s == float("inf") else round(self.last_frame_age_s, 3)
            ),
            "consecutive_failures": self.consecutive_failures,
            "device_path": self.device_path,
            "opened_at": self.opened_at,
            "reopens": self.reopens,
            "reason": self.reason,
            "detail": self.detail,
            "frames": self.frames,
            "open_failures": self.open_failures,
            "orphaned_readers": self.orphaned_readers,
            "orphans_total": self.orphans_total,
            "identity": self.identity,
            "fps": round(self.fps, 2),
            "target_fps": round(self.target_fps, 2),
            "terminal": self.terminal,
            "since": self.since,
            "healthy": self.state == STATE_OK,
        }


def _fingerprint(frame: np.ndarray) -> str:
    """Cheap content fingerprint used only to detect *bit-identical* frames.

    Strided sampling keeps this ~12 kB of hashing per 2.7 MB frame.  Two frames
    from a real sensor pointed at a motionless bench still differ in sensor
    noise; two frames that are the same buffer repeated do not.  A collision
    would produce a false "frozen" report, which is a loud failure and not a
    silent one — the safe direction for this particular check.
    """
    view = np.ascontiguousarray(frame[::17, ::13])
    h = hashlib.blake2b(view.tobytes(), digest_size=8)
    h.update(str(frame.shape).encode())
    return h.hexdigest()


# ── the wrapper ───────────────────────────────────────────────────────────────


class SupervisedCamera(CameraDriver):
    """Wrap any ``CameraDriver`` with a bounded read, reopen-with-backoff and health.

    Args:
        factory: zero-argument callable returning a NEW driver instance.  It has
            to be a factory, not a driver: ``OpencvCamera`` and
            ``RealSenseCamera`` both open their device in ``__post_init__``, so
            reopening means constructing a new object.  The factory is called on
            the supervisor thread and may block; ``open_timeout_s`` bounds it.
        name: node/camera name, used in every log line and in the health record.
        read_deadline_s: how long :meth:`read` waits for a fresh frame before it
            declares device loss.  Must be comfortably longer than one frame
            period: at 15 Hz a frame is 67 ms, and USB jitter of a few hundred
            ms is normal.  Default 1.0 s, which detects loss well inside the 2 s
            budget the acceptance bar asks for.
        freeze_timeout_s: how long bit-identical frames may continue before the
            camera is declared frozen.
        reopen_backoff_s: backoff schedule between reopen attempts; the last
            value is the cap and repeats forever.
        give_up_after: consecutive failed opens before the state goes ``failed``
            and an ERROR is logged.  Retries continue (see module docstring).
        expected_shape: ``(height, width)`` the frames must have, or None to
            accept whatever the first good frame had (learned on first frame).
        target_fps: paces the pump so a driver whose read() returns instantly
            (a fake, or a v4l2 device in a strange state) cannot spin a core.
        identity_fn: ``callable(driver) -> str`` returning a stable identity for
            the physical device.  Compared across reopens; a change is terminal.
        max_orphans: abandoned pump threads tolerated before the camera is
            declared failed.
    """

    def __init__(
        self,
        factory: Callable[[], CameraDriver],
        *,
        name: str = "camera",
        read_deadline_s: float = 1.0,
        freeze_timeout_s: float = 3.0,
        reopen_backoff_s: Tuple[float, ...] = (0.25, 0.5, 1.0, 2.0, 4.0, 8.0),
        open_timeout_s: float = 20.0,
        give_up_after: int = 5,
        geometry_give_up: int = 3,
        freeze_give_up: int = 2,
        min_healthy_frames: int = 3,
        stall_window: int = 30,
        stall_distinct_ratio: float = 0.34,
        slow_ratio: float = 0.5,
        slow_recover_ratio: float = 0.75,
        slow_grace_s: float = 2.0,
        expected_shape: Optional[Tuple[int, int]] = None,
        target_fps: Optional[float] = None,
        identity_fn: Optional[Callable[[CameraDriver], str]] = None,
        presence_check: Optional[Callable[[], bool]] = None,
        device_path: str = "",
        max_orphans: int = 4,
        autostart: bool = True,
    ) -> None:
        self._factory = factory
        self.name = name
        self._read_deadline_s = float(read_deadline_s)
        self._freeze_timeout_s = float(freeze_timeout_s)
        self._backoff = tuple(reopen_backoff_s) or (1.0,)
        self._open_timeout_s = float(open_timeout_s)
        self._give_up_after = int(give_up_after)
        self._geometry_give_up = int(geometry_give_up)
        # How many freeze incidents may be "cured" by a reopen before the camera
        # is declared failed instead of being reopened round and round.
        self._freeze_give_up = int(freeze_give_up)
        # Consecutive DISTINCT frames required before a recovering camera is
        # allowed to call itself ok. See _accept.
        self._min_healthy_frames = max(1, int(min_healthy_frames))
        # Window over which frame distinctness is judged, to catch a device
        # cycling a ring of buffers rather than sitting on one. See _accept.
        self._stall_distinct_ratio = float(stall_distinct_ratio)
        self._recent_fps: deque[str] = deque(maxlen=max(4, int(stall_window)))
        self._stalled_since: Optional[float] = None

        # DELIVERED RATE. Found by starving a node of CPU: it kept delivering
        # fresh, distinct, well-formed frames — just far too few of them — and
        # every check passed, because nothing in the health model had an opinion
        # about HOW MANY frames a camera owes you. A camera running at a
        # sixtieth of its configured rate is not a healthy camera; it is a
        # camera that will quietly ruin a training set.
        #
        # Reported as `degraded`, never `failed`, and it never triggers a reopen:
        # the cause is load, not the device, and reopening under load would make
        # it worse. Hysteresis (fall below `slow_ratio`, recover above
        # `slow_recover_ratio`) so a camera hovering at the threshold does not
        # rattle between states.
        self._target_fps = float(target_fps) if target_fps else None
        self._slow_grace_s = float(slow_grace_s)
        self._slow_ratio = float(slow_ratio)
        self._slow_recover_ratio = float(slow_recover_ratio)
        self._frame_times: deque[float] = deque(maxlen=45)
        self._slow_since: Optional[float] = None
        self._is_slow = False
        self._expected_shape = tuple(expected_shape) if expected_shape else None
        self._min_period_s = (1.0 / float(target_fps)) if target_fps else 0.0
        self._identity_fn = identity_fn
        self._presence_check = presence_check
        self._max_orphans = int(max_orphans)

        self._driver: Optional[CameraDriver] = None
        self._generation = 0
        self._lock = threading.RLock()
        self._frame_event = threading.Event()
        self._slot: Optional[Tuple[int, CameraData, float]] = None   # (gen, data, monotonic)
        self._pump_error: Optional[Tuple[int, BaseException]] = None
        self._pump_thread: Optional[threading.Thread] = None
        self._pump_stop = threading.Event()

        self._reopen_request = threading.Event()
        self._stop_event = threading.Event()
        self._supervisor: Optional[threading.Thread] = None

        self._last_consumed_t = 0.0
        self._last_frame_mono: Optional[float] = None
        self._last_fingerprint: Optional[str] = None
        self._frozen_since: Optional[float] = None
        self._geometry_failures = 0
        self._freeze_incidents = 0
        self._healthy_streak = 0
        self._orphans: list[threading.Thread] = []

        self._health = CameraHealth(
            name=name, device_path=device_path, since=time.time()
        )
        self._first_open_done = threading.Event()

        if autostart:
            self.start()

    # ── lifecycle ────────────────────────────────────────────────────────────

    def start(self) -> None:
        """Start the supervisor thread.  Returns immediately; open is async."""
        if self._supervisor is not None:
            return
        self._reopen_request.set()
        self._supervisor = threading.Thread(
            target=self._supervise, name=f"cam-sup-{self.name}", daemon=True
        )
        self._supervisor.start()

    def wait_until_open(self, timeout: float = 30.0) -> bool:
        """Block until the first open attempt has resolved (either way)."""
        return self._first_open_done.wait(timeout)

    def stop(self) -> None:
        self._stop_event.set()
        self._reopen_request.set()
        sup = self._supervisor
        if sup is not None and sup.is_alive() and sup is not threading.current_thread():
            sup.join(timeout=5.0)
        self._retire_pump(join_timeout=2.0)
        with self._lock:
            driver, self._driver = self._driver, None
        if driver is not None:
            self._safe_stop_driver(driver)

    # ── CameraDriver protocol ────────────────────────────────────────────────

    def read(self) -> CameraData:
        """Return the next fresh frame, or raise :class:`CameraUnavailable`.

        Bounded by ``read_deadline_s`` under every fault.  This is the single
        line of defence that kills failure #4: whatever the driver does — spin,
        block, raise, lie — this returns.
        """
        deadline = time.monotonic() + self._read_deadline_s
        while True:
            with self._lock:
                gen = self._generation
                err = self._pump_error
                slot = self._slot
            if err is not None and err[0] == gen:
                exc = err[1]
                self._on_failure("read_error", f"{type(exc).__name__}: {exc}")
                raise CameraUnavailable("read_error", self._health.state,
                                        f"{type(exc).__name__}: {exc}")
            if slot is not None and slot[0] == gen and slot[2] > self._last_consumed_t:
                self._last_consumed_t = slot[2]
                return self._accept(slot[1], slot[2])
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            self._frame_event.wait(min(remaining, 0.05))
            self._frame_event.clear()

        # No fresh frame inside the deadline.  This covers BOTH shapes of the
        # hang: a driver blocked inside a C call, and a driver spinning on a
        # stale handle returning (False, None) — the pump makes them identical
        # from here, which is the point.
        with self._lock:
            have_driver = self._driver is not None
            prior_reason = self._health.reason
        if have_driver:
            reason = "timeout"
            detail = f"no frame within {self._read_deadline_s:.2f}s"
        else:
            # DON'T RELABEL THE CAUSE. With no device open at all, "no frame
            # within 0.40s" is true but useless — it describes the symptom and
            # buries the fact that the camera could not be opened in the first
            # place. Keep the reason the supervisor established.
            reason = prior_reason or "no_device"
            detail = "no camera device is open"
        self._on_failure(reason, detail)
        raise CameraUnavailable(reason, self._health.state, detail)

    def read_calibration_data_intrinsics(self) -> Dict[str, Any]:
        with self._lock:
            driver = self._driver
        if driver is None:
            return {}
        try:
            return driver.read_calibration_data_intrinsics()
        except Exception:
            return {}

    def get_camera_info(self) -> Dict[str, Any]:
        with self._lock:
            driver = self._driver
        info: Dict[str, Any] = {}
        if driver is not None:
            try:
                info = dict(driver.get_camera_info())
            except Exception:
                info = {}
        info["supervised"] = True
        info["health"] = self.health()
        return info

    def __getattr__(self, item: str) -> Any:
        """Delegate unknown attributes to the wrapped driver.

        ``CameraNode`` reads ``driver.intrinsic_data`` and ``driver.extrinsics``
        directly; wrapping must not make those disappear.  Guarded against
        recursion during __init__ by going through the instance dict.
        """
        if item.startswith("_"):
            raise AttributeError(item)
        driver = self.__dict__.get("_driver")
        if driver is None:
            raise AttributeError(item)
        return getattr(driver, item)

    # ── health ───────────────────────────────────────────────────────────────

    def health(self) -> Dict[str, Any]:
        with self._lock:
            h = self._health
            if self._last_frame_mono is None:
                h.last_frame_age_s = float("inf")
            else:
                h.last_frame_age_s = time.monotonic() - self._last_frame_mono
            h.orphaned_readers = sum(1 for t in self._orphans if t.is_alive())
            return h.as_dict()

    @property
    def state(self) -> str:
        with self._lock:
            return self._health.state

    @property
    def is_healthy(self) -> bool:
        return self.state == STATE_OK

    def _set_state(self, state: str, reason: str = "", detail: str = "") -> None:
        assert state in _ALL_STATES
        with self._lock:
            prev = self._health.state
            # FAILED IS STICKY. A camera that has been declared failed keeps
            # retrying, and each retry passes through `reopening` and a `warmup`
            # `degraded` — so without this, a hard-down camera flickers between
            # three states forever and an operator glancing at the cockpit sees
            # a busy-looking panel rather than a broken one. Only an earned
            # recovery to `ok` (or a different terminal reason) clears it. The
            # reason/detail below still update, so the record stays current.
            if prev == STATE_FAILED and state in (STATE_DEGRADED, STATE_REOPENING):
                self._health.reason = reason or self._health.reason
                self._health.detail = (detail or self._health.detail)[:300]
                return
            self._health.state = state
            self._health.reason = reason
            self._health.detail = detail[:300]
            if state != prev:
                self._health.since = time.time()
        if state == prev:
            return
        # Every transition is logged.  A camera silently changing state is the
        # class of bug this whole module exists to remove.
        if state == STATE_FAILED:
            logger.error("[%s] camera FAILED (%s) %s", self.name, reason, detail)
        elif state == STATE_OK:
            logger.info("[%s] camera healthy (was %s)", self.name, prev)
        else:
            logger.warning("[%s] camera %s (%s) %s", self.name, state, reason, detail)

    # ── frame validation ─────────────────────────────────────────────────────

    def _accept(self, data: CameraData, mono: float) -> CameraData:
        """Validate a frame off the pump; update health; return it or raise."""
        frame = None
        if data is not None and getattr(data, "images", None):
            frame = data.images.get("rgb")
            if frame is None and data.images:
                frame = next(iter(data.images.values()))

        if frame is None:
            self._on_failure("malformed", "driver returned no image")
            raise CameraUnavailable("malformed", self._health.state, "driver returned no image")

        arr = np.asarray(frame)
        if arr.size == 0 or arr.ndim != 3 or arr.shape[2] != 3:
            self._on_failure("malformed", f"bad frame shape {arr.shape}")
            raise CameraUnavailable("malformed", self._health.state, f"bad frame shape {arr.shape}")
        if arr.dtype != np.uint8:
            self._on_failure("malformed", f"bad frame dtype {arr.dtype}")
            raise CameraUnavailable("malformed", self._health.state, f"bad frame dtype {arr.dtype}")

        hw = (int(arr.shape[0]), int(arr.shape[1]))
        if self._expected_shape is None:
            self._expected_shape = hw
        elif hw != self._expected_shape:
            self._geometry_failures += 1
            detail = f"expected {self._expected_shape}, got {hw}"
            if self._geometry_failures >= self._geometry_give_up:
                self._fail_terminal("geometry", detail)
            else:
                self._on_failure("geometry", detail)
            raise CameraUnavailable("geometry", self._health.state, detail)

        fp = _fingerprint(arr)
        distinct = fp != self._last_fingerprint

        # A RING OF BUFFERS IS A FREEZE WITH EXTRA STEPS. Comparing each frame
        # only against the one before it catches a device stuck on ONE buffer and
        # misses a device cycling a small ring — A,B,A,B forever is "distinct"
        # at every single step, so the camera reported `ok` indefinitely while
        # carrying no new information at all. (Found by inventing the fault:
        # rings of 2, 3 and 5 all sailed through the consecutive-frame check.)
        #
        # So distinctness is measured over a WINDOW. A real sensor pointed at a
        # motionless bench still produces all-distinct frames — the noise floor
        # guarantees it — so a window in which a third of the frames are repeats
        # is not a working camera. Sustained for freeze_timeout_s, same as the
        # ring-of-1 case, so a transient cannot trip it.
        self._recent_fps.append(fp)
        if len(self._recent_fps) == self._recent_fps.maxlen:
            ratio = len(set(self._recent_fps)) / len(self._recent_fps)
            if ratio <= self._stall_distinct_ratio:
                if self._stalled_since is None:
                    self._stalled_since = mono
                elif mono - self._stalled_since >= self._freeze_timeout_s:
                    self._freeze_incidents += 1
                    n_distinct = len(set(self._recent_fps))
                    detail = (
                        f"only {n_distinct} distinct frames in the last "
                        f"{len(self._recent_fps)} — the device is cycling a ring of "
                        f"buffers (incident {self._freeze_incidents})"
                    )
                    if self._freeze_incidents >= self._freeze_give_up:
                        self._set_state(STATE_FAILED, "stalled", detail)
                        self._reopen_request.set()
                        raise CameraUnavailable("stalled", STATE_FAILED, detail)
                    self._on_failure("stalled", detail)
                    raise CameraUnavailable("stalled", self._health.state, detail)
            else:
                self._stalled_since = None

        if not distinct:
            if self._frozen_since is None:
                self._frozen_since = mono
            elif mono - self._frozen_since >= self._freeze_timeout_s:
                self._freeze_incidents += 1
                detail = (
                    f"identical frames for {mono - self._frozen_since:.1f}s "
                    f"(incident {self._freeze_incidents})"
                )
                # A REOPEN DOES NOT ALWAYS CURE A FREEZE, and when it does not,
                # the naive version of this code flapped: detect -> reopen ->
                # accept the first frame as healthy -> freeze again, forever.
                # The camera then spent most of its life reporting `ok` while
                # publishing one repeated image, which is the exact failure the
                # freeze check exists to catch, reintroduced by the cure.
                if self._freeze_incidents >= self._freeze_give_up:
                    self._set_state(
                        STATE_FAILED, "frozen",
                        f"{self._freeze_incidents} freeze incidents; reopening is not "
                        f"clearing it — the device is delivering one repeated buffer",
                    )
                    self._reopen_request.set()
                    raise CameraUnavailable("frozen", STATE_FAILED, detail)
                self._on_failure("frozen", detail)
                raise CameraUnavailable("frozen", self._health.state, detail)
            # Inside the freeze grace period the frame is still delivered: a
            # genuinely motionless scene must not be dropped.
        else:
            self._frozen_since = None
        self._last_fingerprint = fp

        with self._lock:
            self._last_frame_mono = mono
            self._health.frames += 1
            self._health.consecutive_failures = 0
            self._geometry_failures = 0
            # OK IS EARNED, NOT ASSUMED. Only a frame whose CONTENT differs from
            # the previous one counts towards recovery. One frame proves the
            # handle is open; it does not prove the sensor is producing. A
            # camera stuck on one buffer therefore never accumulates a streak and
            # never climbs back to `ok`, which is what stops the flap above.
            self._healthy_streak = self._healthy_streak + 1 if distinct else 0
            streak = self._healthy_streak

        # TWO SEPARATE GATES, because they answer different questions.
        #
        # Entering `ok` must be FAST: a camera coming up from cold has an empty
        # window, and requiring a full one made every node sit in `degraded
        # (warmup)` for a second or two after startup. That is long enough for
        # an auto-record episode — which starts 0.3 s after the session does —
        # to be stamped degraded every single time. A marker that cries wolf on
        # every clean take is worth nothing, and worse, it trains the operator
        # to ignore it. So a partial window does not block recovery.
        #
        # Clearing the FREEZE INCIDENT COUNT must be SLOW: that is what makes a
        # persistent fault converge instead of flapping. A device cycling a ring
        # of buffers satisfies "N consecutive distinct frames" trivially
        # (A,B,A,B is distinct at every step), so if the counter reset the moment
        # the state went `ok`, the camera would loop reopen → ok → stall →
        # reopen forever. The count only clears once a FULL window has proved
        # genuinely varied.
        window_full = len(self._recent_fps) == self._recent_fps.maxlen
        window_ratio = (
            len(set(self._recent_fps)) / len(self._recent_fps) if self._recent_fps else 1.0
        )
        # WHICH GATE APPLIES DEPENDS ON WHAT THIS CAMERA HAS JUST DONE. A camera
        # with no freeze history gets the lenient one (fast start, see above). A
        # camera that has ALREADY been caught delivering stalled content has to
        # prove itself over a full, varied window — otherwise it walks back to
        # `ok` on the first three frames after each reopen and the flap is
        # straight back, this time clearing a `failed` state on its way through.
        strict = self._freeze_incidents > 0
        window_ok = (
            (window_full and window_ratio > self._stall_distinct_ratio) if strict
            else ((not window_full) or window_ratio > self._stall_distinct_ratio)
        )

        slow = self._update_rate(mono)

        if (
            streak >= self._min_healthy_frames
            and window_ok
            and not slow
            and self._health.state != STATE_OK
            and not self._health.terminal
        ):
            self._set_state(STATE_OK)
        if window_full and window_ratio > self._stall_distinct_ratio:
            self._freeze_incidents = 0
        return data

    def _update_rate(self, mono: float) -> bool:
        """Track delivered fps against the configured rate. True while too slow.

        Deliberately NOT a failure and never a reopen trigger: the cause is load
        (or a bus that cannot carry the configured rate), not the device, and reopening
        under load makes it worse. But it must not read as `ok` either — a camera
        delivering three frames a second where thirty were asked for is not a
        camera anyone should be recording training data from.
        """
        self._frame_times.append(mono)
        if self._target_fps is None or len(self._frame_times) < 8:
            return False
        span = self._frame_times[-1] - self._frame_times[0]
        if span <= 0:
            return False
        achieved = (len(self._frame_times) - 1) / span
        with self._lock:
            self._health.fps = achieved
            self._health.target_fps = self._target_fps

        floor = self._target_fps * self._slow_ratio
        ceiling = self._target_fps * self._slow_recover_ratio
        if not self._is_slow:
            if achieved < floor:
                if self._slow_since is None:
                    self._slow_since = mono
                elif mono - self._slow_since >= self._slow_grace_s:
                    self._is_slow = True
                    self._slow_since = None
                    self._set_state(
                        STATE_DEGRADED, "slow",
                        f"delivering {achieved:.1f} Hz against a configured "
                        f"{self._target_fps:.1f} Hz",
                    )
            else:
                self._slow_since = None
        elif achieved >= ceiling:
            # Hysteresis: recover well clear of the threshold so a camera sitting
            # on it does not rattle between states.
            self._is_slow = False
            self._slow_since = None
        return self._is_slow

    # ── failure handling ─────────────────────────────────────────────────────

    def _on_failure(self, reason: str, detail: str) -> None:
        """Count a failure and ask the supervisor for a reopen."""
        with self._lock:
            if self._health.terminal:
                return
            self._health.consecutive_failures += 1
            self._healthy_streak = 0
            n = self._health.consecutive_failures
        logger.warning("[%s] camera read failure #%d (%s): %s", self.name, n, reason, detail)
        self._set_state(STATE_REOPENING, reason, detail)
        self._reopen_request.set()

    def _fail_terminal(self, reason: str, detail: str) -> None:
        with self._lock:
            self._health.terminal = True
        self._set_state(STATE_FAILED, reason, detail)
        logger.error(
            "[%s] TERMINAL camera failure (%s): %s — this cannot be fixed by "
            "reopening and the supervisor has stopped retrying.",
            self.name, reason, detail,
        )

    # ── pump ─────────────────────────────────────────────────────────────────

    def _start_pump(self, driver: CameraDriver, generation: int) -> None:
        stop = threading.Event()
        self._pump_stop = stop

        def _pump() -> None:
            next_t = time.monotonic()
            while not stop.is_set():
                try:
                    data = driver.read()
                except BaseException as exc:                      # noqa: BLE001
                    with self._lock:
                        self._pump_error = (generation, exc)
                    self._frame_event.set()
                    return
                now = time.monotonic()
                with self._lock:
                    self._slot = (generation, data, now)
                self._frame_event.set()
                if self._min_period_s:
                    next_t = max(next_t + self._min_period_s, now)
                    delay = next_t - time.monotonic()
                    if delay > 0:
                        stop.wait(delay)

        t = threading.Thread(target=_pump, name=f"cam-pump-{self.name}-{generation}", daemon=True)
        self._pump_thread = t
        t.start()

    def _retire_pump(self, join_timeout: float = 0.5) -> None:
        """Ask the current pump to stop; count it as an orphan if it will not."""
        self._pump_stop.set()
        t = self._pump_thread
        self._pump_thread = None
        if t is None or not t.is_alive():
            return
        t.join(timeout=join_timeout)
        if t.is_alive():
            # Stuck inside a blocking driver call.  Python cannot kill it; the
            # honest move is to count it and say so.
            self._orphans.append(t)
            with self._lock:
                self._health.orphans_total += 1
            alive = sum(1 for o in self._orphans if o.is_alive())
            logger.error(
                "[%s] reader thread is wedged inside the driver and has been "
                "abandoned (%d orphaned reader thread(s) now). The old device "
                "handle stays held until that call returns.",
                self.name, alive,
            )
            if alive > self._max_orphans:
                self._fail_terminal(
                    "orphaned_readers",
                    f"{alive} wedged reader threads — refusing to leak more",
                )

    @staticmethod
    def _safe_stop_driver(driver: CameraDriver) -> None:
        """Close a driver without ever blocking the caller.

        ``cap.release()`` on a wedged UVC handle can block indefinitely, and the
        supervisor must stay responsive, so the close runs on a throwaway thread.
        """
        def _close() -> None:
            try:
                driver.stop()
            except Exception as exc:                              # noqa: BLE001
                logger.debug("driver stop failed: %s", exc)

        threading.Thread(target=_close, name="cam-close", daemon=True).start()

    # ── supervisor ───────────────────────────────────────────────────────────

    def _supervise(self) -> None:
        attempt = 0
        while not self._stop_event.is_set():
            self._reopen_request.wait(0.25)
            if self._stop_event.is_set():
                break
            if not self._reopen_request.is_set():
                continue
            self._reopen_request.clear()

            with self._lock:
                if self._health.terminal:
                    continue

            # Tear the old one down before opening a new one: two open handles
            # on one RealSense is an error, and on a UVC device it is a race.
            self._retire_pump()
            with self._lock:
                old, self._driver = self._driver, None
                self._generation += 1
                gen = self._generation
                self._slot = None
                self._pump_error = None
                self._last_fingerprint = None
                self._frozen_since = None
                self._healthy_streak = 0
                self._recent_fps.clear()
                self._stalled_since = None
                self._frame_times.clear()
                self._slow_since = None
                self._is_slow = False
            if old is not None:
                self._safe_stop_driver(old)
                self._set_state(STATE_REOPENING, self._health.reason, self._health.detail)

            if attempt:
                wait = self._backoff[min(attempt - 1, len(self._backoff) - 1)]
                if self._stop_event.wait(wait):
                    break

            # Do not pay for an open we can already tell will fail.
            if self._device_is_present() is False:
                attempt += 1
                with self._lock:
                    self._health.open_failures += 1
                    failures = self._health.open_failures
                self._set_state(
                    STATE_FAILED if failures >= self._give_up_after else STATE_REOPENING,
                    "device_absent",
                    "the device is not on the bus; not attempting to open it",
                )
                self._first_open_done.set()
                self._reopen_request.set()
                continue

            driver = self._open_once()
            if driver is None:
                attempt += 1
                with self._lock:
                    self._health.open_failures += 1
                    failures = self._health.open_failures
                if failures >= self._give_up_after and self._health.state != STATE_FAILED:
                    self._set_state(
                        STATE_FAILED, "open_failed",
                        f"{failures} consecutive open attempts failed; still retrying "
                        f"every {self._backoff[-1]:.0f}s",
                    )
                self._first_open_done.set()
                self._reopen_request.set()
                continue

            identity = ""
            if self._identity_fn is not None:
                try:
                    identity = str(self._identity_fn(driver))
                except Exception as exc:                          # noqa: BLE001
                    logger.warning("[%s] identity check failed: %s", self.name, exc)
                    identity = ""
            with self._lock:
                known = self._health.identity
            if identity and known and identity != known:
                self._safe_stop_driver(driver)
                self._fail_terminal(
                    "identity_mismatch",
                    f"expected device {known!r}, found {identity!r} on this path",
                )
                self._first_open_done.set()
                continue

            with self._lock:
                self._driver = driver
                if identity:
                    self._health.identity = identity
                self._health.opened_at = time.time()
                self._health.open_failures = 0
                self._health.consecutive_failures = 0
                if gen > 1:
                    self._health.reopens += 1
                path = getattr(driver, "device_path", None) or getattr(driver, "device_id", None)
                if path:
                    self._health.device_path = str(path)
            attempt = 0
            self._start_pump(driver, gen)
            # Not ok yet — ok is only asserted by a validated frame arriving.
            self._set_state(STATE_DEGRADED, "warmup", "waiting for first frame")
            self._first_open_done.set()

    def _device_is_present(self) -> Optional[bool]:
        """Cheap "is the device even there" check, or None if we cannot tell.

        WHY THIS IS WORTH A ROUND TRIP. Opening an ABSENT RealSense is not cheap
        and not quiet: ``_open_with_retries`` makes five attempts, and each one
        enumerates every connected device's sensors and stream profiles before
        failing. That is seconds of C code holding the GIL, during which every
        Python thread in the node — including the one publishing health — is
        frozen. Measured on this rig with the top-down camera absent: **four
        health messages in twenty-five seconds**, against the two-per-second the
        rest of the system manages. The health topic went stale, which every
        consumer correctly reported as `stale` — nothing lied — but "the camera
        that cannot be opened is also the camera whose health stops arriving" is
        a poor trade, and it hammers a controller that may be dying.

        One cheap check first turns five expensive failures into zero.
        """
        if self._presence_check is None:
            return None
        try:
            return bool(self._presence_check())
        except Exception as exc:                                  # noqa: BLE001
            logger.debug("[%s] presence check failed: %s", self.name, exc)
            return None

    def _open_once(self) -> Optional[CameraDriver]:
        """Call the factory with a timeout; None on failure.

        The factory is run on a worker thread because opening a camera is a
        blocking C call that can hang (``cv2.VideoCapture`` on a half-dead port,
        ``pipe.start`` on a RealSense that is enumerating).  A supervisor that
        can hang inside open() is a supervisor that cannot report anything.
        """
        box: Dict[str, Any] = {}

        def _open() -> None:
            try:
                box["driver"] = self._factory()
            except BaseException as exc:                           # noqa: BLE001
                box["error"] = exc

        t = threading.Thread(target=_open, name=f"cam-open-{self.name}", daemon=True)
        t.start()
        t.join(self._open_timeout_s)
        if t.is_alive():
            logger.error(
                "[%s] camera open did not return within %.0fs — abandoning that "
                "attempt (thread orphaned)", self.name, self._open_timeout_s,
            )
            self._orphans.append(t)
            self._set_state(STATE_REOPENING, "open_timeout",
                            f"open blocked >{self._open_timeout_s:.0f}s")
            return None
        if "error" in box:
            exc = box["error"]
            logger.warning("[%s] camera open failed: %s: %s", self.name, type(exc).__name__, exc)
            self._set_state(STATE_REOPENING, "open_failed", f"{type(exc).__name__}: {exc}")
            return None
        return box.get("driver")


# ── identity helpers ──────────────────────────────────────────────────────────


def v4l2_identity(driver: CameraDriver) -> str:
    """Stable-ish identity for a V4L2 device behind a by-path node.

    The two Innomaker wrist cameras report the SAME USB serial (``SN0001``), so
    the serial is useless and the by-path port IS the identity.  What can still
    change under a by-path node is *which device the kernel bound there*, so
    this returns the device's ``name`` plus the udev-stable path, which together
    catch "a different model answered on this port".

    Deliberately NOT the resolved ``/dev/videoN`` number: that legitimately
    changes on every replug, and asserting on it would turn a healthy
    auto-recovery into a false alarm — the exact identity trap that already bit
    this rig.
    """
    import os

    path = getattr(driver, "device_path", "") or ""
    if not path:
        return ""
    try:
        real = os.path.realpath(path)
        node = os.path.basename(real)
        name_file = f"/sys/class/video4linux/{node}/name"
        with open(name_file, encoding="utf-8", errors="replace") as f:
            name = f.read().strip()
    except Exception:
        name = ""
    return f"{name}@{path}" if name else ""


def realsense_identity(driver: CameraDriver) -> str:
    """Identity for a RealSense: its serial number, which is genuinely unique."""
    dev = getattr(driver, "device_id", None)
    return str(dev) if dev else ""
