"""CameraNode — wraps any CameraDriver and publishes frames to the bus.

Hardware timestamp from the driver (RealSense / ZED SDK) is used directly,
giving sub-millisecond accurate per-frame timestamps for post-hoc alignment.

poll_freq is None by default: the driver's blocking read() call paces the loop
at the hardware frame rate.  Set poll_freq only for drivers (e.g. bare OpenCV)
where read() returns immediately and you want an explicit rate cap.

Optional ``publish_resize`` shrinks frames before they hit the bus so consumers
(e.g. an OpenPI policy that resizes to 224×224 anyway) don't pay full-VGA
serialization + TCP cost. The on-disk MP4 keeps the full-resolution frame —
only the bus payload is downsized. Two modes match AsyncDiffusionAgent's
``image_preprocess``: ``center_crop`` (crop to min(H,W) square then resize)
and ``pad`` (resize-with-pad / letterbox).
"""

from __future__ import annotations

import importlib
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np

from robots_realtime.runtime.node import Node, NodeRole
from robots_realtime.sensors.cameras.camera import CameraData, CameraDriver
from robots_realtime.sensors.cameras.supervised_camera import (
    STATE_FAILED,
    CameraUnavailable,
    SupervisedCamera,
    realsense_identity,
    v4l2_identity,
)

_logger = logging.getLogger(__name__)

#: How often ``<node>/health`` goes out even when nothing has changed.  The
#: acceptance bar asks for a dead camera to be visible within ~2 s; 2 Hz plus
#: an immediate publish on every state change leaves plenty of margin for the
#: cockpit's own poll period on top.
_HEALTH_PERIOD_S = 0.5

_CAMERA_DRIVER_REGISTRY: dict[str, str] = {
    "ZedCamera":        "robots_realtime.sensors.cameras.zed_camera:ZedCamera",
    "OpenCVCamera":     "robots_realtime.sensors.cameras.opencv_camera:OpencvCamera",
    "RealSenseCamera":  "robots_realtime.sensors.cameras.realsense_camera:RealSenseCamera",
}

_NODE_ONLY_KEYS = {
    "name",
    "type",
    "poll_freq",
    "publish_resize",
    "publish_resize_mode",
    "extrinsics",
    "extrinsics_file",
    "pinned_cpu",
    "realtime_priority",
    "require_realtime",
    # Supervision knobs live on the node, not on the driver — they must not be
    # forwarded into the driver's constructor kwargs.
    "supervise",
    "read_deadline_s",
    "freeze_timeout_s",
}


def _center_crop_and_resize(img: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    """Center-crop to the largest square that fits, then resize to (target_h, target_w).

    Mirrors ``AsyncDiffusionAgent._center_crop_and_resize`` so frames published
    with mode=center_crop are bit-identical to what the policy would produce
    if it received the full-res frame and ran its own preprocessing.
    """
    from openpi_client.image_tools import resize_with_pad  # noqa: PLC0415

    h, w = img.shape[:2]
    side = min(h, w)
    h0 = (h - side) // 2
    w0 = (w - side) // 2
    cropped = img[h0:h0 + side, w0:w0 + side]
    return resize_with_pad(cropped, target_h, target_w)


def _resize_with_pad(img: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    from openpi_client.image_tools import resize_with_pad  # noqa: PLC0415
    return resize_with_pad(img, target_h, target_w)


_RESIZE_MODES = {
    "center_crop": _center_crop_and_resize,
    "pad": _resize_with_pad,
}


def _instantiate_camera_driver(spec: dict) -> CameraDriver:
    """Instantiate a camera driver from a spec dict (driver name + kwargs)."""
    driver_name: str = spec["driver"]
    if driver_name in _CAMERA_DRIVER_REGISTRY:
        ref = _CAMERA_DRIVER_REGISTRY[driver_name]
    elif ":" in driver_name:
        ref = driver_name
    else:
        raise ValueError(
            f"Unknown camera driver '{driver_name}'. "
            f"Known drivers: {list(_CAMERA_DRIVER_REGISTRY.keys())}"
        )
    module_path, cls_name = ref.rsplit(":", 1)
    mod = importlib.import_module(module_path)
    kwargs = {k: v for k, v in spec.items() if k != "driver"}
    return getattr(mod, cls_name)(**kwargs)


def _extrinsics_from_spec(spec: dict[str, Any]) -> dict[str, np.ndarray]:
    import viser.transforms as vtf  # noqa: PLC0415

    position = np.asarray(spec["position"], dtype=np.float64)
    if position.shape != (3,):
        raise ValueError(f"camera extrinsics position must have shape (3,), got {position.shape}")

    if "wxyz" in spec:
        wxyz = np.asarray(spec["wxyz"], dtype=np.float64)
    elif "rotation" in spec:
        wxyz = np.asarray(spec["rotation"], dtype=np.float64)
    elif "rpy_radians" in spec:
        rpy = np.asarray(spec["rpy_radians"], dtype=np.float64)
        if rpy.shape != (3,):
            raise ValueError(f"camera extrinsics rpy_radians must have shape (3,), got {rpy.shape}")
        wxyz = vtf.SO3.from_rpy_radians(*rpy).wxyz
    else:
        raise ValueError("camera extrinsics must define one of: wxyz, rotation, rpy_radians")

    if wxyz.shape != (4,):
        raise ValueError(f"camera extrinsics quaternion must have shape (4,), got {wxyz.shape}")

    pose_mat = vtf.SE3(wxyz_xyz=np.concatenate([wxyz, position])).as_matrix()
    return {"position": position, "wxyz": wxyz, "pose_mat": pose_mat}


def _load_extrinsics_file(path_str: str) -> dict[str, np.ndarray]:
    import yaml  # noqa: PLC0415

    path = Path(path_str)
    if not path.exists():
        raise FileNotFoundError(f"camera extrinsics file not found: {path}")
    with path.open(encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"camera extrinsics file must contain a mapping: {path}")
    return _extrinsics_from_spec(data)


class CameraNode(Node):
    """Publish camera frames from any CameraDriver onto the bus.

    Published topics:
        ``{name}/rgb``    — dict with ``frame`` (H,W,3 uint8) and ``ts`` float
        ``{name}/info``   — camera info dict (published once on setup)

    Optionally also publishes:
        ``{name}/depth``  — if driver provides it in CameraData.other_sensors
        ``{name}/imu``    — if driver provides IMUData

    Args:
        driver:    Camera driver implementing read() -> CameraData.
        name:      Node name on the bus.
        poll_freq: Optional rate cap for drivers where read() is non-blocking.
        writer:    Optional Writer injected at construction for recording.
    """

    role = NodeRole.SENSOR
    published_topics: list[str] = ["rgb", "health"]
    poll_freq: float | None = None

    def __init__(
        self,
        driver: CameraDriver | None = None,
        name: str = "camera",
        poll_freq: float | None = None,
        writer=None,
        _driver_spec: dict | None = None,
        publish_resize: tuple[int, int] | list[int] | None = None,
        publish_resize_mode: str = "center_crop",
        extrinsics: dict[str, Any] | None = None,
        extrinsics_file: str | None = None,
        supervise: bool = True,
        read_deadline_s: float = 1.0,
        freeze_timeout_s: float = 3.0,
        driver_factory=None,
        **kwargs,
    ) -> None:
        super().__init__(name=name, writer=writer, **kwargs)
        self._driver = driver
        self._driver_spec = _driver_spec
        self._driver_factory = driver_factory
        self._supervise = bool(supervise)
        self._read_deadline_s = float(read_deadline_s)
        self._freeze_timeout_s = float(freeze_timeout_s)
        self._supervised: SupervisedCamera | None = None
        self._last_health_pub = 0.0
        self._last_health_state = ""
        self.poll_freq = poll_freq
        if extrinsics is not None and extrinsics_file is not None:
            raise ValueError("CameraNode accepts either extrinsics or extrinsics_file, not both")
        if extrinsics_file is not None:
            self._extrinsics = _load_extrinsics_file(extrinsics_file)
        elif extrinsics is not None:
            self._extrinsics = _extrinsics_from_spec(extrinsics)
        else:
            self._extrinsics = None

        if publish_resize is not None:
            if publish_resize_mode not in _RESIZE_MODES:
                raise ValueError(
                    f"[{name}] publish_resize_mode must be one of "
                    f"{sorted(_RESIZE_MODES)}, got {publish_resize_mode!r}"
                )
            h, w = publish_resize
            self._publish_resize: tuple[int, int] | None = (int(h), int(w))
            self._publish_resize_fn = _RESIZE_MODES[publish_resize_mode]
        else:
            self._publish_resize = None
            self._publish_resize_fn = None

    # ------------------------------------------------------------------
    # Supervision
    # ------------------------------------------------------------------

    def _make_factory(self):
        """Return a zero-arg callable that builds a FRESH driver.

        Reopening a camera means constructing a new driver object, because both
        ``OpencvCamera`` and ``RealSenseCamera`` open their device in
        ``__post_init__``.  A driver injected as an instance (tests, and the
        ``driver=`` kwarg) cannot be rebuilt, so it is handed back as-is and
        reopen degenerates to "reuse the same object" — which is honest for a
        fake and irrelevant for a real rig, where the driver always comes from
        ``_driver_spec``.
        """
        if self._driver_factory is not None:
            return self._driver_factory
        if self._driver_spec is not None:
            spec = dict(self._driver_spec)
            return lambda: _instantiate_camera_driver(spec)
        driver = self._driver
        if driver is None:
            raise RuntimeError(
                f"[{self.name}] CameraNode.driver is None — inject a camera driver before starting."
            )
        return lambda: driver

    def _presence_check(self):
        """A cheap callable answering "is this device on the bus at all".

        Only for the shapes we can answer cheaply and correctly. Returning None
        (no check) is always safe — it just means the open attempt happens.
        """
        spec = self._driver_spec or {}
        path = spec.get("device_path") or getattr(self._driver, "device_path", None)
        if path and str(path).startswith("/dev/"):
            import os  # noqa: PLC0415
            return lambda p=str(path): os.path.exists(p)

        serial = spec.get("device_id") or getattr(self._driver, "device_id", None)
        if serial:
            def _realsense_present(want=str(serial)) -> bool:
                import pyrealsense2 as rs  # noqa: PLC0415
                # query_devices() only — deliberately NOT get_stream_profiles(),
                # which is the expensive part of the failing open path.
                return want in [
                    d.get_info(rs.camera_info.serial_number)
                    for d in rs.context().query_devices()
                ]
            return _realsense_present
        return None

    def _identity_fn(self):
        """Pick the identity check that matches how this camera is addressed."""
        spec = self._driver_spec or {}
        if spec.get("device_path") or getattr(self._driver, "device_path", None):
            return v4l2_identity
        if spec.get("device_id") or getattr(self._driver, "device_id", None):
            return realsense_identity
        return None

    def setup(self) -> None:
        if not self._supervise:
            # Escape hatch only. An unsupervised camera has no bounded read and
            # no health topic, which is precisely the configuration that
            # produced failure #4. Say so at ERROR, every time.
            _logger.error(
                "[%s] running UNSUPERVISED (supervise: false) — no bounded read, "
                "no health topic. A stale device handle will hang this node "
                "silently. This is not a supported production setting.",
                self.name,
            )
            if self._driver is None:
                if self._driver_spec is None:
                    raise RuntimeError(
                        f"[{self.name}] CameraNode.driver is None — inject a camera driver before starting."
                    )
                self._driver = _instantiate_camera_driver(self._driver_spec)
            return

        spec = self._driver_spec or {}
        resolution = spec.get("resolution")
        expected_shape = None
        if isinstance(resolution, (list, tuple)) and len(resolution) == 2:
            # Driver configs carry (width, height); frames are (height, width).
            expected_shape = (int(resolution[1]), int(resolution[0]))

        supervised = SupervisedCamera(
            self._make_factory(),
            name=self.name,
            read_deadline_s=self._read_deadline_s,
            freeze_timeout_s=self._freeze_timeout_s,
            expected_shape=expected_shape,
            target_fps=spec.get("fps"),
            identity_fn=self._identity_fn(),
            presence_check=self._presence_check(),
            device_path=str(spec.get("device_path") or spec.get("device_id") or ""),
        )
        self._supervised = supervised
        self._driver = supervised
        # Wait for the FIRST open to resolve so a camera that cannot be opened at
        # all is reported before the session declares itself started. Do not
        # raise on failure: the node must stay alive to keep publishing
        # health=failed, which is the whole point — a node that dies at setup is
        # exactly the invisible failure this replaces.
        if not supervised.wait_until_open(timeout=30.0):
            _logger.error("[%s] camera did not finish its first open within 30s", self.name)
        self._publish_health(force=True)

    def _publish_health(self, force: bool = False) -> None:
        """Publish ``<name>/health``: every state change, and at 2 Hz regardless.

        ``record=False`` deliberately: the writers are video writers keyed on
        image topics, and health is bus-only telemetry. It must never end up in
        the mp4 path, both because it would confuse the writer and because
        health has to be measurable INDEPENDENTLY of what got recorded — the
        recording path already lies (Publisher.publish writes before the ZMQ
        send, so a perfect mp4 proves nothing about delivery).
        """
        if self._supervised is None:
            return
        health = self._supervised.health()
        state = str(health.get("state", ""))
        now = time.monotonic()
        if not force and state == self._last_health_state and now - self._last_health_pub < _HEALTH_PERIOD_S:
            return
        self._last_health_pub = now
        self._last_health_state = state
        try:
            self.publish("health", health, record=False)
        except Exception as exc:                                   # noqa: BLE001
            # A publish failure must not take the node down on the HEALTH path;
            # the rgb path below is still allowed to, because a node that cannot
            # publish frames should die loudly rather than pretend.
            _logger.warning("[%s] health publish failed: %s", self.name, exc)

    def step(self) -> None:
        if self._driver is None:
            raise RuntimeError(f"[{self.name}] CameraNode.step() called before setup")
        driver = self._driver
        try:
            data: CameraData = driver.read()
        except CameraUnavailable as exc:
            # THE FIX FOR FAILURE #4. The old code let a stale handle spin inside
            # read() forever: step() never returned, _tick() never ran, and the
            # node published nothing at all — not even its heartbeat — while
            # staying alive and green in the TUI.
            #
            # Now every device-loss shape arrives here as a bounded, named event.
            # We publish health, return normally so _tick() runs and the
            # heartbeat keeps flowing, and let the supervisor reopen. A camera
            # that is failing is now a camera that is SAYING it is failing.
            self._publish_health(force=True)
            level = _logger.error if exc.state == STATE_FAILED else _logger.warning
            level("[%s] no frame (%s): %s", self.name, exc.reason, exc.detail)
            # Back off a little so a hard-down camera does not spin this loop at
            # full tilt; the deadline inside read() already paces the normal case.
            time.sleep(0.05)
            return
        except Exception as exc:
            _logger.error("[%s] camera read failed: %s", self.name, exc)
            if hasattr(driver, "stop"):
                driver.stop()
            raise

        # Hardware timestamp from driver (ms) → seconds
        ts = data.timestamp / 1000.0 if data.timestamp else time.time()

        # Publish one consolidated message that matches the format agents and
        # visualization code already expect (same shape as old CameraNode._get_latest_data).
        msg: dict = {"images": data.images, "timestamp": ts}

        depth = (data.other_sensors or {}).get("depth")
        if depth is None:
            depth = getattr(data, "depth_data", None)
        if depth is not None:
            msg["depth_data"] = depth

        intrinsics = getattr(driver, "intrinsic_data", None)
        if depth is not None and data.other_sensors is not None:
            intrinsics = data.other_sensors.get("depth_intrinsics", intrinsics)
        if intrinsics is not None:
            msg["intrinsics"] = intrinsics
        extrinsics = self._extrinsics if self._extrinsics is not None else getattr(driver, "extrinsics", None)
        if extrinsics is not None:
            msg["extrinsics"] = extrinsics

        if self._publish_resize is None:
            self.publish("rgb", msg, ts=ts)
        else:
            if self._publish_resize_fn is None:
                raise RuntimeError(f"[{self.name}] publish resize function is not configured")
            # Bus payload: resized RGB only — depth and intrinsics would need
            # geometric rescaling to stay consistent with the new pixel grid,
            # so they're dropped from the bus version. The disk recording
            # (record_data=msg) keeps everything at full resolution.
            target_h, target_w = self._publish_resize
            resized = {
                k: self._publish_resize_fn(img, target_h, target_w)
                for k, img in data.images.items()
            }
            bus_msg: dict = {"images": resized, "timestamp": ts}
            if extrinsics is not None:
                bus_msg["extrinsics"] = extrinsics  # pose is resolution-invariant
            self.publish("rgb", bus_msg, ts=ts, record_data=msg)

        if data.imu_data is not None:
            imu = data.imu_data
            self.publish("imu", {
                "accel": imu.acceleration,
                "gyro": imu.gyroscope,
                "ts": imu.timestamp,
            }, ts=ts)

        self._publish_health()

    def cleanup(self) -> None:
        # SupervisedCamera.stop() also joins its supervisor thread and closes the
        # inner driver, so one call covers both the supervised and the bare case.
        driver = self._driver
        if driver is not None and hasattr(driver, "stop"):
            driver.stop()
        self._supervised = None

    @classmethod
    def build_kwargs(cls, params: dict) -> dict:
        kwargs: dict = {
            "name": params["name"],
            "poll_freq": params.get("poll_freq"),
            "publish_resize": params.get("publish_resize"),
            "publish_resize_mode": params.get("publish_resize_mode", "center_crop"),
            "extrinsics": params.get("extrinsics"),
            "extrinsics_file": params.get("extrinsics_file"),
            "supervise": params.get("supervise", True),
            "read_deadline_s": params.get("read_deadline_s", 1.0),
            "freeze_timeout_s": params.get("freeze_timeout_s", 3.0),
        }
        if "driver" in params:
            driver_kwargs = {k: v for k, v in params.items() if k not in _NODE_ONLY_KEYS}
            kwargs["_driver_spec"] = driver_kwargs
        return kwargs
