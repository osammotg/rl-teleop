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

_logger = logging.getLogger(__name__)

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
    published_topics: list[str] = ["rgb"]
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
        **kwargs,
    ) -> None:
        super().__init__(name=name, writer=writer, **kwargs)
        self._driver = driver
        self._driver_spec = _driver_spec
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

    def setup(self) -> None:
        if self._driver is None:
            if self._driver_spec is None:
                raise RuntimeError(
                    f"[{self.name}] CameraNode.driver is None — inject a camera driver before starting."
                )
            self._driver = _instantiate_camera_driver(self._driver_spec)

    def step(self) -> None:
        if self._driver is None:
            raise RuntimeError(f"[{self.name}] CameraNode.step() called before setup")
        driver = self._driver
        try:
            data: CameraData = driver.read()
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

    def cleanup(self) -> None:
        driver = self._driver
        if driver is not None and hasattr(driver, "stop"):
            driver.stop()

    @classmethod
    def build_kwargs(cls, params: dict) -> dict:
        kwargs: dict = {
            "name": params["name"],
            "poll_freq": params.get("poll_freq"),
            "publish_resize": params.get("publish_resize"),
            "publish_resize_mode": params.get("publish_resize_mode", "center_crop"),
            "extrinsics": params.get("extrinsics"),
            "extrinsics_file": params.get("extrinsics_file"),
        }
        if "driver" in params:
            driver_kwargs = {k: v for k, v in params.items() if k not in _NODE_ONLY_KEYS}
            kwargs["_driver_spec"] = driver_kwargs
        return kwargs
