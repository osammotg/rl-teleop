import logging
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from robots_realtime.sensors.cameras.camera import CameraData, CameraDriver


@dataclass
class OpencvCamera(CameraDriver):
    """
    bash v4l2-ctl --list-devices -> get the device id
    bash v4l2-ctl --device=/dev/video12 --list-formats-ext -> get the resolution and fps

    """

    device_path: str = "/dev/video-zed2i"
    camera_type: str = "zed_camera"
    image_transfer_time_offset: int = 80  # ms typical transfer time, can change based on computer type and load
    resolution: Tuple[int, int] = (640, 480)
    fps: int = 30
    fourcc: Optional[str] = None
    v4l2_controls: Optional[Dict[str, int]] = None
    auto_exposure_enabled: bool = False
    auto_exposure_target: float = 110.0
    auto_exposure_deadband: float = 8.0
    auto_exposure_min: int = 5
    auto_exposure_max: int = 200
    auto_exposure_speed: float = 0.25
    auto_exposure_period_s: float = 0.5
    name: Optional[str] = None

    def __repr__(self) -> str:
        return f"OpencvCamera(device_path={self.device_path!r}, name={self.name!r}, resolution={self.resolution}, fps={self.fps})"

    def __post_init__(self):
        self._last_auto_exposure_s = 0.0
        self._exposure: Optional[int] = None
        self._apply_v4l2_controls()
        self.cap = cv2.VideoCapture(self.device_path)
        if self.fourcc:
            self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*self.fourcc))
        self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])

        # Try setting FPS to 30
        self.cap.set(cv2.CAP_PROP_FPS, self.fps)
        if self.auto_exposure_enabled:
            self._exposure = self._get_v4l2_control("exposure_time_absolute")

    def _resolved_device_path(self) -> str:
        path = Path(self.device_path)
        try:
            return str(path.resolve())
        except Exception:
            return self.device_path

    def _set_v4l2_control(self, control: str, value: int) -> None:
        device = self._resolved_device_path()
        subprocess.run(
            ["v4l2-ctl", "-d", device, f"--set-ctrl={control}={int(value)}"],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

    def _get_v4l2_control(self, control: str) -> Optional[int]:
        device = self._resolved_device_path()
        try:
            result = subprocess.run(
                ["v4l2-ctl", "-d", device, f"--get-ctrl={control}"],
                check=True,
                capture_output=True,
                text=True,
            )
        except Exception:
            return None
        try:
            return int(result.stdout.rsplit(":", 1)[1].strip())
        except Exception:
            return None

    def _apply_v4l2_controls(self) -> None:
        for control, value in (self.v4l2_controls or {}).items():
            try:
                self._set_v4l2_control(control, int(value))
            except Exception as exc:
                logging.warning("%s: failed to set %s=%s: %s", self, control, value, exc)

    def _update_auto_exposure(self, frame: np.ndarray) -> None:
        if not self.auto_exposure_enabled:
            return
        now = time.monotonic()
        if now - self._last_auto_exposure_s < self.auto_exposure_period_s:
            return
        self._last_auto_exposure_s = now

        exposure = self._exposure
        if exposure is None:
            exposure = self._get_v4l2_control("exposure_time_absolute")
        if exposure is None:
            self.auto_exposure_enabled = False
            logging.warning("%s: exposure_time_absolute unavailable; disabling auto exposure", self)
            return

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        brightness = float(np.mean(gray))
        error = self.auto_exposure_target - brightness
        if abs(error) <= self.auto_exposure_deadband:
            self._exposure = exposure
            return

        scale = 1.0 + self.auto_exposure_speed * error / max(self.auto_exposure_target, 1.0)
        next_exposure = round(exposure * scale)
        next_exposure = max(self.auto_exposure_min, min(self.auto_exposure_max, next_exposure))
        if next_exposure == exposure:
            self._exposure = exposure
            return

        try:
            self._set_v4l2_control("exposure_time_absolute", next_exposure)
            self._exposure = next_exposure
        except Exception as exc:
            self.auto_exposure_enabled = False
            logging.warning("%s: auto exposure update failed; disabling: %s", self, exc)

    def list_cameras(self) -> List[int]:
        return []

    def read(self) -> CameraData:
        try:
            ret, frame = self.cap.read()
            capture_time_ms = time.time() * 1000
            while not ret:
                # If read failed, retry and update capture time
                ret, frame = self.cap.read()
                capture_time_ms = time.time() * 1000
                time.sleep(0.01)
        except Exception as e:
            logging.error(f"Error reading frame: {e}")
            raise e
        frame = np.ascontiguousarray(frame)
        self._update_auto_exposure(frame)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Adjust timestamp using the offset
        timestamp_ms = capture_time_ms - self.image_transfer_time_offset
        return CameraData(images={"rgb": frame}, timestamp=timestamp_ms)

    def get_camera_info(self) -> dict:
        info = {}
        info.update(
            {
                "camera_type": self.camera_type,
                "device_path": self.device_path,
                "width": self.resolution[0],
                "height": self.resolution[1],
                "fps": self.fps,
                "fourcc": self.fourcc,
            }
        )
        return info

    def stop(self) -> None:
        self.cap.release()

    def read_calibration_data_intrinsics(self) -> Dict[str, Any]:
        raise NotImplementedError(f"Calibration data reading is not implemented for {self}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--device_path", type=str, default="/dev/video-zed2i")
    args = parser.parse_args()

    camera = OpencvCamera(device_path=args.device_path)

    while True:
        data = camera.read()
        print(data)
        time.sleep(1 / camera.fps)
