"""V4L2 / udev camera enumeration with capture-capable filtering."""

from __future__ import annotations

import glob
import logging
import subprocess
import time
from collections import Counter
from pathlib import Path
from typing import Callable, Dict, List, Optional

import cv2
import numpy as np

from cv_camera_mapping.schemas import CameraDevice, EnumerateResult, NodeKind

logger = logging.getLogger(__name__)

CAPTURE_PROBE_FRAMES = 3
CAPTURE_PROBE_TIMEOUT_S = 2.0


def _run_command(cmd: List[str], timeout: float = 5.0) -> tuple[int, str]:
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
        output = (result.stdout or "") + (result.stderr or "")
        return result.returncode, output
    except (subprocess.TimeoutExpired, FileNotFoundError) as exc:
        return 1, str(exc)


def parse_udev_properties(text: str) -> Dict[str, str]:
    props: Dict[str, str] = {}
    for line in text.splitlines():
        if "=" in line:
            key, _, value = line.partition("=")
            props[key.strip()] = value.strip()
    return props


def udevadm_info(
    device_path: str,
    runner: Callable[[List[str], float], tuple[int, str]] = _run_command,
) -> Dict[str, str]:
    _, output = runner(
        ["udevadm", "info", "--query=property", f"--name={device_path}"],
        5.0,
    )
    return parse_udev_properties(output)


def v4l2_ctl_info(
    device_path: str,
    runner: Callable[[List[str], float], tuple[int, str]] = _run_command,
) -> str:
    code, output = runner(
        ["v4l2-ctl", "--device", device_path, "--all"],
        5.0,
    )
    return output if code == 0 else ""


def is_capture_capable_udev(props: Dict[str, str]) -> bool:
    caps = props.get("ID_V4L_CAPABILITIES", "")
    if "capture" in caps.lower():
        return True
    return False


def probe_opencv_capture(
    device_path: str,
    backend: int = cv2.CAP_V4L2,
    frames: int = CAPTURE_PROBE_FRAMES,
    timeout_s: float = CAPTURE_PROBE_TIMEOUT_S,
    video_capture_factory: Optional[Callable[..., object]] = None,
) -> tuple[bool, Optional[int], Optional[int]]:
    """Open device and read at least one valid frame within timeout."""
    factory = video_capture_factory or cv2.VideoCapture
    deadline = time.monotonic() + timeout_s
    cap = factory(device_path, backend)
    try:
        if not cap.isOpened():
            return False, None, None
        width = height = None
        while time.monotonic() < deadline:
            for _ in range(frames):
                ret, frame = cap.read()
                if ret and frame is not None and getattr(frame, "size", 0) > 0:
                    height, width = frame.shape[:2]
                    return True, width, height
            time.sleep(0.05)
        return False, width, height
    finally:
        cap.release()


def build_camera_device(
    device_path: str,
    props: Dict[str, str],
    capture_capable: bool,
    width: Optional[int],
    height: Optional[int],
    v4l2_text: str = "",
) -> CameraDevice:
    resolved = str(Path(device_path).resolve()) if Path(device_path).exists() else device_path
    product = props.get("ID_MODEL_FROM_DATABASE") or props.get("ID_MODEL")
    if not product and "Card type" in v4l2_text:
        for line in v4l2_text.splitlines():
            if "Card type" in line:
                product = line.split(":", 1)[-1].strip()
                break
    devlinks_raw = props.get("DEVLINKS", "")
    devlinks = devlinks_raw.split() if devlinks_raw else []
    node_kind = NodeKind.CAPTURE if capture_capable else NodeKind.NON_CAPTURE_OR_METADATA
    return CameraDevice(
        device_path=device_path,
        resolved_path=resolved,
        capture_capable=capture_capable,
        serial=props.get("ID_SERIAL"),
        serial_short=props.get("ID_SERIAL_SHORT"),
        usb_path=props.get("ID_PATH"),
        vendor_id=props.get("ID_VENDOR_ID"),
        model_id=props.get("ID_MODEL_ID"),
        product=product,
        devlinks=devlinks,
        backend="v4l2",
        width=width,
        height=height,
        node_kind=node_kind,
    )


def list_video_device_paths() -> List[str]:
    paths = sorted(glob.glob("/dev/video*"), key=lambda p: int(p.replace("/dev/video", "") or "0"))
    return paths


def enumerate_cameras(
    include_non_capture: bool = False,
    udev_runner: Callable[[List[str], float], tuple[int, str]] = _run_command,
    v4l2_runner: Callable[[List[str], float], tuple[int, str]] = _run_command,
    video_capture_factory: Optional[Callable[..., object]] = None,
    device_paths: Optional[List[str]] = None,
) -> EnumerateResult:
    paths = device_paths if device_paths is not None else list_video_device_paths()
    candidates: List[CameraDevice] = []
    serial_counts: Counter[str] = Counter()

    for device_path in paths:
        props = udevadm_info(device_path, runner=udev_runner)
        v4l2_text = v4l2_ctl_info(device_path, runner=v4l2_runner)
        udev_says_capture = is_capture_capable_udev(props)
        cv_ok, width, height = probe_opencv_capture(
            device_path,
            video_capture_factory=video_capture_factory,
        )
        capture_capable = cv_ok or (udev_says_capture and cv_ok)
        # Require OpenCV frame read for capture-capable classification
        capture_capable = cv_ok
        if not include_non_capture and not capture_capable:
            continue
        device = build_camera_device(
            device_path,
            props,
            capture_capable=capture_capable,
            width=width,
            height=height,
            v4l2_text=v4l2_text,
        )
        if device.serial_short:
            serial_counts[device.serial_short] += 1
        candidates.append(device)

    duplicate_serials = [s for s, c in serial_counts.items() if c > 1]
    for device in candidates:
        if device.serial_short:
            device.serial_unique = device.serial_short not in duplicate_serials

    message = None
    if not candidates:
        message = "no_capture_capable_candidates"

    return EnumerateResult(
        candidates=candidates,
        duplicate_serials=duplicate_serials,
        message=message,
    )


def capture_frame(
    device_path: str,
    resolution: tuple[int, int] = (640, 480),
    fps: int = 30,
    warmup_frames: int = 5,
    video_capture_factory: Optional[Callable[..., object]] = None,
) -> np.ndarray:
    factory = video_capture_factory or cv2.VideoCapture
    cap = factory(device_path, cv2.CAP_V4L2)
    try:
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open {device_path}")
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])
        cap.set(cv2.CAP_PROP_FPS, fps)
        frame = None
        for _ in range(warmup_frames + 10):
            ret, img = cap.read()
            if ret and img is not None and img.size > 0:
                frame = img
        if frame is None:
            raise RuntimeError(f"No frame from {device_path}")
        return np.ascontiguousarray(frame)
    finally:
        cap.release()
