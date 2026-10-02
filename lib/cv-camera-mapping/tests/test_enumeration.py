from __future__ import annotations

import numpy as np

from cv_camera_mapping.enumeration import enumerate_cameras, parse_udev_properties


def _udev_output(serial: str, path_suffix: str, capture: bool) -> str:
    caps = ":capture:" if capture else ":"
    return "\n".join(
        [
            f"ID_SERIAL_SHORT={serial}",
            f"ID_SERIAL=innomaker_{serial}",
            f"ID_PATH=pci-0000:00:14.0-usb-0:1:{path_suffix}",
            "ID_VENDOR_ID=2bdf",
            "ID_MODEL_ID=0002",
            f"ID_V4L_CAPABILITIES={caps}",
            "DEVLINKS=/dev/video-left /dev/video-right",
        ]
    )


def _fake_runner(responses: dict):
    def runner(cmd, timeout):
        dev = cmd[-1]
        if dev.startswith("--name="):
            dev = dev.split("=", 1)[1]
        if "udevadm" in cmd[0]:
            return 0, responses.get(dev, "")
        return 1, ""

    return runner


class FakeCap:
    def __init__(self, path, backend=None):
        self.path = path
        self._open = path in ("/dev/video0", "/dev/video8")

    def isOpened(self):
        return self._open

    def set(self, *args, **kwargs):
        pass

    def read(self):
        if self._open:
            return True, np.zeros((480, 640, 3), dtype=np.uint8) + 50
        return False, None

    def release(self):
        pass


def test_filters_capture_nodes_only():
    responses = {
        "/dev/video0": _udev_output("SN0001", "1.1.1", True),
        "/dev/video1": _udev_output("SN0001", "1.1.1", False),
        "/dev/video8": _udev_output("SN0001", "1.2.1", True),
        "/dev/video9": _udev_output("SN0001", "1.2.1", False),
    }
    result = enumerate_cameras(
        device_paths=["/dev/video0", "/dev/video1", "/dev/video8", "/dev/video9"],
        udev_runner=_fake_runner(responses),
        v4l2_runner=_fake_runner({}),
        video_capture_factory=FakeCap,
    )
    paths = [c.device_path for c in result.candidates]
    assert paths == ["/dev/video0", "/dev/video8"]
    assert "SN0001" in result.duplicate_serials


def test_parse_udev():
    props = parse_udev_properties("ID_SERIAL_SHORT=SN0001\nID_PATH=pci-foo")
    assert props["ID_SERIAL_SHORT"] == "SN0001"
