import json
from unittest.mock import patch

import numpy as np

from cv_camera_mapping.cli import main
from cv_camera_mapping.schemas import CameraDevice, EnumerateResult


def test_identify_success_json(tmp_path):
    import cv2

    rng = np.random.default_rng(10)
    left = np.zeros((480, 640, 3), dtype=np.uint8)
    left[:, :320] = rng.integers(40, 220, (480, 320, 3), dtype=np.uint8)
    right = np.zeros((480, 640, 3), dtype=np.uint8)
    rng2 = np.random.default_rng(11)
    right[:, 320:] = rng2.integers(40, 220, (480, 320, 3), dtype=np.uint8)
    for y in range(0, 480, 24):
        for x in range(10, 300, 18):
            cv2.circle(left, (x, y), 5, (255, 180, 40), -1)
    for y in range(0, 480, 22):
        for x in range(340, 620, 16):
            cv2.rectangle(right, (x, y), (x + 8, y + 8), (50, 255, 90), -1)

    refs_dir = tmp_path / "refs"
    from cv_camera_mapping.reference_capture import capture_references

    capture_references(
        refs_dir,
        ["camera_left", "camera_right"],
        {"camera_left": "/dev/video0", "camera_right": "/dev/video8"},
        frame_loader=lambda p: left.copy() if "video0" in p else right.copy(),
    )

    candidates = [
        CameraDevice(device_path="/dev/video0", capture_capable=True, serial_short="SN0001", serial_unique=False),
        CameraDevice(device_path="/dev/video8", capture_capable=True, serial_short="SN0001", serial_unique=False),
    ]
    enum = EnumerateResult(candidates=candidates, duplicate_serials=["SN0001"])

    def fake_enum(**kwargs):
        return enum

    def loader(path):
        return left.copy() if path == "/dev/video0" else right.copy()

    out = tmp_path / "mapping.json"
    with (
        patch("cv_camera_mapping.enumeration.enumerate_cameras", fake_enum),
        patch("cv_camera_mapping.assignment.enumerate_cameras", fake_enum),
        patch("cv_camera_mapping.assignment.capture_frame", loader),
    ):
        code = main(
            [
                "identify",
                "--references",
                str(refs_dir),
                "--json",
                "--output",
                str(out),
                "--confidence-threshold",
                "0.15",
                "--margin-threshold",
                "0.02",
                "--min-keypoints",
                "10",
                "--min-raw-matches",
                "8",
                "--min-inliers",
                "6",
            ]
        )
    assert code == 0
    data = json.loads(out.read_text())
    assert data["ambiguous"] is False
    assert "camera_left" in data["assignments"]
    assert data["schema_version"] == 1


def test_identify_ambiguous_exits_nonzero(tmp_path):

    gray = np.full((480, 640, 3), 128, dtype=np.uint8)
    refs_dir = tmp_path / "refs"
    from cv_camera_mapping.reference_capture import capture_references

    capture_references(
        refs_dir,
        ["camera_left", "camera_right"],
        {"camera_left": "/dev/video0", "camera_right": "/dev/video8"},
        frame_loader=lambda _: gray.copy(),
    )
    candidates = [
        CameraDevice(device_path="/dev/video0", capture_capable=True, serial_short="SN0001", serial_unique=False),
        CameraDevice(device_path="/dev/video8", capture_capable=True, serial_short="SN0001", serial_unique=False),
    ]

    fake_enum = lambda **k: EnumerateResult(
        candidates=candidates, duplicate_serials=["SN0001"]
    )
    with (
        patch("cv_camera_mapping.enumeration.enumerate_cameras", fake_enum),
        patch("cv_camera_mapping.assignment.enumerate_cameras", fake_enum),
        patch("cv_camera_mapping.assignment.capture_frame", lambda p: gray.copy()),
    ):
        code = main(
            [
                "identify",
                "--references",
                str(refs_dir),
                "--json",
            ]
        )
    assert code != 0
