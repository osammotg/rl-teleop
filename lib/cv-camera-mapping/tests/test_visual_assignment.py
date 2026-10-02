from pathlib import Path

import cv2
import numpy as np

from cv_camera_mapping.assignment import score_visual_assignment
from cv_camera_mapping.schemas import CameraDevice, FrameMetadata, MatcherSettings, ReferenceSet, RoleReference


def _refs(tmp_path: Path, left_img, right_img) -> ReferenceSet:
    left_dir = tmp_path / "camera_left"
    right_dir = tmp_path / "camera_right"
    left_dir.mkdir(parents=True)
    right_dir.mkdir(parents=True)
    left_path = left_dir / "reference_0.jpg"
    right_path = right_dir / "reference_0.jpg"
    cv2.imwrite(str(left_path), left_img)
    cv2.imwrite(str(right_path), right_img)
    dev0 = CameraDevice(device_path="/dev/video0", capture_capable=True, serial_short="SN0001")
    dev8 = CameraDevice(device_path="/dev/video8", capture_capable=True, serial_short="SN0001")
    return ReferenceSet(
        roles=[
            RoleReference(
                role="camera_left",
                device_path="/dev/video0",
                image_paths=["camera_left/reference_0.jpg"],
                device=dev0,
                frame_metadata=FrameMetadata(width=640, height=480),
                captured_at="t",
            ),
            RoleReference(
                role="camera_right",
                device_path="/dev/video8",
                image_paths=["camera_right/reference_0.jpg"],
                device=dev8,
                frame_metadata=FrameMetadata(width=640, height=480),
                captured_at="t",
            ),
        ],
        matcher_settings=MatcherSettings(
            min_keypoints=10,
            min_raw_matches=8,
            min_inliers=6,
            confidence_threshold=0.15,
            margin_threshold=0.02,
        ),
    )


def _make_left_view():
    rng = np.random.default_rng(1)
    img = rng.integers(40, 220, (480, 320, 3), dtype=np.uint8)
    out = np.zeros((480, 640, 3), dtype=np.uint8)
    out[:, :320] = img
    for y in range(0, 480, 24):
        for x in range(10, 300, 18):
            cv2.circle(out, (x, y), 5, (255, 180, 40), -1)
    return out


def _make_right_view():
    rng = np.random.default_rng(2)
    img = rng.integers(40, 220, (480, 320, 3), dtype=np.uint8)
    out = np.zeros((480, 640, 3), dtype=np.uint8)
    out[:, 320:] = img
    for y in range(0, 480, 22):
        for x in range(340, 620, 16):
            cv2.rectangle(out, (x, y), (x + 8, y + 8), (50, 255, 90), -1)
    return out


def test_swapped_candidates_are_corrected(tmp_path, left_perspective_image, right_perspective_image):
    refs = _refs(tmp_path, _make_left_view(), _make_right_view())
    candidates = [
        CameraDevice(device_path="/dev/video8", capture_capable=True),
        CameraDevice(device_path="/dev/video0", capture_capable=True),
    ]

    def loader(path):
        if path == "/dev/video0":
            return _make_left_view()
        return _make_right_view()

    result = score_visual_assignment(
        candidates, refs, refs.matcher_settings, loader, tmp_path
    )
    assert result.exit_code == 0
    assert result.artifact.assignments["camera_left"].device_path == "/dev/video0"
    assert result.artifact.assignments["camera_right"].device_path == "/dev/video8"
    assert result.artifact.assignments["camera_left"].method == "visual_fingerprint"
