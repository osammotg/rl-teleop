
import cv2

from cv_camera_mapping.assignment import score_visual_assignment
from cv_camera_mapping.schemas import CameraDevice, FrameMetadata, MatcherSettings, ReferenceSet, RoleReference


def _refs(tmp_path, img) -> ReferenceSet:
    for role in ("camera_left", "camera_right"):
        d = tmp_path / role
        d.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(d / "reference_0.jpg"), img)
    dev = CameraDevice(device_path="/dev/video0", capture_capable=True)
    roles = []
    for role, path in (("camera_left", "/dev/video0"), ("camera_right", "/dev/video8")):
        roles.append(
            RoleReference(
                role=role,
                device_path=path,
                image_paths=[f"{role}/reference_0.jpg"],
                device=dev,
                frame_metadata=FrameMetadata(width=640, height=480),
                captured_at="t",
            )
        )
    return ReferenceSet(roles=roles, matcher_settings=MatcherSettings())


def test_identical_images_fail_closed(tmp_path, ambiguous_twin_image):
    refs = _refs(tmp_path, ambiguous_twin_image)
    candidates = [
        CameraDevice(device_path="/dev/video0", capture_capable=True),
        CameraDevice(device_path="/dev/video8", capture_capable=True),
    ]
    result = score_visual_assignment(
        candidates,
        refs,
        refs.matcher_settings,
        lambda _: ambiguous_twin_image.copy(),
        tmp_path,
    )
    assert result.exit_code != 0
    assert result.artifact.ambiguous is True


def test_black_mat_low_features_fail_closed(tmp_path, black_mat_image):
    refs = _refs(tmp_path, black_mat_image)
    candidates = [
        CameraDevice(device_path="/dev/video0", capture_capable=True),
        CameraDevice(device_path="/dev/video8", capture_capable=True),
    ]
    result = score_visual_assignment(
        candidates,
        refs,
        refs.matcher_settings,
        lambda _: black_mat_image.copy(),
        tmp_path,
    )
    assert result.exit_code != 0
    assert result.artifact.failure_reason is not None
