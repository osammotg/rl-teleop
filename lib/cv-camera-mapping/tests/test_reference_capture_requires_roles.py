import pytest

from cv_camera_mapping.reference_capture import capture_references


def test_capture_reference_requires_role_device(tmp_path):
    with pytest.raises(ValueError, match="role_device_mapping_required"):
        capture_references(tmp_path, ["camera_left", "camera_right"], {})
