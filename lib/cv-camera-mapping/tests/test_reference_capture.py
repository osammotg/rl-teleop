
import numpy as np

from cv_camera_mapping.reference_capture import capture_references


def test_non_interactive_reference_capture(tmp_path):
    frame = np.zeros((480, 640, 3), dtype=np.uint8) + 80

    def loader(path):
        return frame.copy()

    capture_references(
        tmp_path,
        ["camera_left", "camera_right"],
        {
            "camera_left": "/dev/video0",
            "camera_right": "/dev/video8",
        },
        frame_loader=loader,
    )
    assert (tmp_path / "reference_set.json").exists()
    assert (tmp_path / "camera_left" / "reference_0.jpg").exists()
    assert (tmp_path / "camera_right" / "reference_0.jpg").exists()
