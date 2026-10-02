from cv_camera_mapping.assignment import assign_by_unique_serial
from cv_camera_mapping.schemas import CameraDevice, FrameMetadata, ReferenceSet, RoleReference


def _device(path: str, serial: str, unique: bool) -> CameraDevice:
    return CameraDevice(
        device_path=path,
        capture_capable=True,
        serial_short=serial,
        serial_unique=unique,
    )


def test_duplicate_serials_do_not_assign_by_serial():
    candidates = [
        _device("/dev/video0", "SN0001", False),
        _device("/dev/video8", "SN0001", False),
    ]
    refs = ReferenceSet(
        roles=[
            RoleReference(
                role="camera_left",
                device_path="/dev/video0",
                image_paths=["camera_left/reference_0.jpg"],
                device=_device("/dev/video0", "SN0001", False),
                frame_metadata=FrameMetadata(width=640, height=480),
                captured_at="t",
            ),
            RoleReference(
                role="camera_right",
                device_path="/dev/video8",
                image_paths=["camera_right/reference_0.jpg"],
                device=_device("/dev/video8", "SN0001", False),
                frame_metadata=FrameMetadata(width=640, height=480),
                captured_at="t",
            ),
        ]
    )
    assert assign_by_unique_serial(candidates, refs) is None
