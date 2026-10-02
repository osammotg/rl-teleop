from cv_camera_mapping.assignment import filter_candidates_from_references
from cv_camera_mapping.schemas import CameraDevice, FrameMetadata, ReferenceSet, RoleReference


def test_filters_extra_realsense_by_enrolled_product():
    innomaker = CameraDevice(
        device_path="/dev/video0",
        capture_capable=True,
        product="Innomaker-U20CAM-1080p-S1",
    )
    realsense = CameraDevice(
        device_path="/dev/video6",
        capture_capable=True,
        product="Intel_R__RealSense_TM__Depth_Camera_455",
    )
    refs = ReferenceSet(
        roles=[
            RoleReference(
                role="camera_left",
                device_path="/dev/video0",
                image_paths=["camera_left/reference_0.jpg"],
                device=innomaker,
                frame_metadata=FrameMetadata(width=640, height=480),
                captured_at="t",
            )
        ]
    )
    filtered = filter_candidates_from_references(
        [innomaker, realsense, CameraDevice(device_path="/dev/video8", capture_capable=True, product="Innomaker-U20CAM-1080p-S1")],
        refs,
    )
    assert len(filtered) == 2
    assert all("Innomaker" in (c.product or "") for c in filtered)
