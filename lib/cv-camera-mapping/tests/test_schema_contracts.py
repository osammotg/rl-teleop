from cv_camera_mapping.schemas import FailureReason, MappingArtifact


def test_success_artifact_shape():
    artifact = MappingArtifact.model_validate(
        {
            "schema_version": 1,
            "assignments": {
                "camera_left": {
                    "device_path": "/dev/video0",
                    "confidence": 0.91,
                    "method": "visual_fingerprint",
                }
            },
            "ambiguous": False,
        }
    )
    assert artifact.assignments["camera_left"].device_path == "/dev/video0"


def test_failure_artifact_shape():
    artifact = MappingArtifact.model_validate(
        {
            "schema_version": 1,
            "assignments": {},
            "ambiguous": True,
            "failure_reason": FailureReason.SCORE_MARGIN_BELOW_THRESHOLD,
        }
    )
    assert artifact.failure_reason == FailureReason.SCORE_MARGIN_BELOW_THRESHOLD


def test_failure_reasons_enum():
    expected = {
        "duplicate_serial",
        "insufficient_candidates",
        "too_many_candidates",
        "insufficient_features",
        "insufficient_geometric_inliers",
        "score_below_threshold",
        "score_margin_below_threshold",
        "mapping_artifact_missing",
        "reference_artifact_invalid",
    }
    assert expected.issubset({r.value for r in FailureReason})
