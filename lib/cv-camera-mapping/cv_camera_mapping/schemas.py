"""JSON artifact schemas and resolver precedence documentation.

Resolver precedence (main repo + identify):
1. Explicit ``device_path`` in config — highest; unchanged when set alone.
2. Unique serial among capture-capable candidates — only when exactly one match.
3. USB path (``ID_PATH``) — port-stable only, not camera-identity-stable after port swaps.
4. Visual fingerprint — assigns ``camera_left`` / ``camera_right`` for duplicate serials.
5. Ambiguity — fail closed; never guess.

Serial burning / firmware flashing is excluded. Runtime VLM is excluded.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field

SCHEMA_VERSION = 1
REFERENCE_ROLES = ("camera_left", "camera_right")


class FailureReason(str, Enum):
    DUPLICATE_SERIAL = "duplicate_serial"
    INSUFFICIENT_CANDIDATES = "insufficient_candidates"
    TOO_MANY_CANDIDATES = "too_many_candidates"
    INSUFFICIENT_FEATURES = "insufficient_features"
    INSUFFICIENT_GEOMETRIC_INLIERS = "insufficient_geometric_inliers"
    SCORE_BELOW_THRESHOLD = "score_below_threshold"
    SCORE_MARGIN_BELOW_THRESHOLD = "score_margin_below_threshold"
    MAPPING_ARTIFACT_MISSING = "mapping_artifact_missing"
    REFERENCE_ARTIFACT_INVALID = "reference_artifact_invalid"
    ROLE_DEVICE_MAPPING_REQUIRED = "role_device_mapping_required"
    GEOMETRY_VERIFICATION_FAILED = "geometry_verification_failed"


class NodeKind(str, Enum):
    CAPTURE = "capture"
    NON_CAPTURE_OR_METADATA = "non_capture_or_metadata"


class CameraDevice(BaseModel):
    device_path: str
    resolved_path: Optional[str] = None
    capture_capable: bool = False
    serial: Optional[str] = None
    serial_short: Optional[str] = None
    serial_unique: Optional[bool] = None
    usb_path: Optional[str] = None
    vendor_id: Optional[str] = None
    model_id: Optional[str] = None
    product: Optional[str] = None
    devlinks: List[str] = Field(default_factory=list)
    backend: str = "v4l2"
    width: Optional[int] = None
    height: Optional[int] = None
    node_kind: NodeKind = NodeKind.NON_CAPTURE_OR_METADATA


class EnumerateResult(BaseModel):
    schema_version: int = SCHEMA_VERSION
    candidates: List[CameraDevice] = Field(default_factory=list)
    duplicate_serials: List[str] = Field(default_factory=list)
    message: Optional[str] = None


class ReferenceRole(str, Enum):
    CAMERA_LEFT = "camera_left"
    CAMERA_RIGHT = "camera_right"


class FrameMetadata(BaseModel):
    width: int
    height: int
    index: int = 0


class RoleReference(BaseModel):
    role: Literal["camera_left", "camera_right"]
    device_path: str
    image_paths: List[str] = Field(default_factory=list)
    device: CameraDevice
    frame_metadata: FrameMetadata
    captured_at: str


class MatcherSettings(BaseModel):
    feature_detector: str = "akaze"
    ratio_threshold: float = 0.75
    confidence_threshold: float = 0.75
    margin_threshold: float = 0.15
    min_inliers: int = 20
    min_keypoints: int = 50
    min_raw_matches: int = 30
    dark_mask_threshold: int = 30


class ReferenceSet(BaseModel):
    schema_version: int = SCHEMA_VERSION
    roles: List[RoleReference] = Field(default_factory=list)
    matcher_settings: MatcherSettings = Field(default_factory=MatcherSettings)
    created_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


class RoleScoreDetail(BaseModel):
    role: Literal["camera_left", "camera_right"]
    device_path: str
    confidence: float
    score: float
    match_count: int
    inlier_count: int
    keypoint_count: int
    method: str = "visual_fingerprint"


class Assignment(BaseModel):
    device_path: str
    confidence: float
    method: str
    score: Optional[float] = None
    score_margin: Optional[float] = None
    inlier_count: Optional[int] = None
    match_count: Optional[int] = None


class MappingArtifact(BaseModel):
    schema_version: int = SCHEMA_VERSION
    assignments: Dict[str, Assignment] = Field(default_factory=dict)
    ambiguous: bool = False
    failure_reason: Optional[FailureReason] = None
    mapper_version: Optional[str] = None
    created_at: Optional[str] = None
    score_details: List[RoleScoreDetail] = Field(default_factory=list)
    source_candidates: List[CameraDevice] = Field(default_factory=list)


class IdentificationResult(BaseModel):
    artifact: MappingArtifact
    exit_code: int
