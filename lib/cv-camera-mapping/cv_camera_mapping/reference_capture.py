"""Reference enrollment for camera_left / camera_right."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import cv2

from cv_camera_mapping.enumeration import build_camera_device, capture_frame, udevadm_info
from cv_camera_mapping.schemas import (
    FailureReason,
    FrameMetadata,
    MappingArtifact,
    MatcherSettings,
    ReferenceSet,
    RoleReference,
)


def parse_role_devices(role_device_args: List[str]) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    for item in role_device_args:
        if "=" not in item:
            raise ValueError(f"Invalid --role-device value: {item!r}")
        role, path = item.split("=", 1)
        if role not in ("camera_left", "camera_right"):
            raise ValueError(f"Unknown role {role!r}")
        mapping[role] = path
    return mapping


def capture_references(
    output_dir: Path,
    roles: List[str],
    role_devices: Dict[str, str],
    resolution: tuple[int, int] = (640, 480),
    fps: int = 30,
    settings: Optional[MatcherSettings] = None,
    frame_loader=None,
) -> ReferenceSet:
    if not role_devices:
        raise ValueError(FailureReason.ROLE_DEVICE_MAPPING_REQUIRED.value)

    required = set(roles)
    if not required.issubset(role_devices.keys()):
        missing = required - role_devices.keys()
        raise ValueError(
            f"role_device_mapping_required: missing {sorted(missing)}"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    matcher_settings = settings or MatcherSettings()
    loader = frame_loader or (lambda p: capture_frame(p, resolution=resolution, fps=fps))
    role_refs: List[RoleReference] = []

    for role in sorted(role_devices.keys()):
        device_path = role_devices[role]
        props = udevadm_info(device_path)
        frame = loader(device_path)
        role_dir = output_dir / role
        role_dir.mkdir(parents=True, exist_ok=True)
        image_path = role_dir / "reference_0.jpg"
        cv2.imwrite(str(image_path), frame)
        h, w = frame.shape[:2]
        device = build_camera_device(
            device_path,
            props,
            capture_capable=True,
            width=w,
            height=h,
        )
        role_refs.append(
            RoleReference(
                role=role,
                device_path=device_path,
                image_paths=[str(image_path.relative_to(output_dir))],
                device=device,
                frame_metadata=FrameMetadata(width=w, height=h, index=0),
                captured_at=datetime.now(timezone.utc).isoformat(),
            )
        )

    ref_set = ReferenceSet(
        roles=role_refs,
        matcher_settings=matcher_settings,
        created_at=datetime.now(timezone.utc).isoformat(),
    )
    manifest = output_dir / "reference_set.json"
    manifest.write_text(ref_set.model_dump_json(indent=2))
    return ref_set


def validate_mapping_artifact(path: Path) -> MappingArtifact:
    if not path.exists():
        raise FileNotFoundError(str(path))
    data = json.loads(path.read_text())
    artifact = MappingArtifact.model_validate(data)
    if artifact.schema_version != 1:
        raise ValueError(f"Unsupported schema_version: {artifact.schema_version}")
    if artifact.ambiguous:
        reason = artifact.failure_reason
        msg = reason.value if reason is not None else "ambiguous mapping"
        raise ValueError(msg)
    return artifact
