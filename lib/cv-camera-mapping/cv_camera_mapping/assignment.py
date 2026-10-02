"""Role assignment: serial (unique only), USB path, visual fingerprint."""

from __future__ import annotations

import itertools
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np

from cv_camera_mapping import __version__ as mapper_version
from cv_camera_mapping.enumeration import capture_frame, enumerate_cameras
from cv_camera_mapping.matcher import match_images
from cv_camera_mapping.schemas import (
    SCHEMA_VERSION,
    Assignment,
    CameraDevice,
    FailureReason,
    IdentificationResult,
    MappingArtifact,
    MatcherSettings,
    ReferenceSet,
    RoleScoreDetail,
)


def load_reference_set(references_dir: Path) -> ReferenceSet:
    manifest = references_dir / "reference_set.json"
    if not manifest.exists():
        raise ValueError(f"Missing reference_set.json in {references_dir}")
    return ReferenceSet.model_validate_json(manifest.read_text())


def load_reference_image(path: Path) -> np.ndarray:
    img = cv2.imread(str(path))
    if img is None:
        raise ValueError(f"Cannot read reference image: {path}")
    return img


def assign_by_unique_serial(
    candidates: List[CameraDevice],
    references: ReferenceSet,
) -> Optional[Dict[str, Assignment]]:
    """Assign when each role's enrolled serial is unique among candidates."""
    if any(d.serial_short and d.serial_unique is False for d in candidates):
        return None
    by_serial = {d.serial_short: d for d in candidates if d.serial_short}
    assignments: Dict[str, Assignment] = {}
    for role_ref in references.roles:
        serial = role_ref.device.serial_short
        if not serial or serial not in by_serial:
            return None
        device = by_serial[serial]
        assignments[role_ref.role] = Assignment(
            device_path=device.device_path,
            confidence=1.0,
            method="unique_serial",
        )
    if set(assignments.keys()) != {"camera_left", "camera_right"}:
        return None
    paths = [a.device_path for a in assignments.values()]
    if len(paths) != len(set(paths)):
        return None
    return assignments


def score_visual_assignment(
    candidates: List[CameraDevice],
    references: ReferenceSet,
    settings: MatcherSettings,
    frame_loader,
    references_dir: Path,
) -> IdentificationResult:
    """Match live frames to references; try both permutations for two cameras."""
    if len(candidates) < 2:
        return _failure(
            FailureReason.INSUFFICIENT_CANDIDATES,
            candidates,
            settings,
        )
    if len(candidates) > 2:
        return _failure(
            FailureReason.TOO_MANY_CANDIDATES,
            candidates,
            settings,
        )

    ref_by_role = {r.role: r for r in references.roles}
    if set(ref_by_role.keys()) != {"camera_left", "camera_right"}:
        return _failure(
            FailureReason.REFERENCE_ARTIFACT_INVALID,
            candidates,
            settings,
        )

    live_frames: Dict[str, np.ndarray] = {}
    for device in candidates:
        live_frames[device.device_path] = frame_loader(device.device_path)

    ref_images = {
        role: load_reference_image(references_dir / ref_by_role[role].image_paths[0])
        for role in ("camera_left", "camera_right")
    }

    permutations = list(itertools.permutations(candidates, 2))
    permutation_scores: List[tuple[float, float, Dict[str, Assignment], List[RoleScoreDetail]]] = []

    for left_dev, right_dev in permutations:
        mapping = {
            "camera_left": left_dev.device_path,
            "camera_right": right_dev.device_path,
        }
        details: List[RoleScoreDetail] = []
        scores: List[float] = []
        failed = False

        for role, device_path in mapping.items():
            result = match_images(
                ref_images[role],
                live_frames[device_path],
                settings,
            )
            details.append(
                RoleScoreDetail(
                    role=role,
                    device_path=device_path,
                    confidence=result.confidence,
                    score=result.score,
                    match_count=result.match_count,
                    inlier_count=result.inlier_count,
                    keypoint_count=result.keypoint_count,
                )
            )
            if not result.success:
                failed = True
            scores.append(result.confidence)

        if failed:
            continue
        total = sum(scores) / len(scores)
        margin = min(scores) - max(0.0, min(scores)) if len(scores) == 2 else scores[0]
        # margin between this permutation and others computed later
        assignments = {
            role: Assignment(
                device_path=path,
                confidence=details[i].confidence,
                method="visual_fingerprint",
                score=details[i].score,
                inlier_count=details[i].inlier_count,
                match_count=details[i].match_count,
            )
            for i, (role, path) in enumerate(mapping.items())
        }
        permutation_scores.append((total, min(scores), assignments, details))

    if not permutation_scores:
        return _failure(
            FailureReason.INSUFFICIENT_FEATURES,
            candidates,
            settings,
        )

    permutation_scores.sort(key=lambda x: x[0], reverse=True)
    best_total, best_min, best_assignments, best_details = permutation_scores[0]
    second_total = permutation_scores[1][0] if len(permutation_scores) > 1 else 0.0
    margin = best_total - second_total

    if best_min < settings.confidence_threshold:
        return _failure(
            FailureReason.SCORE_BELOW_THRESHOLD,
            candidates,
            settings,
            score_details=best_details,
        )
    if margin < settings.margin_threshold:
        return _failure(
            FailureReason.SCORE_MARGIN_BELOW_THRESHOLD,
            candidates,
            settings,
            score_details=best_details,
        )

    for role, asn in best_assignments.items():
        asn.score_margin = margin

    artifact = MappingArtifact(
        schema_version=SCHEMA_VERSION,
        assignments=best_assignments,
        ambiguous=False,
        mapper_version=mapper_version,
        created_at=datetime.now(timezone.utc).isoformat(),
        score_details=best_details,
        source_candidates=candidates,
    )
    return IdentificationResult(artifact=artifact, exit_code=0)


def filter_candidates(
    candidates: List[CameraDevice],
    *,
    include_devices: Optional[List[str]] = None,
    exclude_devices: Optional[List[str]] = None,
    product_substring: Optional[str] = None,
) -> List[CameraDevice]:
    out = list(candidates)
    if include_devices:
        allowed = set(include_devices)
        out = [c for c in out if c.device_path in allowed]
    if exclude_devices:
        blocked = set(exclude_devices)
        out = [c for c in out if c.device_path not in blocked]
    if product_substring:
        needle = product_substring.lower()
        out = [
            c
            for c in out
            if (c.product and needle in c.product.lower())
            or (c.serial and needle in c.serial.lower())
        ]
    return out


def filter_candidates_from_references(
    candidates: List[CameraDevice],
    references: ReferenceSet,
) -> List[CameraDevice]:
    """When extra cameras are present (e.g. RealSense), keep enrolled product family."""
    ref_products = {r.device.product for r in references.roles if r.device.product}
    if not ref_products:
        return candidates
    filtered = [c for c in candidates if c.product in ref_products]
    return filtered if len(filtered) >= 2 else candidates


def identify_roles(
    references_dir: Path,
    settings: Optional[MatcherSettings] = None,
    include_non_capture: bool = False,
    frame_loader=None,
    enumerate_fn=None,
    include_devices: Optional[List[str]] = None,
    exclude_devices: Optional[List[str]] = None,
    product_filter: Optional[str] = None,
) -> IdentificationResult:
    if enumerate_fn is None:
        enumerate_fn = enumerate_cameras
    references = load_reference_set(references_dir)
    matcher_settings = settings or references.matcher_settings
    enum = enumerate_fn(include_non_capture=include_non_capture)
    candidates = [c for c in enum.candidates if c.capture_capable]
    candidates = filter_candidates(
        candidates,
        include_devices=include_devices,
        exclude_devices=exclude_devices,
        product_substring=product_filter,
    )
    if len(candidates) > 2:
        candidates = filter_candidates_from_references(candidates, references)

    if enum.duplicate_serials:
        # duplicate serial cannot distinguish left/right — use visual only
        pass

    serial_assign = assign_by_unique_serial(candidates, references)
    if serial_assign is not None:
        artifact = MappingArtifact(
            schema_version=SCHEMA_VERSION,
            assignments=serial_assign,
            ambiguous=False,
            mapper_version=mapper_version,
            created_at=datetime.now(timezone.utc).isoformat(),
            source_candidates=candidates,
        )
        return IdentificationResult(artifact=artifact, exit_code=0)

    loader = frame_loader or (lambda p: capture_frame(p))
    if len(candidates) == 2:
        return score_visual_assignment(
            candidates,
            references,
            matcher_settings,
            loader,
            references_dir,
        )
    if len(candidates) < 2:
        return _failure(FailureReason.INSUFFICIENT_CANDIDATES, candidates, matcher_settings)
    return _failure(FailureReason.TOO_MANY_CANDIDATES, candidates, matcher_settings)


def _failure(
    reason: FailureReason,
    candidates: List[CameraDevice],
    settings: MatcherSettings,
    score_details: Optional[List[RoleScoreDetail]] = None,
) -> IdentificationResult:
    artifact = MappingArtifact(
        schema_version=SCHEMA_VERSION,
        assignments={},
        ambiguous=True,
        failure_reason=reason,
        mapper_version=mapper_version,
        created_at=datetime.now(timezone.utc).isoformat(),
        score_details=score_details or [],
        source_candidates=candidates,
    )
    return IdentificationResult(artifact=artifact, exit_code=1)
