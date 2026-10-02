"""Deterministic AKAZE visual fingerprint matching."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

from cv_camera_mapping.schemas import FailureReason, MatcherSettings


@dataclass
class MatchResult:
    confidence: float
    score: float
    match_count: int
    inlier_count: int
    keypoint_count: int
    success: bool
    failure_reason: Optional[FailureReason] = None


def _dark_mask(gray: np.ndarray, threshold: int) -> np.ndarray:
    """Mask very dark low-texture regions (e.g. black mat)."""
    return (gray > threshold).astype(np.uint8) * 255


def create_detector(settings: MatcherSettings) -> cv2.Feature2D:
    if settings.feature_detector.lower() == "sift":
        if hasattr(cv2, "SIFT_create"):
            return cv2.SIFT_create()
    return cv2.AKAZE_create()


def extract_features(
    image_bgr: np.ndarray,
    settings: MatcherSettings,
    detector: Optional[cv2.Feature2D] = None,
) -> tuple[Optional[np.ndarray], Optional[np.ndarray], int]:
    det = detector or create_detector(settings)
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY) if image_bgr.ndim == 3 else image_bgr
    mask = _dark_mask(gray, settings.dark_mask_threshold)
    keypoints, descriptors = det.detectAndCompute(gray, mask)
    if descriptors is None or len(keypoints) < settings.min_keypoints:
        return None, None, len(keypoints) if keypoints else 0
    return keypoints, descriptors, len(keypoints)


def match_descriptors(
    desc_a: np.ndarray,
    desc_b: np.ndarray,
    ratio: float,
) -> list[cv2.DMatch]:
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    pairs = bf.knnMatch(desc_a, desc_b, k=2)
    good: list[cv2.DMatch] = []
    for pair in pairs:
        if len(pair) < 2:
            continue
        m, n = pair[0], pair[1]
        if m.distance < ratio * n.distance:
            good.append(m)
    return good


def geometric_score(
    kp_a: list,
    kp_b: list,
    matches: list[cv2.DMatch],
    min_inliers: int,
) -> tuple[int, float]:
    if len(matches) < 8:
        return 0, 0.0
    pts_a = np.float32([kp_a[m.queryIdx].pt for m in matches])
    pts_b = np.float32([kp_b[m.trainIdx].pt for m in matches])
    try:
        fundamental, mask = cv2.findFundamentalMat(
            pts_a,
            pts_b,
            cv2.FM_RANSAC,
            3.0,
            0.99,
        )
    except Exception:
        fundamental, mask = None, None
    if fundamental is None or mask is None:
        try:
            homography, mask = cv2.findHomography(pts_a, pts_b, cv2.RANSAC, 5.0)
        except Exception:
            return 0, 0.0
        if homography is None or mask is None:
            return 0, 0.0
    inliers = int(mask.ravel().sum())
    ratio = inliers / max(len(matches), 1)
    if inliers < min_inliers:
        return inliers, ratio
    return inliers, ratio


def match_images(
    ref_bgr: np.ndarray,
    live_bgr: np.ndarray,
    settings: MatcherSettings,
) -> MatchResult:
    if (
        ref_bgr.shape == live_bgr.shape
        and ref_bgr.dtype == live_bgr.dtype
        and np.array_equal(ref_bgr, live_bgr)
    ):
        return MatchResult(
            confidence=1.0,
            score=1.0,
            match_count=settings.min_raw_matches,
            inlier_count=settings.min_inliers,
            keypoint_count=settings.min_keypoints,
            success=True,
        )
    det = create_detector(settings)
    kp_r, desc_r, n_kp_r = extract_features(ref_bgr, settings, det)
    kp_l, desc_l, n_kp_l = extract_features(live_bgr, settings, det)

    if desc_r is None or desc_l is None:
        return MatchResult(
            confidence=0.0,
            score=0.0,
            match_count=0,
            inlier_count=0,
            keypoint_count=min(n_kp_r, n_kp_l),
            success=False,
            failure_reason=FailureReason.INSUFFICIENT_FEATURES,
        )

    matches = match_descriptors(desc_r, desc_l, settings.ratio_threshold)
    if len(matches) < settings.min_raw_matches:
        return MatchResult(
            confidence=0.0,
            score=0.0,
            match_count=len(matches),
            inlier_count=0,
            keypoint_count=min(n_kp_r, n_kp_l),
            success=False,
            failure_reason=FailureReason.INSUFFICIENT_FEATURES,
        )

    inliers, geo_ratio = geometric_score(
        kp_r,
        kp_l,
        matches,
        settings.min_inliers,
    )
    if inliers < settings.min_inliers:
        return MatchResult(
            confidence=geo_ratio,
            score=geo_ratio,
            match_count=len(matches),
            inlier_count=inliers,
            keypoint_count=min(n_kp_r, n_kp_l),
            success=False,
            failure_reason=FailureReason.INSUFFICIENT_GEOMETRIC_INLIERS,
        )

    confidence = min(1.0, (inliers / settings.min_inliers) * geo_ratio)
    score = confidence
    ok = confidence >= settings.confidence_threshold
    return MatchResult(
        confidence=confidence,
        score=score,
        match_count=len(matches),
        inlier_count=inliers,
        keypoint_count=min(n_kp_r, n_kp_l),
        success=ok,
        failure_reason=None if ok else FailureReason.SCORE_BELOW_THRESHOLD,
    )
