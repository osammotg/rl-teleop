from __future__ import annotations

import cv2
import numpy as np
import pytest


@pytest.fixture
def left_perspective_image() -> np.ndarray:
    """Distinct left-wrist-like view: strong features on the left side."""
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    img[:, :200] = [40, 180, 220]
    for y in range(0, 480, 40):
        for x in range(20, 180, 25):
            cv2.circle(img, (x, y), 8, (255, 200, 50), -1)
    return img


@pytest.fixture
def right_perspective_image() -> np.ndarray:
    """Distinct right-wrist-like view: strong features on the right side."""
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    img[:, 440:] = [220, 80, 160]
    for y in range(0, 480, 35):
        for x in range(460, 620, 30):
            cv2.rectangle(img, (x, y), (x + 12, y + 12), (80, 255, 120), -1)
    return img


@pytest.fixture
def black_mat_image() -> np.ndarray:
    return np.zeros((480, 640, 3), dtype=np.uint8)


@pytest.fixture
def ambiguous_twin_image() -> np.ndarray:
    img = np.full((480, 640, 3), 128, dtype=np.uint8)
    return img
