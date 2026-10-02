"""FK correctness + the regression guard against URDF/joint-order drift."""
from __future__ import annotations

import numpy as np
import pytest

from robots_realtime.labeling.fk import ForwardKinematics, tracking_error

# Baked zero-configuration link_6 pose. If the URDF, joint order, or EE link
# changes, this shifts and the test fails loudly — the guard against every
# grasp/place pose being silently offset. (Regenerate deliberately if the model
# legitimately changes.)
ZERO_POSE = [0.110297, 0.000002, 0.164001, 0.500002, 0.499996, 0.500001, 0.500001]


@pytest.fixture(scope="module")
def fk():
    return ForwardKinematics("urdf/yam.urdf")


def test_zero_config_pose_guard(fk):
    p = fk.ee_pose([0, 0, 0, 0, 0, 0, 0])
    assert np.allclose(p, ZERO_POSE, atol=1e-4), (
        f"link_6 zero-config pose drifted: got {p}. If the URDF/joint order "
        f"changed on purpose, update ZERO_POSE; otherwise FK is now wrong."
    )


def test_gripper_joint_ignored(fk):
    # 7th value (gripper) must not affect the arm EE pose.
    a = fk.ee_pose([0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 0.0])
    b = fk.ee_pose([0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 0.9])
    assert np.allclose(a, b)


def test_deterministic(fk):
    j = [0.3, -0.2, 0.5, 0.1, -0.4, 0.2]
    assert fk.ee_pose(j) == fk.ee_pose(j)


def test_within_reach_envelope(fk):
    for j in ([0, 0, 0, 0, 0, 0], [0.5, 0.5, 0.5, 0.5, 0.5, 0.5], [-1, 1, -1, 1, -1, 1]):
        pos = np.array(fk.ee_pose(j)[:3])
        assert 0.0 < np.linalg.norm(pos) < 1.5   # YAM arm reach

    q = np.array(fk.ee_pose([0.2, 0.3, 0.4, 0.5, 0.6, 0.7])[3:])
    assert abs(np.linalg.norm(q) - 1.0) < 1e-6   # unit quaternion


def test_tracking_error():
    # 2cm constant offset in x
    commanded = np.zeros((10, 3))
    achieved = np.tile([0.02, 0.0, 0.0], (10, 1))
    rms, mx = tracking_error(commanded, achieved)
    assert abs(rms - 0.02) < 1e-9
    assert abs(mx - 0.02) < 1e-9
    assert tracking_error(np.zeros((0, 3)), np.zeros((0, 3))) == (0.0, 0.0)
