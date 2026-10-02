"""Forward kinematics for the YAM arm (labeling side).

Given the 6 arm joint angles (the first 6 of the recorded 7-DoF joint vector;
index 6 is the gripper), compute the end-effector pose at ``link_6`` in the
robot base frame. Used to attach a 6-DoF pose to every grasp/release keyframe
and to compute commanded-vs-achieved tracking error.

FK is via yourdfpy (already a repo dependency), kinematics only — meshes are not
loaded. The URDF gives the wrist link; the real grasp point is offset down the
gripper, so callers may add a tool offset.

GUARD: a regression test bakes in the zero-configuration EE pose. If the URDF,
the joint order, or the EE link ever changes, that pose shifts and the test
fails loudly — this is the guard against the silent "all poses shifted" bug.
"""
from __future__ import annotations

import functools
from pathlib import Path

import numpy as np

from robots_realtime.labeling import constants as C

_ARM_JOINTS = [f"joint{i}" for i in range(1, C.N_ARM_JOINTS + 1)]  # joint1..joint6


def _mat_to_quat(m: np.ndarray) -> np.ndarray:
    """3x3 rotation matrix → quaternion [w, x, y, z]."""
    tr = m[0, 0] + m[1, 1] + m[2, 2]
    if tr > 0:
        s = np.sqrt(tr + 1.0) * 2
        w = 0.25 * s
        x = (m[2, 1] - m[1, 2]) / s
        y = (m[0, 2] - m[2, 0]) / s
        z = (m[1, 0] - m[0, 1]) / s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
        w = (m[2, 1] - m[1, 2]) / s
        x = 0.25 * s
        y = (m[0, 1] + m[1, 0]) / s
        z = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
        w = (m[0, 2] - m[2, 0]) / s
        x = (m[0, 1] + m[1, 0]) / s
        y = 0.25 * s
        z = (m[1, 2] + m[2, 1]) / s
    else:
        s = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
        w = (m[1, 0] - m[0, 1]) / s
        x = (m[0, 2] + m[2, 0]) / s
        y = (m[1, 2] + m[2, 1]) / s
        z = 0.25 * s
    q = np.array([w, x, y, z], dtype=float)
    return q / np.linalg.norm(q)


@functools.lru_cache(maxsize=2)
def _load_urdf(urdf_path: str):
    import yourdfpy
    return yourdfpy.URDF.load(
        urdf_path, load_meshes=False, build_collision_scene_graph=False
    )


class ForwardKinematics:
    """Cached FK for one URDF. FK is a pure function of the 6 arm joints."""

    def __init__(self, urdf_path: str | Path = "urdf/yam.urdf",
                 ee_link: str = C.FK_EE_LINK):
        self._robot = _load_urdf(str(urdf_path))
        self._ee_link = ee_link

    def ee_pose(self, joints) -> list[float]:
        """6 arm joints (or 7 with gripper; extra are ignored) → [x,y,z,qw,qx,qy,qz]."""
        j = np.asarray(joints, dtype=float)[: C.N_ARM_JOINTS]
        cfg = {name: float(val) for name, val in zip(_ARM_JOINTS, j)}
        self._robot.update_cfg(cfg)
        T = np.asarray(self._robot.get_transform(self._ee_link, "base_link"))
        pos = T[:3, 3]
        quat = _mat_to_quat(T[:3, :3])
        return [*pos.tolist(), *quat.tolist()]

    def ee_positions(self, joints_seq) -> np.ndarray:
        """FK a timeline of joint vectors → (N, 3) positions (for lift/tracking)."""
        return np.array([self.ee_pose(j)[:3] for j in joints_seq], dtype=float)


def tracking_error(commanded_pos: np.ndarray, achieved_pos: np.ndarray) -> tuple[float, float]:
    """RMS and max Euclidean position error between two (N,3) EE-position paths."""
    c = np.asarray(commanded_pos, float)
    a = np.asarray(achieved_pos, float)
    n = min(len(c), len(a))
    if n == 0:
        return 0.0, 0.0
    d = np.linalg.norm(c[:n] - a[:n], axis=1)
    return float(np.sqrt(np.mean(d ** 2))), float(np.max(d))
