"""End-to-end labeler: pure golden fixture + full MCAP round-trip."""
from __future__ import annotations

import numpy as np
import pytest

from robots_realtime.labeling.fk import ForwardKinematics
from robots_realtime.labeling.label_episode import label_episode_dir, label_from_arrays
from robots_realtime.labeling.schema import KitItem

DT = 0.02
OPEN, HELD = 1.0, 0.35          # gripper raw: open / holding a bag


@pytest.fixture(scope="module")
def fk():
    return ForwardKinematics("urdf/yam.urdf")


def _low_high(fk) -> tuple[np.ndarray, np.ndarray]:
    """(low, high) configs: home (zeros) is near the top, so 'high' = home and
    'low' is a config whose EE sits >=6cm below it. A lift goes low → high."""
    high = np.zeros(6)
    z_high = fk.ee_pose(high)[2]
    for v in np.linspace(0.1, 1.5, 71):
        low = np.zeros(6); low[1] = v           # joint2 lowers the wrist
        if z_high - fk.ee_pose(low)[2] > 0.06:
            return low, high
    raise AssertionError("could not find a lowered config")


def _two_bag_episode(fk):
    """Synthesize a 2-bag pick-place joint timeline (times, positions Nx7)."""
    low, high = _low_high(fk)
    # (duration_s, arm_joints, gripper_raw). Gripper closes while LOW (grasp),
    # THEN the arm rises to HIGH (lift) before lowering to place — so the lift
    # is a real rise AFTER t_close, matching how detection works.
    def bag():
        return [
            (0.8, low,  OPEN),   # reach
            (0.5, low,  HELD),   # grasp (gripper closes while low)
            (1.0, high, HELD),   # lift + transport
            (0.5, low,  HELD),   # lower to place height
            (0.4, low,  OPEN),   # release
        ]
    plan = bag() + [(0.6, low, OPEN)] + bag()   # two bags, brief gap between
    times, positions, t = [], [], 100.0   # start at epoch-ish 100s
    for dur, joints, grip in plan:
        for _ in range(int(dur / DT)):
            positions.append([*joints[:6], grip])   # 7-vec: 6 arm joints + gripper
            times.append(t); t += DT
    return np.array(times), np.array(positions)


def test_golden_two_bag(fk):
    times, positions = _two_bag_episode(fk)
    ann = label_from_arrays(
        times, positions, fk=fk, arm="left", episode_id="golden",
        gripper_open_ref=OPEN, gripper_closed_ref=0.0,
        kitting_list=[KitItem(1, "P1", compartment=5), KitItem(2, "P2", compartment=3)],
    )
    assert len(ann.place_events) == 2
    assert [p.bag_id for p in ann.place_events] == [1, 2]
    assert [p.target_compartment for p in ann.place_events] == [5, 3]
    assert len(ann.grasp_attempts) == 2
    assert all(g.outcome == "success" for g in ann.grasp_attempts)
    # poses attached and finite
    for g in ann.grasp_attempts:
        assert g.ee_pose is not None and len(g.ee_pose) == 7
    assert ann.episode_meta.outcome == "success"
    assert not [f for f in ann.flags if f.kind == "overlapping_grasp"]


def test_labeler_is_idempotent(fk):
    times, positions = _two_bag_episode(fk)
    a = label_from_arrays(times, positions, fk=fk, gripper_open_ref=OPEN, gripper_closed_ref=0.0)
    b = label_from_arrays(times, positions, fk=fk, gripper_open_ref=OPEN, gripper_closed_ref=0.0)
    assert a.to_dict() == b.to_dict()


def test_full_mcap_round_trip(fk, tmp_path):
    """Write a real yam_left.mcap via the recorder, then label the dir."""
    from robots_realtime.runtime.recording import McapWriter

    times, positions = _two_bag_episode(fk)
    w = McapWriter()
    w.open(str(tmp_path), "yam_left")
    for t, pos in zip(times, positions):
        w.write("joint_state", float(t), {"joint_pos": list(pos)})
    w.close()

    ann = label_episode_dir(tmp_path, arm="left",
                            gripper_open_ref=OPEN, gripper_closed_ref=0.0)
    assert (tmp_path / "annotations.json").exists()
    assert len(ann.place_events) == 2
    assert len(ann.grasp_attempts) == 2
