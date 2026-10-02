"""Two-arm export: a take where only one arm moves at a time.

The operator cannot teleop both arms at once, so the bimanual recording flow is
ONE continuous episode with a handoff in the middle: drive the right arm from
the source box to the mat, let go, then drive the left arm from the mat into the
kit box. Every test here is about that shape of data.

The load-bearing decision, tested rather than only documented: the idle arm is
represented by its own recorded values, not masked and not given a separate
action space. See window_rows()'s docstring for the argument; the tests below
pin the consequences -- 14-DoF vectors, left before right, and a veto on the one
case where "just record what happened" is unsafe (a leader parked away from its
follower, which would train the policy to snap the idle arm).

The compatibility guarantee runs through all of it: --arms left is the default
and must reproduce the single-arm datasets exactly.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from export_lerobot import (  # noqa: E402
    ARM_SETS, CAMERA_SETS, IDLE_ARM_DIVERGENCE_MAX_RAD, JOINT_NAMES, N_DOF,
    Report, arm_activity, build_features, idle_arm_veto, joint_names, n_dof,
    plan_episode, resolve_arms, window_rows,
)
from robots_realtime.labeling import constants as C  # noqa: E402
from robots_realtime.labeling.label_episode import annotations_path  # noqa: E402


# ── fixtures ────────────────────────────────────────────────────────────────
def _ann(attempts):
    return {"episode_meta": {"outcome": "success"}, "segments": [],
            "grasp_attempts": attempts, "place_events": [], "tracking": [], "flags": []}


def _grasp(t, x=0.43, y=-0.25):
    return {"bag_id": 1, "attempt": 1, "arm": "left", "t": t, "outcome": "success",
            "ee_pose": [x, y, 0.12, 0.0, 0.0, 0.0, 1.0]}


def _traj(n, moving: bool, base: float, t0=0.0, hz=200.0, lead_offset=0.0):
    """(times, positions) for one arm. `moving` sweeps joint 1; otherwise it
    holds `base` with encoder-scale noise, which is what a parked follower does.
    `lead_offset` is added to make a LEADER stream that disagrees with it."""
    t = t0 + np.arange(n) / hz
    p = np.full((n, C.N_ARM_JOINTS + 1), base, dtype=float)
    rng = np.random.default_rng(3)
    p += rng.normal(0, 1e-4, p.shape)
    if moving:
        p[:, 0] = base + np.linspace(0.0, 0.9, n)
    p[:, 0] += lead_offset
    # A real open -> close -> open gripper sweep so normalize_gripper is happy.
    w = np.full(n, 0.998)
    w[n // 3: 2 * n // 3] = 0.35
    p[:, C.GRIPPER_JOINT_INDEX] = w
    return t, p


def _write_mcap(path: Path, node: str, t: np.ndarray, p: np.ndarray) -> None:
    """A minimal MCAP in the JSON shape read_positions falls back to.

    Deliberately hand-written rather than driven through recording.McapWriter:
    that class needs a live node and would pull the whole runtime into a unit
    test. What matters is that read_positions accepts it, which is the contract
    the exporter depends on.
    """
    from mcap.writer import Writer
    with open(path, "wb") as fh:
        w = Writer(fh)
        w.start()
        sid = w.register_schema(name="joint_pos", encoding="jsonschema", data=b"{}")
        cid = w.register_channel(topic=f"/{node}/joint_pos", message_encoding="json",
                                 schema_id=sid)
        for ti, pi in zip(t, p):
            ns = int(ti * 1e9)
            w.add_message(channel_id=cid, log_time=ns, publish_time=ns,
                          data=json.dumps({"position": [float(x) for x in pi]}).encode())
        w.finish()


def bimanual_episode(tmp: Path, *, right_moves=True, left_moves=True,
                     right_lead_offset=0.0, n=1200) -> Path:
    """A handoff take: the right arm works the first half, the left the second."""
    ep = tmp / "episode_bimanual"
    ep.mkdir(parents=True)
    for arm, moves, offset, base in (("right", right_moves, right_lead_offset, 0.2),
                                     ("left", left_moves, 0.0, -0.2)):
        t, p = _traj(n, moves, base)
        _write_mcap(ep / f"yam_{arm}.mcap", f"yam_{arm}", t, p)
        tl, pl = _traj(n, moves, base, lead_offset=offset)
        _write_mcap(ep / f"gello_{arm}.mcap", f"gello_{arm}", tl, pl)
        annotations_path(ep, arm).write_text(json.dumps(_ann([_grasp(2.0)])))
    return ep


# ── the feature contract ────────────────────────────────────────────────────
def test_one_arm_keeps_the_historical_names_and_width():
    """THE compatibility guarantee. Any change here silently invalidates every
    existing checkpoint, because LeRobot matches features by name and shape."""
    assert resolve_arms(None) == ("left",)
    assert joint_names(None) == JOINT_NAMES
    assert n_dof(None) == N_DOF == 7


def test_two_arms_concatenate_left_then_right():
    names = joint_names(ARM_SETS["both"])
    assert n_dof(ARM_SETS["both"]) == 14
    assert names[:7] == [f"left_{n}" for n in JOINT_NAMES]
    assert names[7:] == [f"right_{n}" for n in JOINT_NAMES]


def test_two_arm_features_are_14_wide_on_both_state_and_action():
    feats = build_features({"camera_left": (480, 640), "camera_right": (480, 640)},
                           CAMERA_SETS["wrists"], ARM_SETS["both"])
    assert feats["observation.state"]["shape"] == (14,)
    assert feats["action"]["shape"] == (14,)
    assert set(feats) == {"observation.state", "action",
                          "observation.images.wrist_left",
                          "observation.images.wrist_right"}


def test_bimanual_wrist_keys_are_not_the_single_arm_key():
    """A single-arm checkpoint must REFUSE a bimanual dataset rather than load
    the right wrist into weights trained on the left. LeRobot keys by name, so
    keeping "wrist" for one of the two wrists is how that happens silently."""
    assert "observation.images.wrist" not in build_features(
        {"camera_left": (480, 640), "camera_right": (480, 640)},
        CAMERA_SETS["wrists"], ARM_SETS["both"])


def test_no_active_arm_feature_is_exposed():
    """Deliberate omission. "Which arm is being driven" is knowable at export
    time and NOT knowable at inference time -- the policy has to infer it from
    the images. Feeding it in would train a policy that cannot run."""
    feats = build_features({"camera_left": (480, 640), "camera_right": (480, 640)},
                           CAMERA_SETS["wrists"], ARM_SETS["both"])
    assert not [k for k in feats if "arm" in k or "active" in k]


# ── activity detection ──────────────────────────────────────────────────────
def test_a_parked_arm_reads_as_idle_and_a_driven_arm_as_moving():
    idle = np.tile(np.array([[0.2] * 7]), (500, 1)) + np.random.default_rng(0).normal(0, 1e-4, (500, 7))
    act = arm_activity(idle, idle)
    assert not act["moving"] and act["ptp_rad"] < 0.01

    moving = idle.copy()
    moving[:, 0] += np.linspace(0, 0.9, 500)
    act = arm_activity(moving, moving)
    assert act["moving"] and act["ptp_rad"] > 0.5


def test_an_idle_arm_whose_leader_agrees_is_accepted():
    """The normal case, and the reason the idle arm needs no special encoding:
    the follower tracks the parked leader, so the recorded action already IS
    'hold here'. Nothing has to be fabricated."""
    n = 400
    state = np.full((n, 7), 0.2) + np.random.default_rng(1).normal(0, 1e-4, (n, 7))
    action = state + 1e-3
    assert idle_arm_veto({"right": arm_activity(state, action)},
                         IDLE_ARM_DIVERGENCE_MAX_RAD) is None


def test_an_idle_arm_whose_leader_was_parked_elsewhere_is_vetoed():
    """The one case where recording the truth is unsafe. The operator lets go of
    the right leader somewhere other than where the right follower is; the
    leader keeps publishing that pose as the commanded action. Training on it
    teaches the policy to snap the idle arm across the workspace the instant the
    other arm starts working, and no loss curve shows it."""
    n = 400
    state = np.full((n, 7), 0.2)
    action = state.copy()
    action[:, 0] += 0.4                      # leader parked 0.4 rad away
    veto = idle_arm_veto({"right": arm_activity(state, action)},
                         IDLE_ARM_DIVERGENCE_MAX_RAD)
    assert veto is not None and "right" in veto


def test_the_veto_does_not_fire_on_a_MOVING_arm():
    """Tracking error on an arm being driven is normal and large; vetoing on it
    would throw away every real demonstration."""
    n = 400
    state = np.full((n, 7), 0.2)
    state[:, 0] += np.linspace(0, 0.9, n)
    action = state.copy()
    action[:, 0] += 0.4                      # lag, not a parked leader
    assert idle_arm_veto({"left": arm_activity(state, action)},
                         IDLE_ARM_DIVERGENCE_MAX_RAD) is None


def test_the_veto_can_be_switched_off():
    n = 10
    state, action = np.full((n, 7), 0.2), np.full((n, 7), 9.9)
    assert idle_arm_veto({"r": arm_activity(state, action)}, 0.0) is None


# ── end to end over a real two-arm episode on disk ──────────────────────────
def test_a_one_arm_at_a_time_episode_plans_and_resamples_to_14_dof(tmp_path):
    ep = bimanual_episode(tmp_path, right_moves=True, left_moves=False)
    rep = Report()
    plan = plan_episode(ep, 3.0, 2.0, 30, rep, cameras={}, arms=ARM_SETS["both"],
                        window_mode="full")
    assert plan is not None, [r.reason for r in rep.rejected]
    lo, hi = plan["windows"][0]
    grid = lo + np.arange(int((hi - lo) * 30)) / 30.0

    state, action, activity = window_rows(plan, grid)
    assert state.shape == (grid.size, 14)
    assert action.shape == (grid.size, 14)
    # left first, right second -- the order joint_names() promises
    assert np.allclose(state[:, 0], -0.2, atol=0.05)      # left, parked at -0.2
    assert state[:, 7].max() - state[:, 7].min() > 0.5    # right, swept
    assert activity["right"]["moving"] and not activity["left"]["moving"]


def test_the_idle_arms_values_are_its_own_recorded_values_not_zeros(tmp_path):
    """The modelling decision, made checkable. A masked or zeroed idle arm would
    put 0.0 here; 0.0 is a real joint angle and the arm is not at it."""
    ep = bimanual_episode(tmp_path, right_moves=True, left_moves=False)
    rep = Report()
    plan = plan_episode(ep, 3.0, 2.0, 30, rep, cameras={}, arms=ARM_SETS["both"],
                        window_mode="full")
    grid = np.linspace(*plan["windows"][0], 50)
    state, action, _ = window_rows(plan, grid)
    assert np.allclose(state[:, :C.N_ARM_JOINTS], -0.2, atol=0.05)
    assert np.allclose(action[:, :C.N_ARM_JOINTS], -0.2, atol=0.05)
    assert not np.allclose(state[:, :C.N_ARM_JOINTS], 0.0)


def test_a_parked_leader_that_disagrees_is_dropped_end_to_end(tmp_path):
    ep = bimanual_episode(tmp_path, right_moves=False, left_moves=True,
                          right_lead_offset=0.5)
    rep = Report()
    plan = plan_episode(ep, 3.0, 2.0, 30, rep, cameras={}, arms=ARM_SETS["both"],
                        window_mode="full")
    grid = np.linspace(*plan["windows"][0], 50)
    _, _, activity = window_rows(plan, grid)
    assert idle_arm_veto(activity, IDLE_ARM_DIVERGENCE_MAX_RAD) is not None


def test_full_window_mode_keeps_the_handoff_in_one_episode(tmp_path):
    """Cutting a handoff take into per-grasp windows deletes the sequence, which
    is the only thing that needed two arms to record."""
    ep = bimanual_episode(tmp_path)
    rep = Report()
    full = plan_episode(ep, 3.0, 2.0, 30, rep, cameras={}, arms=ARM_SETS["both"],
                        window_mode="full")
    grasp = plan_episode(ep, 3.0, 2.0, 30, rep, cameras={}, arms=ARM_SETS["both"],
                         window_mode="grasp")
    assert len(full["windows"]) == 1
    lo, hi = full["windows"][0]
    assert hi - lo == pytest.approx(1200 / 200.0, abs=0.05)   # the whole take
    # both arms' grasps land in the pool, and each is a shorter window
    assert len(grasp["windows"]) >= 1
    assert all(h - l <= 5.1 for l, h in grasp["windows"])


def test_each_arm_gets_its_own_annotations_file(tmp_path):
    """Two arms in one episode dir means two label sets. Left keeps the bare
    name so every existing episode and tool still resolves."""
    ep = bimanual_episode(tmp_path)
    assert annotations_path(ep, "left").name == "annotations.json"
    assert annotations_path(ep, "right").name == "annotations_right.json"
    assert annotations_path(ep, "left") != annotations_path(ep, "right")
    assert annotations_path(ep, "right").exists()


def test_single_arm_planning_of_the_same_episode_is_unaffected(tmp_path):
    """The production path. A second arm in the directory must be invisible to
    a --arms left export."""
    ep = bimanual_episode(tmp_path)
    rep = Report()
    plan = plan_episode(ep, 3.0, 2.0, 30, rep, cameras={}, arms=("left",))
    assert plan is not None and plan["arms"] == ("left",)
    grid = np.linspace(*plan["windows"][0], 20)
    state, action, _ = window_rows(plan, grid)
    assert state.shape[1] == 7 and action.shape[1] == 7
