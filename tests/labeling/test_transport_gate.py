"""Transport gate: a re-grasp at the pick must NOT advance the kit pointer.

The operator often needs two tries to grab a flat bag. Before this gate, every
close/open cycle advanced the cockpit, so the kit list ran ahead of reality.
Measured on the recorded corpus: 13 of 39 'success' releases (33%) never carried
the packet anywhere.

Threshold rationale (measured, not guessed) — the two populations are cleanly
bimodal with an 8.7 cm empty gap, so 8/10/12/14 cm all classify identically:

    re-grasp / fumble              real pick -> place
    0.1 cm ...... 5.8 cm      14.5 cm ...... 48.1 cm
                     |-- gap --|
                          ^ C.MIN_TRANSPORT_M = 0.10
"""
from __future__ import annotations

import glob
import os

import numpy as np
import pytest

from robots_realtime.labeling import constants as C
from robots_realtime.labeling.live import LiveLabeler, OnlineGripSegmenter
from robots_realtime.labeling.segmentation import (
    transport_distance_m,
    transport_ok,
    transported,
)

KIT = [{"bag_id": i, "part": f"P{i}", "comp": i} for i in range(1, 5)]


# ── the threshold rule ────────────────────────────────────────────────────────

def test_transport_ok_above_and_below_threshold():
    assert transport_ok(0.15, 0.10) is True
    assert transport_ok(0.05, 0.10) is False


def test_transport_ok_is_inclusive_at_the_boundary():
    assert transport_ok(0.10, 0.10) is True


def test_transport_ok_fails_open_when_distance_unknown():
    """No EE pose must never freeze the pointer — degrade to old behavior."""
    assert transport_ok(None, 0.10) is True


def test_transport_ok_disabled_gate_always_passes():
    assert transport_ok(0.0, 0.0) is True
    assert transport_ok(None, 0.0) is True


def test_transport_distance_is_horizontal_only():
    """Pure vertical travel is a lift, not a transport."""
    assert transport_distance_m([0, 0, 0], [0, 0, 5]) == pytest.approx(0.0)
    assert transport_distance_m([0, 0, 0], [3, 4, 99]) == pytest.approx(5.0)
    assert transport_distance_m(None, [1, 1, 1]) is None


def test_offline_and_live_share_one_threshold_rule():
    """transported() (offline, pose pair) must agree with transport_ok() (live,
    scalar) for the same geometry — this is what stops the two labelers drifting."""
    a, b = [0.0, 0.0, 0.0], [0.09, 0.0, 0.0]
    assert transported(a, b, 0.10) == transport_ok(transport_distance_m(a, b), 0.10)
    assert transported(a, b, 0.10) is False


# ── the online segmenter carries the measurement ──────────────────────────────

HOLD_W = 0.20   # normalized width of a gripper closed on a bag: above
                # GRIPPER_EMPTY_CLOSE (0.08) so it is not read as "empty",
                # below GRIPPER_CLOSE_ENTER (0.45) so it counts as closed.
N_HOLD = 40     # classify_hold takes the MEDIAN, and the release sample lands in
                # the buffer too (same as the offline detector). A 3-sample cycle
                # lets that one open sample dominate; real feeds are 200 Hz.


def _samples(t0, hold, n=N_HOLD):
    return [t0 + 0.01 + (hold - 0.01) * i / max(n - 1, 1) for i in range(n)]


def _grip_cycle(seg, t0, ee_close, ee_open, hold=1.0, w=HOLD_W, ee_hold=None):
    """Drive one full open->closed->open cycle, return the release event."""
    seg.push(t0, 1.0, ee_close)                          # open
    for ts in _samples(t0, hold):                        # closed, held
        seg.push(ts, w, ee_hold if ee_hold is not None else ee_close)
    return seg.push(t0 + hold + 0.01, 1.0, ee_open)      # released


def test_release_event_reports_measured_travel():
    seg = OnlineGripSegmenter(open_ref=1.0, closed_ref=0.0)
    ev = _grip_cycle(seg, 0.0, [0.0, 0.0, 0.0], [0.30, 0.0, 0.0])
    assert ev.kind == "release"
    assert ev.dxy_m == pytest.approx(0.30)


def test_release_event_reports_no_travel_for_a_regrasp():
    seg = OnlineGripSegmenter(open_ref=1.0, closed_ref=0.0)
    ev = _grip_cycle(seg, 0.0, [0.2, 0.2, 0.1], [0.201, 0.2, 0.1])
    assert ev.dxy_m == pytest.approx(0.001, abs=1e-6)


def test_release_event_dxy_is_none_without_poses():
    seg = OnlineGripSegmenter(open_ref=1.0, closed_ref=0.0)
    ev = _grip_cycle(seg, 0.0, None, None)
    assert ev.dxy_m is None


def test_pose_is_captured_at_close_not_at_commit():
    """t_close is backdated by MIN_HOLD_S. If the pose were snapped when the close
    EVENT fires, the arm's travel during that window would be lost."""
    seg = OnlineGripSegmenter(open_ref=1.0, closed_ref=0.0)
    seg.push(0.0, 1.0, [0.0, 0.0, 0.0])
    seg.push(0.01, HOLD_W, [0.0, 0.0, 0.0])     # closes HERE, at x=0
    for ts in _samples(0.0, 1.0):
        seg.push(ts, HOLD_W, [0.5, 0.0, 0.0])   # commit fires later, at x=0.5
    ev = seg.push(1.01, 1.0, [0.5, 0.0, 0.0])
    assert ev.dxy_m == pytest.approx(0.5)       # 0.5, not 0.0


def test_lift_overrides_empty_matching_the_offline_classifier():
    """classify_hold treats a lift as proof an object was held. The live segmenter
    had no lift check at all, so it labeled real picks of flat bags 'empty'."""
    seg = OnlineGripSegmenter(open_ref=1.0, closed_ref=0.0)
    seg.push(0.0, 1.0, [0.0, 0.0, 0.0])
    # width 0.0 = fully closed on a flat bag, which on width alone reads "empty".
    # z must START low and RISE, so the lift is visible inside the hold buffer.
    for ts in _samples(0.0, 1.0):
        seg.push(ts, 0.0, [0.0, 0.0, 0.20 * min((ts - 0.01) / C.LIFT_WINDOW_S, 1.0)])
    ev = seg.push(1.01, 1.0, [0.3, 0.0, 0.20])
    assert ev.lifted is True, "a 20 cm rise during the hold is a lift"
    assert ev.outcome == "success", "lift must override the width-only 'empty'"


def test_lift_uses_the_same_window_as_offline():
    """The lift window is LIFT_WINDOW_S from t_close, NOT the whole hold. Measuring
    over the whole hold made an 8 s hold read lifted=True live / False offline, and
    the live labeler advanced on a grasp the offline labeler had rejected."""
    seg = OnlineGripSegmenter(open_ref=1.0, closed_ref=0.0)
    seg.push(0.0, 1.0, [0.0, 0.0, 0.0])
    long_hold = C.LIFT_WINDOW_S * 4
    for ts in _samples(0.0, long_hold, n=200):
        # flat for the whole lift window, only rising well after it has passed
        z = 0.0 if ts - 0.01 <= C.LIFT_WINDOW_S else 0.30
        seg.push(ts, 0.0, [0.0, 0.0, z])
    ev = seg.push(long_hold + 0.01, 1.0, [0.0, 0.0, 0.30])
    assert ev.lifted is False, "a late rise is outside the lift window"


# ── the labeler's advance decision ────────────────────────────────────────────

def _labeler():
    lab = LiveLabeler(open_ref=1.0, closed_ref=0.0)
    lab.seed(KIT)
    return lab


def _cycle(lab, t0, ee_close, ee_open, hold=1.0, lift=0.10):
    """One grasp: open -> close on a bag -> lift it -> release at ee_open.

    The lift matters: the advance rule mirrors fuse._is_terminal, which rejects a
    grasp known NOT to have lifted. A synthetic cycle that never raises z would be
    rejected for the wrong reason and hide whether the transport gate works.
    """
    lab.push(t0, 1.0, ee_close)
    for ts in _samples(t0, hold):
        if ee_close is None:
            lab.push(ts, HOLD_W, None)
            continue
        frac = min((ts - t0) / C.LIFT_WINDOW_S, 1.0)
        lab.push(ts, HOLD_W, [ee_close[0], ee_close[1], ee_close[2] + lift * frac])
    lab.push(t0 + hold + 0.01, 1.0, ee_open)


def test_real_transport_advances_the_pointer():
    lab = _labeler()
    _cycle(lab, 0.0, [0.0, 0.0, 0.0], [0.30, 0.0, 0.05])
    assert lab.ti == 1
    assert lab.regrasps == 0
    assert lab.packets[0]["status"] == "placed"


def test_regrasp_holds_the_pointer_and_counts():
    """The whole point: two tries on the same packet stay on that packet."""
    lab = _labeler()
    _cycle(lab, 0.0, [0.0, 0.0, 0.0], [0.01, 0.0, 0.0])
    assert lab.ti == 0, "re-grasp must not advance"
    assert lab.regrasps == 1
    assert lab.packets[0]["status"] != "placed"
    assert [e["type"] for e in lab.events if e["type"] == "regrasp"] == ["regrasp"]


def test_regrasp_then_success_advances_once_and_resets_the_counter():
    lab = _labeler()
    _cycle(lab, 0.0, [0.0, 0.0, 0.0], [0.01, 0.0, 0.0])    # fumble
    _cycle(lab, 10.0, [0.0, 0.0, 0.0], [0.02, 0.0, 0.0])   # fumble again
    assert lab.ti == 0 and lab.regrasps == 2
    _cycle(lab, 20.0, [0.0, 0.0, 0.0], [0.35, 0.0, 0.05])  # got it
    assert lab.ti == 1
    assert lab.regrasps == 0, "counter resets on the new packet"


def test_missing_pose_fails_open_and_raises_the_flag():
    """No joint data -> behave like before (advance) but say so loudly."""
    lab = _labeler()
    assert lab.gate_off is False
    _cycle(lab, 0.0, None, None)
    assert lab.ti == 1, "must degrade to old behavior, not freeze"
    assert lab.gate_off is True


def test_state_exposes_regrasps_and_gate_status():
    lab = _labeler()
    _cycle(lab, 0.0, [0.0, 0.0, 0.0], [0.01, 0.0, 0.0])
    s = lab.state()
    assert s["regrasps"] == 1
    assert s["gate_off"] is False
    assert s["min_transport_m"] == C.MIN_TRANSPORT_M


def test_manual_advance_resets_the_regrasp_counter():
    lab = _labeler()
    _cycle(lab, 0.0, [0.0, 0.0, 0.0], [0.01, 0.0, 0.0])
    assert lab.regrasps == 1
    lab.advance()
    assert lab.ti == 1 and lab.regrasps == 0


def test_gate_can_be_disabled():
    lab = LiveLabeler(open_ref=1.0, closed_ref=0.0, min_transport_m=0.0)
    lab.seed(KIT)
    _cycle(lab, 0.0, [0.0, 0.0, 0.0], [0.01, 0.0, 0.0])
    assert lab.ti == 1, "gate disabled -> old advance-on-any-success behavior"


# ── real-data regression: live must agree with offline ────────────────────────

def _recorded_episodes():
    root = os.path.join(os.path.dirname(__file__), "..", "..", "recordings")
    return sorted(glob.glob(os.path.join(root, "*", "episode_*", "yam_left.mcap")))


@pytest.mark.skipif(not _recorded_episodes(), reason="no recordings on this machine")
@pytest.mark.parametrize("mcap", _recorded_episodes())
def test_live_advance_count_matches_offline_place_count(mcap):
    """THE invariant: replaying a recording through the LIVE labeler must produce
    the same number of placements the OFFLINE labeler finds in it.

    This is what catches the two segmenters drifting apart, using real operator
    data rather than synthetic cycles.
    """
    from robots_realtime.labeling.fk import ForwardKinematics
    from robots_realtime.labeling.label_episode import label_from_arrays
    from robots_realtime.labeling.mcap_io import read_positions

    ep_dir = os.path.dirname(mcap)
    try:
        times, positions = read_positions(mcap, "yam_left")
    except Exception:
        pytest.skip("unreadable mcap")
    if times.size == 0:
        pytest.skip("empty mcap")

    fk = ForwardKinematics(os.path.join(os.path.dirname(__file__), "..", "..",
                                        "urdf", "yam.urdf"))
    offline = label_from_arrays(
        times, positions, fk=fk, arm="left", episode_id=os.path.basename(ep_dir),
        gripper_open_ref=1.0, gripper_closed_ref=0.0,
        min_transport_m=C.MIN_TRANSPORT_M,
    )

    ee = fk.ee_positions(np.asarray(positions, float)[:, : C.N_ARM_JOINTS])
    grip = np.asarray(positions, float)[:, C.GRIPPER_JOINT_INDEX]
    lab = LiveLabeler(open_ref=1.0, closed_ref=0.0)
    lab.seed([{"bag_id": i, "part": f"P{i}", "comp": i} for i in range(1, 64)])
    for i in range(times.size):
        lab.push(float(times[i]), float(grip[i]), ee[i])

    live_places = sum(1 for e in lab.events if e["type"] == "place_confirmed")
    assert live_places == len(offline.place_events), (
        f"{os.path.basename(ep_dir)}: live advanced {live_places}x but offline "
        f"found {len(offline.place_events)} placements"
    )
