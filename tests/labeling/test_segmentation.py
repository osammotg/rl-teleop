"""Gripper segmentation against synthetic width signals."""
from __future__ import annotations

import numpy as np

from robots_realtime.labeling.segmentation import detect_grip_intervals, normalize_width


def _series(segments, dt=0.02):
    """Build (times, width) from [(duration_s, raw_width), ...]."""
    times, width = [], []
    t = 0.0
    for dur, val in segments:
        n = int(dur / dt)
        for _ in range(n):
            times.append(t); width.append(val); t += dt
    return np.array(times), np.array(width)


# Known physical gripper limits (as label_episode passes from the robot config).
REFS = dict(open_ref=1.0, closed_ref=0.0)


def _detect(t, w, **kw):
    return detect_grip_intervals(t, w, **REFS, **kw)


def test_normalize_orients_open_high():
    # raw where open=1000 (start), closed=0
    t, w = _series([(1.0, 1000), (0.5, 0), (1.0, 1000)])
    n = normalize_width(w)
    assert n[0] > 0.9            # starts open
    assert n.min() < 0.1


def test_normalize_handles_inverted_signal():
    # inverted: open=0 (start), closed=1000
    t, w = _series([(1.0, 0), (0.5, 1000), (1.0, 0)])
    n = normalize_width(w)
    assert n[0] > 0.9            # still oriented open-high
    assert n.min() < 0.1


def test_clean_grasp_detected():
    # open → hold a bag (partial close ~0.35) → open
    t, w = _series([(1.0, 1.0), (1.5, 0.35), (1.0, 1.0)])
    ivs = _detect(t, w)
    assert len(ivs) == 1
    assert ivs[0].outcome == "success"
    assert 0.25 < ivs[0].hold_norm < 0.45


def test_empty_close_flagged_not_grasp():
    # closes on nothing (goes fully closed ~0.0)
    t, w = _series([(1.0, 1.0), (1.5, 0.02), (1.0, 1.0)])
    ivs = _detect(t, w)
    assert len(ivs) == 1
    assert ivs[0].outcome == "empty"


def test_slip_detected():
    # hold a bag at 0.35, then it falls → width collapses to 0.02, then release
    t, w = _series([(1.0, 1.0), (0.8, 0.35), (0.8, 0.02), (1.0, 1.0)])
    ivs = _detect(t, w)
    assert len(ivs) == 1
    assert ivs[0].outcome == "slip"


def test_hysteresis_no_double_count_on_jitter():
    # jitter right around the deadband (0.45–0.60) must not spawn events
    t, w = _series([(1.0, 1.0), (2.0, 0.52), (1.0, 1.0)])  # sits in deadband
    ivs = _detect(t, w)
    assert len(ivs) == 0        # never crossed CLOSE_ENTER


def test_lift_confirmation():
    t, w = _series([(1.0, 1.0), (1.5, 0.35), (1.0, 1.0)])
    # ee_z rises 10cm shortly AFTER the grasp (lift starts after close)
    z = np.where((t > 1.3) & (t < 2.5), 0.25, 0.15)
    ivs = _detect(t, w, ee_z=z)
    assert ivs[0].lifted is True

    z_flat = np.full_like(t, 0.15)
    ivs2 = _detect(t, w, ee_z=z_flat)
    assert ivs2[0].lifted is False


def test_two_grasps_two_intervals():
    t, w = _series([(1.0, 1.0), (1.0, 0.35), (1.0, 1.0), (1.0, 0.35), (1.0, 1.0)])
    ivs = _detect(t, w)
    assert len(ivs) == 2
    assert all(iv.outcome == "success" for iv in ivs)


def test_still_closed_at_end():
    t, w = _series([(1.0, 1.0), (1.5, 0.35)])   # never reopens
    ivs = _detect(t, w)
    assert len(ivs) == 1
    assert ivs[0].t_open is None


def test_lift_overrides_empty_for_flat_bags():
    """A tight close (width reads 'empty') that LIFTS is a real grasp, not empty —
    flat bags close nearly full, so the lift is the reliable evidence it held one."""
    import numpy as np
    from robots_realtime.labeling.segmentation import classify_hold as _classify
    from robots_realtime.labeling import constants as C
    t = np.linspace(0, 2.0, 40)
    hold = np.full(20, C.GRIPPER_EMPTY_CLOSE * 0.3)      # very tight → width says "empty"
    ee_z = np.concatenate([np.zeros(20), np.full(20, C.MIN_LIFT_M + 0.05)])  # clear lift
    _, _, outcome, lifted = _classify(hold, float(t[0]), t, ee_z)
    assert lifted is True and outcome == "success"       # lift wins over width-empty

    # same tight close but NO lift → still empty
    _, _, oc2, lifted2 = _classify(hold, float(t[0]), t, np.zeros(40))
    assert lifted2 is False and oc2 == "empty"
