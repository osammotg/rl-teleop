"""The one gripper normaliser, and the failure it used to hide.

REGRESSION FOR AUDIT.md S1.1 / S1.3 / S1.4 / S1.6 and DATA-PIPELINE.md 2.7.

There used to be two normalisers over the same recordings, agreeing on the
convention (1 = open, 0 = closed) and disagreeing on the only case where either
had no information:

    segmentation.normalize_width      -> np.zeros_like  = jaws FULLY SHUT
    export_lerobot.normalize_gripper  -> np.ones_like   = jaws WIDE OPEN

So one recording session produced a grasp corpus that said the jaws were shut
for the whole take and ACT tensors that said they were open for the whole take.
Neither function had any information. Both returned a value that looks exactly
like a confident measurement, and no test anywhere passed a constant array
through either of them.

Measured over recordings/ on 2026-08-08: 13 of the 28 readable episodes have a
gripper channel that never left the open stop. Six are bit-identical and hit the
all-zeros branch, so each was labelled as ONE whole-episode "empty" grasp with
the jaws shut for the entire take. The other seven carry ~7e-5 of sensor noise,
which clears `hi - lo < 1e-9` by five orders of magnitude, so the percentile
fallback rescaled that noise to full scale: one episode produced 26,443
open/close transitions in the hysteresis state machine (all debounced away by
MIN_HOLD_S, so the episode simply read as "nothing happened"). In all 13 the
guard written to catch exactly this stayed silent.

Every test below fails against the pre-2026-08-08 code.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from robots_realtime.labeling import constants as C  # noqa: E402
from robots_realtime.labeling.segmentation import (  # noqa: E402
    GripperRangeUnknown,
    detect_grip_intervals,
    is_unknown,
    normalize_width,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
from export_lerobot import normalize_gripper  # noqa: E402

# The real rig's limits. qa_label.py is the one consumer that always got this
# right; everything else was left to guess.
REFS = dict(open_ref=1.0, closed_ref=0.0)


# ── the degenerate case is not a value ──────────────────────────────────────
def test_a_gripper_that_never_moved_is_not_reported_as_closed():
    """S1.1. THE bug. An operator records a take without touching the trigger;
    raw pos[:,6] sits at the open stop for every sample. The old code returned
    all-zeros -- byte-identical to "most confident possible full close"."""
    with pytest.raises(GripperRangeUnknown):
        normalize_width(np.full(100, 0.993))


def test_a_gripper_that_never_moved_is_not_reported_as_open_either():
    """S1.3, the other half. The export path's answer was the opposite constant.
    Being right by accident on this rig is not being right."""
    with pytest.raises(GripperRangeUnknown):
        normalize_gripper(np.full(100, 0.993, dtype=np.float32))


def test_the_two_normalisers_now_give_the_same_answer_on_a_dead_gripper():
    """The contradiction itself, expressed as a test. Before: zeros vs ones."""
    dead = np.full(50, 0.9989)
    with pytest.raises(GripperRangeUnknown):
        normalize_width(dead)
    with pytest.raises(GripperRangeUnknown):
        normalize_gripper(dead.astype(np.float32))


def test_real_sensor_noise_does_not_clear_the_degeneracy_guard():
    """The measured shape of the failure. episode_233409_4edd62b9's gripper sits
    at 0.9989..0.9990 -- a range of 7.3e-05, which is 5 orders of magnitude
    above the old 1e-9 threshold. The old guard passed it and the fallback
    rescaled that noise to the full [0,1] range -- 26,443 open/close transitions
    in one real episode, all of them noise."""
    rng = np.random.default_rng(0)
    noisy = 0.99895 + rng.uniform(-3.6e-5, 3.6e-5, size=5000)
    assert noisy.ptp() > 1e-9, "fixture must clear the OLD threshold to be a regression"
    with pytest.raises(GripperRangeUnknown):
        normalize_width(noisy)


def test_unknown_mode_returns_an_explicit_sentinel_not_a_plausible_value():
    out = normalize_width(np.full(20, 0.5), on_degenerate="unknown")
    assert is_unknown(out)
    assert np.all(np.isnan(out)), "unknown must not be confusable with 0.0 or 1.0"


def test_is_unknown_is_required_because_nan_reads_as_never_closed():
    """Why the sentinel needs a helper: NaN silently passes every threshold in
    the segmenter as False, i.e. 'the gripper never closed'. A caller that asks
    for `unknown` and then thresholds without checking gets the old bug back in
    a new shape."""
    out = normalize_width(np.full(20, 0.5), on_degenerate="unknown")
    assert not np.any(out < C.GRIPPER_CLOSE_ENTER)      # the trap
    assert is_unknown(out)                              # the check that catches it


def test_empty_input_is_unknown_not_empty_success():
    with pytest.raises(GripperRangeUnknown):
        normalize_width(np.array([]))


def test_non_finite_input_is_rejected():
    """S1.6. One NaN made np.percentile return NaN, which skipped the degenerate
    branch, propagated NaN through the whole array, and turned a real run into
    'nothing was ever grasped' with no error anywhere."""
    w = np.linspace(0.0, 1.0, 100)
    w[42] = np.nan
    with pytest.raises(GripperRangeUnknown, match="non-finite"):
        normalize_width(w)


def test_identical_refs_are_rejected_rather_than_dividing_by_zero():
    with pytest.raises(GripperRangeUnknown):
        normalize_width(np.linspace(0, 1, 10), open_ref=0.5, closed_ref=0.5)


# ── with refs, a constant is answerable and the answer is OPEN ──────────────
def test_with_refs_a_resting_open_gripper_normalises_to_open():
    """The reason to pass refs at all: with the physical limits in hand, a
    never-moving gripper at 0.993 is not ambiguous. It is open, and it says so.
    grasp_dataset.py:165 is the call that built the corpus without them."""
    out = normalize_width(np.full(100, 0.993), **REFS)
    assert np.allclose(out, 0.993)
    assert out.min() > 0.9


def test_with_refs_a_resting_closed_gripper_normalises_to_closed():
    out = normalize_width(np.full(100, 0.004), **REFS)
    assert out.max() < 0.1


# ── the healthy path is unchanged ───────────────────────────────────────────
def test_a_real_sweep_still_spans_the_unit_range():
    w = np.concatenate([np.full(200, 0.998), np.full(200, 0.003), np.full(200, 0.998)])
    n = normalize_width(w)
    assert n[0] > 0.9 and n.min() < 0.1 and n.max() <= 1.0


def test_an_inverted_raw_signal_is_still_oriented_open_high():
    w = np.concatenate([np.full(200, 0.0), np.full(200, 1000.0), np.full(200, 0.0)])
    n = normalize_width(w)
    assert n[0] > 0.9 and n.min() < 0.1


def test_normalisation_is_affine_invariant():
    """Two boots calibrate the raw range differently (observed 5.2218/-0.0235
    on one, ~1.0/0.0 on another). The same physical motion must normalise the
    same way or nothing is comparable across sessions."""
    a = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    b = 2.62 * a - 0.02
    assert np.allclose(normalize_gripper(a.astype(np.float32)),
                       normalize_gripper(b.astype(np.float32)), atol=1e-6)


def test_export_and_label_paths_agree_on_a_healthy_episode():
    """They are one function now; this pins that they stay one."""
    rng = np.random.default_rng(7)
    w = np.concatenate([np.full(500, 0.998), np.full(300, 0.35), np.full(500, 0.998)])
    w = w + rng.normal(0, 1e-4, w.size)
    assert np.allclose(normalize_width(w), normalize_gripper(w.astype(np.float32)),
                       atol=1e-6)


# ── the segmenter propagates rather than inventing ──────────────────────────
def test_detect_grip_intervals_raises_instead_of_returning_no_grasps():
    """'no gripper signal' and 'no grasps' are different facts. Collapsing them
    into [] is how a dead channel became an ordinary quiet episode."""
    t = np.arange(0, 10, 0.02)
    with pytest.raises(GripperRangeUnknown):
        detect_grip_intervals(t, np.full(t.size, 0.9989))


def test_detect_grip_intervals_with_refs_reads_a_dead_gripper_as_never_closed():
    """With refs there is no ambiguity, so it must NOT raise -- it must simply
    find nothing, because a gripper resting open really did not grasp."""
    t = np.arange(0, 10, 0.02)
    assert detect_grip_intervals(t, np.full(t.size, 0.993), **REFS) == []


# ── the guard that could not fire ───────────────────────────────────────────
def test_the_flag_now_fires_on_the_input_it_was_written_for():
    """S1.4. The old guard read `normalize_width(gripper).min() > 0.15`, and on
    a dead gripper normalize_width returned zeros, so min() was 0.0 and
    `0.0 > 0.15` was False. It certified the exact failure it existed to catch.
    Measured: it stayed silent on all 28 readable episodes, including all 13
    dead ones, six of which it then labelled as a whole-episode empty grasp."""
    from robots_realtime.labeling.fk import ForwardKinematics
    from robots_realtime.labeling.label_episode import label_from_arrays

    fk = ForwardKinematics(Path(__file__).resolve().parents[2] / "urdf" / "yam.urdf")
    t = np.arange(0, 20, 0.02)
    pos = np.zeros((t.size, 7))
    pos[:, 6] = 0.9989 + np.random.default_rng(1).uniform(-3.6e-5, 3.6e-5, t.size)

    ann = label_from_arrays(t, pos, fk=fk, arm="left", episode_id="dead")
    kinds = [f.kind for f in ann.flags]
    assert "gripper_range_unknown" in kinds, f"no flag raised; flags={kinds}"
    assert ann.grasp_attempts == [], "a dead channel must not manufacture grasps"


def test_a_healthy_episode_is_not_flagged():
    """The other half of a guard: it has to stay quiet on good data, or it gets
    ignored and stops being a guard."""
    from robots_realtime.labeling.fk import ForwardKinematics
    from robots_realtime.labeling.label_episode import label_from_arrays

    fk = ForwardKinematics(Path(__file__).resolve().parents[2] / "urdf" / "yam.urdf")
    t = np.arange(0, 20, 0.02)
    pos = np.zeros((t.size, 7))
    w = np.full(t.size, 0.998)
    w[300:700] = 0.35                       # one hold on a bag
    pos[:, 6] = w
    ann = label_from_arrays(t, pos, fk=fk, arm="left", episode_id="ok")
    assert "gripper_range_unknown" not in [f.kind for f in ann.flags]
    # The hold IS seen. (It classifies `empty` because this synthetic arm never
    # lifts, which is a different question from whether the channel was read.)
    assert len(detect_grip_intervals(t, pos[:, 6])) == 1


# ── against the real corpus ─────────────────────────────────────────────────
CORPUS = Path(__file__).resolve().parents[2] / "recordings"
needs_corpus = pytest.mark.skipif(not CORPUS.exists(), reason="recordings/ not present")


@needs_corpus
def test_every_episode_the_exporter_keeps_has_a_live_gripper():
    """The reassuring half of the measurement: the 15 episodes that make up
    yam_grasp_v1 / v2_wrist all have a real open->closed sweep, so tightening
    the guard removes nothing from the shipped datasets. If this ever fails, a
    dataset was built on a dead gripper channel."""
    from export_lerobot import episode_dirs, usable_grasps
    from robots_realtime.labeling.mcap_io import read_positions

    checked = 0
    for ep in episode_dirs(CORPUS):
        _, why = usable_grasps(ep)
        if why:
            continue
        t, p = read_positions(ep / "yam_left.mcap", "yam_left")
        normalize_width(p[:, C.GRIPPER_JOINT_INDEX])      # must not raise
        checked += 1
    assert checked == 15
