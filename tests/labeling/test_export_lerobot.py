"""Coverage for the mcap -> LeRobot grasp exporter.

Every filter and every branch here corresponds to a failure actually present in
the recorded corpus, not a hypothetical one. The counts in the comments are
measured over recordings/ as of 2026-07-27 (30 episodes, 28 with annotations).

The two failures worth the most: action/state swapped, and BGR/RGB swapped.
Neither raises. Both produce a dataset that trains to a plausible loss curve and
a policy that does not work, discovered an hour of GPU time later.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))

from export_lerobot import (  # noqa: E402
    CAMERAS, MIN_WINDOW_FRAMES, N_DOF, build_features, grasp_windows,
    in_zone, nearest_index, normalize_gripper, usable_grasps, zone_label,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from robots_realtime.labeling import constants as C  # noqa: E402


# ── fixtures ────────────────────────────────────────────────────────────────
def write_episode(tmp: Path, name: str, *, annotations=None, flags=None) -> Path:
    ep = tmp / name
    ep.mkdir(parents=True)
    if annotations is not None:
        (ep / "annotations.json").write_text(json.dumps(annotations))
    if flags is not None:
        (ep / "operator_flags.json").write_text(json.dumps(flags))
    return ep


def ann(outcome="success", attempts=(), **meta):
    return {"episode_meta": {"outcome": outcome, **meta},
            "segments": [], "grasp_attempts": list(attempts),
            "place_events": [], "tracking": [], "flags": []}


def grasp(t, outcome="success", x=0.43, y=-0.25):
    """A grasp attempt. Defaults sit mid-zone: the corpus mean is (0.432, -0.244)
    and the median y is -0.244, so the default survives every gate below."""
    return {"bag_id": 1, "attempt": 1, "arm": "left", "t": t, "outcome": outcome,
            "ee_pose": [x, y, 0.12, 0.0, 0.0, 0.0, 1.0]}


# ── episode selection ───────────────────────────────────────────────────────
def test_accepts_a_clean_episode(tmp_path):
    ep = write_episode(tmp_path, "episode_a", annotations=ann(attempts=[grasp(10.0)]))
    good, why = usable_grasps(ep)
    assert why is None and len(good) == 1


def test_rejects_zero_grasp_episode_labelled_success(tmp_path):
    """The 17% case: `outcome` DEFAULTS to success and is not derived from
    whether anything was grasped. 3 of 18 'success' episodes hold no attempts.
    Training on these teaches the policy to stand still."""
    ep = write_episode(tmp_path, "episode_b", annotations=ann(outcome="success"))
    good, why = usable_grasps(ep)
    assert good == [] and "zero grasp_attempts" in why


def test_rejects_aborted(tmp_path):
    ep = write_episode(tmp_path, "episode_c",
                       annotations=ann(outcome="aborted", attempts=[grasp(1.0)]))
    good, why = usable_grasps(ep)
    assert good == [] and "aborted" in why


def test_rejects_missing_annotations(tmp_path):
    ep = write_episode(tmp_path, "episode_d")           # 2 of 30 episodes
    good, why = usable_grasps(ep)
    assert good == [] and "no annotations.json" in why


def test_rejects_the_tag_the_recorder_actually_writes(tmp_path):
    """THE regression for AUDIT.md S7.1. The operator presses the KEY 'x';
    tui.py:259 maps it to the TAG 'bad'; session.py:389 writes {"tag": "bad"};
    control_server.py:59 rejects a literal "x" with a 400. The exporter filtered
    on "x", so the tag it looked for could not exist and the filter never once
    fired -- while review_corpus.py, reading the same files, correctly named
    those episodes in the QA report.

    This fixture is the bytes session.py writes, not a hand-made "x". The old
    test fabricated {"tag": "x"} and therefore passed while pinning the bug in
    place; that is why this file's fixture had to change with the code."""
    ep = write_episode(tmp_path, "episode_e",
                       annotations=ann(attempts=[grasp(5.0)]),
                       flags={"flags": [{"tag": "bad", "t": 1785097885.49}]})
    good, why = usable_grasps(ep)
    assert good == [] and "bad" in why


def test_the_tag_matches_what_the_session_writer_uses():
    """Pins the two ends of the chain to one constant so they cannot drift
    apart again. Reads the writer's own map rather than restating "bad"."""
    from robots_realtime.runtime.tui import _FLAG_KEYS
    assert _FLAG_KEYS["x"] in C.OPERATOR_BAD_TAGS
    assert _FLAG_KEYS["g"] not in C.OPERATOR_BAD_TAGS
    assert _FLAG_KEYS["s"] not in C.OPERATOR_BAD_TAGS


def test_a_bad_take_is_excluded_from_the_exported_plan(tmp_path):
    """usable_grasps is not the only door into the exporter -- plan_episode has
    its own path for window_mode='full', which never calls it. Prove the take is
    actually excluded from what gets written, not merely from one filter."""
    from export_lerobot import Report, plan_episode
    ep = write_episode(tmp_path, "episode_bad",
                       annotations=ann(attempts=[grasp(5.0)]),
                       flags={"flags": [{"tag": "bad", "t": 5.2}]})
    for mode in ("grasp", "full"):
        rep = Report()
        assert plan_episode(ep, 3.0, 2.0, 30, rep, window_mode=mode) is None
        assert any("bad take" in r.reason for r in rep.rejected), mode


def test_other_flags_do_not_reject(tmp_path):
    """'re_grasp' and 'slow' are annotations, not rejections. Both appear in the
    real corpus (two episodes carry 'slow')."""
    ep = write_episode(tmp_path, "episode_f",
                       annotations=ann(attempts=[grasp(5.0)]),
                       flags={"flags": [{"tag": "re_grasp", "t": 5.2},
                                        {"tag": "slow", "t": 6.0}]})
    good, why = usable_grasps(ep)
    assert why is None and len(good) == 1


def test_a_malformed_flag_entry_does_not_crash_the_export(tmp_path):
    ep = write_episode(tmp_path, "episode_f2",
                       annotations=ann(attempts=[grasp(5.0)]),
                       flags={"flags": ["bad", {"t": 1.0}, {"tag": None}]})
    good, why = usable_grasps(ep)
    assert why is None and len(good) == 1


def test_keeps_only_successful_attempts_within_an_episode(tmp_path):
    ep = write_episode(tmp_path, "episode_g", annotations=ann(
        attempts=[grasp(1.0, "success"), grasp(5.0, "empty"), grasp(9.0, "success")]))
    good, why = usable_grasps(ep)
    assert why is None and [g["t"] for g in good] == [1.0, 9.0]


def test_rejects_when_no_attempt_succeeded(tmp_path):
    ep = write_episode(tmp_path, "episode_h",
                       annotations=ann(attempts=[grasp(1.0, "empty"), grasp(2.0, "empty")]))
    good, why = usable_grasps(ep)
    assert good == [] and "none with outcome=success" in why


def test_corrupt_annotations_rejected_not_raised(tmp_path):
    ep = tmp_path / "episode_i"
    ep.mkdir()
    (ep / "annotations.json").write_text("{not json")
    good, why = usable_grasps(ep)
    assert good == [] and why is not None


# ── windowing ───────────────────────────────────────────────────────────────
def test_window_is_pre_and_post_around_the_close_instant():
    """`segments` with phase='grasp' are INSTANTS (t_start == t_end) in the real
    data, so the trainable window has to be constructed around them."""
    (lo, hi), = grasp_windows([grasp(100.0)], t0=0.0, t1=200.0, pre_s=3.0, post_s=2.0)
    assert (lo, hi) == (97.0, 102.0)


def test_window_clipped_to_recorded_span():
    """A grasp 1s into the episode cannot have 3s of approach in front of it."""
    (lo, hi), = grasp_windows([grasp(1.0)], t0=0.0, t1=10.0, pre_s=3.0, post_s=2.0)
    assert lo == 0.0 and hi == 3.0


def test_adjacent_windows_never_overlap():
    """Overlap would put identical frames in two training episodes with
    different action labels — silent, and it inflates the dataset while teaching
    the policy contradictions."""
    ws = grasp_windows([grasp(10.0), grasp(12.0)], t0=0.0, t1=30.0,
                       pre_s=3.0, post_s=2.0)
    assert len(ws) == 2
    assert ws[0][1] <= ws[1][0], f"windows overlap: {ws}"


def test_many_grasps_yield_many_windows():
    """One recorded kitting run produces several training episodes — measured
    81 windows from 15 real episodes."""
    ts = [10.0 * i for i in range(1, 6)]
    assert len(grasp_windows([grasp(t) for t in ts], 0.0, 100.0, 3.0, 2.0)) == 5


def test_grasp_outside_recorded_span_is_dropped():
    assert grasp_windows([grasp(500.0)], t0=0.0, t1=100.0, pre_s=3.0, post_s=2.0) == []


# ── signals ─────────────────────────────────────────────────────────────────
def test_gripper_normalised_to_unit_range():
    """Raw limits are auto-calibrated on every boot (observed 5.2218/-0.0235),
    so raw values are not comparable across sessions."""
    out = normalize_gripper(np.array([-0.02, 2.6, 5.22], dtype=np.float32))
    assert out.min() == pytest.approx(0.0) and out.max() == pytest.approx(1.0)


def test_gripper_normalisation_is_scale_invariant():
    """Two sessions with different calibration must produce the same normalised
    signal for the same physical motion. This is the whole reason to normalise."""
    a = normalize_gripper(np.array([0.0, 1.0, 2.0], dtype=np.float32))
    b = normalize_gripper(np.array([-0.02, 2.6, 5.22], dtype=np.float32))
    assert np.allclose(a, b, atol=1e-6)


def test_degenerate_gripper_range_refuses_instead_of_inventing_all_open():
    """Was: `assert np.all(out == 1.0)`. That pinned half of AUDIT.md S1.3 --
    this file answered "wide open" and the labeller answered "fully shut" for
    the same recording, and both were guesses. Full coverage in
    test_gripper_normalisation.py."""
    from robots_realtime.labeling.segmentation import GripperRangeUnknown
    with pytest.raises(GripperRangeUnknown):
        normalize_gripper(np.full(5, 3.0, dtype=np.float32))


def test_nearest_index_picks_the_closest_sample():
    t = np.array([0.0, 1.0, 2.0, 3.0])
    assert list(nearest_index(t, np.array([-5.0, 0.4, 0.6, 2.9, 99.0]))) == [0, 0, 1, 3, 3]


def test_nearest_index_handles_rate_mismatch():
    """200Hz state resampled to a 30Hz grid must stay in order and in range."""
    fast = np.arange(0, 10, 1 / 200)
    grid = np.arange(0, 10, 1 / 30)
    idx = nearest_index(fast, grid)
    assert np.all(np.diff(idx) >= 0)
    assert np.max(np.abs(fast[idx] - grid)) < 1 / 200


# ── schema ──────────────────────────────────────────────────────────────────
def test_features_use_per_camera_resolution():
    """The rig records camera_top at 1280x720 and camera_left at 640x480.
    Assuming one shared resolution makes LeRobotDataset reject every wrist frame
    at add_frame() — caught only when it hit real data."""
    feats = build_features({"camera_top": (720, 1280), "camera_left": (480, 640)})
    assert feats["observation.images.top"]["shape"] == (720, 1280, 3)
    assert feats["observation.images.wrist"]["shape"] == (480, 640, 3)


def test_state_and_action_are_separate_features_of_the_right_width():
    feats = build_features({c: (64, 64) for c in CAMERAS})
    assert feats["observation.state"]["shape"] == (N_DOF,)
    assert feats["action"]["shape"] == (N_DOF,)
    assert feats["observation.state"]["names"][-1] == "gripper"


def test_scan_camera_is_excluded():
    """The scan cam watches the packet mat, not the workspace. It is ~49% of
    every episode's bytes and contributes nothing to a grasp policy."""
    assert "camera_scan" not in CAMERAS
    assert all("scan" not in k for k in build_features({c: (64, 64) for c in CAMERAS}))


def test_min_window_frames_guards_stub_episodes():
    assert MIN_WINDOW_FRAMES >= 10


# ── video ───────────────────────────────────────────────────────────────────
def _write_video(path: Path, colors_bgr, size=(64, 48)):
    cv2 = pytest.importorskip("cv2")
    w, h = size
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (w, h))
    if not vw.isOpened():
        pytest.skip("no mp4 writer available")
    for bgr in colors_bgr:
        frame = np.zeros((h, w, 3), np.uint8)
        frame[:, :] = bgr
        vw.write(frame)
    vw.release()


def test_camera_stream_returns_rgb_not_bgr(tmp_path):
    """cv2 decodes BGR; LeRobot stores RGB. Swapping them raises nothing, trains
    to a normal-looking loss, and produces a policy that fails on real images.
    Written as pure blue in BGR (255,0,0) — read back it must be RED-channel 0
    and BLUE-channel high."""
    from export_lerobot import CameraStream
    mp4 = tmp_path / "v.mp4"
    _write_video(mp4, [(255, 0, 0)] * 5)            # BGR blue
    stamps = tmp_path / "v.npy"
    np.save(stamps, np.arange(5) / 30.0)

    s = CameraStream(mp4, stamps)
    try:
        img = s.frame_at(0)
    finally:
        s.close()
    assert img is not None
    r, g, b = (int(img[..., i].mean()) for i in range(3))
    assert b > 200 and r < 60, f"channels look BGR-ordered: r={r} g={g} b={b}"


def test_camera_stream_reads_forward_and_refuses_backward(tmp_path):
    """Sequential-only by design: cv2 seeking on h264 lands on the nearest
    keyframe and silently returns the wrong frame."""
    from export_lerobot import CameraStream
    mp4 = tmp_path / "v.mp4"
    _write_video(mp4, [(0, 0, 0), (0, 0, 0), (0, 0, 0), (0, 0, 0)])
    stamps = tmp_path / "v.npy"
    np.save(stamps, np.arange(4) / 30.0)

    s = CameraStream(mp4, stamps)
    try:
        assert s.frame_at(0) is not None
        assert s.frame_at(2) is not None            # forward is fine
        with pytest.raises(RuntimeError, match="backward"):
            s.frame_at(1)
    finally:
        s.close()


def test_camera_stream_past_end_returns_none(tmp_path):
    from export_lerobot import CameraStream
    mp4 = tmp_path / "v.mp4"
    _write_video(mp4, [(0, 0, 0)] * 3)
    stamps = tmp_path / "v.npy"
    np.save(stamps, np.arange(3) / 30.0)

    s = CameraStream(mp4, stamps)
    try:
        assert s.frame_at(99) is None               # must not raise or hang
    finally:
        s.close()


# ── workspace gate ──────────────────────────────────────────────────────────
# Added 2026-07-28. The corpus holds 4 successful grasps at x 0.107..0.178 with
# z 0.162..0.226 against a corpus z mean of 0.120 -- near AND high, i.e. mid-air
# or mislabelled, not table grasps. Sorted x has a 176mm empty band between them
# and the other 77, so the gate is a fixed threshold rather than a moving
# statistical cut. See constants.GRASP_WORKSPACE_X_MIN.

def test_gate_keeps_a_grasp_on_the_mat():
    assert in_zone(grasp(1.0, x=0.43)) is True


def test_gate_drops_a_grasp_off_the_mat():
    assert in_zone(grasp(1.0, x=0.15)) is False


def test_gate_boundary_is_inclusive():
    assert in_zone(grasp(1.0, x=C.GRASP_WORKSPACE_X_MIN)) is True
    assert in_zone(grasp(1.0, x=C.GRASP_WORKSPACE_X_MIN - 1e-6)) is False


def test_gate_fails_open_on_a_missing_pose():
    """Unknown is not out-of-bounds. A grasp with no ee_pose is a labeller
    problem and must not be silently reclassified as a zone result."""
    g = grasp(1.0)
    del g["ee_pose"]
    assert in_zone(g) is True


# ── lateral gate ────────────────────────────────────────────────────────────
# Added 2026-08-03 for the zone-filtered retrain. OFF by default so every
# dataset exported before that date still reproduces byte-for-byte.

def test_lateral_gate_is_off_by_default():
    """THE compatibility guarantee. y=-0.099 is a real corner grasp from
    episode_181321; with no y_max passed it must still export, or yam_grasp_v1
    silently stops reproducing."""
    assert C.GRASP_ZONE_Y_MAX is None
    assert in_zone(grasp(1.0, x=0.54, y=-0.099)) is True


def test_lateral_gate_drops_the_corner_when_asked():
    assert in_zone(grasp(1.0, y=-0.099), y_max=-0.13) is False


def test_lateral_gate_keeps_the_main_band():
    assert in_zone(grasp(1.0, y=-0.30), y_max=-0.13) is True


def test_lateral_boundary_is_inclusive_like_x():
    """Both bounds keep the grasp that sits exactly on them. Asymmetry here
    would be a silent off-by-one between the two gates."""
    assert in_zone(grasp(1.0, y=-0.13), y_max=-0.13) is True
    assert in_zone(grasp(1.0, y=-0.13 + 1e-6), y_max=-0.13) is False


def test_lateral_gate_fails_open_on_a_missing_pose():
    g = grasp(1.0)
    del g["ee_pose"]
    assert in_zone(g, y_max=-0.13) is True


def test_bounds_are_independent():
    """A grasp outside on x is dropped no matter how good its y is, and the
    reverse. Collapsing these into one test would hide either bound going dead."""
    assert in_zone(grasp(1.0, x=0.15, y=-0.30), y_max=-0.13) is False   # x only
    assert in_zone(grasp(1.0, x=0.43, y=-0.09), y_max=-0.13) is False   # y only
    assert in_zone(grasp(1.0, x=0.15, y=-0.09), y_max=-0.13) is False   # both
    assert in_zone(grasp(1.0, x=0.43, y=-0.30), y_max=-0.13) is True    # neither


def test_zone_label_agrees_with_in_zone():
    """The review page badges with zone_label and the exporter gates with
    in_zone. If they disagree the page shows a selection the dataset does not
    contain -- the exact failure the review tool exists to prevent."""
    cases = [(0.43, -0.25), (0.15, -0.25), (0.43, -0.09), (0.15, -0.09),
             (0.25, -0.13), (0.54, -0.085), (0.354, -0.379)]
    for x, y in cases:
        g = grasp(1.0, x=x, y=y)
        assert (zone_label(g, y_max=-0.13) == "in") is in_zone(g, y_max=-0.13), (x, y)

    g = grasp(1.0)
    del g["ee_pose"]
    assert zone_label(g) == "nopose"
    assert in_zone(g) is True


def test_zone_label_names_which_bound_dropped_it():
    assert zone_label(grasp(1.0, x=0.15), y_max=-0.13) == "near"
    assert zone_label(grasp(1.0, y=-0.09), y_max=-0.13) == "corner"
    assert zone_label(grasp(1.0), y_max=-0.13) == "in"


def test_episode_rejection_reason_names_both_bounds(tmp_path):
    """A run that exports nothing must say WHY in terms of the bounds actually
    in force, not the default ones."""
    ep = write_episode(tmp_path, "episode_corner", annotations=ann(
        attempts=[grasp(1.0, y=-0.09), grasp(5.0, y=-0.10)]))
    good, why = usable_grasps(ep, y_max=-0.13)
    assert good == []
    assert "y > -0.13" in why


def test_gate_is_grasp_level_not_episode_level(tmp_path):
    """THE regression this gate exists to avoid getting wrong. Both corpus
    episodes carrying out-of-workspace grasps also carry good ones. Rejecting
    whole episodes would drop 23 usable windows to remove 4 bad ones."""
    ep = write_episode(tmp_path, "episode_mixed", annotations=ann(attempts=[
        grasp(1.0, x=0.43),      # on the mat
        grasp(5.0, x=0.15),      # off the mat
        grasp(9.0, x=0.47),      # on the mat
    ]))
    good, why = usable_grasps(ep)
    assert why is None
    assert [g["t"] for g in good] == [1.0, 9.0]


def test_episode_with_only_off_mat_grasps_is_rejected(tmp_path):
    ep = write_episode(tmp_path, "episode_alloff", annotations=ann(
        attempts=[grasp(1.0, x=0.10), grasp(5.0, x=0.18)]))
    good, why = usable_grasps(ep)
    assert good == []
    assert why is not None and "outside the zone" in why


def test_gate_can_be_disabled_for_review(tmp_path):
    """review_grasps.py passes workspace_gate=False so it can SHOW what was
    dropped. Nothing that writes training data may pass False."""
    ep = write_episode(tmp_path, "episode_review", annotations=ann(
        attempts=[grasp(1.0, x=0.43), grasp(5.0, x=0.15)]))
    good, why = usable_grasps(ep, workspace_gate=False)
    assert why is None and [g["t"] for g in good] == [1.0, 5.0]


# ── the corpus itself ───────────────────────────────────────────────────────
# Everything above runs on synthetic episodes. These two run on the REAL
# recordings, because the numbers the review page shows an operator are the
# numbers the dataset must contain, and a synthetic fixture cannot pin that.
# Skipped rather than failed when recordings/ is absent (CI, fresh clone).

CORPUS = Path(__file__).resolve().parents[2] / "recordings"
needs_corpus = pytest.mark.skipif(not CORPUS.exists(), reason="recordings/ not present")


def _corpus_grasps(x_min=None, y_max=None):
    from export_lerobot import episode_dirs
    n = 0
    for ep in episode_dirs(CORPUS):
        good, why = usable_grasps(ep, x_min=x_min, y_max=y_max)
        if not why:
            n += len(good)
    return n


@needs_corpus
def test_corpus_default_zone_still_yields_v1():
    """yam_grasp_v1 is 77 windows. If this number moves, the lateral gate has
    leaked into the default path and v1 has stopped reproducing."""
    assert _corpus_grasps() == 77


@needs_corpus
def test_corpus_zone_x325_y13_yields_69():
    """The zone reviewed on 2026-08-03. This is the count the HTML page reports
    and the count `--zone-x-min 0.325 --zone-y-max -0.13` must export. If the
    two ever disagree, an operator approved a selection the model never saw."""
    assert _corpus_grasps(x_min=0.325, y_max=-0.13) == 69


@needs_corpus
def test_corpus_x_bound_alone_changes_nothing_between_025_and_0325():
    """Documents WHY the x bound is not the interesting knob: the corpus has a
    176 mm empty band there, so every threshold in it selects the same grasps.
    The lateral bound is where the real choice lives."""
    assert _corpus_grasps(x_min=0.25) == _corpus_grasps(x_min=0.325) == 77


# ── camera sets ─────────────────────────────────────────────────────────────
# Added 2026-08-03. A wrist-only dataset trains a policy that reads the gripper
# camera and the joints, nothing else. ACT supports it with no config change:
# the backbone and encoder_img_feat_input_proj are SHARED across cameras and the
# camera position embedding is sinusoidal, so no weight is camera-keyed
# (verified against the 50k checkpoint: zero tensors mention a camera).

def test_default_camera_set_is_both():
    """Compatibility guarantee, same shape as the zone one: omitting --cameras
    must reproduce the two-camera datasets exported before this flag existed."""
    from export_lerobot import CAMERA_SETS, resolve_cameras
    assert resolve_cameras(None) == CAMERAS
    assert CAMERA_SETS["both"] == CAMERAS


def test_wrist_set_is_the_gripper_camera_only():
    from export_lerobot import CAMERA_SETS
    assert CAMERA_SETS["wrist"] == {"camera_left": "wrist"}


def test_scan_camera_is_in_no_camera_set():
    """camera_scan looks at the packet mat and must never reach the policy,
    whichever set is chosen."""
    from export_lerobot import CAMERA_SETS
    for name, cams in CAMERA_SETS.items():
        assert "camera_scan" not in cams, name


def test_features_follow_the_selected_camera_set():
    from export_lerobot import CAMERA_SETS
    shapes = {"camera_top": (720, 1280), "camera_left": (480, 640)}

    both = build_features(shapes, CAMERA_SETS["both"])
    assert set(both) == {"observation.state", "action",
                         "observation.images.top", "observation.images.wrist"}

    wrist = build_features(shapes, CAMERA_SETS["wrist"])
    assert set(wrist) == {"observation.state", "action", "observation.images.wrist"}
    # the joints are NOT dropped with the camera -- that is the whole point
    assert wrist["observation.state"]["shape"] == (N_DOF,)
    assert wrist["action"]["shape"] == (N_DOF,)
    # and the surviving camera keeps its own 640x480, not the top camera's
    assert wrist["observation.images.wrist"]["shape"] == (480, 640, 3)


def test_wrist_features_are_a_subset_of_both():
    """A wrist-only policy must see exactly the feature it would have seen in a
    two-camera dataset, byte for byte. If the suffix or shape drifted, weights
    from the two-camera checkpoint could not be fine-tuned onto it."""
    from export_lerobot import CAMERA_SETS
    shapes = {"camera_top": (720, 1280), "camera_left": (480, 640)}
    both = build_features(shapes, CAMERA_SETS["both"])
    wrist = build_features(shapes, CAMERA_SETS["wrist"])
    for k, v in wrist.items():
        assert both[k] == v, k


def test_build_features_names_the_mismatch_not_the_missing_key():
    """Regression, 2026-08-03: export() probed shapes for the wrist camera but
    called build_features() without passing `cameras`, so it asked for the top
    camera and died on `KeyError: 'camera_top'` 100 lines from the cause. The
    --dry-run path returns before this line, so the dry run was clean."""
    from export_lerobot import CAMERA_SETS
    with pytest.raises(KeyError, match="pass the same .cameras."):
        build_features({"camera_left": (480, 640)}, CAMERA_SETS["both"])
