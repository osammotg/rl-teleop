"""Hermetic tests for the target-conditioning export path.

No rig, no recordings, no GPU, no network: every cycle below is a fabricated dict in the
shape wrist_ocr/targets.py writes. The point is that the branches which decide whether a
frame carries a target -- and the one that REFUSES -- are exercised without needing a
session to have run.

The branch that matters most is test_episode_absent_is_refused_loudly's sibling in
export_lerobot (the per-episode rejection): an episode with no target_cycles.json inside a
--targets export would be written with target_valid = -1 on every frame, which is
indistinguishable, to the model and to every metric downstream, from a deliberately masked
window. That is the failure mode that silently costs a whole training run.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
import target_windows as tw                                     # noqa: E402


# ── fixtures in the shape targets.py writes ──────────────────────────────────

def cand(n, cx, cy, area=1000, red=0.9):
    return {"n": n, "cx": cx, "cy": cy, "area": area, "red_adj": red,
            "freeness": 1.0, "bbox": [int(cx) - 10, int(cy) - 10, 20, 20]}


def cycle(k=2, phase="pick", t0=1000.0, t1=1010.0, moved_n=1, cands=None,
          moved_px=None, bad=False, slug=None, **extra):
    c = {"k": k, "phase": phase, "t_start": t0, "t_end": t1, "bad": bad,
         "moved_n": moved_n, "candidates": cands if cands is not None else [cand(1, 240, 135)],
         "slug": slug}
    if moved_px is not None:
        c["moved_px"] = moved_px
    c.update(extra)
    return c


def write_ep(tmp_path, name, cycles):
    import json
    ep = tmp_path / name
    ep.mkdir(parents=True, exist_ok=True)
    (ep / "target_cycles.json").write_text(json.dumps(
        {"schema": "target_cycles/1", "cycles": cycles}))
    return ep


@pytest.fixture
def frame_480x270(monkeypatch):
    """Pin the frame resolver so tests never depend on a PNG being on disk.

    The real resolver reads a real image precisely so the frame is never guessed; here we
    substitute a known answer so the NORMALISATION is what is under test, not the lookup."""
    monkeypatch.setattr(tw, "_frame_size", lambda ep, cyc: (480, 270, "test"))


# ── dedupe / recovery ────────────────────────────────────────────────────────

def test_dedupe_drops_only_the_near_duplicate():
    cands = [cand(1, 296.9, 167.1), cand(2, 299.1, 167.6), cand(3, 349.0, 155.8)]
    keep = tw.dedupe_candidates(cands)
    assert keep == [0, 2], "n=2 sits 2.3 px from n=1 and is the same packet"


def test_dedupe_keeps_the_higher_ranked_of_a_pair():
    cands = [cand(1, 100, 100), cand(2, 102, 101)]
    assert tw.dedupe_candidates(cands) == [0], "rank 1 must never lose to its own sub-mask"


def test_dedupe_keeps_genuinely_distinct_packets():
    # 18.8 px was the smallest measured separation between two real packets
    cands = [cand(1, 100, 100), cand(2, 118.8, 100)]
    assert tw.dedupe_candidates(cands) == [0, 1]


def test_recompute_recovers_the_abstention_a_duplicate_caused():
    """The measured 2026-09-21 case: two co-located candidates both move by nearly the
    same distance, so which_moved's margin rule can never fire."""
    cands = [cand(1, 294.1, 196.9), cand(2, 296.8, 201.1), cand(3, 50, 50)]
    n, why = tw.recompute_moved(cands, [29.8, 33.6, 0.1])
    assert n == 1, why


def test_recompute_returns_none_when_two_distinct_packets_moved():
    cands = [cand(1, 100, 100), cand(2, 200, 100), cand(3, 300, 100)]
    n, _ = tw.recompute_moved(cands, [42.0, 12.9, 38.9])
    assert n is None, "a genuine two-packet move must stay an abstention, not a guess"


def test_recompute_refuses_mismatched_inputs():
    assert tw.recompute_moved([cand(1, 1, 1)], [])[0] is None
    assert tw.recompute_moved([], [1.0])[0] is None
    assert tw.recompute_moved([cand(1, 1, 1)], [1.0, 2.0])[0] is None


# ── load_targets branches ────────────────────────────────────────────────────

def test_supplied_target_is_normalised_against_the_real_frame(tmp_path, frame_480x270):
    write_ep(tmp_path, "ep1", [cycle(cands=[cand(1, 349.0, 155.8)], moved_n=1)])
    rows = tw.load_targets(tmp_path)["targets"]["ep1"]
    (_, _, u, v, valid) = rows[0]
    assert valid == 1.0
    assert u == pytest.approx(2 * 349.0 / 480 - 1, abs=1e-6)
    assert v == pytest.approx(2 * 155.8 / 270 - 1, abs=1e-6)


def test_moved_n_null_and_unrecoverable_is_masked(tmp_path, frame_480x270):
    write_ep(tmp_path, "ep1", [cycle(moved_n=None, moved_px=[1.0], cands=[cand(1, 10, 10)])])
    assert tw.load_targets(tmp_path)["targets"]["ep1"][0][4] == -1.0


def test_bad_cycle_is_masked(tmp_path, frame_480x270):
    write_ep(tmp_path, "ep1", [cycle(bad=True)])
    assert tw.load_targets(tmp_path)["targets"]["ep1"][0][4] == -1.0


def test_phase_outside_the_requested_set_is_masked(tmp_path, frame_480x270):
    write_ep(tmp_path, "ep1", [cycle(phase="isolate")])
    rows = tw.load_targets(tmp_path, phases=("pick",))["targets"]["ep1"]
    assert rows[0][4] == -1.0


def test_moved_n_not_in_candidate_list_is_masked(tmp_path, frame_480x270):
    write_ep(tmp_path, "ep1", [cycle(moved_n=9, cands=[cand(1, 100, 100)])])
    assert tw.load_targets(tmp_path)["targets"]["ep1"][0][4] == -1.0


def test_unresolved_frame_is_masked_never_defaulted(tmp_path, monkeypatch):
    """A cycle whose frame size cannot be established must NOT fall back to 480x270.
    Normalising against the wrong frame trains happily and is wrong by 2.667x."""
    monkeypatch.setattr(tw, "_frame_size", lambda ep, cyc: (None, None, "no png"))
    write_ep(tmp_path, "ep1", [cycle()])
    doc = tw.load_targets(tmp_path)
    assert doc["targets"]["ep1"][0][4] == -1.0
    assert doc["manifest"]["no_frame"] == 1


def test_centroid_outside_the_frame_is_refused_not_clamped(tmp_path, frame_480x270):
    write_ep(tmp_path, "ep1", [cycle(cands=[cand(1, 9999, 135)], moved_n=1)])
    doc = tw.load_targets(tmp_path)
    assert doc["targets"]["ep1"][0][4] == -1.0
    assert doc["manifest"]["no_frame"] == 1


def test_cycle_without_timestamps_is_skipped(tmp_path, frame_480x270):
    write_ep(tmp_path, "ep1", [cycle(t0=None, t1=None), cycle(k=4)])
    assert len(tw.load_targets(tmp_path)["targets"]["ep1"]) == 1


def test_cycle_with_inverted_window_is_skipped(tmp_path, frame_480x270):
    write_ep(tmp_path, "ep1", [cycle(t0=20.0, t1=10.0)])
    assert "ep1" not in tw.load_targets(tmp_path)["targets"]


# ── masking ──────────────────────────────────────────────────────────────────

def test_masking_is_deterministic_for_a_seed(tmp_path, frame_480x270):
    write_ep(tmp_path, "ep1", [cycle(k=k) for k in (2, 4, 6, 8, 10, 12)])
    a = tw.load_targets(tmp_path, p_mask=0.5, seed=7)["targets"]["ep1"]
    b = tw.load_targets(tmp_path, p_mask=0.5, seed=7)["targets"]["ep1"]
    assert a == b


def test_masking_extremes(tmp_path, frame_480x270):
    write_ep(tmp_path, "ep1", [cycle(k=k) for k in (2, 4, 6, 8)])
    none = tw.load_targets(tmp_path, p_mask=0.0)["manifest"]
    allm = tw.load_targets(tmp_path, p_mask=1.0)["manifest"]
    assert none["masked"] == 0 and none["supplied"] == 4
    assert allm["masked"] == 4 and allm["supplied"] == 0


def test_masking_is_per_window_not_per_frame(tmp_path, frame_480x270):
    """The goal is constant over a cycle by construction; per-frame flicker would be a
    train/deploy mismatch the model can detect."""
    write_ep(tmp_path, "ep1", [cycle(t0=0.0, t1=10.0)])
    rows = tw.load_targets(tmp_path, p_mask=1.0)["targets"]["ep1"]
    assert len(rows) == 1, "one row per cycle, not per frame"
    for t in (0.0, 5.0, 10.0):
        assert tw.lookup(rows, t) == (0.0, 0.0, -1.0)


# ── lookup ───────────────────────────────────────────────────────────────────

def test_lookup_is_by_time_containment(tmp_path, frame_480x270):
    rows = [(100.0, 110.0, 0.5, 0.25, 1.0)]
    assert tw.lookup(rows, 100.0) == (0.5, 0.25, 1.0)
    assert tw.lookup(rows, 110.0) == (0.5, 0.25, 1.0)
    assert tw.lookup(rows, 105.0) == (0.5, 0.25, 1.0)


def test_lookup_outside_every_cycle_is_masked():
    rows = [(100.0, 110.0, 0.5, 0.25, 1.0)]
    assert tw.lookup(rows, 99.9) == (0.0, 0.0, -1.0)
    assert tw.lookup(rows, 110.1) == (0.0, 0.0, -1.0)
    assert tw.lookup([], 105.0) == (0.0, 0.0, -1.0)


# ── the export's refusal ─────────────────────────────────────────────────────

def test_episode_absent_from_the_target_map_is_not_silently_unconditioned(tmp_path,
                                                                          frame_480x270):
    """An episode with no target_cycles.json must not appear in the map at all, so
    export_lerobot can reject it by name rather than exporting it all-masked."""
    write_ep(tmp_path, "ep_with", [cycle()])
    (tmp_path / "ep_without").mkdir()
    doc = tw.load_targets(tmp_path)
    assert "ep_with" in doc["targets"]
    assert "ep_without" not in doc["targets"]


def test_build_features_widens_state_but_never_action():
    import export_lerobot as E
    shapes = {"camera_top": (720, 1280), "camera_right": (480, 640)}
    cams, arms = E.CAMERA_SETS["wrist_right_top"], E.ARM_SETS["right"]
    saved = E.TARGETS
    try:
        E.TARGETS = None
        plain = E.build_features(shapes, cams, arms)
        E.TARGETS = {"ep": []}
        cond = E.build_features(shapes, cams, arms)
    finally:
        E.TARGETS = saved
    assert plain["observation.state"]["shape"] == (7,)
    assert cond["observation.state"]["shape"] == (10,)
    assert cond["observation.state"]["names"][7:] == list(E.TARGET_DIM_NAMES)
    # the deploy path truncates the policy's output to 7 (vla_policy_server.py:214), so a
    # wider action would train dimensions that are thrown away at inference
    assert cond["action"]["shape"] == plain["action"]["shape"] == (7,)
