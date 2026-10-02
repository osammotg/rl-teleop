"""Association engine: intervals + intent → per-bag annotations, with guards."""
from __future__ import annotations

from robots_realtime.labeling.fuse import GraspCandidate, build_annotations
from robots_realtime.labeling.placement import Compartment
from robots_realtime.labeling.schema import KitItem


def _grasp(t_close, t_open, outcome="success", lifted=True, gp=None, rp=None):
    return GraspCandidate(t_close, t_open, outcome, lifted, gp, rp)


def test_sequential_no_cockpit():
    cands = [_grasp(1, 2), _grasp(3, 4), _grasp(5, 6)]
    ann = build_annotations("ep", "left", 0, 10, cands)
    assert len(ann.place_events) == 3
    assert [p.bag_id for p in ann.place_events] == [1, 2, 3]
    assert len(ann.grasp_attempts) == 3
    assert ann.flags == []


def test_kitting_list_supplies_compartments():
    kit = [KitItem(1, "P1", compartment=5), KitItem(2, "P2", compartment=3)]
    cands = [_grasp(1, 2), _grasp(3, 4)]
    ann = build_annotations("ep", "left", 0, 10, cands, kitting_list=kit)
    assert ann.place_events[0].target_compartment == 5
    assert ann.place_events[1].target_compartment == 3


def test_cockpit_events_assign_bag_and_compartment():
    cands = [_grasp(1, 2), _grasp(3, 4)]
    events = [
        {"t": 2.1, "type": "place_confirmed", "bag_id": 7, "comp": 4},
        {"t": 4.1, "type": "place_confirmed", "bag_id": 9, "comp": 2},
    ]
    ann = build_annotations("ep", "left", 0, 10, cands, cockpit_events=events)
    assert ann.place_events[0].bag_id == 7
    assert ann.place_events[0].target_compartment == 4
    assert ann.place_events[1].bag_id == 9


def test_slip_then_regrasp_same_bag():
    # bag 1: slip (attempt 1), then success (attempt 2) → one bag, one place
    cands = [_grasp(1, 2, outcome="slip"), _grasp(2.5, 4, outcome="success")]
    ann = build_annotations("ep", "left", 0, 10, cands)
    assert len(ann.place_events) == 1                 # only one bag placed
    assert len(ann.grasp_attempts) == 2               # two attempts on it
    assert ann.grasp_attempts[0].bag_id == ann.grasp_attempts[1].bag_id
    assert ann.grasp_attempts[0].attempt == 1 and ann.grasp_attempts[0].outcome == "slip"
    assert ann.grasp_attempts[1].attempt == 2 and ann.grasp_attempts[1].regrasp_of == 1


def test_placement_classified_against_compartments():
    comps = [Compartment(5, 0.4, 0.6, -0.2, 0.0)]
    cands = [_grasp(1, 2, gp=[0.5, -0.3, 0.2, 1, 0, 0, 0],
                    rp=[0.5, -0.1, 0.15, 1, 0, 0, 0])]
    kit = [KitItem(1, "P1", compartment=5)]
    ann = build_annotations("ep", "left", 0, 10, cands, kitting_list=kit, compartments=comps)
    p = ann.place_events[0]
    assert p.in_target_region is True
    assert p.release_height_m == 0.15


def test_overlapping_grasp_flagged():
    # second grasp starts before the first is released → invariant violation
    cands = [_grasp(1, 5), _grasp(3, 6)]
    ann = build_annotations("ep", "left", 0, 10, cands)
    assert any(f.kind == "overlapping_grasp" for f in ann.flags)


def test_clock_out_of_window_event_flagged_and_ignored():
    cands = [_grasp(1, 2)]
    events = [{"t": 999, "type": "place_confirmed", "bag_id": 3, "comp": 1}]
    ann = build_annotations("ep", "left", 0, 10, cands, cockpit_events=events)
    assert any(f.kind == "clock_out_of_window" for f in ann.flags)
    # event ignored → bag falls back to sequential id 1, no compartment from event
    assert ann.place_events[0].bag_id == 1


def test_ocr_null_flagged():
    kit = [KitItem(1, None, compartment=None)]   # unreadable
    ann = build_annotations("ep", "left", 0, 10, [_grasp(1, 2)], kitting_list=kit)
    assert any(f.kind == "ocr_null" and f.bag_id == 1 for f in ann.flags)


def test_unplaced_grasp_flagged():
    # a slip that never resolves to a place → flagged, not dropped
    cands = [_grasp(1, 2, outcome="slip")]
    ann = build_annotations("ep", "left", 0, 10, cands)
    assert ann.place_events == []
    assert any(f.kind == "unplaced_grasp" for f in ann.flags)


def test_retargeting_flagged():
    """A slip that gets re-grasped and then placed = retargeting → flagged, and both
    attempts recorded (regrasp chain), so clean vs retargeted demos are separable."""
    cands = [_grasp(1, 2, outcome="slip"), _grasp(3, 4, outcome="success")]  # slip → re-grasp → place
    ann = build_annotations("ep", "left", 0, 10, cands)
    assert len(ann.place_events) == 1                       # one bag placed
    assert len(ann.grasp_attempts) == 2                     # both attempts kept
    assert ann.grasp_attempts[1].regrasp_of == 1           # 2nd re-grasps the 1st
    rt = [f for f in ann.flags if f.kind == "retargeting"]
    assert len(rt) == 1 and rt[0].bag_id == 1              # loudly flagged


def test_clean_grasp_not_flagged():
    ann = build_annotations("ep", "left", 0, 10, [_grasp(1, 2, outcome="success")])
    assert [f.kind for f in ann.flags if f.kind == "retargeting"] == []   # single attempt = clean
