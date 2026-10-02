"""Online segmenter consistency + live labeler progress."""
from __future__ import annotations

from robots_realtime.labeling.live import LiveLabeler, OnlineGripSegmenter

DT = 0.02


def _feed(segmenter, segments):
    """Feed [(dur, raw), ...] and collect emitted events."""
    evs, t = [], 0.0
    for dur, val in segments:
        for _ in range(int(dur / DT)):
            e = segmenter.push(t, val)
            if e:
                evs.append(e)
            t += DT
    return evs


def test_online_detects_close_and_release():
    seg = OnlineGripSegmenter(open_ref=1.0, closed_ref=0.0)
    evs = _feed(seg, [(1.0, 1.0), (1.0, 0.35), (1.0, 1.0)])
    kinds = [e.kind for e in evs]
    assert kinds == ["close", "release"]
    assert evs[1].outcome == "success"


def test_online_empty_and_slip():
    seg = OnlineGripSegmenter(open_ref=1.0, closed_ref=0.0)
    empty = _feed(seg, [(1.0, 1.0), (1.0, 0.02), (1.0, 1.0)])
    assert empty[-1].outcome == "empty"

    seg2 = OnlineGripSegmenter(open_ref=1.0, closed_ref=0.0)
    slip = _feed(seg2, [(1.0, 1.0), (0.6, 0.35), (0.6, 0.02), (1.0, 1.0)])
    assert slip[-1].outcome == "slip"


def test_online_running_normalization_no_refs():
    # no refs → running min/max, first sample = open
    seg = OnlineGripSegmenter()
    evs = _feed(seg, [(1.0, 1000), (1.0, 350), (1.0, 1000)])
    assert [e.kind for e in evs] == ["close", "release"]


def test_live_labeler_advances_on_place():
    lab = LiveLabeler(open_ref=1.0, closed_ref=0.0)
    lab.seed([{"part": "P1", "comp": 5, "bag_id": 1},
              {"part": "P2", "comp": 3, "bag_id": 2}])
    assert lab.state()["ti"] == 0
    # two clean grasp-place cycles
    t = 0.0
    for dur, val in [(1.0, 1.0), (1.0, 0.35), (1.0, 1.0), (1.0, 0.35), (1.0, 1.0)]:
        for _ in range(int(dur / DT)):
            lab.push(t, val); t += DT
    s = lab.state()
    assert s["ti"] == 2
    assert s["done"] == 2
    assert all(p["status"] == "placed" for p in s["packets"])
    places = [e for e in lab.events if e["type"] == "place_confirmed"]
    assert [e["comp"] for e in places] == [5, 3]


def test_live_labeler_slip_stays_on_bag():
    lab = LiveLabeler(open_ref=1.0, closed_ref=0.0)
    lab.seed([{"part": "P1", "comp": 5, "bag_id": 1}])
    t = 0.0
    # slip first (bag not advanced), then successful regrasp
    for dur, val in [(1.0, 1.0), (0.6, 0.35), (0.6, 0.02), (1.0, 1.0),  # slip
                     (1.0, 0.35), (1.0, 1.0)]:                          # success
        for _ in range(int(dur / DT)):
            lab.push(t, val); t += DT
    s = lab.state()
    assert s["ti"] == 1 and s["done"] == 1
    assert any(e["type"] == "grasp_failed" for e in lab.events)
