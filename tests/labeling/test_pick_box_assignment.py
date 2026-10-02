"""Regression: the cockpit's orange 'pick this next' box must stay on the CURRENT
packet even with duplicate part ids + a scarce OCR window.

Bug (fixed): _state_with_detections iterated ALL packets including status=='placed'
ones. A placed bag is physically off the mat, but its entry still claimed a live
detection by part id — so with 3 identical bags, once one was placed it stole the
one box OCR read that window from the still-present CURRENT packet, and the orange
overlay vanished. Placed packets must not claim detections.
"""
from __future__ import annotations

from robots_realtime.labeling.live import LiveLabeler
from robots_realtime.labeling.live_server import _state_with_detections
from robots_realtime.labeling.detector import Detection


class _FakeDetector:
    """Minimal stand-in for PacketDetector: exposes current() + status()."""

    def __init__(self, dets, wh=(1280, 720)):
        self._dets = dets
        self._wh = wh

    def current(self):
        return list(self._dets), self._wh

    def status(self):
        return {"mode": "contour"}


def _kit():
    # The real kit runs three identical Blindniet bags, then a distinct one.
    return [
        {"bag_id": 1, "part": "UNN-16022-009", "name": "Blindniet", "comp": 7},
        {"bag_id": 2, "part": "UNN-16022-009", "name": "Blindniet", "comp": 7},
        {"bag_id": 3, "part": "UNN-16022-009", "name": "Blindniet", "comp": 7},
        {"bag_id": 4, "part": "UNN-10015-007", "name": "Schraube", "comp": 6},
    ]


def _current(st):
    return st["packets"][st["ti"]]


def test_placed_duplicate_does_not_steal_current_box():
    """bag1 placed, only one Blindniet id read this window -> current bag2 keeps a box."""
    lab = LiveLabeler()
    lab.seed(_kit())
    lab.advance()  # place bag1 (physically removed from the mat), ti -> bag2

    dets = [
        Detection([441, 205, 142, 61], "UNN-16022-009", 0.99),  # one Blindniet read OK
        Detection([799, 419, 145, 90], None, 0.0),              # other Blindniet: box held, id unread
        Detection([901, 292, 133, 61], "UNN-10015-007", 0.98),  # the Schraube
    ]
    st = _state_with_detections(lab, _FakeDetector(dets))

    assert st["ti"] == 1, "bag2 should be current after placing bag1"
    assert _current(st)["bbox_px"] == [441, 205, 142, 61], \
        "current packet lost its orange box to the placed duplicate"


def test_placed_packets_never_carry_a_box():
    lab = LiveLabeler()
    lab.seed(_kit())
    lab.advance()
    lab.advance()  # place bag1 + bag2
    dets = [
        Detection([441, 205, 142, 61], "UNN-16022-009", 0.99),
        Detection([901, 292, 133, 61], "UNN-10015-007", 0.98),
    ]
    st = _state_with_detections(lab, _FakeDetector(dets))
    placed = [p for p in st["packets"] if p["status"] == "placed"]
    assert placed and all(p["bbox_px"] is None for p in placed)


def test_fresh_scan_all_boxed():
    """No placements yet + all ids read -> every packet (incl. current) is boxed."""
    lab = LiveLabeler()
    lab.seed(_kit())
    dets = [
        Detection([1, 1, 10, 10], "UNN-16022-009", 0.9),
        Detection([2, 2, 10, 10], "UNN-16022-009", 0.9),
        Detection([3, 3, 10, 10], "UNN-16022-009", 0.9),
        Detection([9, 9, 10, 10], "UNN-10015-007", 0.9),
    ]
    st = _state_with_detections(lab, _FakeDetector(dets))
    assert _current(st)["bbox_px"] is not None
    assert all(p["bbox_px"] is not None for p in st["packets"])
