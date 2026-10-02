"""A bag the scan cannot name must not stall the kit.

The rule this file pins down: scan everything, map it against the catalog, and
for anything that does not map — keep the row, flag it, let the operator step
past it. The failure it exists to prevent is silent: an unreadable bag used to
vanish from the pick list entirely, so the operator faced five bags on the mat
and one row on screen, with nothing on the interface admitting a bag was lost.
"""
from __future__ import annotations

import numpy as np
import pytest

from robots_realtime.labeling.detector import (
    Detection, PacketDetector, detect_blobs, kit_from_detections,
)
from robots_realtime.labeling.live import LiveLabeler
from robots_realtime.labeling.live_server import (
    KIT_CATALOG, scanned_kit, _state_with_detections,
)


class _Det:
    """Stand-in for PacketDetector.current()/status()."""

    def __init__(self, dets):
        self._dets = dets

    def current(self):
        return self._dets, [1280, 720]

    def status(self):
        return {"boxes": len(self._dets), "mode": "ocr"}


def _frame_with(*rects, size=(720, 1280)):
    """Black mat with bright rectangles — bright enough for detect_blobs (thresh 140)."""
    f = np.zeros((size[0], size[1], 3), np.uint8)
    for (x, y, w, h) in rects:
        f[y:y + h, x:x + w] = 255
    return f


# ── blob-level: telling a bag from scene furniture ──────────────────────────

def test_max_area_frac_rejects_scene_furniture_but_keeps_bags():
    # a calibration board left in shot measures ~0.34 of frame; a bag ~0.14
    board = (0, 0, 500, 620)          # 0.336 of a 1280x720 frame
    bag = (700, 100, 380, 330)        # 0.136
    f = _frame_with(board, bag)
    assert len(detect_blobs(f, max_area_frac=0.6)) == 2      # old cap keeps both
    kept = detect_blobs(f, max_area_frac=0.25)
    assert len(kept) == 1 and kept[0][0] >= 700              # only the bag survives


def test_default_max_area_frac_is_unchanged():
    """The 0.6 default is the pre-existing background guard — callers that never
    pass max_area_frac must behave exactly as before."""
    f = np.zeros((480, 640, 3), np.uint8)
    f[:, :] = 255                                            # full-frame → still rejected
    assert detect_blobs(f) == []


# ── detector: unreadable bags become unnamed detections ─────────────────────

def test_unnamed_blobs_emitted_when_ocr_reads_nothing():
    """The real-world case: OCR returns no part numbers at all. Previously this
    produced zero detections and a blank interface."""
    f = _frame_with((100, 100, 300, 260), (700, 300, 320, 240))
    det = PacketDetector(lambda: None, lambda: None)
    det._reader = type("R", (), {"readtext": staticmethod(lambda *a, **k: [])})()
    dets = det.detect_once(f, None)
    assert len(dets) == 2
    assert all(d.part is None and d.conf == 0.0 for d in dets)


def test_named_read_is_not_also_emitted_as_unnamed():
    """A bag that WAS read must produce exactly one detection, not one named plus
    one unnamed for the same silhouette."""
    f = _frame_with((200, 200, 300, 260))
    det = PacketDetector(lambda: None, lambda: None)
    named = Detection([220, 220, 200, 180], "UNN-10126-151", 0.9)
    out = named_plus = det._unnamed_blobs(f, [named])
    assert named_plus == [], f"silhouette double-counted: {out}"


# ── kit: the row survives, flagged ──────────────────────────────────────────

def test_scanned_kit_flags_unreadable_and_withholds_compartment():
    kit = scanned_kit([
        Detection([100, 100, 80, 60], "UNN-10015-007", 0.9),
        Detection([300, 100, 80, 60], None, 0.0),
    ])
    assert len(kit) == 2, "an unreadable bag must not be dropped from the kit"
    ok, unknown = kit[0], kit[1]
    assert ok["ident"] == "ok" and ok["comp"] == KIT_CATALOG["UNN-10015-007"]
    assert unknown["ident"] == "unknown"
    assert unknown["comp"] is None, "no catalog match must never yield a compartment"
    assert unknown["name"] == "Nicht erkannt"


def test_read_but_not_in_catalog_is_also_unknown():
    """'Map them against our database' — a number read cleanly but absent from the
    catalog is still not identified, and must not be routed anywhere."""
    kit = scanned_kit([Detection([10, 10, 50, 50], "ZZZ-99999-999", 0.95)])
    assert kit[0]["ident"] == "unknown" and kit[0]["comp"] is None
    # `read` carries the pipeline's reading, which has already had its prefix
    # snapped to the nearest real Bühler prefix by _clean_dets (ZZZ→UNN). That
    # normalisation is deliberate and predates this flag; what matters here is
    # that a snapped prefix never smuggles a part into the catalog — the middle
    # and suffix still do not match, so the row stays unidentified.
    assert kit[0]["read"] == "UNN-99999-999"


def test_unknown_entry_still_gets_a_box_to_point_at():
    dets = [Detection([500, 100, 80, 60], None, 0.0)]
    lab = LiveLabeler()
    lab.seed(scanned_kit(dets))
    st = _state_with_detections(lab, _Det(dets))
    assert st["packets"][0]["bbox_px"] == [500, 100, 80, 60]


def test_state_exposes_ident_through_the_whitelist():
    """LiveLabeler.state() whitelists packet fields; a flag the cockpit never
    receives is a flag that does not exist."""
    lab = LiveLabeler()
    lab.seed(scanned_kit([Detection([1, 1, 9, 9], None, 0.0)]))
    p = lab.state()["packets"][0]
    assert p["ident"] == "unknown"
    assert lab.state()["unknown"] == 1


# ── skip: the pipeline keeps moving ─────────────────────────────────────────

def test_skip_advances_without_counting_as_placed():
    dets = [Detection([100, 100, 80, 60], None, 0.0),
            Detection([300, 100, 80, 60], "UNN-16022-009", 0.9)]
    lab = LiveLabeler()
    lab.seed(scanned_kit(dets))
    lab.skip()
    st = lab.state()
    assert st["ti"] == 1, "skip must move the pointer on"
    assert st["done"] == 0, "a skipped bag was never placed"
    assert st["skipped"] == 1
    assert st["packets"][0]["status"] == "skipped"
    assert st["packets"][0]["skip_reason"] == "unidentified"


def test_skipped_kit_never_reads_as_complete():
    """The whole point of a distinct status: finishing a kit with an unread bag
    must not present as a clean full count."""
    dets = [Detection([100, 100, 80, 60], None, 0.0),
            Detection([300, 100, 80, 60], "UNN-16022-009", 0.9)]
    lab = LiveLabeler()
    lab.seed(scanned_kit(dets))
    lab.skip()
    lab.advance()
    st = lab.state()
    assert st["done"] < st["total"]
    assert [e["type"] for e in lab.events] == ["seed", "skipped", "place_confirmed"]


def test_skip_at_end_of_kit_is_a_noop():
    lab = LiveLabeler()
    lab.seed(scanned_kit([Detection([1, 1, 9, 9], None, 0.0)]))
    lab.skip()
    lab.skip()                       # already past the end
    assert lab.state()["ti"] == 1 and lab.state()["skipped"] == 1


def test_skipped_row_does_not_steal_the_current_packets_box():
    """A skipped bag is still on the mat, so it keeps a box — but it sits BEFORE
    the pointer, and in list order it would claim the scarce detection first and
    blank the box on the packet being picked right now."""
    dets = [Detection([100, 100, 80, 60], None, 0.0),
            Detection([300, 100, 80, 60], None, 0.0)]
    lab = LiveLabeler()
    lab.seed(scanned_kit(dets))
    lab.skip()                                   # row 0 skipped, pointer now on row 1
    st = _state_with_detections(lab, _Det(dets))
    current = st["packets"][st["ti"]]
    assert current["bbox_px"] is not None, "the packet being picked must keep its box"


def test_placed_packet_claims_no_box():
    """Unchanged pre-existing rule, re-pinned because skip() touches the same loop."""
    dets = [Detection([100, 100, 80, 60], "UNN-10015-007", 0.9)]
    lab = LiveLabeler()
    lab.seed(scanned_kit(dets))
    lab.advance()
    st = _state_with_detections(lab, _Det(dets))
    assert st["packets"][0]["bbox_px"] is None
