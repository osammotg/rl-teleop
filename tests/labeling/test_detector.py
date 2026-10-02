"""Unit tests for the scan-cam packet detector's pure logic (blob detection + the
closed-SKU matching that turns OCR tokens into a part id)."""
import threading
import time

import numpy as np

from robots_realtime.labeling.detector import (
    parse_part, match_sku, detect_blobs, PacketDetector, Detection,
    kit_from_detections, normalize_prefix, _clean_dets,
)


def test_normalize_prefix():
    assert normalize_prefix("UMM-10126-151") == "UNN-10126-151"   # UMM → UNN
    assert normalize_prefix("MDO-11065-001") == "MDDY-11065-001"  # MDO → MDDY
    assert normalize_prefix("UNN-10015-007") == "UNN-10015-007"   # already good


def test_clean_dets_dedupes_thresholds_normalizes():
    dets = [Detection([480, 157, 100, 56], "UMM-10126-000", 0.76),   # real packet
            Detection([479, 157, 102, 56], "UMM-10426-000", 0.65),   # same packet twin
            Detection([520, 680, 78, 32], "VSDA-02290-000", 0.36),   # noise < threshold
            Detection([800, 590, 100, 64], "UMH-10015-007", 0.98)]   # another packet
    out = _clean_dets(dets, min_conf=0.5)
    assert sorted(d.part for d in out) == ["UNN-10015-007", "UNN-10126-000"]


class _FakeReader:
    """Stub easyocr: readtext returns fixed (box, text, conf) triples."""
    def __init__(self, results):
        self._results = results

    def readtext(self, _img):
        return self._results


def _word(txt, x, y, w=40, h=20, conf=0.9):
    return ([[x, y], [x + w, y], [x + w, y + h], [x, y + h]], txt, conf)


def test_parse_part():
    assert parse_part("UNN-10126-151") == ("UNN", "10126", "151")
    assert parse_part("unn 10126 151") == ("UNN", "10126", "151")   # spaced/lowercase
    assert parse_part("MDDY-11065-001") == ("MDDY", "11065", "001")
    assert parse_part("garbage") is None


def test_match_sku_by_middle():
    known = ["UNN-10126-151", "UNN-10015-007", "DNN-15122-009"]
    # the 5-digit middle is the discriminator
    part, conf = match_sku([("UNN", 0.9), ("10126", 0.95), ("151", 0.5)], known)
    assert part == "UNN-10126-151" and conf == 0.95


def test_match_sku_needs_middle():
    known = ["UNN-10126-151"]
    # prefix alone (no middle) is not enough to trust the match
    assert match_sku([("UNN", 0.9)], known) == (None, 0.0)
    assert match_sku([("99999", 0.9)], known) == (None, 0.0)


def test_match_sku_disambiguates_duplicate_middle():
    # two SKUs share middle 10015; only the suffix separates them
    known = ["UNN-10015-007", "UNN-10015-231"]
    p007, _ = match_sku([("UNN", 0.9), ("10015", 0.95), ("007", 0.6)], known)
    p231, _ = match_sku([("UNN", 0.9), ("10015", 0.95), ("231", 0.6)], known)
    assert p007 == "UNN-10015-007"
    assert p231 == "UNN-10015-231"


def test_detect_blobs_finds_bright_rectangles():
    frame = np.zeros((480, 640, 3), np.uint8)          # dark mat
    frame[100:200, 120:320] = 230                      # one bright packet
    frame[300:380, 400:560] = 210                      # another
    boxes = detect_blobs(frame, min_area=1500)
    assert len(boxes) == 2
    # boxes are [x,y,w,h] and roughly cover the bright regions
    xs = sorted(b[0] for b in boxes)
    assert xs[0] < 200 and xs[1] > 350


def test_detect_blobs_ignores_full_frame():
    frame = np.full((480, 640, 3), 255, np.uint8)      # entirely bright
    assert detect_blobs(frame) == []                   # a near-full-frame blob is skipped


def test_bbox_for_prefers_highest_conf():
    det = PacketDetector(lambda: None, lambda: [])
    det._dets = [Detection([0, 0, 10, 10], "UNN-10015-007", 0.6),
                 Detection([5, 5, 20, 20], "UNN-10015-007", 0.9)]
    bbox, conf = det.bbox_for("UNN-10015-007")
    assert bbox == [5, 5, 20, 20] and conf == 0.9
    assert det.bbox_for("MISSING") == (None, 0.0)


def test_detect_once_reads_all_packets_generically():
    # two real packets + one order-number noise line (10 digits, no adjacent prefix)
    results = [
        _word("UNN", 100, 100), _word("10126", 145, 100), _word("151", 205, 100),
        _word("MDDY", 100, 300), _word("11065", 150, 300), _word("001", 210, 300),
        _word("3050765961", 100, 500),   # noise — not a 5-digit middle, ignored
    ]
    det = PacketDetector(lambda: None, lambda: None)
    det._reader = _FakeReader(results)
    dets = det.detect_once(np.zeros((600, 640, 3), np.uint8), None)
    assert sorted(d.part for d in dets) == ["MDDY-11065-001", "UNN-10126-151"]


def test_ocr_mode_box_is_original_heuristic():
    """Mode 'ocr' (the default, the tracker that worked well) expands the 5-digit box onto
    the bag: left/up 1×, right 2×, down 3×."""
    frame = np.zeros((480, 640, 3), np.uint8)
    det = PacketDetector(lambda: None, lambda: None)                 # mode='ocr' default
    det._reader = _FakeReader([_word("UNN", 200, 150),
                               _word("10126", 250, 150, w=40, h=20), _word("151", 330, 150)])
    dets = det.detect_once(frame, None)
    assert len(dets) == 1 and dets[0].part == "UNN-10126-151"
    # mid box [250,150,40,20] → [250-40, 150-20, 40+2*40, 20+3*20] = [210,130,120,80]
    assert tuple(dets[0].bbox) == (210, 130, 120, 80)


def test_contour_mode_segments_each_bag():
    """Mode 'contour' watersheds the bag silhouettes seeded by each digit centroid, so two
    touching-ish bags get two separate boxes, each inside its own bag (never merged)."""
    frame = np.zeros((480, 640, 3), np.uint8)
    frame[100:200, 100:250] = 90                          # bag A (brighter than mat, < label)
    frame[100:200, 300:450] = 90                          # bag B, dark gap between
    det = PacketDetector(lambda: None, lambda: None, mode="contour")
    det._reader = _FakeReader([
        _word("UNN", 120, 140), _word("10126", 150, 140), _word("151", 205, 140),   # in A
        _word("MDDY", 320, 140), _word("11065", 355, 140), _word("001", 415, 140),  # in B
    ])
    dets = det.detect_once(frame, None)
    assert sorted(d.part for d in dets) == ["MDDY-11065-001", "UNN-10126-151"]
    for d in dets:
        assert d.bbox[2] < 220                            # ~one bag wide (~150), not both (~350)


def test_margin_expands_every_box():
    frame = np.zeros((480, 640, 3), np.uint8)
    det = PacketDetector(lambda: None, lambda: None, margin=10)
    det._reader = _FakeReader([_word("UNN", 200, 150),
                               _word("10126", 250, 150, w=40, h=20), _word("151", 330, 150)])
    # base ocr box [210,130,120,80] grown by 10px/side → [200,120,140,100]
    assert tuple(det.detect_once(frame, None)[0].bbox) == (200, 120, 140, 100)


def test_set_mode_and_margin():
    det = PacketDetector(lambda: None, lambda: None)
    assert det.set_mode("contour") == {"ok": True, "mode": "contour"} and det._reset is True
    assert det.set_mode("bogus")["ok"] is False           # unknown mode rejected, keeps current
    assert det._mode == "contour"
    assert det.set_margin(25) == {"ok": True, "margin": 25} and det._margin == 25


def test_recalibrate_now_kicks_and_resets():
    det = PacketDetector(lambda: None, lambda: None)
    det._expected = 5
    det.recalibrate_now()
    assert det._reset is True and det._kick.is_set()


def test_autotune_picks_gamma_with_most_reads():
    """autotune sweeps exposure and keeps the gamma reading the most distinct parts; on a
    tie it prefers gamma closest to 1.0. Also sets self._gamma and triggers a re-scan."""
    frame = np.zeros((480, 640, 3), np.uint8)
    det = PacketDetector(lambda: frame, lambda: None)
    det._reader = _FakeReader([_word("UNN", 100, 100), _word("10126", 150, 100),
                               _word("151", 210, 100)])
    out = det.autotune(gammas=(0.5, 1.0, 1.7))           # fake OCR ignores gamma → all tie at 1
    assert out["ok"] and out["reads"] == 1 and out["gamma"] == 1.0
    assert det._gamma == 1.0 and det._reset is True       # applied + re-scan queued
    # the sweep is scored concurrently, so it must still report gammas in the ORDER given
    assert [s["gamma"] for s in out["sweep"]] == [0.5, 1.0, 1.7]
    assert out["secs"] >= 0                               # cockpit shows how long it took


def test_sweep_shares_an_injected_reader_instead_of_racing_it():
    """The autotune sweep parallelises by giving each worker its OWN easyocr reader. A
    caller-injected reader has no clonable equivalent, so the sweep must fall back to
    serial rather than hand one non-thread-safe object to eight threads."""
    det = PacketDetector(lambda: None, lambda: None)
    det._reader = _FakeReader([])
    assert det._can_clone_reader() is False
    det.autotune(gammas=(0.5, 1.0))
    assert det._sweep_readers == {}                       # no per-thread readers were made


def _running_detector(period_s=0.02, frame_source=None):
    frame = np.zeros((480, 640, 3), np.uint8)
    det = PacketDetector(frame_source or (lambda: frame), lambda: None, period_s=period_s)
    det._reader = _FakeReader([_word("UNN", 100, 100), _word("10126", 150, 100),
                               _word("151", 210, 100)])
    return det


def test_recalc_wait_blocks_until_the_rescan_actually_landed():
    """/recalc used to answer the instant it set the flag, so it reported the box count
    from BEFORE the rescan and the cockpit's ⟳ button looked like it did nothing.
    recalibrate_now(wait_s=...) must return only once a cycle has served the request."""
    det = _running_detector().start()
    try:
        assert det.recalibrate_now(wait_s=3.0) is True
        assert det._served_gen >= det._reset_gen           # this request was served
        assert len(det.current()[0]) == 1                  # post-rescan boxes are live
    finally:
        det.stop()


def test_recalc_wait_times_out_instead_of_hanging():
    """No loop running → nothing can serve the request. Bounded wait, honest False, and
    the reset stays queued for whenever the loop does start."""
    det = PacketDetector(lambda: None, lambda: None)
    t0 = time.monotonic()
    assert det.recalibrate_now(wait_s=0.2) is False
    assert 0.15 <= time.monotonic() - t0 < 2.0
    assert det._reset is True and det._kick.is_set()


def test_recalc_arriving_mid_cycle_waits_for_the_NEXT_cycle():
    """A cycle already in flight read its frame before the operator hit ⟳ and never dropped
    the held boxes, so it must not be credited with serving that request."""
    in_cycle, release = threading.Event(), threading.Event()

    def frames():
        in_cycle.set()
        release.wait(5)
        return np.zeros((480, 640, 3), np.uint8)

    det = _running_detector(frame_source=frames).start()
    done = []
    try:
        assert in_cycle.wait(3)                            # a cycle is now mid-flight
        t = threading.Thread(target=lambda: done.append(det.recalibrate_now(wait_s=3.0)))
        t.start()
        time.sleep(0.3)
        assert done == []                                  # in-flight cycle must NOT satisfy it
        release.set()                                      # let it finish; next cycle serves us
        t.join(5)
        assert done == [True]
    finally:
        release.set()
        det.stop()


def test_detect_once_needs_prefix_near_middle():
    # a 5-digit number with NO letter prefix nearby is not a part number
    det = PacketDetector(lambda: None, lambda: None)
    det._reader = _FakeReader([_word("48291", 100, 100), _word("77310", 400, 400)])
    assert det.detect_once(np.zeros((480, 640, 3), np.uint8), None) == []


def test_kit_from_detections_grouped_and_ordered():
    dets = [Detection([300, 50, 40, 20], "UNN-10015-007", 0.9),    # top row, right
            Detection([100, 50, 40, 20], "UNN-10126-151", 0.9),    # top row, left
            Detection([100, 300, 40, 20], "UNN-10015-007", 0.9)]   # lower row, dup part
    kit = kit_from_detections(dets)
    assert [k["part"] for k in kit] == ["UNN-10126-151", "UNN-10015-007", "UNN-10015-007"]
    assert kit[0]["comp"] == 1      # first distinct part
    assert kit[1]["comp"] == 2      # second distinct part
    assert kit[2]["comp"] == 2      # duplicate shares its part's compartment


def test_bbox_for_matches_by_middle_despite_suffix_flicker():
    det = PacketDetector(lambda: None, lambda: None)
    det._dets = [Detection([1, 2, 3, 4], "UNN-10015-000", 0.8)]   # suffix misread as 000
    bbox, _conf = det.bbox_for("UNN-10015-007")                   # kit wants -007
    assert bbox == [1, 2, 3, 4]                                   # matched on middle 10015


def test_detect_once_falls_back_when_ocr_unavailable():
    frame = np.zeros((480, 640, 3), np.uint8)
    frame[100:200, 120:320] = 230
    det = PacketDetector(lambda: None, lambda: ["UNN-10126-151"])
    det._ocr = lambda: (_ for _ in ()).throw(RuntimeError("no easyocr"))  # force failure
    dets = det.detect_once(frame, ["UNN-10126-151"])
    # graceful: still returns unidentified blob boxes (part=None) for order fallback
    assert dets and all(d.part is None for d in dets)
