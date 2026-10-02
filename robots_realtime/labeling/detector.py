"""Live packet detector for the kitting cockpit "pick this next" box.

The scan camera (D435i, top-down over the box) sees the loose Bühler packets. This
finds each packet's bounding box and reads its part id, so the cockpit can highlight
the exact packet the current kit step wants.

Pipeline (per frame, ~0.3-0.5 Hz — packets are static during a session):

    scan RGB frame
      │
      ▼  detect_blobs()  — bright packets on the dark mat (threshold + contours)
    [bbox_px, ...]
      │
      ▼  read_part_id()  — easyocr the crop, pull digit tokens, match the CLOSED
      │                    SKU set by the discriminative 5-digit middle field
    [{part, bbox, conf}, ...]

Barcode decoding was tried first and does not resolve at this camera range; OCR of
the large printed part number does. easyocr is optional — if it is missing or the GPU
kernel is unavailable, detections still carry bboxes with part=None and the cockpit
falls back to kit-order highlighting.
"""
from __future__ import annotations

import re
import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import numpy as np

# A Bühler part id is PREFIX-MIDDLE-SUFFIX, e.g. UNN-10126-151. The MIDDLE 5 digits
# are the discriminative field OCR reads most reliably (conf ~0.9 on the real cam).
_PART_RE = re.compile(r"([A-Z]{2,4})[-\s]?(\d{5})[-\s]?(\d{3})")


def parse_part(part_id: str) -> tuple[str, str, str] | None:
    m = _PART_RE.search((part_id or "").upper().replace(" ", ""))
    return (m.group(1), m.group(2), m.group(3)) if m else None


# Real Bühler part-id prefixes on the packets. OCR frequently mangles these (UNN → UMM /
# UMH / UMN / UNM; MDDY → MDO / MD / MDOT) so we snap each read to the nearest real prefix.
# NOTE: "DNN" is NOT a real prefix (it came from an old synthetic DEFAULT_KIT). Keeping it
# here made real UNN reads snap to the phantom DNN — the exact UNN→DNN misread we saw live.
KNOWN_PREFIXES = ["UNN", "MDDY"]


def _edit_dist(a: str, b: str) -> int:
    dp = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        prev, dp[0] = dp[0], i
        for j, cb in enumerate(b, 1):
            prev, dp[j] = dp[j], min(dp[j] + 1, dp[j - 1] + 1, prev + (ca != cb))
    return dp[-1]


def normalize_prefix(part: str) -> str:
    p = parse_part(part)
    if not p:
        return part
    pre, mid, suf = p
    best = min(KNOWN_PREFIXES, key=lambda k: _edit_dist(pre, k))
    return f"{best}-{mid}-{suf}"


def snap_to_catalog(part: str, known) -> str:
    """Snap a raw part read to the nearest SKU in the closed catalog ``known``.

    The 5-digit MIDDLE is the field OCR reads most reliably, so we first restrict
    to catalog SKUs that share the read middle, then pick the smallest edit
    distance on the full id — that repairs prefix/suffix slips (UNN→DNN, -009→-000)
    without inventing a SKU. If the middle itself was misread, we fall back to the
    nearest SKU over the whole catalog. No catalog → return the read unchanged."""
    if not known:
        return part
    p = parse_part(part)
    if not p:
        return part
    _pre, mid, _suf = p
    same_middle = [k for k in known if (parse_part(k) or (None, None, None))[1] == mid]
    pool = same_middle or list(known)
    return min(pool, key=lambda k: _edit_dist(part, k))


@dataclass
class Detection:
    bbox: list[int]        # [x, y, w, h] in scan-frame pixels
    part: str | None       # matched kit part id, or None if unidentified
    conf: float            # OCR/match confidence 0..1


def detect_blobs(frame: np.ndarray, min_area: int = 1500,
                 thresh: int = 140, max_area_frac: float = 0.6) -> list[list[int]]:
    """Bright packets on the dark mat → list of [x,y,w,h] pixel bboxes.

    ``max_area_frac`` caps a blob's share of the frame. The default 0.6 only
    rejects a near-full-frame background; callers that need to tell a BAG from
    scene furniture (a calibration board left in shot measures ~0.34 of frame,
    a bag ~0.14) should pass something tighter.
    """
    import cv2
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY) if frame.ndim == 3 and frame.shape[2] == 3 else frame
    _, th = cv2.threshold(gray, thresh, 255, cv2.THRESH_BINARY)
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = []
    H, W = gray.shape[:2]
    for c in cnts:
        if cv2.contourArea(c) < min_area:
            continue
        x, y, w, h = cv2.boundingRect(c)
        if w * h > max_area_frac * W * H:      # scene furniture / background, not a packet
            continue
        out.append([int(x), int(y), int(w), int(h)])
    return out


def _center_in(box: list[int], other: list[int]) -> bool:
    """Is box's centre inside other?"""
    cx, cy = box[0] + box[2] / 2, box[1] + box[3] / 2
    return (other[0] <= cx <= other[0] + other[2]
            and other[1] <= cy <= other[1] + other[3])


def enhance_for_ocr(frame, gamma: float = 1.0):
    """Make the printed part numbers more readable for OCR: gray-world white balance
    (kills the D435i green cast), a gamma (software 'exposure' — <1 brighter, >1 darker),
    and CLAHE local contrast so the black-on-white text pops. Does NOT touch the camera
    or the recorded frames — this is applied only to the OCR input."""
    import cv2
    img = np.asarray(frame)
    if img.ndim != 3 or img.shape[2] != 3:
        return img
    f = img.astype(np.float32)
    means = f.reshape(-1, 3).mean(0) + 1e-6           # gray-world WB
    f = np.clip(f * (means.mean() / means), 0, 255)
    if gamma != 1.0:
        f = np.clip(((f / 255.0) ** gamma) * 255.0, 0, 255)
    out = f.astype(np.uint8)
    lab = cv2.cvtColor(out, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8)).apply(l)
    return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2RGB)


def match_sku(tokens: list[tuple[str, float]], known: list[str]) -> tuple[str | None, float]:
    """Match OCR tokens against the closed SKU set.

    Score each known part id: +2 if its 5-digit middle appears in the tokens,
    +1 for the 3-digit suffix, +0.5 for the prefix. The middle field is the
    reliable discriminator; suffix/prefix disambiguate duplicate middles.
    Returns (best_part_or_None, confidence).
    """
    digits5 = {t for (t, _) in tokens if re.fullmatch(r"\d{5}", t)}
    digits3 = {t for (t, _) in tokens if re.fullmatch(r"\d{3}", t)}
    prefixes = {t for (t, _) in tokens if re.fullmatch(r"[A-Z]{2,4}", t)}
    conf5 = {t: c for (t, c) in tokens if re.fullmatch(r"\d{5}", t)}
    best, best_score, best_conf = None, 0.0, 0.0
    for part in known:
        p = parse_part(part)
        if not p:
            continue
        pre, mid, suf = p
        score = 0.0
        if mid in digits5:
            score += 2.0
        if suf in digits3:
            score += 1.0
        if pre in prefixes:
            score += 0.5
        if score > best_score:
            best, best_score = part, score
            best_conf = conf5.get(mid, 0.5)
    # require at least the middle field to trust the match
    if best is None or best_score < 2.0:
        return None, 0.0
    return best, round(float(best_conf), 2)


def _clean_dets(dets: list["Detection"], min_conf: float,
                merge_px: int = 75) -> list["Detection"]:
    """Turn raw OCR reads into one clean detection per physical packet:
    drop low-confidence noise (field labels, order numbers), keep the
    highest-confidence read per spatial cluster (dedupe prefix-misread twins),
    and snap the prefix to the nearest real Bühler prefix."""
    good = sorted((d for d in dets if d.conf >= min_conf), key=lambda d: -d.conf)
    kept: list[Detection] = []
    for d in good:
        cx, cy = d.bbox[0] + d.bbox[2] / 2, d.bbox[1] + d.bbox[3] / 2
        if any(abs(cx - (k.bbox[0] + k.bbox[2] / 2)) < merge_px and
               abs(cy - (k.bbox[1] + k.bbox[3] / 2)) < merge_px for k in kept):
            continue                                   # same packet, lower-conf twin
        kept.append(Detection(d.bbox, normalize_prefix(d.part) if d.part else None, d.conf))
    return kept


def _key(ctr) -> tuple[int, int]:
    """Hashable pixel key for a centroid (used to map a digit centroid → its seed label)."""
    return (int(round(ctr[0])), int(round(ctr[1])))


def _region_bbox_by_label(labels: np.ndarray, k: int | None) -> list[int] | None:
    """Bounding box [x,y,w,h] of all pixels with value k in a watershed/label image."""
    if k is None:
        return None
    ys, xs = np.where(labels == k)
    if len(xs) == 0:
        return None
    x0, y0 = int(xs.min()), int(ys.min())
    return [x0, y0, int(xs.max()) - x0, int(ys.max()) - y0]


def _smallest_mask_containing(masks, ctr) -> list[int] | None:
    """Among SAM (x,y,w,h,mask) instances covering the centroid, the smallest one's box —
    smallest = the individual bag, not a big multi-bag blob."""
    cx, cy = int(ctr[0]), int(ctr[1])
    best, best_area = None, None
    for (x, y, w, h, m) in masks:
        if 0 <= cy < m.shape[0] and 0 <= cx < m.shape[1] and m[cy, cx] > 0.5:
            area = w * h
            if best_area is None or area < best_area:
                best, best_area = [x, y, w, h], area
    return best


class PacketDetector:
    """Detects packets on the scan camera and reads their part ids.

    Thread-safe: a background thread refreshes ``detections`` from the latest
    scan frame; the HTTP handler reads ``current()``. ``frame_source`` is a
    callable returning the latest scan RGB frame (H,W,3) uint8 or None.
    """

    # Concurrency for the autotune gamma sweep. Each worker holds its own easyocr
    # Reader (~290MB RSS), created on first use and kept for later sweeps, so this
    # is a speed-vs-memory dial, not a free one.
    SWEEP_WORKERS = 4

    def __init__(self, frame_source, known_source, period_s: float = 2.0,
                 gpu: bool = False, hold_seconds: float = 5.0, min_conf: float = 0.5,
                 mode: str = "ocr", margin: int = 0):
        self._frame_source = frame_source          # () -> np.ndarray | None
        self._known_source = known_source          # () -> list[str] | None
        self._period = period_s
        # BOX GEOMETRY is pluggable so we can A/B/C the tracker live from the cockpit. The
        # OCR read (identity + where the digits are) feeds all modes; only how the BOX is
        # derived differs. Switch with /trackmode:
        #   "ocr"     — expand the part-number's OCR text box onto the bag (the original,
        #               zero-dep heuristic that worked well).
        #   "contour" — classical CV: threshold the bag silhouette on the dark mat and
        #               watershed-split touching bags, seeded by each read's digit centroid.
        #   "sam"     — FastSAM point-prompted at each digit centroid → precise bag mask.
        # `margin` grows every mode's final box by N px/side (the cockpit wider/narrower).
        self._mode = mode if mode in ("ocr", "contour", "sam") else "ocr"
        self._margin = margin
        self._sam_model = None
        self._bag_thresh = 60          # brightness that separates bag (translucent) from black mat
        self._gamma = 1.0              # base OCR exposure; /autotune picks the best for the light
        # Hold a box this long (WALL-CLOCK) after it was last seen, then drop it. Kept in
        # seconds, NOT cycles, so raising period_s (to ease CPU) never silently lingers a
        # stale box longer — the two knobs are decoupled. A box older than this over a
        # scene that has changed (packet picked/occluded) is the "random box" bug.
        self._hold_seconds = hold_seconds
        self.min_conf = min_conf                   # drop OCR reads below this confidence
        self._lock = threading.Lock()
        self._dets: list[Detection] = []
        self._wh: list[int] = [0, 0]
        self._reader = None
        self._gpu = gpu
        self._stop = threading.Event()
        self._kick = threading.Event()   # wake the loop early (manual recalc)
        self._reset = False              # drop held boxes + reset expected on next cycle
        # Recalc completion tracking. The cockpit's ⟳ Recalculate used to return
        # status() the instant it set the flag above — i.e. the box count from BEFORE
        # the rescan — so the panel showed a stale number and the button looked dead.
        # recalibrate_now(wait_s=...) now blocks until the loop has actually finished a
        # cycle that consumed this request, so the caller reports the real new count.
        self._reset_gen = 0              # bumped per recalc request
        self._served_gen = 0            # highest request the loop has fully re-scanned
        self._cycle_cv = threading.Condition()
        self._sweep_readers: dict[int, object] = {}   # per-thread OCR readers for autotune
        self._expected = 0             # high-water mark of packets seen (recalibration target)
        self._recalibrate = True       # sweep exposure/gamma when we come up short
        self._vote_window = 15         # per-packet reads kept for the temporal identity vote

    def _ocr(self):
        if self._reader is None:
            import easyocr
            # Cap CPU threads so the OCR doesn't grab all cores and starve the
            # live-camera MJPEG streaming that shares this process (was freezing the
            # cockpit). Leaves cores free for the HTTP streaming threads.
            try:
                import cv2
                import torch
                torch.set_num_threads(4)
                cv2.setNumThreads(2)
            except Exception:
                pass
            self._reader = easyocr.Reader(["en"], gpu=self._gpu, verbose=False)
        return self._reader

    def detect_once(self, frame: np.ndarray, known=None,
                    gamma: float = 1.0, reader=None) -> list[Detection]:
        """Read EVERY packet on the scan frame: enhance for OCR (white-balance +
        gamma + contrast), OCR, group each printed part number (prefix + 5-digit
        middle + 3-digit suffix) into one box, then dedupe/clean. ``gamma`` is the
        software exposure used by the recalibration sweep. ``known`` (optional) snaps
        a read to the closest catalog SKU. ``reader`` overrides the shared easyocr
        reader — the parallel autotune sweep passes one reader per worker thread,
        since easyocr Readers are not thread-safe."""
        try:
            res = (reader or self._ocr()).readtext(enhance_for_ocr(frame, gamma))
        except Exception:
            # OCR unavailable → fall back to unidentified blobs (cockpit uses order)
            return [Detection(b, None, 0.0) for b in detect_blobs(frame)]

        # tokens with pixel boxes + centers
        toks = []   # (text_upper, conf, [x,y,w,h], (cx,cy))
        for (box, txt, conf) in res:
            xs = [p[0] for p in box]; ys = [p[1] for p in box]
            bb = [int(min(xs)), int(min(ys)), int(max(xs) - min(xs)), int(max(ys) - min(ys))]
            cx, cy = bb[0] + bb[2] / 2, bb[1] + bb[3] / 2
            for t in re.split(r"[^A-Za-z0-9]+", txt.upper()):
                if t:
                    toks.append((t, float(conf), bb, (cx, cy)))

        mids = [(t, c, bb, ctr) for (t, c, bb, ctr) in toks if re.fullmatch(r"\d{5}", t)]
        pres = [(t, c, bb, ctr) for (t, c, bb, ctr) in toks if re.fullmatch(r"[A-Z]{2,4}", t)]
        sufs = [(t, c, bb, ctr) for (t, c, bb, ctr) in toks if re.fullmatch(r"\d{3}", t)]
        H, W = frame.shape[:2]

        def nearest(cands, ctr, maxd):
            best, bd = None, maxd
            for c in cands:
                d = abs(c[3][0] - ctr[0]) + 2 * abs(c[3][1] - ctr[1])   # weight y (same line)
                if d < bd:
                    bd, best = d, c
            return best

        # First pass: which reads are real part numbers (5-digit middle WITH a letter prefix
        # next to it, which rejects order-number noise). Collect their digit centroids so the
        # region-based trackers (contour/sam) can segment one bag per read in a single pass.
        hits = []   # (mid, mconf, mbb, mctr, suf)
        for (mid, mconf, mbb, mctr) in mids:
            pre = nearest(pres, mctr, maxd=max(140, 3 * mbb[2]))
            if pre is None:
                continue
            suf = nearest(sufs, mctr, maxd=max(140, 3 * mbb[2]))
            hits.append((mid, mconf, mbb, mctr, pre, suf))

        centroids = [h[3] for h in hits]
        boxer = self._make_boxer(frame, centroids)   # mode-specific box(dbb, dctr) -> [x,y,w,h]

        dets: list[Detection] = []
        for (mid, mconf, mbb, mctr, pre, suf) in hits:
            # Keep the RAW read here (prefix normalized later in _clean_dets). Catalog
            # snapping is deferred to the temporal vote (_vote): if we snapped per-frame,
            # a garbage suffix on one frame would collapse two same-middle SKUs (007/231)
            # before the vote ever saw the frames that read the suffix correctly.
            part = f"{pre[0]}-{mid}-{(suf[0] if suf else '000')}"
            box = boxer(mbb, mctr)
            dets.append(Detection([int(v) for v in box], part, round(mconf, 2)))
        named = _clean_dets(dets, self.min_conf)
        return named + self._unnamed_blobs(frame, named)

    # A blob larger than this share of the frame is scene furniture (a calibration
    # board, the tray, a shadow), not a bag. Measured: board ~0.34, bag ~0.14.
    MAX_BAG_AREA_FRAC = 0.25

    def _unnamed_blobs(self, frame, named: list["Detection"]) -> list["Detection"]:
        """Bags the silhouette pass can see but OCR could not name.

        Without this an unreadable bag simply does not exist: the kit is built from
        detections, so it is silently absent from the pick list and the operator is
        left with five bags in front of them and one row on screen. Emitting it with
        part=None keeps it in the pipeline as an explicit "not identified" entry the
        cockpit can flag and the operator can skip — strictly better than a missing
        row nobody can act on, and it never invents an identity.
        """
        try:
            blobs = detect_blobs(frame, max_area_frac=self.MAX_BAG_AREA_FRAC)
        except Exception:
            return []
        out = []
        for b in blobs:
            b = [int(v) for v in b]
            # A blob that already carries a named read IS that packet. Test both
            # directions: the OCR heuristic box can be smaller than the silhouette
            # or, with `margin` cranked up, larger than it.
            if any(_center_in(b, d.bbox) or _center_in(d.bbox, b) for d in named):
                continue
            if any(_center_in(b, o.bbox) for o in out):     # one entry per silhouette
                continue
            out.append(Detection(b, None, 0.0))
        return out

    # ---- pluggable box trackers (switch live via /trackmode) ----------------------------

    def _make_boxer(self, frame, centroids):
        """Return a per-read box(dbb, dctr) -> [x,y,w,h] for the current mode. Region modes
        (contour/sam) segment the whole frame ONCE here, then assign the region under each
        digit centroid; any read with no region falls back to the OCR heuristic box."""
        H, W = frame.shape[:2]

        def _clamp_expand(box):                             # apply margin, clamp to frame
            x, y, w, h = box
            m = self._margin
            x = max(0, x - m); y = max(0, y - m)
            w = min(W - x, w + 2 * m); h = min(H - y, h + 2 * m)
            return [int(x), int(y), int(w), int(h)]

        def _ocr_box(dbb, _dctr):
            """The original heuristic: expand the 5-digit box onto the bag (left/up 1x,
            right 2x, down 3x — the number sits top-left on the label)."""
            mx, my = dbb[2], dbb[3]
            box = [max(0, dbb[0] - mx), max(0, dbb[1] - my), dbb[2] + 2 * mx, dbb[3] + 3 * my]
            box[2] = min(box[2], W - box[0]); box[3] = min(box[3], H - box[1])
            return _clamp_expand(box)

        if self._mode == "ocr" or not centroids:
            return _ocr_box

        try:
            if self._mode == "contour":
                labels, idx_of = self._segment_contour(frame, centroids)

                def _contour_box(dbb, dctr):
                    box = _region_bbox_by_label(labels, idx_of.get(_key(dctr)))
                    return _clamp_expand(box) if box else _ocr_box(dbb, dctr)
                return _contour_box

            if self._mode == "sam":
                masks = self._segment_sam(frame)            # list of (x,y,w,h,mask)

                def _sam_box(dbb, dctr):
                    box = _smallest_mask_containing(masks, dctr)
                    return _clamp_expand(box) if box else _ocr_box(dbb, dctr)
                return _sam_box
        except Exception as e:                              # any tracker failure → safe default
            print(f"[detector] mode '{self._mode}' failed ({type(e).__name__}: {e}); using ocr")
        return _ocr_box

    def _segment_contour(self, frame, centroids):
        """Watershed the bag silhouettes on the dark mat, seeded by one marker per digit
        centroid, so touching bags are split at their seam. Returns (labels, {key→seed})."""
        import cv2
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY) if frame.ndim == 3 else frame
        _, bag = cv2.threshold(gray, self._bag_thresh, 255, cv2.THRESH_BINARY)
        bag = cv2.morphologyEx(bag, cv2.MORPH_CLOSE, np.ones((11, 11), np.uint8))
        markers = np.zeros(gray.shape, np.int32)
        idx_of = {}
        for k, (cx, cy) in enumerate(centroids, start=2):   # 1 is reserved for background
            cv2.circle(markers, (int(cx), int(cy)), 6, k, -1)
            idx_of[_key((cx, cy))] = k
        markers[bag == 0] = 1                               # mat is background
        cv2.watershed(cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR), markers)
        return markers, idx_of

    def _sam(self):
        if self._sam_model is None:
            from ultralytics import FastSAM
            self._sam_model = FastSAM("FastSAM-s.pt")       # ~23MB, auto-downloads once
        return self._sam_model

    def _segment_sam(self, frame):
        """FastSAM 'segment everything', returned as (x,y,w,h,mask) per instance. We assign
        the smallest mask under each digit centroid, so each bag gets its own tight box."""
        # Force CPU: the RTX 5090 (sm_120) has no kernels in this torch build, so 'cuda'
        # crashes/falls back anyway. ~1s/frame on CPU is fine at the scan's slow rate.
        res = self._sam()(frame, device="cpu", retina_masks=True,
                          imgsz=1024, conf=0.4, iou=0.9, verbose=False)
        out = []
        masks = res[0].masks if res else None
        if masks is not None:
            for m in masks.data.cpu().numpy():
                ys, xs = np.where(m > 0.5)
                if len(xs):
                    out.append((int(xs.min()), int(ys.min()),
                                int(xs.max() - xs.min()), int(ys.max() - ys.min()), m))
        return out

    def _vote(self, entry: dict, known) -> "Detection":
        """Collapse a packet's recent reads into ONE identity by weighted majority vote.

        Per-frame OCR jitters the prefix/suffix, so a single frame is untrustworthy; a
        vote over the last _vote_window reads (weighted by that frame's OCR confidence)
        is stable. The winning raw read is then snapped to the closed catalog. The
        emitted confidence is the vote AGREEMENT (winner weight / total) — a real
        trust signal, unlike the saturated per-frame OCR score, so a split/uncertain
        packet gets a low conf and is dropped by the min_conf gate (abstain, not
        mis-route)."""
        tally: dict[str, float] = {}
        for part, conf in entry["reads"]:
            if part:
                tally[part] = tally.get(part, 0.0) + max(float(conf), 0.05)
        if not tally:
            return Detection(list(entry["bbox"]), None, 0.0)
        total = sum(tally.values())
        winner = max(tally, key=lambda k: tally[k])
        agree = tally[winner] / total
        return Detection(list(entry["bbox"]), snap_to_catalog(winner, known), round(agree, 2))

    def _loop(self):
        # Spatial tracker: one entry per PHYSICAL packet (keyed by box POSITION, not the
        # part text — OCR jitters the prefix/suffix across passes, so text keys would
        # accumulate duplicates of the same packet). Each position keeps a WINDOW of its
        # recent reads and reports the vote winner (see _vote); a packet holds its box for
        # _hold_seconds after it was last seen (rides out arm occlusion) then drops once
        # it's been gone that long (picked / removed).
        held: list[dict] = []          # {"bbox": [x,y,w,h], "reads": deque, "seen": monotonic}
        while not self._stop.is_set():
            # Snapshot the recalc request this cycle will satisfy BEFORE doing the work,
            # so a request arriving mid-cycle is served by the NEXT cycle, not credited to
            # this one (which already read its frame and never dropped the held boxes).
            pending = self._reset_gen
            if self._reset:                        # manual /recalc: forget everything, re-scan clean
                held = []
                self._expected = 0
                self._reset = False
            frame = self._frame_source()
            known = self._known_source() or None
            if frame is not None:
                try:
                    frame = np.asarray(frame)
                    fresh = self.detect_once(frame, known, gamma=self._gamma)
                    # Auto-recalibrate: if we found fewer packets than we've seen before,
                    # a packet is likely washed out / too dark — sweep software exposure
                    # (gamma) and merge whatever extra reads that surfaces.
                    if self._recalibrate and len(fresh) < self._expected:
                        pool = list(fresh)
                        for g in (0.55, 1.7):
                            pool += self.detect_once(frame, known, gamma=g)
                        fresh = _clean_dets(pool, self.min_conf)
                    self._expected = max(self._expected, len(fresh))
                    used = [False] * len(fresh)

                    def _near(bbox):
                        cx, cy = bbox[0] + bbox[2] / 2, bbox[1] + bbox[3] / 2
                        for i, f in enumerate(fresh):
                            if used[i]:
                                continue
                            fx, fy = f.bbox[0] + f.bbox[2] / 2, f.bbox[1] + f.bbox[3] / 2
                            if abs(cx - fx) < 75 and abs(cy - fy) < 75:
                                return i
                        return -1

                    now = time.monotonic()
                    new_held: list[dict] = []
                    for entry in held:
                        i = _near(entry["bbox"])
                        if i >= 0:
                            used[i] = True
                            f = fresh[i]
                            entry["bbox"] = f.bbox                  # latest position
                            entry["reads"].append((f.part, f.conf))  # one more vote for this packet
                            entry["seen"] = now
                            new_held.append(entry)
                        elif now - entry["seen"] < self._hold_seconds:
                            new_held.append(entry)                 # hold last-good until it ages out
                    for i, f in enumerate(fresh):
                        if not used[i]:                            # newly seen packet
                            reads = deque(maxlen=self._vote_window)
                            reads.append((f.part, f.conf))
                            new_held.append({"bbox": f.bbox, "reads": reads, "seen": now})
                    held = new_held
                    voted = [self._vote(e, known) for e in held]
                    with self._lock:
                        # final box-dedupe guard: the tracker can hold two co-located
                        # entries if a cycle briefly saw a packet twice; collapse them.
                        self._dets = _clean_dets(voted, self.min_conf)
                        self._wh = [int(frame.shape[1]), int(frame.shape[0])]
                except Exception:
                    pass
            # publish: this cycle's boxes are live, so anyone blocked in
            # recalibrate_now(wait_s=...) can now read a post-rescan count
            with self._cycle_cv:
                self._served_gen = max(self._served_gen, pending)
                self._cycle_cv.notify_all()
            # sleep until the next cycle OR until a manual recalc kicks us awake
            if self._kick.wait(self._period):
                self._kick.clear()

    def start(self) -> "PacketDetector":
        threading.Thread(target=self._loop, daemon=True).start()
        return self

    def stop(self):
        self._stop.set()

    # ---- runtime controls (driven by the cockpit debug panel) --------------------------

    def recalibrate_now(self, wait_s: float = 0.0) -> bool:
        """Drop all held boxes and re-scan from scratch on the next (immediate) cycle.
        Use after moving/adding bags so stale or merged boxes clear at once.

        With ``wait_s`` > 0, block until the loop has finished a cycle that consumed
        THIS request (so ``status()`` afterwards reports the rescanned boxes, not the
        stale ones) and return whether that happened inside the budget. The default 0
        keeps mode/margin switches instant — they only need the rescan queued."""
        with self._cycle_cv:
            self._reset_gen += 1
            target = self._reset_gen
        self._reset = True
        self._kick.set()
        if wait_s <= 0:
            return False
        deadline = time.monotonic() + wait_s
        with self._cycle_cv:
            while self._served_gen < target:
                left = deadline - time.monotonic()
                if left <= 0:
                    return False                   # still scanning; caller reports "pending"
                self._cycle_cv.wait(left)
        return True

    def set_mode(self, mode: str) -> dict:
        """Switch the box tracker: 'ocr' | 'contour' | 'sam'. Applies on the next cycle.
        For 'sam', warm the model now so the first cycle isn't a multi-second stall (and so
        we can report if the weights/deps are missing)."""
        if mode not in ("ocr", "contour", "sam"):
            return {"ok": False, "reason": f"unknown mode {mode!r}", "mode": self._mode}
        warm = None
        if mode == "sam":
            try:
                self._sam()
            except Exception as e:
                return {"ok": False, "reason": f"sam unavailable: {type(e).__name__}: {e}",
                        "mode": self._mode}
        self._mode = mode
        self.recalibrate_now()
        return {"ok": True, "mode": self._mode, **({"warm": warm} if warm else {})}

    def set_margin(self, margin: int) -> dict:
        """Grow/shrink every mode's box by N px per side (cockpit wider/narrower)."""
        self._margin = int(margin)
        self._kick.set()
        return {"ok": True, "margin": self._margin}

    def autotune(self, gammas=(0.4, 0.55, 0.7, 0.85, 1.0, 1.3, 1.7, 2.2)) -> dict:
        """Find the OCR exposure (gamma) best suited to the CURRENT lighting: sweep gamma,
        run a detection pass at each, and keep the one that reads the MOST distinct valid
        part numbers (ties → gamma closest to 1.0). Sets self._gamma and forces a re-scan.
        Returns the chosen gamma, its read count, and the full sweep for the debug panel."""
        frame = self._frame_source()
        if frame is None:
            return {"ok": False, "reason": "no scan frame"}
        frame = np.asarray(frame)
        known = self._known_source() or None
        t0 = time.monotonic()
        # The passes are independent, so run them concurrently. Measured on the rig
        # (32 cores, 1280x720, 8 gammas): 15.7s serial -> 6.5s at 8 workers x 2 torch
        # threads, with an IDENTICAL sweep. Downscaling the frame was tried first and is
        # not an option — at 0.75 scale the part numbers stop reading (4 reads -> 1) and
        # below that they vanish entirely, so the sweep would tune on noise.
        scored = self._sweep(frame, known, gammas)
        best_g, best_n = self._gamma, -1
        for g, n in scored:
            if n > best_n or (n == best_n and abs(g - 1.0) < abs(best_g - 1.0)):
                best_g, best_n = g, n
        self._gamma = best_g
        self.recalibrate_now()
        return {"ok": True, "gamma": round(best_g, 2), "reads": best_n,
                "sweep": [{"gamma": round(g, 2), "reads": n} for g, n in scored],
                "secs": round(time.monotonic() - t0, 1)}

    def _can_clone_reader(self) -> bool:
        """Can we mint INDEPENDENT OCR readers for the sweep workers? Only if the primary
        reader is a real easyocr one. A caller-injected reader (the tests' fake) has no
        clonable equivalent, so the sweep shares it and runs serially instead of racing
        one non-thread-safe object across eight threads."""
        try:
            import easyocr
        except Exception:
            return False
        primary = self._reader
        return primary is None or isinstance(primary, easyocr.Reader)

    def _sweep(self, frame, known, gammas) -> list[tuple[float, int]]:
        """Score every gamma (distinct valid part numbers read) in the given order. Each
        worker thread keeps its own easyocr Reader — they are not thread-safe, and a shared
        one would serialise the sweep anyway. Readers are cached across calls so only the
        first autotune pays the model-load cost."""
        parallel = self._can_clone_reader()

        def score(g):
            r = self._reader
            if parallel:
                tid = threading.get_ident()
                r = self._sweep_readers.get(tid)
                if r is None:
                    import easyocr
                    r = easyocr.Reader(["en"], gpu=self._gpu, verbose=False)
                    self._sweep_readers[tid] = r
            dets = self.detect_once(frame, known, gamma=g, reader=r)
            return (g, len({d.part for d in dets if d.part}))

        workers = max(1, min(len(gammas), self.SWEEP_WORKERS)) if parallel else 1
        if workers == 1:
            return [score(g) for g in gammas]
        # Trim torch's intra-op threads for the duration: the detect loop runs at 4 so it
        # does not starve the camera streaming in this process, but 8 concurrent passes x
        # 4 threads oversubscribes 32 cores and ran SLOWER (7.1s) than 8 x 2 (6.5s).
        prev = None
        try:
            import torch
            prev = torch.get_num_threads()
            torch.set_num_threads(2)
        except Exception:
            pass
        try:
            with ThreadPoolExecutor(max_workers=workers,
                                    thread_name_prefix="autotune") as ex:
                return list(ex.map(score, gammas))
        finally:
            if prev is not None:
                try:
                    torch.set_num_threads(prev)
                except Exception:
                    pass

    def status(self) -> dict:
        with self._lock:
            n = len(self._dets)
        return {"gamma": round(self._gamma, 2), "boxes": n,
                "mode": self._mode, "margin": self._margin}

    def current(self) -> tuple[list[Detection], list[int]]:
        with self._lock:
            return list(self._dets), list(self._wh)

    def bbox_for(self, part: str) -> tuple[list[int] | None, float]:
        """Best detected bbox for ``part``, matched on the reliable 5-digit MIDDLE
        (OCR reads the middle far better than the suffix). Prefers an exact suffix
        match to disambiguate duplicate middles, else highest confidence."""
        p = parse_part(part)
        if not p:
            return None, 0.0
        _pre, mid, suf = p
        with self._lock:
            cands = [d for d in self._dets
                     if (parse_part(d.part) or (None, None, None))[1] == mid]
        if not cands:
            return None, 0.0
        exact = [d for d in cands if (parse_part(d.part) or (None, None, None))[2] == suf]
        best = max(exact or cands, key=lambda d: d.conf)
        return best.bbox, best.conf


def kit_from_detections(dets: list["Detection"], max_comp: int = 7,
                        comp_of: dict[str, int] | None = None) -> list[dict]:
    """Build the pick-list from what the scan cam actually sees: one entry per
    detected packet, ordered top-to-bottom then left-to-right.

    Compartment assignment:
      - ``comp_of`` given → route each SKU to its FIXED physical compartment (the
        box calibration). This is the correct path: a jittery read never reshuffles
        the whole numbering, and the pick target matches the real box.
      - ``comp_of`` None → fall back to the legacy "grouped, first-seen" rule
        (identical parts share a compartment, numbered in first-seen order)."""
    dets = _clean_dets(dets, 0.0)      # box-dedupe guard: never two entries per packet
    ordered = sorted(dets, key=lambda d: (d.bbox[1] // 120, d.bbox[0]))
    seen: dict[str, int] = {}
    kit = []
    for i, d in enumerate(ordered):
        comp = None
        if d.part:
            if comp_of is not None:
                comp = comp_of.get(d.part)         # fixed physical compartment for this SKU
            else:
                if d.part not in seen:
                    seen[d.part] = min(len(seen) + 1, max_comp)
                comp = seen[d.part]
        kit.append({"bag_id": i + 1, "part": d.part, "name": "",
                    "comp": comp, "bbox": [0.0, 0.0, 0.0, 0.0]})
    return kit
