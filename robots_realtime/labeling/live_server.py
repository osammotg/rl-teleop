"""Live label backend for the kitting cockpit.

Serves the exact contract the cockpit polls:
    GET  /state            -> {seeded, packets:[{part,name,comp,bbox,status}], ti, ...}
    POST /seed             -> seed the kit, returns state
    GET  /cam/<id>         -> MJPEG proxied from --cam-base, else 404
    GET  /events           -> the live cockpit_events (debug / offline consistency)
    POST /episodemode?m=   -> flip auto-advance live: grasp | full (cockpit "Takt")

Point the cockpit's "live URL" at http://localhost:<port>. As the operator
teleoperates, a feed pushes gripper samples into the LiveLabeler and the cockpit
shows each grasp/place labeled live (ti advances, packets flip to 'placed').

Feeds:
    replay_feed(...)   replay a recorded/synthetic episode (verifiable demo).
    bus_feed(...)      subscribe to the live rl-teleop joint stream (real wire-up).

Run a demo:
    uv run python -m robots_realtime.labeling.live_server --replay <episode_dir>
"""
from __future__ import annotations

import argparse
import json
import threading
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np

from robots_realtime.labeling import constants as C
from robots_realtime.labeling.fk import ForwardKinematics
from robots_realtime.labeling.live import LiveLabeler

DEFAULT_KIT = [
    {"bag_id": 1, "part": "UNN-10126-151", "name": "Flügelmutter M8", "comp": 1, "bbox": [0.30, 0.55, 0.10, 0.10]},
    {"bag_id": 2, "part": "UNN-10015-007", "name": "Sechskantschraube", "comp": 6, "bbox": [0.55, 0.55, 0.10, 0.10]},
    {"bag_id": 3, "part": "MDDY-11065-001", "name": "Kappe", "comp": 5, "bbox": [0.20, 0.30, 0.10, 0.10]},
]

# Closed SKU catalog for the CURRENT kit + each SKU's fixed physical compartment.
#   - SKU set  mirrors yams-sorting-robot/configs/sorting.yaml (parts_to_cell keys).
#   - comp ids are the cockpit's calibrated 7-cell box (const ASSIGN in
#     Buehler-Kitting-Cockpit.html) — NOT the uncalibrated [row,col] grid in grid.yaml.
# This is the single place to edit when the kit changes. Wiring it into the detector's
# known_source turns on catalog-snapping (kills UNN→DNN / -009→-000 misreads); using it
# as comp_of routes every read to its real compartment instead of first-seen order.
KIT_CATALOG = {
    "UNN-16022-009": 7,   # Blindniet D4.0x12.5 Al
    "UNN-10015-007": 6,   # Sechskantschraube M6x10
    "MDDY-11065-001": 5,  # Kappe
    "UNN-10015-231": 4,   # Sechskantflanschschraube M6
    "UNN-10126-151": 1,   # Flügelmutter M8 nichtrostend
}
KNOWN_SKUS = list(KIT_CATALOG)
KIT_NAMES = {
    "UNN-16022-009": "Blindniet D4.0x12.5 Al",
    "UNN-10015-007": "Sechskantschraube M6x10",
    "MDDY-11065-001": "Kappe",
    "UNN-10015-231": "Sechskantflanschschraube M6",
    "UNN-10126-151": "Flügelmutter M8 nichtrostend",
}


def scanned_kit(dets):
    """kit_from_detections with the current kit's fixed compartments + display names.

    Every physical packet the scan sees becomes a row, including one whose number
    could not be read or does not map to the catalog. Those are marked
    ident="unknown" rather than dropped: a bag missing from the pick list is a bag
    the operator cannot act on, and one unreadable label must not stall the kit.
    Identity is only ever asserted from a real catalog match — never guessed.
    """
    from robots_realtime.labeling.detector import kit_from_detections
    kit = kit_from_detections(dets, comp_of=KIT_CATALOG)
    for p in kit:
        part = p.get("part")
        known = bool(part) and part in KIT_CATALOG
        p["ident"] = "ok" if known else "unknown"
        p["name"] = KIT_NAMES.get(part, "") if known else "Nicht erkannt"
        if not known:
            # No catalog match → no compartment. Showing one would be a guess, and a
            # bag placed in a guessed compartment is a defect that leaves the cell.
            p["comp"] = None
            p["read"] = part or None      # keep the raw read (if any) for the operator
    return kit


def encode_frame_jpeg(frame, quality: int = 80) -> bytes | None:
    """RGB (H,W,3) uint8 → JPEG bytes (BGR-corrected for the browser)."""
    import cv2
    import numpy as np
    arr = np.asarray(frame)
    if arr.ndim != 3 or arr.shape[2] != 3:
        return None
    # coerce to a clean uint8 C-contiguous buffer so cv2 never trips on an odd
    # stride/dtype (the frame content itself is trusted — see wrist MJPEG note).
    if arr.dtype != np.uint8:
        arr = arr.astype(np.uint8)
    arr = np.ascontiguousarray(arr)
    bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return buf.tobytes() if ok else None


class CameraBridge:
    """Serve the running session's camera topics as single JPEGs at /cam/<id>.

    During teleop the physical cameras are held by rr-session, so we can't open
    /dev/video directly — we subscribe to the ``<name>/rgb`` topics on the ZMQ bus
    ({"frame": (H,W,3) uint8}) and JPEG-encode the latest frame on request. The
    cockpit's <img src=".../cam/<id>?t=..."> cache-buster re-fetches it live.
    """

    #: How old the newest envelope may be before /cam/<id> stops serving it.
    #:
    #: THE BRIDGE USED TO SERVE THE LAST FRAME FOREVER, at a convincing 15 fps,
    #: with HTTP 200 and a valid 21 KB JPEG. On 2026-08-10 the cockpit showed a
    #: wrist panel that was one frame repeated for minutes while the bus carried
    #: nothing at all. An image that keeps arriving is not evidence that a camera
    #: is alive; it is evidence that something has a copy of an old picture.
    #:
    #: 2.0 s is above any legitimate gap (the slowest camera here is 15 Hz) and
    #: inside the 2 s detection budget the acceptance bar asks for.
    STALE_AFTER_S = 2.0

    def __init__(self, id_to_topic: dict[str, str], host: str = "127.0.0.1",
                 port: int | None = None, stale_after_s: float | None = None):
        from robots_realtime.runtime.transport.message_bus import DEFAULT_SUB_PORT
        from robots_realtime.runtime.transport.subscriber import Subscriber
        self._id_to_topic = dict(id_to_topic)
        topics = sorted(set(id_to_topic.values()))
        # Also subscribe to every mapped camera's health topic, so the cockpit
        # can show WHY a panel is dark instead of inferring it from a missing
        # JPEG. "no image" and "the camera says it is reopening" are different
        # facts and the operator needs the second one.
        health_topics = sorted({f"{t.split('/')[0]}/health" for t in topics if "/" in t})
        self._health_topics = {
            cam_id: f"{topic.split('/')[0]}/health"
            for cam_id, topic in self._id_to_topic.items() if "/" in topic
        }
        self._sub = Subscriber(topics + health_topics, host=host, port=port or DEFAULT_SUB_PORT)
        self._stale_after_s = float(
            stale_after_s if stale_after_s is not None else self.STALE_AFTER_S
        )
        # Wall clock, deliberately: the envelope's `ts` is wall clock too, and
        # comparing monotonic against wall clock is how you get an age of 1.7e9.
        self._started = time.time()

    # ── staleness ────────────────────────────────────────────────────────────

    def age(self, cam_id: str) -> float | None:
        """Seconds since the newest envelope for this panel, or None if never."""
        topic = self._id_to_topic.get(cam_id)
        if topic is None:
            return None
        env = self._sub.get_latest(topic)
        if not env:
            return None
        ts = env.get("ts")
        if not isinstance(ts, (int, float)):
            return None
        return max(0.0, time.time() - float(ts))

    def camera_health(self, cam_id: str) -> dict | None:
        """The camera node's own health record for this panel, if it publishes one.

        The record carries its OWN age and a ``stale`` flag. A health topic that
        has stopped updating is not evidence of health — SIGSTOP a camera node
        and it publishes nothing at all, leaving a cheerful ``ok`` sitting on the
        bus for as long as anyone cares to read it.
        """
        topic = self._health_topics.get(cam_id)
        if topic is None:
            return None
        env = self._sub.get_latest(topic)
        if not env:
            return None
        data = env.get("data")
        if not isinstance(data, dict):
            return None
        out = dict(data)
        ts = env.get("ts")
        if isinstance(ts, (int, float)):
            age = max(0.0, time.time() - float(ts))
            out["health_age_s"] = round(age, 3)
            if age > self._stale_after_s:
                out["stale"] = True
                out["healthy"] = False
                out["state"] = "stale"
        return out

    def state(self, cam_id: str) -> dict:
        """One honest verdict per panel, for the cockpit and for /cam_health.

        ``state`` is one of:
          ``unmapped``  — this panel id is not wired to any topic (503)
          ``no_data``   — mapped, but nothing has ever arrived (503)
          ``stale``     — the newest frame is older than STALE_AFTER_S (503)
          ``ok``        — a fresh frame is available

        ``camera`` carries the node's own health record when it publishes one,
        so a stale panel can say "reopening" rather than just "dark".
        """
        if cam_id not in self._id_to_topic:
            return {"id": cam_id, "state": "unmapped", "age_s": None, "camera": None}
        age = self.age(cam_id)
        health = self.camera_health(cam_id)
        if age is None:
            state = "no_data"
        elif age > self._stale_after_s:
            state = "stale"
        else:
            state = "ok"
        # The camera's own verdict can only make things WORSE, never better: a
        # node that says `failed` while a frame from 100 ms ago is still in the
        # buffer is not a healthy panel.
        if state == "ok" and health is not None and not health.get("healthy", True):
            state = "unhealthy"
        return {
            "id": cam_id,
            "state": state,
            "age_s": None if age is None else round(age, 3),
            "topic": self._id_to_topic.get(cam_id),
            "camera": health,
        }

    def all_states(self) -> dict:
        return {
            "t": time.time(),
            "stale_after_s": self._stale_after_s,
            "cams": {cid: self.state(cid) for cid in sorted(self._id_to_topic)},
        }

    def frame(self, cam_id: str):
        """Latest RGB (H,W,3) uint8 frame for a cam id, or None if unmapped.

        NO FALLBACK TO 'default'. Until 2026-08-10 an unmapped id served the
        default topic "so no panel is blank" — so on a right-arm session the
        Scan and Handgelenk-links panels both rendered camera_top, and the
        cockpit showed three copies of the same top-down view while looking
        entirely healthy. An operator cannot tell that from three working
        cameras, and it is a worse failure than an empty panel: a blank frame
        says "no source", a wrong frame asserts a source that isn't there.
        Unmapped now returns None, which the /cam route turns into a 404.
        """
        topic = self._id_to_topic.get(cam_id)
        if topic is None:
            return None
        env = self._sub.get_latest(topic)
        if not env:
            return None
        # STALENESS IS A HARD GATE, not a hint. Serving the last envelope forever
        # is what made a dead wrist camera look like a working one for minutes:
        # HTTP 200, valid JPEG, 15 fps, and not one message on the bus behind it.
        # A panel with no source must be visibly empty, not plausibly full.
        ts = env.get("ts")
        if isinstance(ts, (int, float)) and (time.time() - float(ts)) > self._stale_after_s:
            return None
        # AND the camera's own verdict. Freshness of the ENVELOPE is not enough:
        # a frozen camera publishes a brand-new message every 33 ms carrying the
        # same picture, so the age check waves it straight through. That is the
        # exact failure this bridge is supposed to stop lying about, arriving by
        # a different road. One definition of healthy, honoured everywhere.
        health = self.camera_health(cam_id)
        if health is not None and not health.get("healthy", True):
            return None
        data = env.get("data") or {}
        # CameraNode publishes {"images": {"rgb": (H,W,3) uint8}, ...}; older
        # payloads used a bare {"frame": ...}.
        frame = data.get("frame")
        if frame is None:
            imgs = data.get("images")
            if isinstance(imgs, dict) and imgs:
                frame = imgs.get("rgb")
                if frame is None:
                    frame = next(iter(imgs.values()))
        return frame

    def jpeg(self, cam_id: str) -> bytes | None:
        frame = self.frame(cam_id)
        return encode_frame_jpeg(frame) if frame is not None else None


def _state_with_detections(labeler: LiveLabeler, detector) -> dict:
    """labeler.state() augmented with live scan detections: each kit packet gets a
    pixel bbox on the scan frame so the cockpit can draw the 'pick this next' box on
    the actual packet.

    Assignment matches each kit entry to a DISTINCT physical detection, claimed so no
    two entries share one box. Priority:
      1. exact part id  (middle AND suffix) — the suffix is the discriminator that tells
         UNN-10015-007 from UNN-10015-231; matching on it prevents same-middle packets
         from getting their boxes/identities swapped (→ wrong compartment shown).
      2. same middle only — fallback when the suffix wasn't read or doesn't line up.
    An entry with no distinct detection left gets NO box (a wrong box is worse than none).
    wh = scan size."""
    from robots_realtime.labeling.detector import parse_part
    st = labeler.state()
    if detector is None:
        return st
    dets, wh = detector.current()
    st["wh"] = wh
    try:
        st["det"] = detector.status()      # mode/margin/gamma/boxes → keep the debug panel honest
    except Exception:
        pass
    parsed = [(parse_part(d.part or ""), d) for d in dets]      # (parsed_id_or_None, Detection)
    claimed = [False] * len(parsed)

    def _take(pred) -> "Detection | None":
        # claim the first unclaimed detection satisfying pred, in stable top-to-bottom,
        # left-to-right order (so the positional fallback is deterministic).
        cands = sorted((j for j, (p, d) in enumerate(parsed) if not claimed[j] and pred(p, d)),
                       key=lambda j: (parsed[j][1].bbox[1], parsed[j][1].bbox[0]))
        if not cands:
            return None
        claimed[cands[0]] = True
        return parsed[cands[0]][1]

    # A SKIPPED bag is still physically on the mat, so it keeps its box — but it must
    # claim LAST. Skipped rows sit before the pointer, so in list order they would grab
    # the scarce detections first and blank the box on the packet being picked right now.
    # The entries are mutated in place, so visiting them in a different order is safe.
    for pk in sorted(st["packets"], key=lambda p: p.get("status") == "skipped"):
        # A PLACED packet is physically off the mat, so it must not claim a detection.
        # With duplicate part ids (this kit runs 3 identical bags), a placed entry would
        # otherwise steal the one box OCR read this window from the still-present CURRENT
        # packet — and its orange "pick this next" overlay vanishes. Skipping placed
        # entries also makes the current packet (first non-placed in kit order) claim
        # first, so it wins the box whenever detections are scarce.
        if pk.get("status") == "placed":
            pk["bbox_px"] = None
            pk["det_conf"] = 0.0
            continue
        kp = parse_part(pk.get("part") or "")
        d = None
        if kp and pk.get("ident") != "unknown":
            mid, suf = kp[1], kp[2]
            d = _take(lambda p, _d: bool(p) and p[1] == mid and p[2] == suf)   # 1) exact id
            if d is None:
                d = _take(lambda p, _d: bool(p) and p[1] == mid)               # 2) middle-only
        else:
            # An unidentified entry still has a physical bag behind it, so give it the
            # matching unidentified detection. The operator has to SEE the bag the
            # system could not name — that is the whole point of keeping the row.
            d = _take(lambda p, _d: p is None)
        pk["bbox_px"] = d.bbox if d else None
        pk["det_conf"] = d.conf if d else 0.0
    return st


class EpisodeMode:
    """The auto-advance switch, flippable while a session runs.

    This used to be a launch-time argument only: --episode-mode grasp wired the
    labeler's on_place to rr-session's /record/next, and nothing could unwire it
    short of restarting the backend. But auto-advance is exactly the feature an
    operator wants to switch OFF the moment it misfires — a placement the gate
    scores as real when it was a re-grip ends the take early, and the fix is to
    take the takt back by hand, not to stop the session.

    So the callback is now always installed and reads this object each time. The
    read is a plain attribute under a lock: the joint feed fires it from its own
    thread, the HTTP handler writes it from another.

        full   the operator ends every take   (→ / Nächste / [1])
        grasp  every gated placement saves the take and starts the next
    """

    VALID = ("full", "grasp")

    def __init__(self, mode: str = "full"):
        self._lock = threading.Lock()
        self._mode = mode if mode in self.VALID else "full"

    @property
    def mode(self) -> str:
        with self._lock:
            return self._mode

    @property
    def auto(self) -> bool:
        return self.mode == "grasp"

    def set(self, mode: str) -> dict:
        if mode not in self.VALID:
            return {"ok": False, "error": f"mode must be one of {list(self.VALID)}",
                    "episode_mode": self.mode}
        with self._lock:
            changed = mode != self._mode
            self._mode = mode
        if changed:
            print(f"[live] episode-mode → {mode}"
                  + ("  (each gated placement saves the take and starts the next)"
                     if mode == "grasp" else "  (you end every take yourself)"))
        return {"ok": True, "episode_mode": mode, "changed": changed}


def make_auto_advance(control_url: str, episode_mode: EpisodeMode):
    """The labeler's on_place callback: save this take and start the next one.

    Always installed, but a no-op unless the switch is on. Keeping the wiring
    constant and gating inside is what makes the toggle safe to flip mid-take —
    there is no window where on_place is being reassigned under the feed thread.
    """
    import urllib.request as _u

    ctl = control_url.rstrip("/")

    def _advance() -> None:
        if not episode_mode.auto:
            return
        req = _u.Request(ctl + "/record/next", data=b"{}", method="POST",
                         headers={"Content-Type": "application/json"})
        try:
            with _u.urlopen(req, timeout=2) as r:
                r.read()
        except Exception as e:
            # Never fatal: the operator can still end the take by hand.
            print(f"[live] auto-advance failed ({e}) — end the take with [1] or the cockpit")

    return _advance


def _make_handler(labeler: LiveLabeler, cam_base: str | None,
                  bridge: "CameraBridge | None" = None, detector=None,
                  episode_mode: EpisodeMode | None = None):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _cors(self):
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")

        def _json(self, obj, code=200):
            body = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self._cors()
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_OPTIONS(self):
            self.send_response(204); self._cors(); self.end_headers()

        def do_GET(self):
            if self.path == "/state":
                st = _state_with_detections(labeler, detector)
                # The cockpit renders the takt switch from this, never from a
                # local flag — same rule as the recorder transport.
                st["episode_mode"] = episode_mode.mode if episode_mode else "full"
                self._json(st)
            elif self.path == "/events":
                self._json(labeler.events)
            elif self.path.startswith("/cam_health"):
                # ONE honest verdict per cockpit panel, polled by cockpit-cams.js.
                #
                # The cockpit used to infer camera health from "did a JPEG
                # arrive", which is exactly the signal that lies: the bridge
                # served the last frame forever, so every panel looked alive
                # while the bus carried nothing. Health has to be read, not
                # inferred from the artefact whose freshness is in question.
                if bridge is None:
                    self._json({"t": time.time(), "cams": {}, "bridge": False})
                else:
                    st = bridge.all_states()
                    st["bridge"] = True
                    self._json(st)
            elif self.path.startswith("/cam/"):
                self._proxy_cam(self.path.split("/cam/", 1)[1])
            else:
                self.send_error(404)

        def do_POST(self):
            if self.path == "/seed":
                n = int(self.headers.get("Content-Length", 0) or 0)
                raw = self.rfile.read(n) if n else b""
                kit = None
                if raw:
                    try:
                        kit = json.loads(raw)["packets"]
                    except Exception:
                        kit = None
                if kit is None and detector is not None:
                    # operator started recording → freeze the kit as the scan sees it NOW
                    dets, _wh = detector.current()
                    kit = scanned_kit(dets) or None
                labeler.seed(kit if kit is not None else DEFAULT_KIT)
                labeler.locked = True
                self._json(_state_with_detections(labeler, detector))
            elif self.path == "/advance":
                labeler.advance()
                self._json(labeler.state())
            elif self.path.startswith("/skip"):
                # Step past the current packet without claiming it was placed, so one
                # unreadable label can never hold up the rest of the kit.
                from urllib.parse import parse_qs, urlparse
                q = parse_qs(urlparse(self.path).query)
                labeler.skip((q.get("reason") or ["unidentified"])[0])
                self._json(_state_with_detections(labeler, detector))
            elif self.path.startswith("/recalc"):
                # Manual re-scan: drop held/merged/stale boxes and detect fresh right now.
                # WAIT for that scan to land before answering — returning immediately meant
                # replying with the pre-rescan box count, so the cockpit panel showed an
                # unchanged number and the ⟳ button looked like it had done nothing. One
                # cycle is ~2-3s (full-res OCR); past the budget we say so instead of lying.
                if detector is None:
                    self._json({"ok": False, "reason": "no detector"})
                else:
                    done = detector.recalibrate_now(wait_s=8.0)
                    self._json({"ok": True, "rescanned": done, **detector.status()})
            elif self.path.startswith("/autotune"):
                # sweep exposure to match the current lighting, pick the best, re-scan
                self._json(detector.autotune() if detector is not None
                           else {"ok": False, "reason": "no detector"})
            elif self.path.startswith("/episodemode"):
                # Flip auto-advance live: /episodemode?m=grasp|full  (cockpit "Takt")
                from urllib.parse import parse_qs, urlparse
                q = parse_qs(urlparse(self.path).query)
                m = (q.get("m") or [""])[0]
                if episode_mode is None:
                    self._json({"ok": False, "error": "no episode-mode switch on this server"})
                else:
                    self._json(episode_mode.set(m))
            elif self.path.startswith("/trackmode"):
                # switch the box tracker: /trackmode?m=ocr|contour|sam
                from urllib.parse import parse_qs, urlparse
                q = parse_qs(urlparse(self.path).query)
                m = (q.get("m") or ["ocr"])[0]
                self._json(detector.set_mode(m) if detector is not None
                           else {"ok": False, "reason": "no detector"})
            elif self.path.startswith("/margin"):
                # grow/shrink every mode's box: /margin?px=20  (wider/narrower buttons)
                from urllib.parse import parse_qs, urlparse
                q = parse_qs(urlparse(self.path).query)
                px = int((q.get("px") or ["0"])[0])
                self._json(detector.set_margin(px) if detector is not None else {"ok": False})
            else:
                self.send_error(404)

        def _proxy_cam(self, cam_id: str):
            # 1) live bus bridge. Firefox does NOT render multipart/x-mixed-replace
            #    inside an <img>, so serve a SINGLE JPEG per request — the cockpit
            #    re-polls with a ?t= cache-buster for live-ish video (works in all
            #    browsers). ?stream=1 opts into MJPEG for chrome/direct viewing.
            stream = self.path.split("?", 1)[1].startswith("stream=1") if "?" in self.path else False
            cam_id = cam_id.split("?", 1)[0]
            if bridge is not None:
                jpg = bridge.jpeg(cam_id)
                if jpg is not None and not stream:
                    self.send_response(200)
                    self.send_header("Content-Type", "image/jpeg")
                    self._cors()
                    self.send_header("Cache-Control", "no-store")
                    self.send_header("Content-Length", str(len(jpg)))
                    self.end_headers()
                    self.wfile.write(jpg)
                    return
                if jpg is not None and stream:
                    self.send_response(200)
                    self.send_header("Content-Type",
                                     "multipart/x-mixed-replace; boundary=frame")
                    self._cors(); self.send_header("Cache-Control", "no-store")
                    self.end_headers()
                    try:
                        while True:
                            f = bridge.jpeg(cam_id)
                            if f is None:
                                # END THE STREAM rather than holding the socket
                                # open emitting nothing. The old loop kept the
                                # connection alive forever with no frames, which
                                # reads to a browser as "still connecting" and to
                                # a checker as a slow stream. A closed stream is
                                # an unambiguous "this camera has stopped".
                                return
                            self.wfile.write(
                                b"--frame\r\nContent-Type: image/jpeg\r\n"
                                b"Content-Length: " + str(len(f)).encode()
                                + b"\r\n\r\n" + f + b"\r\n")
                            time.sleep(1 / 15.0)
                    except (BrokenPipeError, ConnectionResetError):
                        return
                    return
            # 2) optional proxy fallback
            if not cam_base:
                # 503 with a REASON. "unmapped", "no_data" and "stale" are three
                # different operator problems (wrong --bus-cams, node never
                # started, camera stopped mid-session) and a bare 503 makes them
                # look like one.
                reason = "no_bridge"
                if bridge is not None:
                    reason = str(bridge.state(cam_id).get("state", "unknown"))
                self.send_response(503)
                self.send_header("Content-Type", "text/plain")
                self.send_header("X-Cam-State", reason)
                self._cors()
                body = f"camera {cam_id}: {reason}\n".encode()
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            try:
                up = urllib.request.urlopen(f"{cam_base.rstrip('/')}/cam/{cam_id}", timeout=3)
            except Exception:
                self.send_error(502); return
            self.send_response(200)
            self.send_header("Content-Type", up.headers.get("Content-Type", "multipart/x-mixed-replace"))
            self._cors(); self.end_headers()
            try:
                while True:
                    chunk = up.read(65536)
                    if not chunk:
                        break
                    self.wfile.write(chunk)
            except Exception:
                pass

    return Handler


class LiveLabelServer:
    def __init__(self, labeler: LiveLabeler, port: int = 8791, cam_base: str | None = None,
                 bridge: "CameraBridge | None" = None, host: str = "127.0.0.1", detector=None,
                 episode_mode: EpisodeMode | None = None):
        self.labeler = labeler
        self._httpd = ThreadingHTTPServer(
            (host, port), _make_handler(labeler, cam_base, bridge, detector, episode_mode))
        self.port = port

    def serve_forever(self):
        self._httpd.serve_forever()

    def start_background(self) -> threading.Thread:
        t = threading.Thread(target=self.serve_forever, daemon=True)
        t.start()
        return t

    def shutdown(self):
        self._httpd.shutdown()


def _feed_from_source(labeler: LiveLabeler, next_sample, stop=None,
                      poll_s: float = 0.005) -> None:
    """Pump (ts, gripper_width, ee_pos) from ``next_sample()`` into the labeler
    until stop is set. ``next_sample`` returns (ts, width, ee_pos) or None when
    nothing new; ee_pos may be None when joint data is unavailable (the transport
    gate then fails open and the cockpit flags it).
    Deduplicates on ts so the latest-per-topic subscriber isn't re-pushed."""
    last_ts = None
    while stop is None or not stop.is_set():
        s = next_sample()
        if s is not None:
            # Accept (ts, width) as well as (ts, width, ee_pos): a source that
            # cannot supply poses must degrade to the ungated behavior, never
            # crash the feed thread and take live labeling down with it.
            ts, width, ee_pos = s if len(s) == 3 else (s[0], s[1], None)
            if ts != last_ts:
                last_ts = ts
                labeler.push(float(ts), float(width), ee_pos)
        time.sleep(poll_s)


def bus_feed(labeler: LiveLabeler, arm: str = "left", host: str = "127.0.0.1",
             port: int | None = None, stop=None,
             urdf_path: str = "urdf/yam.urdf") -> None:
    """Subscribe to the live rl-teleop joint stream and drive the labeler.

    The follower publishes ``yam_{arm}/joint_state`` = {"joint_pos": [...7...]}
    (gripper at index 6) on the ZMQ XPUB/XSUB broker. This taps it live so the
    cockpit shows each grasp/place labeled as it happens.
    """
    from robots_realtime.runtime.transport.message_bus import DEFAULT_SUB_PORT
    from robots_realtime.runtime.transport.subscriber import Subscriber

    topic = f"yam_{arm}/joint_state"
    sub = Subscriber([topic], host=host, port=port or DEFAULT_SUB_PORT)

    # The transport gate needs where the gripper IS, not just how open it is. The
    # arm joints already ride in the same envelope as gripper_pos, so this costs
    # one FK per sample (~89 us, 1.8% of a core at 200 Hz) and no new subscription.
    fk = None
    try:
        fk = ForwardKinematics(urdf_path)
    except Exception as e:                      # missing/!parseable URDF
        print(f"[live] no FK ({e}) — transport gate disabled, re-grasps will advance")

    def next_sample():
        env = sub.get_latest(topic)
        if env is None:
            return None
        ts = env.get("ts")
        data = env.get("data") or {}
        # Follower publishes joint_pos (6 arm) + gripper_pos (1) as SEPARATE fields;
        # the gripper is NOT joint_pos[6].
        grip = data.get("gripper_pos")
        if grip is None or ts is None:
            return None
        width = float(grip[0]) if hasattr(grip, "__len__") else float(grip)
        ee_pos = None
        joints = data.get("joint_pos")
        if fk is not None and joints is not None and len(joints) >= C.N_ARM_JOINTS:
            try:
                ee_pos = fk.ee_pose(list(joints)[: C.N_ARM_JOINTS])[:3]
            except Exception:
                ee_pos = None                   # never let FK kill the feed
        return float(ts), width, ee_pos

    _feed_from_source(labeler, next_sample, stop=stop)


def write_cockpit_events(labeler: LiveLabeler, path: str | Path) -> None:
    """Persist the live place/grasp events as cockpit_events.jsonl so the
    offline labeler can fuse intent (part → compartment) later."""
    lines = [json.dumps(e) for e in labeler.events]
    Path(path).write_text("\n".join(lines) + ("\n" if lines else ""))


def replay_feed(labeler: LiveLabeler, times, positions, realtime: bool = True,
                speed: float = 1.0, urdf_path: str | None = "urdf/yam.urdf") -> None:
    """Push a joint timeline into the labeler, optionally pacing in real time.

    Computes EE positions so a replay exercises the SAME transport gate as the
    live bus feed — a replay that skipped the gate would report advance counts the
    real rig never produces. Pass urdf_path=None to replay without the gate.
    """
    times = np.asarray(times, float)
    pos = np.asarray(positions, float)
    gripper = pos[:, C.GRIPPER_JOINT_INDEX]
    ee = None
    if urdf_path is not None:
        try:
            ee = ForwardKinematics(urdf_path).ee_positions(pos[:, : C.N_ARM_JOINTS])
        except Exception:
            ee = None
    t0 = times[0] if times.size else 0.0
    wall0 = time.monotonic()
    for i in range(times.size):
        if realtime:
            target = (times[i] - t0) / max(speed, 1e-6)
            dt = target - (time.monotonic() - wall0)
            if dt > 0:
                time.sleep(dt)
        labeler.push(float(times[i]), float(gripper[i]),
                     None if ee is None else ee[i])


def _demo_episode():
    """Synthesize a 3-bag joint timeline (gripper only matters for live)."""
    dt = 0.02
    OPEN, HELD = 1.0, 0.35
    plan = []
    for _ in range(3):
        plan += [(1.0, OPEN), (0.5, HELD), (1.0, HELD), (0.6, OPEN)]
    times, pos, t = [], [], 0.0
    for dur, grip in plan:
        for _ in range(int(dur / dt)):
            pos.append([0, 0, 0, 0, 0, 0, grip]); times.append(t); t += dt
    return np.array(times), np.array(pos)


def episode_kit_json(packets: list[dict]) -> list[dict]:
    """Convert the live scanned kit → the kit.json shape label_episode reads
    ([{bag_id, part_no, name, compartment}, ...]) so each bag's part identity is
    saved with the recording and fused into annotations.json."""
    return [{"bag_id": p.get("bag_id", i + 1),
             "part_no": p.get("part"),
             "name": p.get("name") or None,
             "compartment": p.get("comp")}
            for i, p in enumerate(packets)]


def _label_episode_argv(episode_dir: Path, arm: str,
                        labeler: "LiveLabeler | None" = None) -> list[str]:
    """The exact argv the auto-labeller shells out with.

    Split out so it is testable, because what it forwards is the whole bug:
    until 2026-08-08 this was `label_episode <dir> --arm <arm>` and nothing
    else. The LIVE labeller runs with open_ref=1.0 / closed_ref=0.0 and
    MIN_TRANSPORT_M=0.10; label_episode's own defaults are None/None/0.0. So the
    same --auto-label run showed the operator one set of labels live and wrote a
    DIFFERENT set into annotations.json — the authoritative file the corpus is
    built from — using the percentile guess instead of the known limits.
    (DATA-PIPELINE.md 2.3.) The live path already had the right answer; it just
    never handed it over.
    """
    argv = ["robots_realtime.labeling.label_episode", str(episode_dir), "--arm", arm]
    if labeler is not None:
        if labeler.open_ref is not None and labeler.closed_ref is not None:
            argv += ["--open-ref", repr(float(labeler.open_ref)),
                     "--closed-ref", repr(float(labeler.closed_ref))]
        argv += ["--min-transport", repr(float(labeler.min_transport_m))]
    return argv


def _run_label_episode(episode_dir: Path, arm: str,
                       labeler: "LiveLabeler | None" = None) -> None:
    """Fire label_episode as an isolated subprocess → writes annotations.json.
    Isolated so a bad/incomplete episode (e.g. missing yam_<arm>.mcap) logs an
    error instead of taking down the server."""
    import subprocess
    import sys
    try:
        # low priority (nice) so the labeler's OCR doesn't starve the live camera
        # streaming at save time (that was freezing the cockpit).
        subprocess.Popen(
            ["nice", "-n", "19", sys.executable, "-m"]
            + _label_episode_argv(episode_dir, arm, labeler),
            cwd=str(Path.cwd()),
            stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
        print(f"[auto-label] {episode_dir.name} → annotations.json")
    except Exception as e:
        print(f"[auto-label] failed to launch labeler for {episode_dir}: {e}")


def record_watcher(save_root: str, labeler: LiveLabeler, arm: str,
                   auto_label: bool, stop=None) -> None:
    """Watch save_root for episodes rr-session creates, decoupled via the filesystem.

    IMPORTANT timing: rr-session writes session_meta.json TWICE — once at episode
    START (_make_episode_dir) and again at END (end_episode, on save). So "meta
    exists" means recording is IN PROGRESS, not done. We label only when session_meta
    is RE-WRITTEN (its mtime advances past what we first saw), which happens after the
    mcap is flushed on save. A discarded episode is rmtree'd → never re-written → never
    labeled. Idempotent across live_server restarts: an episode that already has
    annotations.json / kit.json is not reprocessed."""
    root = Path(save_root)
    track: dict[str, dict] = {}          # dir → {"meta0": float|None, "labeled": bool}
    while stop is None or not stop.is_set():
        time.sleep(1.0)
        for d in sorted(root.glob("*/episode_*")) if root.exists() else []:
            if not d.is_dir():
                continue
            key, meta = str(d), d / "session_meta.json"
            if key not in track:
                already = (d / "annotations.json").exists()
                track[key] = {"meta0": (meta.stat().st_mtime if meta.exists() else None),
                              "labeled": already}
                # write the scanned kit ONCE, and never clobber an existing one (restart-safe)
                if not (d / "kit.json").exists():
                    try:
                        (d / "kit.json").write_text(
                            json.dumps(episode_kit_json(labeler.packets), indent=2))
                        print(f"[auto-label] wrote {d.name}/kit.json ({len(labeler.packets)} bags)")
                    except Exception as e:
                        print(f"[auto-label] kit.json write failed for {d}: {e}")
                # copy a canonical compartments.json (box calibration) in if present
                canon = root / "compartments.json"
                if canon.exists() and not (d / "compartments.json").exists():
                    try:
                        (d / "compartments.json").write_text(canon.read_text())
                    except Exception as e:
                        print(f"[auto-label] compartments copy failed for {d}: {e}")
                # BACKFILL saved-but-unlabeled episodes. If an episode was saved while this
                # watcher was down, it never sees the meta-rewrite trigger below (meta0 is
                # initialized to its already-final mtime, so `m > meta0 + 0.5` can never
                # fire) → stuck unlabeled forever. Catch it here: meta present + a SETTLED
                # mcap (finished writing >20s ago) + no annotations = saved, not recording.
                # The staleness guard is what keeps this from labeling an in-flight recording
                # (whose mcap is either absent-until-save or actively growing).
                mcap = d / f"yam_{arm}.mcap"
                if (auto_label and not already and meta.exists() and mcap.exists()
                        and mcap.stat().st_size > 0
                        and time.time() - mcap.stat().st_mtime > 20.0):
                    print(f"[auto-label] backfill saved-but-unlabeled {d.name}")
                    track[key]["labeled"] = True
                    _run_label_episode(d, arm, labeler)
            t = track[key]
            if not t["labeled"] and meta.exists():
                m = meta.stat().st_mtime
                if t["meta0"] is None:                 # first time meta appeared (start write)
                    t["meta0"] = m
                elif m > t["meta0"] + 0.5:              # re-written at end_episode → episode saved
                    t["labeled"] = True
                    labeler.locked = False             # episode done → let the NEXT setup re-scan
                    if auto_label:
                        _run_label_episode(d, arm, labeler)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8791)
    ap.add_argument("--live", action="store_true",
                    help="tap the running rl-teleop joint bus (real teleop)")
    ap.add_argument("--replay", type=str, default=None,
                    help="episode dir to replay (yam_left.mcap)")
    ap.add_argument("--record-events", type=str, default=None,
                    help="on exit, write cockpit_events.jsonl into this episode dir")
    ap.add_argument("--arm", default="left")
    ap.add_argument("--cam-base", default=None, help="proxy /cam/<id> to this base URL")
    ap.add_argument("--host", default="127.0.0.1",
                    help="bind address (use 0.0.0.0 to reach it over Tailscale/LAN)")
    ap.add_argument("--bus-cams", default=None,
                    help="serve /cam/<id> from bus topics, e.g. "
                         "'top=camera_top/rgb,ego=camera_top/rgb,left=camera_left/rgb'")
    ap.add_argument("--speed", type=float, default=3.0)
    ap.add_argument("--open-ref", type=float, default=1.0)
    ap.add_argument("--closed-ref", type=float, default=0.0)
    ap.add_argument("--detect", action="store_true",
                    help="run the scan-cam packet detector (needs easyocr)")
    ap.add_argument("--detect-period", type=float, default=2.0,
                    help="seconds between scan-cam detection passes")
    ap.add_argument("--detect-gpu", action="store_true",
                    help="use GPU for OCR (leave off on the RTX 5090 — no torch kernels)")
    ap.add_argument("--detect-mode", default="contour", choices=["ocr", "contour", "sam"],
                    help="box tracker: ocr (heuristic) | contour (watershed, default) | sam (FastSAM)")
    ap.add_argument("--save-root", default=None,
                    help="rr-session recordings root; watch it to save kit.json + auto-label")
    ap.add_argument("--auto-label", action="store_true",
                    help="run label_episode on each saved episode → annotations.json")
    ap.add_argument("--episode-mode", default="full", choices=["full", "grasp"],
                    help=(
                        "STARTING position of a switch that stays live all session "
                        "(cockpit Takt control / POST /episodemode?m=...).  "
                        "full  = one episode per BOX; the operator ends it (default, "
                        "current behaviour).  "
                        "grasp = one episode per GRASP-AND-PLACE cycle; each gated "
                        "successful placement saves the take and starts the next, "
                        "hands-free. Needs --control-url (rr-session's control port)."
                    ))
    ap.add_argument("--control-url", default="http://localhost:8792",
                    help="rr-session control surface, used by --episode-mode grasp")
    ap.add_argument("--bus-sub-port", type=int, default=None,
                    help="bus XPUB port to subscribe to (default 5556). Point this at "
                         "5566 to bridge a soak session without touching the live bus.")
    ap.add_argument("--cam-stale-after", type=float, default=CameraBridge.STALE_AFTER_S,
                    help="seconds before /cam/<id> stops serving the last frame and "
                         "starts returning 503. Serving a stale frame forever is the "
                         "bug this exists to prevent; raise it only with a reason.")
    args = ap.parse_args(argv)

    labeler = LiveLabeler(open_ref=args.open_ref, closed_ref=args.closed_ref)

    # Auto-advance ("grasp" mode): one saved episode per grasp-and-place cycle.
    # The labeler already knows the exact instant a placement passes the
    # transport gate; all that was missing was telling the recorder.
    #
    # --episode-mode only picks the STARTING position now. The switch is live
    # for the whole session — POST /episodemode?m=grasp|full, or the Takt
    # control in the cockpit — because the moment auto-advance misfires is the
    # moment you want the takt back, without restarting anything.
    episode_mode = EpisodeMode(args.episode_mode)
    labeler.on_place = make_auto_advance(args.control_url, episode_mode)
    if episode_mode.auto:
        print(f"[live] episode-mode=grasp — each placement saves a take via "
              f"{args.control_url.rstrip('/')} (switchable: cockpit Takt / POST /episodemode)")
    else:
        print("[live] episode-mode=full — one episode per box; you end it ([1] or cockpit) "
              "(switchable: cockpit Takt / POST /episodemode)")
    labeler.seed(DEFAULT_KIT)

    bridge = None
    if args.bus_cams:
        id_to_topic = dict(p.split("=", 1) for p in args.bus_cams.split(",") if "=" in p)
        bridge = CameraBridge(
            id_to_topic, host="127.0.0.1",
            port=args.bus_sub_port, stale_after_s=args.cam_stale_after,
        )
        print(f"Camera bridge: {id_to_topic} (bus sub port "
              f"{args.bus_sub_port or 'default'}, stale after {args.cam_stale_after}s)")

    # Packet detector: reads the scan cam, finds each kit packet's box + part id so
    # the cockpit can draw the "pick this next" box. Optional — needs easyocr; if it
    # is unavailable the cockpit falls back to kit-order highlighting.
    detector = None
    if bridge is not None and args.detect:
        try:
            from robots_realtime.labeling.detector import PacketDetector, kit_from_detections
            detector = PacketDetector(
                frame_source=lambda: bridge.frame("scan"),
                known_source=lambda: KNOWN_SKUS,    # snap every read to the closed kit catalog
                period_s=args.detect_period, gpu=args.detect_gpu,
                mode=args.detect_mode).start()      # default 'contour' (best in testing)
            print(f"Packet detector on the scan cam (period {args.detect_period}s, "
                  f"mode={args.detect_mode}, gpu={args.detect_gpu})")

            # Auto-seed the kit from the scan: the pick-list = the packets actually on the
            # table (not a fixed list). Re-seed while idle so the box always reflects reality;
            # once the operator starts recording, POST /seed locks in the scanned kit.
            def _autoseed():
                while True:
                    time.sleep(args.detect_period)
                    dets, _wh = detector.current()
                    if dets and not labeler.locked:
                        kit = scanned_kit(dets)
                        if kit and [p["part"] for p in kit] != [p.get("part") for p in labeler.packets]:
                            labeler.seed(kit)
            threading.Thread(target=_autoseed, daemon=True).start()
        except Exception as e:
            print(f"Packet detector disabled ({type(e).__name__}: {e})")

    # Watch recordings: save the scanned kit.json into each new episode and (optionally)
    # auto-run the offline labeler when the episode is saved, so every recording ends
    # up with a complete annotations.json (part id + compartment + grasp/place).
    if args.save_root:
        threading.Thread(target=record_watcher,
                         args=(args.save_root, labeler, args.arm, args.auto_label),
                         daemon=True).start()
        print(f"Watching {args.save_root} (kit.json + auto-label={args.auto_label})")

    server = LiveLabelServer(labeler, port=args.port, cam_base=args.cam_base,
                             bridge=bridge, host=args.host, detector=detector,
                             episode_mode=episode_mode)
    server.start_background()
    print(f"Live label backend on http://{args.host}:{args.port}  (point the cockpit here)")

    def _finish():
        if args.record_events:
            write_cockpit_events(labeler, Path(args.record_events) / "cockpit_events.jsonl")
            print(f"wrote {args.record_events}/cockpit_events.jsonl ({len(labeler.events)} events)")
        s = labeler.state()
        print(f"Final: {s['done']}/{s['total']} placed")

    try:
        if args.live:
            print("Tapping the live rl-teleop joint bus... teleoperate now.")
            bus_feed(labeler, arm=args.arm)          # blocks until Ctrl+C
        else:
            if args.replay:
                from robots_realtime.labeling.mcap_io import read_positions
                times, positions = read_positions(
                    Path(args.replay) / f"yam_{args.arm}.mcap", f"yam_{args.arm}")
            else:
                times, positions = _demo_episode()
            print(f"Feeding {len(times)} samples (speed x{args.speed})...")
            replay_feed(labeler, times, positions, realtime=True, speed=args.speed)
            _finish()
            while True:                              # keep serving final state
                time.sleep(1)
    except KeyboardInterrupt:
        _finish()
        server.shutdown()


if __name__ == "__main__":
    main()
