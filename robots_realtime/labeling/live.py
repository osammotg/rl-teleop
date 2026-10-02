"""Live labeling — drive the cockpit's /state view as the operator works.

The cockpit polls  GET liveBase()/state -> {seeded, packets:[{part,name,comp,bbox}], ti}
and  POST /seed. So to make each step show up labeled live we run the SAME gripper
state machine online, and as grasps/releases are detected we advance ``ti`` and mark
packets done. Confirmation stays batched per-kit (design decision) — this is the live
*labeling* feed, not the confirm sweep.

Online normalization uses a running min/max (no episode percentiles available live),
oriented so the first sample is "open". The authoritative labels are still produced
offline by label_episode; this is the live preview + the cockpit_events.jsonl the
offline pass consumes.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field

import numpy as np

from robots_realtime.labeling import constants as C
from robots_realtime.labeling.segmentation import (
    classify_hold,
    transport_distance_m,
    transport_ok,
)


@dataclass
class GripEvent:
    kind: str        # "close" | "release"
    t: float
    outcome: str | None = None   # for "release": success | slip | empty
    dxy_m: float | None = None   # EE horizontal travel close→release (None if no poses)
    lifted: bool | None = None   # did the EE rise during the hold? (None if no poses)


class OnlineGripSegmenter:
    """Incremental hysteresis state machine over the gripper width.

    Feed samples with ``push(t, width_raw)``; it returns a GripEvent on a
    close→open ("release") or the committed open→close ("close"), else None.
    """

    def __init__(self, open_ref: float | None = None, closed_ref: float | None = None):
        self._open_ref = open_ref
        self._closed_ref = closed_ref
        self._lo = None
        self._hi = None
        self._oriented_flip = None
        self._closed = False
        self._t_close = 0.0
        self._committed = False   # emitted the "close" event for this interval
        # Hold buffers, accumulated exactly as the offline detector accumulates its
        # `hold` list, so classify_hold() sees the same shape of input in both paths.
        # Bounded by hold duration (a 10 s hold at 200 Hz is 2000 floats).
        self._hold_w: list[float] = []
        self._hold_t: list[float] = []
        self._hold_z: list[float] = []
        # EE pose snapshot taken the instant the gripper commits to closing — NOT
        # when the "close" event fires, which is MIN_HOLD_S later. Backdating the
        # pose by 0.2 s would shrink every measured transport by however far the
        # arm moved in that window.
        self._pose_close = None

    def _norm(self, raw: float, first: bool) -> float:
        if self._open_ref is not None and self._closed_ref is not None:
            span = self._open_ref - self._closed_ref
            n = (raw - self._closed_ref) / span if abs(span) > 1e-9 else 0.0
        else:
            self._lo = raw if self._lo is None else min(self._lo, raw)
            self._hi = raw if self._hi is None else max(self._hi, raw)
            span = (self._hi - self._lo) if self._hi is not None else 0.0
            n = (raw - self._lo) / span if span > 1e-9 else 1.0
            if first:
                # first sample defines "open"; if it maps low, the signal is inverted
                self._oriented_flip = n < 0.5
            if self._oriented_flip:
                n = 1.0 - n
        return min(max(n, 0.0), 1.0)

    def push(self, t: float, width_raw: float, ee_pos=None) -> GripEvent | None:
        """Feed one sample. ``ee_pos`` is the end-effector [x, y, z] in metres, or
        None when joint data is unavailable (the gate then fails open upstream)."""
        first = self._lo is None and self._open_ref is None
        w = self._norm(float(width_raw), first)
        ev: GripEvent | None = None
        if not self._closed:
            if w < C.GRIPPER_CLOSE_ENTER:
                self._closed = True
                self._t_close = t
                self._committed = False
                self._hold_w, self._hold_t = [w], [t]
                self._hold_z = [float(ee_pos[2])] if ee_pos is not None else []
                self._pose_close = list(ee_pos) if ee_pos is not None else None
        else:
            self._hold_w.append(w)
            self._hold_t.append(t)
            if ee_pos is not None:
                self._hold_z.append(float(ee_pos[2]))
            if not self._committed and t - self._t_close >= C.MIN_HOLD_S:
                self._committed = True
                ev = GripEvent("close", self._t_close)
            if w > C.GRIPPER_CLOSE_EXIT:
                if self._committed:
                    # SAME classifier the offline labeler runs. Not a reimplementation.
                    z = (np.asarray(self._hold_z, float)
                         if len(self._hold_z) == len(self._hold_t) else None)
                    _hn, _mn, outcome, lifted = classify_hold(
                        np.asarray(self._hold_w, float), self._t_close,
                        np.asarray(self._hold_t, float), z)
                    ev = GripEvent("release", t, outcome,
                                   dxy_m=transport_distance_m(self._pose_close, ee_pos),
                                   lifted=lifted)
                self._closed = False
                self._hold_w, self._hold_t, self._hold_z = [], [], []
                self._pose_close = None
        return ev


@dataclass
class LiveLabeler:
    """Holds the live kit state and turns gripper events into cockpit progress.

    Thread-safe: the feed thread calls push(); the HTTP server reads state().
    """
    open_ref: float | None = None
    closed_ref: float | None = None
    _lock: threading.Lock = field(default_factory=threading.Lock)
    packets: list[dict] = field(default_factory=list)
    ti: int = 0
    seeded: bool = False
    locked: bool = False   # once the operator records, freeze the scanned kit (stop auto-reseed)
    events: list[dict] = field(default_factory=list)   # live cockpit_events
    stage: str = "idle"
    # Transport gate: a 'success' release only advances the kit pointer if the EE
    # actually carried the packet somewhere. Without this, every re-grasp attempt
    # on the SAME packet steps the cockpit forward — measured at 33% of advances
    # (13 of 39) across the recorded corpus. Set to 0 to disable the gate.
    min_transport_m: float = C.MIN_TRANSPORT_M
    regrasps: int = 0            # re-grasp attempts on the CURRENT packet
    gate_off: bool = False       # True once a release arrived with no EE pose
    # Fired once per GATED successful placement — i.e. exactly when the pointer
    # steps. live_server wires this to rr-session's /record/advance so one
    # grasp-and-place cycle becomes one saved episode with no keypress at all
    # ("grasp" episode mode). None = the operator ends episodes themselves
    # ("full" mode, one episode per box). Called OFF the lock, on a thread, so a
    # slow HTTP call can never stall the joint feed.
    on_place: object | None = None

    def __post_init__(self):
        self._seg = OnlineGripSegmenter(self.open_ref, self.closed_ref)
        self._grasping = False

    def seed(self, packets: list[dict]) -> None:
        with self._lock:
            self.packets = [dict(p) for p in packets]
            for i, p in enumerate(self.packets):
                p.setdefault("status", "pending")
            self.ti = 0
            self.seeded = True
            self.stage = "reach"
            self.events = [{"t": time.monotonic(), "type": "seed",
                            "packets": len(self.packets)}]

    def advance(self) -> None:
        """Operator manually confirmed the current placement (cockpit 'Placed'
        button → POST /advance): mark the current packet placed and step ti."""
        with self._lock:
            if not self.seeded or self.ti >= len(self.packets):
                return
            p = self.packets[self.ti]
            p["status"] = "placed"
            self.events.append({"t": time.monotonic(), "type": "place_confirmed",
                                "bag_id": p.get("bag_id", self.ti + 1),
                                "comp": p.get("comp"), "outcome": "manual"})
            self.ti = min(self.ti + 1, len(self.packets))
            self.regrasps = 0
            self.stage = "reach"

    def skip(self, reason: str = "unidentified") -> None:
        """Step past the current packet WITHOUT claiming it was placed.

        The kit must never be blocked by one bag. When a label cannot be read the
        row stays in the list flagged as unidentified, and this moves the pointer
        on so the rest of the kit can be worked. It is deliberately NOT advance():
        `skipped` is a distinct status from `placed`, so a skipped bag is never
        counted as done, never emits place_confirmed, and stays visible as
        outstanding work for whoever reconciles the kit.
        """
        with self._lock:
            if not self.seeded or self.ti >= len(self.packets):
                return
            p = self.packets[self.ti]
            p["status"] = "skipped"
            p["skip_reason"] = reason
            self.events.append({"t": time.monotonic(), "type": "skipped",
                                "bag_id": p.get("bag_id", self.ti + 1),
                                "comp": p.get("comp"), "outcome": reason})
            self.ti = min(self.ti + 1, len(self.packets))
            self.regrasps = 0
            self.stage = "reach"

    def push(self, t: float, width_raw: float, ee_pos=None) -> None:
        """Feed one robot sample. ``ee_pos`` = end-effector [x, y, z] in metres.

        Advance rule (see the ASCII gate diagram in segmentation.transported):

            release
               │
               ├── outcome != success ──────────────▶ grasp_failed, HOLD pointer
               │
               └── outcome == success
                      │
                      ├── EE moved >= min_transport_m ──▶ place_confirmed, ti++
                      │
                      └── EE barely moved ─────────────▶ regrasp, HOLD pointer
                                                          (operator needed 2 tries
                                                           on the SAME packet)
        """
        fire_place = False
        with self._lock:
            if not self.seeded:
                return
            ev = self._seg.push(t, width_raw, ee_pos)
            if ev is None:
                return
            if ev.kind == "close":
                self._grasping = True
                self.stage = "grasp"
                self.events.append({"t": t, "type": "grasp", "bag_ti": self.ti})
            elif ev.kind == "release":
                self.stage = "place"
                # No EE pose → the gate cannot run. Fail OPEN (behave as before) but
                # latch a flag so the cockpit shows the gate is off rather than
                # silently reverting to advancing on every re-grasp.
                if ev.dxy_m is None:
                    self.gate_off = True
                # Mirrors fuse._is_terminal exactly: success + not-known-unlifted +
                # actually transported. Dropping the `lifted is not False` term made
                # the live labeler advance on a grasp the offline labeler rejected.
                moved = transport_ok(ev.dxy_m, self.min_transport_m)
                placed = (ev.outcome == "success" and ev.lifted is not False and moved)
                if placed and self.ti < len(self.packets):
                    p = self.packets[self.ti]
                    p["status"] = "placed"
                    self.events.append({
                        "t": t, "type": "place_confirmed",
                        "bag_id": p.get("bag_id", self.ti + 1),
                        "comp": p.get("comp"), "outcome": "success",
                        "dxy_m": ev.dxy_m,
                    })
                    self.ti = min(self.ti + 1, len(self.packets))
                    self.regrasps = 0
                    self.stage = "reach"
                    fire_place = True
                elif ev.outcome == "success":  # grasped but never carried it anywhere
                    # Grasped and released without carrying it anywhere = a re-grip
                    # at the pick. Same packet, try again. Counted so the operator
                    # can tell "held on purpose" from "frozen / dead gripper".
                    self.regrasps += 1
                    self.events.append({"t": t, "type": "regrasp", "bag_ti": self.ti,
                                        "dxy_m": ev.dxy_m})
                    self.stage = "reach"
                else:  # slip/empty — stay on the same bag (regrasp coming)
                    self.regrasps += 1
                    self.events.append({"t": t, "type": "grasp_failed",
                                        "bag_ti": self.ti, "outcome": ev.outcome})
                self._grasping = False

        # OUTSIDE the lock, on its own thread: this reaches across to rr-session
        # over HTTP, and the joint feed must never wait on a socket.
        if fire_place and self.on_place is not None:
            threading.Thread(target=self._fire_place, daemon=True).start()

    def _fire_place(self) -> None:
        try:
            self.on_place()
        except Exception:
            # A failed auto-advance must never take down the labeler: the
            # operator can always still end the episode by hand.
            pass

    def state(self) -> dict:
        with self._lock:
            return {
                "seeded": self.seeded,
                "locked": self.locked,
                # This is a WHITELIST — a field not named here never reaches the
                # cockpit. `ident`/`read`/`skip_reason` carry the not-identified
                # flagging, so they have to be listed explicitly.
                "packets": [{"part": p.get("part"), "name": p.get("name"),
                             "comp": p.get("comp"), "bbox": p.get("bbox"),
                             "status": p.get("status", "pending"),
                             "ident": p.get("ident", "ok"),
                             "read": p.get("read"),
                             "skip_reason": p.get("skip_reason")}
                            for p in self.packets],
                "ti": self.ti,
                "stage": self.stage,
                "done": sum(1 for p in self.packets if p.get("status") == "placed"),
                # Skipped is NOT done: counted separately so a kit finished with
                # unread bags can never read as a clean 5/5.
                "skipped": sum(1 for p in self.packets if p.get("status") == "skipped"),
                "unknown": sum(1 for p in self.packets if p.get("ident") == "unknown"),
                "total": len(self.packets),
                # Re-grasp attempts on the CURRENT packet. The cockpit shows this so a
                # deliberately-held pointer is visibly different from a frozen one.
                "regrasps": self.regrasps,
                # True once a release arrived with no EE pose: the transport gate is
                # not running and re-grasps will advance the pointer again.
                "gate_off": self.gate_off,
                "min_transport_m": self.min_transport_m,
            }
