"""bus_feed pump logic (via the testable _feed_from_source) + event writeback."""
from __future__ import annotations

import json
import threading

from robots_realtime.labeling.live import LiveLabeler
from robots_realtime.labeling.live_server import _feed_from_source, write_cockpit_events


def test_feed_from_source_dedups_and_advances():
    lab = LiveLabeler(open_ref=1.0, closed_ref=0.0)
    lab.seed([{"part": "P1", "comp": 5, "bag_id": 1}])

    # a latest-per-topic source that repeats the same ts (like the real subscriber)
    dt = 0.02
    plan = [(1.0, 1.0), (1.0, 0.35), (1.0, 1.0)]     # one grasp-place
    samples, t = [], 0.0
    for dur, w in plan:
        for _ in range(int(dur / dt)):
            samples.append((round(t, 4), w)); t += dt
    # duplicate every sample (subscriber returns latest repeatedly between updates)
    doubled = [s for s in samples for _ in range(2)]
    it = iter(doubled)
    stop = threading.Event()

    def src():
        nxt = next(it, None)
        if nxt is None:
            stop.set()
        return nxt

    _feed_from_source(lab, src, stop=stop, poll_s=0.0)
    s = lab.state()
    assert s["done"] == 1          # dedup on ts → exactly one place, not double-counted
    assert s["ti"] == 1


def test_write_cockpit_events(tmp_path):
    lab = LiveLabeler(open_ref=1.0, closed_ref=0.0)
    lab.seed([{"part": "P1", "comp": 5, "bag_id": 1}])
    t = 0.0
    for dur, w in [(1.0, 1.0), (1.0, 0.35), (1.0, 1.0)]:
        for _ in range(50):
            lab.push(t, w); t += 0.02
    write_cockpit_events(lab, tmp_path / "cockpit_events.jsonl")
    lines = (tmp_path / "cockpit_events.jsonl").read_text().splitlines()
    events = [json.loads(x) for x in lines]
    assert any(e["type"] == "place_confirmed" and e["comp"] == 5 for e in events)
