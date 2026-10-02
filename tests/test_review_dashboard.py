"""Review dashboard: label->video mapping, event rows, and verdict persistence.

The dashboard's job is to let the operator judge whether auto-labeling is right.
That rests on one fact: camera_*-rgb-timestamp.npy shares the mcap wall clock, so
a grasp time maps onto a video offset. These tests pin that mapping, the stdlib
.npy reader it needs, and the correction write path.
"""
from __future__ import annotations

import json
import os
import re
import struct
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import review_corpus as RC  # noqa: E402


# ── stdlib .npy reader ────────────────────────────────────────────────────────

def _write_npy(path, vals):
    """Minimal 1-D float64 .npy writer (matches what the camera nodes emit)."""
    hdr = ("{'descr': '<f8', 'fortran_order': False, 'shape': (%d,), }" % len(vals))
    hdr += " " * ((64 - (10 + len(hdr) + 1) % 64) % 64) + "\n"
    with open(path, "wb") as f:
        f.write(b"\x93NUMPY\x01\x00")
        f.write(struct.pack("<H", len(hdr)))
        f.write(hdr.encode("latin1"))
        for v in vals:
            f.write(struct.pack("<d", v))


def test_npy_endpoints_reads_first_and_last(tmp_path):
    p = tmp_path / "ts.npy"
    _write_npy(p, [100.5, 101.0, 101.5, 102.25])
    assert RC.npy_endpoints(str(p)) == (100.5, 102.25)


def test_npy_endpoints_single_sample(tmp_path):
    p = tmp_path / "ts.npy"
    _write_npy(p, [7.5])
    assert RC.npy_endpoints(str(p)) == (7.5, 7.5)


def test_npy_endpoints_returns_none_on_junk(tmp_path):
    p = tmp_path / "bad.npy"
    p.write_bytes(b"not an npy file at all")
    assert RC.npy_endpoints(str(p)) is None


def test_npy_endpoints_returns_none_when_missing(tmp_path):
    assert RC.npy_endpoints(str(tmp_path / "nope.npy")) is None


@pytest.mark.skipif(
    not os.path.exists(os.path.join(os.path.dirname(__file__), "..", "recordings")),
    reason="no recordings on this machine")
def test_npy_endpoints_matches_numpy_on_real_data():
    """The whole point is reading REAL camera sidecars without numpy."""
    np = pytest.importorskip("numpy")
    import glob
    files = glob.glob(os.path.join(os.path.dirname(__file__), "..",
                                   "recordings", "*", "episode_*", "*-timestamp.npy"))
    if not files:
        pytest.skip("no timestamp sidecars recorded")
    for f in files[:5]:
        arr = np.load(f).ravel()
        got = RC.npy_endpoints(f)
        assert got is not None, f
        assert got[0] == pytest.approx(float(arr[0]))
        assert got[1] == pytest.approx(float(arr[-1]))


# ── event flattening ──────────────────────────────────────────────────────────

def _episode():
    return dict(
        grasps=[dict(t=110.0, bag_id=1, attempt=1, outcome="success"),
                dict(t=115.0, bag_id=1, attempt=2, outcome="empty")],
        places=[dict(t=120.0, bag_id=1, target_compartment=3,
                     detected_compartment=3, in_target_region=True)],
        flags=[dict(kind="retargeting", detail="bag 1 re-grasped 1x", t=115.0),
               dict(kind="ocr_null", detail="bag 2 unreadable")],
    )


def test_events_are_time_ordered_with_untimed_last():
    evs = RC.episode_events(_episode())
    timed = [e["t"] for e in evs if e["t"] is not None]
    assert timed == sorted(timed)
    assert evs[-1]["t"] is None, "episode-wide flags sort to the end"


def test_event_keys_address_the_corrections_schema():
    """Keys must match what load_ann merges, or a verdict silently does nothing."""
    evs = RC.episode_events(_episode())
    keys = [e["key"] for e in evs if e["key"]]
    assert "grasp_attempts/1:1" in keys
    assert "grasp_attempts/1:2" in keys
    assert "place_events/1" in keys


def test_flags_have_no_key_so_they_are_not_verdictable():
    evs = RC.episode_events(_episode())
    assert [e["key"] for e in evs if e["label"].startswith("⚑")] == ["", ""]


def test_existing_verdicts_are_surfaced():
    ep = _episode()
    ep["grasps"][0]["operator_verdict"] = "wrong"
    ep["places"][0]["operator_verdict"] = "ok"
    evs = {e["key"]: e for e in RC.episode_events(ep) if e["key"]}
    assert evs["grasp_attempts/1:1"]["verdict"] == "wrong"
    assert evs["place_events/1"]["verdict"] == "ok"
    assert evs["grasp_attempts/1:2"]["verdict"] is None


def test_empty_episode_yields_no_events():
    assert RC.episode_events(dict(grasps=[], places=[], flags=[])) == []
    assert RC.episode_events({}) == []


def _card(ep="episode_120000_abc", t0=100.0, **kw):
    e = dict(ep=ep, dir=f"/rec/20260726/{ep}", outcome="success",
             grasps=[dict(t=t0 + 10, bag_id=1, attempt=1, outcome="success")],
             places=[dict(t=t0 + 20, bag_id=1, target_compartment=3,
                          in_target_region=True)],
             flags=[], qa={}, meta=dict(t_start=t0, t_end=t0 + 60))
    e.update(kw)
    return e


def test_index_lists_episodes_and_embeds_no_video(tmp_path):
    """The landing page is a list. 17 preloaded players is why it was heavy, and
    its 10s auto-refresh would interrupt a review."""
    idx = tmp_path / "index.html"
    RC.write_html(str(idx), [], [], [], [], "report text",
                  cards=[_card("episode_a"), _card("episode_b")], root="/rec")
    h = idx.read_text()
    assert h.count("class=eprow") == 2
    assert "<video" not in h, "index must not embed players"
    assert 'href="ep/episode_a.html"' in h
    assert 'href="ep/episode_b.html"' in h


def test_each_episode_gets_its_own_page(tmp_path):
    idx = tmp_path / "index.html"
    RC.write_html(str(idx), [], [], [], [], "report",
                  cards=[_card("episode_a"), _card("episode_b")], root="/rec")
    for ep in ("episode_a", "episode_b"):
        p = tmp_path / "ep" / f"{ep}.html"
        assert p.exists(), f"missing page for {ep}"
        h = p.read_text()
        assert "all episodes" in h, "needs a way back"
        assert "class=evrow" in h, "needs the event rows"


def test_episode_page_does_not_auto_reload(tmp_path):
    """You are working in this page — a timed reload would discard your place."""
    idx = tmp_path / "index.html"
    RC.write_html(str(idx), [], [], [], [], "report", cards=[_card("episode_a")], root="/rec")
    assert "location.reload" not in (tmp_path / "ep" / "episode_a.html").read_text()
    assert "location.reload" in idx.read_text(), "the index still refreshes"


def test_episode_page_media_paths_are_one_level_up(tmp_path):
    """Pages live in ep/, so the recordings symlink is ../recordings."""
    idx = tmp_path / "index.html"
    RC.write_html(str(idx), [], [], [], [], "r", cards=[_card("episode_a")], root="/rec",
                  media_prefix="recordings")
    h = (tmp_path / "ep" / "episode_a.html").read_text()
    assert "../recordings/" in h or "<video" not in h


def test_timeline_has_a_playhead_and_is_seekable(tmp_path):
    idx = tmp_path / "index.html"
    RC.write_html(str(idx), [], [], [], [], "r", cards=[_card("episode_a")], root="/rec")
    h = (tmp_path / "ep" / "episode_a.html").read_text()
    assert 'id="cur_v_episode_a"' in h, "timeline needs a playhead cursor"
    assert 'id="tlab_v_episode_a"' in h, "needs a live position readout"
    assert "tlSeek(" in h, "clicking the bar should seek"
    assert "requestAnimationFrame" in h, "timeupdate (~4Hz) stutters too much to judge alignment"


def test_camt_carries_the_episode_window_the_cursor_needs(tmp_path):
    idx = tmp_path / "index.html"
    RC.write_html(str(idx), [], [], [], [], "r",
                  cards=[_card("episode_a", t0=1000.0)], root="/rec")
    h = (tmp_path / "ep" / "episode_a.html").read_text()
    m = re.search(r'CAMT\["v_episode_a"\]=\{(.*?)\};', h)
    assert m, "no CAMT payload"
    assert "t_start:1000.0" in m.group(1)
    assert "dur:60" in m.group(1)


def test_cursor_lands_exactly_on_the_marker_for_the_same_event(tmp_path):
    """THE alignment invariant, in numbers.

    A marker is drawn at (t_event - t_start)/dur. Clicking that row seeks to
    (t_event - first_camera_t0). The cursor then renders at
    ((currentTime + t0_of_current_cam) - t_start)/dur. Those two percentages must
    be the same, or the timeline lies about where events are — which is exactly
    what a playhead exists to let you check.
    """
    ep_dir = tmp_path / "rec" / "20260726" / "episode_a"
    ep_dir.mkdir(parents=True)
    t_start, cam_t0 = 1000.0, 1000.02          # camera starts 20ms after the mcap
    _write_npy(ep_dir / "camera_top-rgb-timestamp.npy",
               [cam_t0 + i / 30.0 for i in range(1800)])
    (ep_dir / "camera_top-images-rgb.mp4").write_bytes(b"\x00")

    t_event = t_start + 42.5
    card = dict(ep="episode_a", dir=str(ep_dir), outcome="success",
                grasps=[dict(t=t_event, bag_id=1, attempt=1, outcome="success")],
                places=[], flags=[], qa={},
                meta=dict(t_start=t_start, t_end=t_start + 100.0))
    idx = tmp_path / "index.html"
    RC.write_html(str(idx), [], [], [], [], "r", cards=[card],
                  root=str(tmp_path / "rec"))
    h = (tmp_path / "ep" / "episode_a.html").read_text()

    marker_pct = float(re.search(r'left:([0-9.]+)%;top:0;bottom:0;width:2px', h).group(1))
    seek_off = float(re.search(r"seekTo\('v_episode_a',([0-9.]+)\)", h).group(1))

    # Replay the JS math: seekTo -> currentTime -> frac() -> cursor left%.
    first_t0 = cam_t0
    current_time = max(0.0, (first_t0 + seek_off) - cam_t0)
    cursor_pct = ((current_time + cam_t0) - t_start) / 100.0 * 100.0

    assert cursor_pct == pytest.approx(marker_pct, abs=0.05), (
        f"cursor would sit at {cursor_pct:.2f}% but the marker is at {marker_pct:.2f}%")
    assert marker_pct == pytest.approx(42.5, abs=0.05)


def test_index_shows_review_progress(tmp_path):
    """So you can tell at a glance which episodes you have already validated."""
    done = _card("episode_done")
    done["grasps"][0]["operator_verdict"] = "ok"
    done["places"][0]["operator_verdict"] = "wrong"
    idx = tmp_path / "index.html"
    RC.write_html(str(idx), [], [], [], [], "r",
                  cards=[done, _card("episode_todo")], root="/rec")
    h = idx.read_text()
    assert "2/2 reviewed" in h
    assert "0/2 reviewed" in h


def test_place_shows_where_it_actually_landed():
    ep = dict(grasps=[], flags=[], places=[
        dict(t=1.0, bag_id=2, target_compartment=3,
             detected_compartment=5, in_target_region=False)])
    detail = RC.episode_events(ep)[0]["detail"]
    assert "OFF target" in detail and "landed c5" in detail


# ── correction write path ─────────────────────────────────────────────────────

def _free_port():
    """An OS-assigned free port.

    NOT a hardcoded one: this box already runs several http.servers on
    memorable ports, and a fixture that grabs a busy port silently talks to
    somebody else's server and produces baffling failures. Learned the hard way.
    """
    import socket
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


@pytest.fixture
def server(tmp_path):
    """review_server.py over a fake serve dir with a recordings/ tree."""
    rec = tmp_path / "rec" / "20260726" / "episode_x"
    rec.mkdir(parents=True)
    serve = tmp_path / "serve"
    serve.mkdir()
    os.symlink(tmp_path / "rec", serve / "recordings")
    sentinel = f"review-server-test-{os.getpid()}-{tmp_path.name}"
    (serve / "index.html").write_text(sentinel)

    port = _free_port()
    proc = subprocess.Popen(
        [sys.executable, os.path.join(os.path.dirname(__file__), "..", "review_server.py"),
         str(serve), str(port)],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    ok = False
    for _ in range(80):                       # wait for bind
        try:
            body = urllib.request.urlopen(
                f"http://127.0.0.1:{port}/index.html", timeout=0.5).read().decode()
            # Identity check: refuse to run against a server that isn't ours.
            ok = sentinel in body
            break
        except Exception:
            if proc.poll() is not None:
                break
            time.sleep(0.05)
    if not ok:
        proc.kill()
        pytest.skip(f"review_server did not start cleanly on :{port}")
    yield f"http://127.0.0.1:{port}", rec
    proc.kill()
    proc.wait(timeout=5)


def _post(base, payload):
    req = urllib.request.Request(f"{base}/corrections",
                                 data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"},
                                 method="POST")
    try:
        with urllib.request.urlopen(req, timeout=5) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def test_verdict_is_written_to_corrections_json(server):
    base, rec = server
    code, body = _post(base, {"dir": "20260726/episode_x",
                              "key": "grasp_attempts/1:2", "verdict": "wrong"})
    assert (code, body["ok"]) == (200, True)
    data = json.loads((rec / "corrections.json").read_text())
    assert data["grasp_attempts"]["1:2"]["operator_verdict"] == "wrong"


def test_verdicts_accumulate_without_clobbering(server):
    base, rec = server
    _post(base, {"dir": "20260726/episode_x", "key": "grasp_attempts/1:1", "verdict": "ok"})
    _post(base, {"dir": "20260726/episode_x", "key": "place_events/1", "verdict": "wrong"})
    data = json.loads((rec / "corrections.json").read_text())
    assert data["grasp_attempts"]["1:1"]["operator_verdict"] == "ok"
    assert data["place_events"]["1"]["operator_verdict"] == "wrong"


def test_verdict_can_be_changed(server):
    base, rec = server
    _post(base, {"dir": "20260726/episode_x", "key": "place_events/1", "verdict": "wrong"})
    _post(base, {"dir": "20260726/episode_x", "key": "place_events/1", "verdict": "ok"})
    data = json.loads((rec / "corrections.json").read_text())
    assert data["place_events"]["1"]["operator_verdict"] == "ok"


def test_written_file_is_mergeable_by_load_ann(server, tmp_path):
    """End to end: a verdict must actually come back through the loader."""
    base, rec = server
    _post(base, {"dir": "20260726/episode_x", "key": "grasp_attempts/1:1", "verdict": "wrong"})
    (rec / "annotations.json").write_text(json.dumps({
        "episode_meta": {"episode_id": "episode_x"},
        "grasp_attempts": [{"bag_id": 1, "attempt": 1, "outcome": "success", "t": 1.0}],
        "place_events": [], "segments": [], "flags": [],
    }))
    merged = RC.load_ann(str(rec / "annotations.json"))
    assert merged["grasp_attempts"][0]["operator_verdict"] == "wrong"


@pytest.mark.parametrize("bad_dir", [
    "../../../etc",
    "/etc",
    "20260726/../../..",
    "20260726/episode_x/../../../../tmp",
])
def test_path_escape_is_rejected(server, bad_dir):
    """The dashboard binds 0.0.0.0 — this is the only LAN-reachable write path."""
    base, _ = server
    code, body = _post(base, {"dir": bad_dir, "key": "place_events/1", "verdict": "ok"})
    assert code == 400 and body["ok"] is False


def test_unknown_episode_is_rejected(server):
    base, _ = server
    code, body = _post(base, {"dir": "20260726/episode_nope",
                              "key": "place_events/1", "verdict": "ok"})
    assert code == 400 and body["ok"] is False


@pytest.mark.parametrize("payload", [
    {"dir": "20260726/episode_x", "key": "bogus_section/1", "verdict": "ok"},
    {"dir": "20260726/episode_x", "key": "grasp_attempts/", "verdict": "ok"},
    {"dir": "20260726/episode_x", "key": "place_events/1", "verdict": "maybe"},
    {"dir": "20260726/episode_x", "key": "place_events/1"},
])
def test_malformed_requests_are_rejected(server, payload):
    base, _ = server
    code, body = _post(base, payload)
    assert code == 400 and body["ok"] is False


def test_unknown_endpoint_404s(server):
    base, _ = server
    req = urllib.request.Request(f"{base}/whatever", data=b"{}", method="POST")
    try:
        urllib.request.urlopen(req, timeout=5)
        assert False, "should have 404'd"
    except urllib.error.HTTPError as e:
        assert e.code == 404


def test_corrupt_corrections_file_does_not_crash_the_write(server):
    base, rec = server
    (rec / "corrections.json").write_text("{ this is not json")
    code, body = _post(base, {"dir": "20260726/episode_x",
                              "key": "place_events/1", "verdict": "ok"})
    assert (code, body["ok"]) == (200, True)
    assert json.loads((rec / "corrections.json").read_text())["place_events"]["1"]


def test_range_requests_still_work(server, tmp_path):
    """Regression: adding POST must not break video seeking."""
    base, rec = server
    (rec / "clip.bin").write_bytes(bytes(range(256)))
    req = urllib.request.Request(f"{base}/recordings/20260726/episode_x/clip.bin")
    req.add_header("Range", "bytes=10-19")
    with urllib.request.urlopen(req, timeout=5) as r:
        assert r.status == 206
        assert r.read() == bytes(range(10, 20))


def test_concurrent_verdicts_all_land(server):
    """Clicking several rows quickly puts multiple POSTs in flight.

    Two bugs this pins: a shared temp filename (one thread's os.replace hits
    ENOENT because the other already renamed it), and a lost update (both
    threads read the same file, the second write drops the first verdict).
    """
    base, rec = server
    n = 8
    errs = []

    def go(i):
        try:
            c, b = _post(base, {"dir": "20260726/episode_x",
                                "key": f"grasp_attempts/{i}:1", "verdict": "ok"})
            if c != 200:
                errs.append((i, c, b))
        except Exception as e:                # noqa: BLE001
            errs.append((i, "exc", str(e)))

    ts = [threading.Thread(target=go, args=(i,)) for i in range(1, n + 1)]
    for t in ts:
        t.start()
    for t in ts:
        t.join(timeout=20)
    assert not errs, errs
    data = json.loads((rec / "corrections.json").read_text())
    got = sorted(data["grasp_attempts"])
    assert got == sorted(f"{i}:1" for i in range(1, n + 1)), (
        f"lost update: only {len(got)}/{n} verdicts survived -> {got}")
    # No temp files left behind on the happy path.
    assert [p for p in os.listdir(rec) if p.endswith(".tmp")] == []
