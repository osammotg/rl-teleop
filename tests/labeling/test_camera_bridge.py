"""Camera bridge: JPEG encode + /cam/<id> serving (no live bus needed)."""
from __future__ import annotations

import urllib.request

import numpy as np

from robots_realtime.labeling.live import LiveLabeler
from robots_realtime.labeling.live_server import LiveLabelServer, encode_frame_jpeg


def test_encode_frame_jpeg_valid():
    frame = np.random.randint(0, 255, (48, 64, 3), dtype=np.uint8)
    jpg = encode_frame_jpeg(frame)
    assert jpg is not None
    assert jpg[:2] == b"\xff\xd8" and jpg[-2:] == b"\xff\xd9"   # JPEG SOI/EOI


def test_encode_rejects_non_rgb():
    assert encode_frame_jpeg(np.zeros((10, 10), dtype=np.uint8)) is None       # 2D
    assert encode_frame_jpeg(np.zeros((10, 10, 4), dtype=np.uint8)) is None     # RGBA


class _FakeBridge:
    """Duck-typed bridge: serves a fixed JPEG for cam id 'top'.

    ``state()`` is part of the interface now, not an optional extra: /cam/<id>
    returns 503 with an ``X-Cam-State`` header naming WHY there is no picture,
    because "unmapped", "no_data" and "stale" are three different operator
    problems (wrong --bus-cams, node never started, camera died mid-session) and
    a bare 503 makes them look like one.
    """
    def __init__(self):
        self._jpg = encode_frame_jpeg(np.full((32, 32, 3), 128, np.uint8))
    def jpeg(self, cam_id):
        return self._jpg if cam_id == "top" else None
    def state(self, cam_id):
        return {"id": cam_id, "state": "ok" if cam_id == "top" else "unmapped"}


def test_cam_endpoint_serves_jpeg():
    lab = LiveLabeler()
    lab.seed([{"part": "P1", "comp": 1, "bag_id": 1}])
    srv = LiveLabelServer(lab, port=8796, bridge=_FakeBridge())
    srv.start_background()
    try:
        # default = single JPEG (Firefox-safe); cockpit re-polls with ?t=
        r = urllib.request.urlopen("http://127.0.0.1:8796/cam/top?t=1", timeout=2)
        assert r.headers["Content-Type"] == "image/jpeg"
        assert r.headers["Access-Control-Allow-Origin"] == "*"
        body = r.read()
        assert body[:2] == b"\xff\xd8"           # a JPEG
        # ?stream=1 opts into MJPEG
        s = urllib.request.urlopen("http://127.0.0.1:8796/cam/top?stream=1", timeout=2)
        assert "multipart/x-mixed-replace" in s.headers["Content-Type"]
        assert b"--frame" in s.read(8192); s.close()
        # unknown cam id → 503 (no source)
        try:
            urllib.request.urlopen("http://127.0.0.1:8796/cam/nope", timeout=2)
            assert False, "expected 503"
        except urllib.error.HTTPError as e:
            assert e.code == 503
            # and it says WHY, so an unmapped panel is distinguishable from a
            # camera that died mid-session.
            assert e.headers.get("X-Cam-State") == "unmapped"
    finally:
        srv.shutdown()
