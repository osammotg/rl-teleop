#!/usr/bin/env python3
"""Frame-level health check for every camera in a running session.

WHY THIS EXISTS. On 2026-08-10 the cockpit showed a wrist panel that looked
like a working camera and was in fact one frame repeated forever. Every
indicator available to the operator agreed it was fine:

    TUI            camera_right  ● live  29.5 Hz     <- looked healthy
    /cam/wristR    HTTP 200, 21 KB of valid JPEG     <- looked healthy
    recording      998 real frames in the mp4        <- genuinely was fine
    the bus        NOTHING. not one message.         <- the actual truth

The three healthy-looking signals are all measured somewhere OTHER than the
bus, so none of them can see a bus failure:

  * recording never touches the bus. Publisher.publish() calls the writer
    in-process BEFORE the ZMQ send (transport/publisher.py:72-78), so a node
    records perfectly while publishing nothing.
  * /cam serves bridge.jpeg(), which returns the LAST envelope the bridge
    received. With no new messages it returns the same frame forever, at a
    convincing 15 fps.
  * a rate counter counts frames PRODUCED, not frames delivered.

So the only honest test of "is this stream arriving" is: read the bus for a
few seconds and count messages whose CONTENT actually differs. That is what
this does. Identity of frames is the signal — not count, not rate, not HTTP
status.

USAGE
    tools/check_streams.py                 # 6 s check of bus + cockpit
    tools/check_streams.py --secs 20       # longer window
    tools/check_streams.py --watch 30      # re-check every 30 s until Ctrl-C
                                           #   (catches slow degradation)
    tools/check_streams.py --episode DIR   # also verify a recorded episode

Exit status is 0 only if every discovered camera is genuinely streaming, so
this is usable as a gate before a recording session.
"""
from __future__ import annotations

import argparse
import hashlib
import re
import sys
import time
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path

import zmq

from robots_realtime.runtime.transport.serialization import unpack
from robots_realtime.runtime.transport.subscriber import DEFAULT_SUB_PORT

DEFAULT_LAB = "http://localhost:8791"


# ── bus ───────────────────────────────────────────────────────────────────────

def sample_bus(secs: float, port: int = DEFAULT_SUB_PORT) -> dict:
    """Subscribe to EVERYTHING and count messages + distinct payloads per topic.

    Subscribing with an empty filter rather than a per-topic prefix is
    deliberate: a topic that is missing entirely and a topic whose name does
    not match your filter look identical from the outside, and the first
    question this tool has to answer is "is it there at all".

    Distinctness is hashed over the raw frame bytes. A stalled publisher and a
    perfectly still scene both hold `unique` at 1, so a low count is reported
    as SUSPECT rather than FAIL — an unmoving camera pointed at an unmoving
    bench is a legitimate 1.
    """
    ctx = zmq.Context.instance()
    sock = ctx.socket(zmq.SUB)
    sock.connect(f"tcp://127.0.0.1:{port}")
    sock.setsockopt(zmq.SUBSCRIBE, b"")
    counts: Counter = Counter()
    digests: dict[str, set] = defaultdict(set)
    first_seen: dict[str, float] = {}
    t0 = time.time()
    try:
        while time.time() - t0 < secs:
            if not sock.poll(20):
                continue
            while True:                       # drain fully; latest-wins
                try:
                    parts = sock.recv_multipart(zmq.NOBLOCK)
                except zmq.Again:
                    break
                if len(parts) < 2:
                    continue
                topic = parts[0].decode(errors="replace")
                counts[topic] += 1
                first_seen.setdefault(topic, time.time() - t0)
                if not topic.endswith("/rgb"):
                    continue
                try:
                    env = unpack(parts[1])
                except Exception:
                    continue
                data = env.get("data") or {}
                frame = data.get("frame")
                if frame is None:
                    imgs = data.get("images")
                    if isinstance(imgs, dict) and imgs:
                        # Explicit `is None`, never `or`: these are numpy arrays and
                        # `arr or fallback` raises "truth value is ambiguous".
                        frame = imgs.get("rgb")
                        if frame is None:
                            frame = next(iter(imgs.values()))
                if frame is not None:
                    digests[topic].add(hashlib.md5(
                        memoryview(frame).tobytes() if hasattr(frame, "tobytes")
                        else bytes(frame)).hexdigest())
    finally:
        sock.close()
    return {"counts": counts, "unique": {k: len(v) for k, v in digests.items()},
            "first_seen": first_seen, "secs": secs}


# ── cockpit ───────────────────────────────────────────────────────────────────

_CL = re.compile(rb"Content-Length:\s*(\d+)", re.I)


def sample_cockpit(cam_id: str, secs: float, base: str = DEFAULT_LAB) -> dict:
    """Read the MJPEG stream the cockpit reads and count DISTINCT JPEGs.

    Mirrors cockpit-cams.js: find CRLFCRLF, read Content-Length, slice that
    many bytes. Counting frames alone is useless here — a frozen bridge still
    emits a full 15 fps of identical images, which is exactly the failure this
    tool exists to catch.
    """
    url = f"{base}/cam/{cam_id}?stream=1"
    buf = b""
    jpegs: list[bytes] = []
    t0 = time.time()
    try:
        with urllib.request.urlopen(url, timeout=max(2.0, secs + 2)) as r:
            if r.status != 200:
                return {"ok": False, "error": f"HTTP {r.status}"}
            while time.time() - t0 < secs:
                chunk = r.read(65536)
                if not chunk:
                    break
                buf += chunk
                while True:
                    h = buf.find(b"\r\n\r\n")
                    if h < 0:
                        break
                    m = _CL.search(buf[:h])
                    if not m:
                        buf = buf[h + 4:]
                        continue
                    n = int(m.group(1))
                    s, e = h + 4, h + 4 + n
                    if len(buf) < e:
                        break
                    jpegs.append(buf[s:e])
                    buf = buf[e:]
    except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as exc:
        code = getattr(exc, "code", None)
        return {"ok": False, "error": f"HTTP {code}" if code else type(exc).__name__}
    except Exception as exc:                       # noqa: BLE001 - report, don't crash
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
    uniq = len({hashlib.md5(j).hexdigest() for j in jpegs})
    valid = sum(1 for j in jpegs if j[:2] == b"\xff\xd8" and j[-2:] == b"\xff\xd9")
    return {"ok": True, "delivered": len(jpegs), "unique": uniq, "valid_jpeg": valid}


# ── episode ───────────────────────────────────────────────────────────────────

def check_episode(d: Path) -> list[str]:
    """Verify a recorded episode: every camera has video AND it decodes.

    The timestamp sidecar is the tell. AsyncMp4Writer.close() writes one per
    topic that ever received a frame, so a camera with no .npy never had
    write() called at all — the writer was never opened, which is a different
    (and more serious) failure than a short or corrupt video.
    """
    import numpy as np
    out = []
    mp4s = sorted(d.glob("*-images-rgb.mp4"))
    if not mp4s:
        return [f"  FAIL  {d.name}: no camera video at all"]
    for f in mp4s:
        cam = f.name.split("-images-rgb.mp4")[0]
        ts_path = d / f"{cam}-rgb-timestamp.npy"
        if not ts_path.exists():
            out.append(f"  FAIL  {cam}: video but NO timestamp sidecar "
                       f"(writer never received a frame)")
            continue
        ts = np.load(ts_path)
        try:
            import av
            with av.open(str(f)) as c:
                n = sum(1 for _ in c.decode(c.streams.video[0]))
        except Exception as exc:                   # noqa: BLE001
            out.append(f"  FAIL  {cam}: video will not decode: {exc}")
            continue
        dur = float(ts[-1] - ts[0]) if len(ts) > 1 else 0.0
        hz = len(ts) / dur if dur > 0 else 0.0
        tag = "ok  " if n == len(ts) and n > 0 else "FAIL"
        out.append(f"  {tag}  {cam}: {n} frames decoded, {len(ts)} timestamps, "
                   f"{dur:.1f}s, {hz:.1f} Hz"
                   + ("" if n == len(ts) else "  <- MISMATCH"))
    return out


# ── report ────────────────────────────────────────────────────────────────────

def run_once(secs: float, base: str, cams: list[str] | None,
             sub_port: int = DEFAULT_SUB_PORT) -> bool:
    bus = sample_bus(secs, port=sub_port)
    rgb = sorted(t for t in bus["counts"] if t.endswith("/rgb"))
    print(f"── bus ({secs:.0f}s) " + "─" * 46)
    if not rgb:
        print("  FAIL  no camera topics on the bus at all")
    ok = bool(rgb)
    for t in rgb:
        n = bus["counts"][t]
        u = bus["unique"].get(t, 0)
        hz = n / secs
        if u <= 1:
            verdict = "** FROZEN / SUSPECT — identical frames **"
            ok = False
        elif u < n * 0.5:
            verdict = f"** DEGRADED — only {u} distinct of {n} **"
            ok = False
        else:
            verdict = "streaming"
        print(f"  {t:26s} {n:5d} msgs  {hz:5.1f} Hz  {u:5d} distinct   {verdict}")

    # every node publishes _step_hz at 1 Hz; a camera missing it is not looping
    for t in rgb:
        node = t.split("/")[0]
        if f"{node}/_step_hz" not in bus["counts"]:
            print(f"  WARN  {node}: no _step_hz — node loop may be stalled")

    ids = cams or ["top", "wristR"]
    print(f"── cockpit ({secs:.0f}s) " + "─" * 42)
    for cam in ids:
        r = sample_cockpit(cam, secs, base)
        if not r["ok"]:
            print(f"  /cam/{cam:10s} unreachable: {r['error']}")
            continue
        d, u, v = r["delivered"], r["unique"], r["valid_jpeg"]
        if u <= 1:
            verdict = "** FROZEN — same image repeated **"
            ok = False
        elif v != d:
            verdict = f"** {d - v} CORRUPT JPEG **"
            ok = False
        else:
            verdict = "streaming"
        print(f"  /cam/{cam:10s} {d:4d} delivered  {u:4d} distinct  "
              f"{v:4d} valid   {verdict}")
    return ok


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--secs", type=float, default=6.0)
    ap.add_argument("--sub-port", type=int, default=DEFAULT_SUB_PORT,
                    help="bus XPUB port to audit (default 5556, the live bus). Point it "
                         "at a soak bus to audit that instead — without this the tool "
                         "silently audits the WRONG bus and reports 'no camera topics at "
                         "all', which reads like a dead rig rather than a wrong port.")
    ap.add_argument("--base", default=DEFAULT_LAB)
    ap.add_argument("--cam", action="append", dest="cams",
                    help="cockpit cam id (repeatable). Default: top, wristR")
    ap.add_argument("--watch", type=float, metavar="PERIOD",
                    help="re-check every PERIOD seconds until Ctrl-C")
    ap.add_argument("--episode", type=Path, help="also verify a recorded episode")
    a = ap.parse_args(argv)

    if a.episode:
        print(f"── episode {a.episode.name} " + "─" * 34)
        for line in check_episode(a.episode):
            print(line)
        print()

    if not a.watch:
        ok = run_once(a.secs, a.base, a.cams, a.sub_port)
        print(("\nALL STREAMS OK" if ok else "\nPROBLEMS FOUND — see above"))
        return 0 if ok else 1

    print(f"watching every {a.watch:.0f}s — Ctrl-C to stop\n")
    n = 0
    try:
        while True:
            n += 1
            print(f"═══ check #{n}  t+{time.strftime('%H:%M:%S')} " + "═" * 28)
            run_once(a.secs, a.base, a.cams, a.sub_port)
            print()
            time.sleep(max(0.0, a.watch - a.secs))
    except KeyboardInterrupt:
        print("stopped")
    return 0


if __name__ == "__main__":
    sys.exit(main())
