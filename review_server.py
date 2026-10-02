#!/usr/bin/env python3
"""Static server for the review dashboard, with HTTP Range support.

stdlib http.server ignores Range, so a browser can only stream an episode mp4
from the start — no seeking, and the big camera_scan files (hundreds of MB) stall
the page. This adds 206 Partial Content so the <video> scrubber works.

Usage:  python3 review_server.py <serve_dir> <port>
"""
from __future__ import annotations
import json, os, re, sys, threading
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

RANGE_RE = re.compile(r"bytes=(\d*)-(\d*)")

# Serializes the read-modify-write of corrections.json across request threads.
_WRITE_LOCK = threading.Lock()


class RangeHandler(SimpleHTTPRequestHandler):
    protocol_version = "HTTP/1.1"   # keep-alive: video seeks fire many small requests

    def send_head(self):
        rng = self.headers.get("Range")
        if not rng:
            return super().send_head()
        path = self.translate_path(self.path)
        if os.path.isdir(path):
            return super().send_head()
        m = RANGE_RE.fullmatch(rng.strip())
        if not m:
            return super().send_head()
        try:
            f = open(path, "rb")
        except OSError:
            self.send_error(404, "File not found")
            return None
        size = os.fstat(f.fileno()).st_size
        lo, hi = m.group(1), m.group(2)
        if lo:                                   # bytes=lo-[hi]
            start = int(lo)
            end = int(hi) if hi else size - 1
        else:                                    # bytes=-suffix
            start = max(0, size - int(hi or 0))
            end = size - 1
        end = min(end, size - 1)
        if start > end or start >= size:
            f.close()
            self.send_response(416)
            self.send_header("Content-Range", f"bytes */{size}")
            self.send_header("Content-Length", "0")   # keep-alive: client must not wait for a body
            self.end_headers()
            return None
        f.seek(start)
        self.send_response(206)
        self.send_header("Content-Type", self.guess_type(path))
        self.send_header("Content-Range", f"bytes {start}-{end}/{size}")   # Accept-Ranges: end_headers
        self.send_header("Content-Length", str(end - start + 1))
        self.end_headers()
        return _Slice(f, end - start + 1)

    def end_headers(self):
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Cache-Control", "no-cache")
        super().end_headers()

    # ── operator verdicts ────────────────────────────────────────────────────
    #  POST /corrections  {"dir": "<date>/episode_x", "key": "grasp_attempts/1:2",
    #                      "verdict": "ok"|"wrong"}
    #        │
    #        ├─ resolve under the recordings symlink, reject anything outside
    #        ├─ merge into <episode>/corrections.json  (schema that
    #        │   review_corpus.load_ann already knows how to apply)
    #        └─ 200 {"ok": true}
    #
    # Written to disk on every click rather than batched: review_watch.sh
    # regenerates the page on each saved episode, and anything held only in the
    # browser would be silently lost mid-review.
    def do_POST(self):
        if self.path.split("?")[0] != "/corrections":
            self._json(404, {"ok": False, "error": "unknown endpoint"})
            return
        try:
            n = int(self.headers.get("Content-Length") or 0)
            if n <= 0 or n > 64_000:
                raise ValueError("bad body size")
            req = json.loads(self.rfile.read(n))
            ep_dir = self._safe_episode_dir(req.get("dir") or "")
            key = str(req.get("key") or "")
            verdict = str(req.get("verdict") or "")
            section, _, item = key.partition("/")
            if section not in ("grasp_attempts", "place_events") or not item:
                raise ValueError("bad key")
            if verdict not in ("ok", "wrong"):
                raise ValueError("bad verdict")
        except Exception as e:
            self._json(400, {"ok": False, "error": str(e)})
            return

        path = os.path.join(ep_dir, "corrections.json")
        # ThreadingHTTPServer runs these in parallel, and clicking two rows in
        # quick succession really does put two POSTs in flight. Without the lock
        # the read-modify-write races: both threads read the same file and the
        # second write silently drops the first verdict. The unique temp name is
        # the other half — a shared "<path>.tmp" means one thread's os.replace
        # finds the file already renamed by the other and fails with ENOENT.
        with _WRITE_LOCK:
            try:
                data = json.load(open(path)) if os.path.exists(path) else {}
            except Exception:
                data = {}                   # unreadable/corrupt: start clean, never crash
            if not isinstance(data, dict):
                data = {}
            data.setdefault(section, {}).setdefault(item, {})["operator_verdict"] = verdict
            tmp = f"{path}.{os.getpid()}.{threading.get_ident()}.tmp"
            try:
                with open(tmp, "w") as f:
                    json.dump(data, f, indent=2)
                os.replace(tmp, path)       # atomic: a regen never reads a half-written file
            except OSError as e:
                try:
                    os.unlink(tmp)
                except OSError:
                    pass
                self._json(500, {"ok": False, "error": str(e)})
                return
        self._json(200, {"ok": True})

    def _safe_episode_dir(self, rel: str) -> str:
        """Resolve <serve_dir>/recordings/<rel> and refuse anything that escapes it.

        The dashboard binds 0.0.0.0 (review_watch.sh), so this is the one write
        path reachable from the LAN. '../' or an absolute path must not be able
        to place a corrections.json anywhere on the box.
        """
        rec_root = os.path.realpath(os.path.join(os.getcwd(), "recordings"))
        cand = os.path.realpath(os.path.join(rec_root, rel))
        if cand != rec_root and not cand.startswith(rec_root + os.sep):
            raise ValueError("path escapes recordings/")
        if not os.path.isdir(cand):
            raise ValueError("no such episode")
        return cand

    def _json(self, code: int, obj: dict) -> None:
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *a):                   # keep the watcher's console clean
        pass


class _Slice:
    """File wrapper that stops after n bytes (copyfile() reads until EOF)."""

    def __init__(self, f, n):
        self.f, self.left = f, n

    def read(self, sz=-1):
        if self.left <= 0:
            return b""
        if sz is None or sz < 0:
            sz = self.left
        data = self.f.read(min(sz, self.left))
        self.left -= len(data)
        return data

    def close(self):
        self.f.close()


if __name__ == "__main__":
    serve_dir, port = sys.argv[1], int(sys.argv[2])
    os.chdir(serve_dir)
    # THREADING, not HTTPServer: with HTTP/1.1 keep-alive a loaded <video> holds
    # its connection open, and on a single-threaded server the verdict POST would
    # queue behind it and appear to hang.
    ThreadingHTTPServer(("0.0.0.0", port), RangeHandler).serve_forever()
