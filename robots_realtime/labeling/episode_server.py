"""Episode archive server — the cockpit's window onto what has already been recorded.

WHY THIS IS A SEPARATE PROCESS
------------------------------
The obvious home for this was live_server (:8791) — it already knows --save-root
and the cockpit already polls it. But live_server runs *inside* a recording
session: adding an endpoint there means restarting it, which means stopping a
session that is mid-take. This server holds no session state and touches no
hardware, so it can be started, restarted and killed at any point during a
recording without the recorder noticing. That property is worth one more port.

It is also deliberately read-mostly. The single mutating action is delete, and
delete is a MOVE into <root>/.trash/<date>/<episode>, never an rmtree. An
operator culling takes at 7 a.m. on a rig is exactly the person who should be
able to undo it, and a rename on the same filesystem is atomic and instant even
for a 70 MB episode.

Endpoints
---------
    GET  /health                              liveness
    GET  /episodes[?date=YYYYMMDD&limit=N]    newest first, with a summary per take
    GET  /episodes/<date>/<id>/media/<file>   mp4/json with HTTP Range (video seeking)
    GET  /episodes/<date>/<id>/thumb?cam=top  cached JPEG poster frame (ffmpeg)
    POST /episodes/<date>/<id>/delete         → .trash  (reversible)
    POST /episodes/<date>/<id>/restore        ← .trash

Run:
    uv run python -m robots_realtime.labeling.episode_server --save-root recordings
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

# Path components arriving from the browser are matched against these BEFORE
# they are ever joined to a filesystem path. Anything that does not match is a
# 404 — no normalisation, no "..", no symlink games.
DATE_RE = re.compile(r"^\d{8}$")
EP_RE = re.compile(r"^episode_[A-Za-z0-9_.\-]+$")
MEDIA_RE = re.compile(r"^[A-Za-z0-9_.\-]+\.(mp4|json|npy)$")
RANGE_RE = re.compile(r"bytes=(\d*)-(\d*)")

TRASH_DIR = ".trash"
THUMB_DIR = ".thumbs"

# Camera topic → (stable key the cockpit uses, human label, sort order).
# The recorder names files "<topic>-images-rgb.mp4"; the cockpit talks in
# scan/top/wristL because that is what its live panels are called.
CAMERAS = {
    "camera_scan": ("scan", "Scan", 0),
    "camera_top": ("top", "Übersicht", 1),
    "camera_left": ("wristL", "Handgelenk L", 2),
    "camera_right": ("wristR", "Handgelenk R", 3),
}

_MIME = {".mp4": "video/mp4", ".json": "application/json", ".npy": "application/octet-stream",
         ".jpg": "image/jpeg"}


# ---------------------------------------------------------------------------
# summarising one episode
# ---------------------------------------------------------------------------

def _dir_bytes(d: Path) -> int:
    total = 0
    for f in d.iterdir():
        try:
            if f.is_file():
                total += f.stat().st_size
        except OSError:
            pass
    return total


def _read_json(p: Path):
    try:
        return json.loads(p.read_text())
    except Exception:
        return None


def _duration_s(d: Path, meta: dict | None) -> float | None:
    """How long the take actually ran.

    The camera timestamp sidecars are the honest answer — they are the wall
    clock of the frames that were written. They are a few KB, so reading one
    costs nothing. Fall back to (mcap mtime − episode start), which is right to
    within the flush, and finally to None rather than guessing.
    """
    best = None
    for ts in sorted(d.glob("*-timestamp.npy")):
        try:
            import numpy as np
            a = np.load(ts, mmap_mode="r")
            if a.size >= 2:
                span = float(a[-1]) - float(a[0])
                if span > 0 and (best is None or span > best):
                    best = span
        except Exception:
            continue
    if best is not None:
        return round(best, 2)
    start = (meta or {}).get("episode_start_time")
    if start:
        for m in d.glob("yam_*.mcap"):
            try:
                return round(m.stat().st_mtime - float(start), 2)
            except OSError:
                pass
    return None


def _clock_from_id(ep_id: str) -> str:
    """episode_135700_4b04dc48 → 13:57:00. The dir name is the only timestamp a
    crashed take is guaranteed to have."""
    parts = ep_id.split("_")
    if len(parts) >= 2 and len(parts[1]) == 6 and parts[1].isdigit():
        h, m, s = parts[1][:2], parts[1][2:4], parts[1][4:]
        return f"{h}:{m}:{s}"
    return "—"


def _date_label(date: str) -> str:
    return f"{date[6:8]}.{date[4:6]}.{date[0:4]}" if DATE_RE.match(date) else date


def _annotations_path(d: Path, arm: str) -> Path:
    """This arm's annotations file. Delegates to label_episode, which owns the
    naming, so the two can never drift apart. Imported lazily: label_episode
    pulls in numpy and the FK stack, and the archive is meant to start fast."""
    from robots_realtime.labeling.label_episode import annotations_path
    return annotations_path(d, arm)


def summarise(d: Path, arm: str) -> dict:
    """Everything the archive card needs, from the files on disk alone."""
    date, ep_id = d.parent.name, d.name
    meta = _read_json(d / "session_meta.json")
    # Arm-aware: label_episode writes annotations.json for LEFT and
    # annotations_<arm>.json for every other arm (label_episode.annotations_path
    # is the authority). This used to hardcode the left name, so a right-arm
    # episode came back labeled=False / grasps=0 even when its annotations
    # existed — and the archive card renders as an unlabelled stub. Silent, and
    # indistinguishable from "the labeller found nothing".
    ann = _read_json(_annotations_path(d, arm))
    kit = _read_json(d / "kit.json")

    cams = []
    for f in sorted(d.glob("*-images-rgb.mp4")):
        topic = f.name.split("-images-rgb.mp4")[0]
        key, label, order = CAMERAS.get(topic, (topic, topic, 9))
        try:
            size = f.stat().st_size
        except OSError:
            size = 0
        cams.append({"key": key, "label": label, "file": f.name, "bytes": size, "_o": order})
    cams.sort(key=lambda c: c["_o"])
    for c in cams:
        c.pop("_o", None)

    grasps = (ann or {}).get("grasp_attempts") or []
    places = (ann or {}).get("place_events") or []
    # Two different things, both called "flags" on disk and kept apart here:
    #   operator_flags.json  what the OPERATOR pressed during the take (g/x/s)
    #   annotations.flags    what the LABELER could not explain (unplaced grasp…)
    # Collapsing them would let a labeler anomaly read as an operator verdict.
    op_flags = [f.get("tag") for f in ((_read_json(d / "operator_flags.json") or {}).get("flags") or [])
                if isinstance(f, dict) and f.get("tag")]
    anomalies = [{"kind": f.get("kind"), "detail": f.get("detail")}
                 for f in ((ann or {}).get("flags") or []) if isinstance(f, dict)]

    # An episode is COMPLETE once the recorder flushed its mcap. A dir without
    # one is either recording right now or was orphaned by a crash — worth
    # showing (so it can be culled) but never worth mistaking for data.
    mcap = d / f"yam_{arm}.mcap"
    try:
        complete = mcap.exists() and mcap.stat().st_size > 0
    except OSError:
        complete = False

    started = (meta or {}).get("episode_start_time")
    try:
        mtime = d.stat().st_mtime
    except OSError:
        mtime = 0.0

    return {
        "id": ep_id,
        "date": date,
        "date_label": _date_label(date),
        "time_label": _clock_from_id(ep_id),
        "started_at": started,
        "mtime": mtime,
        "duration_s": _duration_s(d, meta),
        "bytes": _dir_bytes(d),
        "complete": complete,
        "labeled": ann is not None,
        "grasps": len(grasps),
        "grasps_ok": sum(1 for g in grasps if g.get("outcome") == "success"),
        "places": len(places),
        "places_on_target": sum(1 for p in places if p.get("in_target_region")),
        "flags": op_flags,
        "anomalies": anomalies,
        "outcome": ((ann or {}).get("episode_meta") or {}).get("outcome") or "",
        "instruction": (meta or {}).get("instruction") or "",
        # kit.json is written in the episode schema's vocabulary (part_no /
        # compartment); the cockpit's live /state uses part / comp. Accept both
        # so a card reads the same whichever produced the file.
        "kit": [{"part": k.get("part_no") or k.get("part"),
                 "name": k.get("name"),
                 "comp": k.get("compartment") if k.get("compartment") is not None else k.get("comp")}
                for k in (kit or [])],
        "cameras": cams,
    }


class Archive:
    """Lists episodes, with a cache so a 4 Hz cockpit poll stays free.

    An episode's summary only changes when its directory does, so the directory
    mtime is the cache key. 29 takes today, a few hundred by the end of the
    project — this keeps a listing at a couple of stat() calls per take instead
    of re-reading every annotations.json.
    """

    def __init__(self, root: Path, arm: str = "left"):
        self.root = root
        self.arm = arm
        self._cache: dict[str, tuple[float, dict]] = {}
        self._lock = threading.Lock()

    def _episode_dirs(self, base: Path) -> list[Path]:
        if not base.exists():
            return []
        out = []
        for day in base.iterdir():
            if not day.is_dir() or day.name in (TRASH_DIR, THUMB_DIR) or not DATE_RE.match(day.name):
                continue
            for ep in day.iterdir():
                if ep.is_dir() and EP_RE.match(ep.name):
                    out.append(ep)
        return out

    def summarise_cached(self, d: Path) -> dict:
        key = str(d)
        try:
            mt = d.stat().st_mtime
        except OSError:
            mt = 0.0
        with self._lock:
            hit = self._cache.get(key)
            if hit and hit[0] == mt:
                return hit[1]
        s = summarise(d, self.arm)
        with self._lock:
            self._cache[key] = (mt, s)
        return s

    def list(self, date: str | None = None, limit: int | None = None,
             trashed: bool = False) -> dict:
        base = self.root / TRASH_DIR if trashed else self.root
        eps = [self.summarise_cached(d) for d in self._episode_dirs(base)]
        # Newest first. started_at is the true clock; mtime carries takes whose
        # session_meta never made it to disk.
        eps.sort(key=lambda e: (e.get("started_at") or e.get("mtime") or 0), reverse=True)

        dates: dict[str, dict] = {}
        for e in eps:
            slot = dates.setdefault(e["date"], {"date": e["date"], "label": e["date_label"],
                                                "count": 0, "bytes": 0})
            slot["count"] += 1
            slot["bytes"] += e.get("bytes") or 0
        date_list = sorted(dates.values(), key=lambda d: d["date"], reverse=True)

        if date:
            eps = [e for e in eps if e["date"] == date]
        if limit:
            eps = eps[:limit]
        for e in eps:
            e["trashed"] = trashed

        trash_n = len(self._episode_dirs(self.root / TRASH_DIR)) if not trashed else len(eps)
        return {"ok": True, "root": str(self.root), "arm": self.arm,
                "dates": date_list, "episodes": eps, "trash_count": trash_n,
                "server_time": time.time()}

    # -- paths -------------------------------------------------------------
    def episode_dir(self, date: str, ep_id: str, trashed: bool = False) -> Path | None:
        if not DATE_RE.match(date or "") or not EP_RE.match(ep_id or ""):
            return None
        base = self.root / TRASH_DIR if trashed else self.root
        d = base / date / ep_id
        return d if d.is_dir() else None

    # -- the one mutating action ------------------------------------------
    def trash(self, date: str, ep_id: str) -> dict:
        d = self.episode_dir(date, ep_id)
        if d is None:
            return {"ok": False, "error": "no such episode"}
        # Refuse anything that looks like it is being written right now. The
        # cockpit already hides delete on an incomplete take; this is the guard
        # that does not depend on the UI being correct.
        try:
            if time.time() - d.stat().st_mtime < 20.0:
                return {"ok": False, "error": "episode is still being written — "
                                              "use ✕ Verwerfen on the recorder instead"}
        except OSError:
            pass
        dest = self.root / TRASH_DIR / date / ep_id
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists():
            dest = dest.with_name(f"{ep_id}.{int(time.time())}")
        try:
            shutil.move(str(d), str(dest))
        except Exception as e:
            return {"ok": False, "error": f"move failed: {e}"}
        with self._lock:
            self._cache.pop(str(d), None)
        print(f"[archive] trashed {date}/{ep_id} → {dest}")
        return {"ok": True, "trashed_to": str(dest)}

    def restore(self, date: str, ep_id: str) -> dict:
        d = self.episode_dir(date, ep_id, trashed=True)
        if d is None:
            return {"ok": False, "error": "not in trash"}
        dest = self.root / date / ep_id
        if dest.exists():
            return {"ok": False, "error": "an episode with that id is already back"}
        dest.parent.mkdir(parents=True, exist_ok=True)
        try:
            shutil.move(str(d), str(dest))
        except Exception as e:
            return {"ok": False, "error": f"move failed: {e}"}
        with self._lock:
            self._cache.pop(str(d), None)
        print(f"[archive] restored {date}/{ep_id}")
        return {"ok": True, "restored_to": str(dest)}


# ---------------------------------------------------------------------------
# thumbnails
# ---------------------------------------------------------------------------

def thumb_path(root: Path, date: str, ep_id: str, cam_file: str) -> Path:
    return root / THUMB_DIR / date / f"{ep_id}__{cam_file}.jpg"


def make_thumb(src: Path, dst: Path, at_s: float = 1.0) -> bool:
    """One 360px-wide JPEG per episode-camera, cached outside the episode dir.

    Letting the browser poster-frame the mp4 instead means every card in the
    grid opens a video decode and range-fetches the moov atom — fine for three
    cards, not for thirty on a rig machine that is also running teleop.
    """
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(".tmp.jpg")
    cmd = ["ffmpeg", "-nostdin", "-loglevel", "error", "-ss", str(at_s), "-i", str(src),
           "-frames:v", "1", "-vf", "scale=360:-2", "-q:v", "5", "-y", str(tmp)]
    try:
        # nice: a thumbnail must never compete with the live camera bridge.
        subprocess.run(["nice", "-n", "19"] + cmd, timeout=25, check=False,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception:
        return False
    if not tmp.exists() or tmp.stat().st_size == 0:
        # A take shorter than the seek point yields nothing — retry at frame 0.
        try:
            subprocess.run(["nice", "-n", "19", "ffmpeg", "-nostdin", "-loglevel", "error",
                            "-i", str(src), "-frames:v", "1", "-vf", "scale=360:-2",
                            "-q:v", "5", "-y", str(tmp)],
                           timeout=25, check=False,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception:
            return False
    if tmp.exists() and tmp.stat().st_size > 0:
        os.replace(tmp, dst)
        return True
    tmp.unlink(missing_ok=True)
    return False


# ---------------------------------------------------------------------------
# HTTP
# ---------------------------------------------------------------------------

def make_handler(archive: Archive):
    class Handler(BaseHTTPRequestHandler):
        # Video seeking fires a burst of small range requests; keep-alive turns
        # that from a connection storm into one connection.
        protocol_version = "HTTP/1.1"

        def log_message(self, *_a):
            return

        def _cors(self):
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Headers", "Content-Type")
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")

        def _json(self, obj, code=200):
            body = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self._cors()
            self.end_headers()
            self.wfile.write(body)

        def _fail(self, msg, code=404):
            self._json({"ok": False, "error": msg}, code)

        def do_OPTIONS(self):
            self.send_response(204)
            self.send_header("Content-Length", "0")
            self._cors()
            self.end_headers()

        # -- GET ------------------------------------------------------------
        def do_GET(self):
            u = urlparse(self.path)
            q = parse_qs(u.query)
            path = u.path.rstrip("/") or "/"

            if path == "/health":
                self._json({"ok": True, "service": "episode archive", "root": str(archive.root)})
                return
            if path == "/episodes":
                date = (q.get("date") or [None])[0]
                limit = (q.get("limit") or [None])[0]
                trashed = (q.get("trash") or ["0"])[0] in ("1", "true", "yes")
                try:
                    limit_i = int(limit) if limit else None
                except ValueError:
                    limit_i = None
                self._json(archive.list(date=date, limit=limit_i, trashed=trashed))
                return

            parts = [p for p in path.split("/") if p]
            # /episodes/<date>/<id>/<what>[/<file>]
            if len(parts) >= 4 and parts[0] == "episodes":
                date, ep_id, what = parts[1], parts[2], parts[3]
                trashed = (q.get("trash") or ["0"])[0] in ("1", "true", "yes")
                d = archive.episode_dir(date, ep_id, trashed)
                if d is None:
                    self._fail("no such episode")
                    return
                if what == "media" and len(parts) == 5:
                    self._send_file(d, parts[4])
                    return
                if what == "annotations":
                    # Same arm-aware lookup as summarise(); see _annotations_path.
                    self._json(_read_json(_annotations_path(d, archive.arm))
                               or {"ok": False, "error": "not labeled"})
                    return
                if what == "thumb":
                    self._send_thumb(d, date, ep_id, (q.get("cam") or [None])[0])
                    return
            self._fail("not found")

        def _send_thumb(self, d: Path, date: str, ep_id: str, cam: str | None):
            files = sorted(d.glob("*-images-rgb.mp4"))
            if not files:
                self._fail("no video")
                return
            src = files[0]
            if cam:
                for f in files:
                    topic = f.name.split("-images-rgb.mp4")[0]
                    if CAMERAS.get(topic, (topic,))[0] == cam or topic == cam:
                        src = f
                        break
            dst = thumb_path(archive.root, date, ep_id, src.name)
            if not dst.exists() and not make_thumb(src, dst):
                self._fail("thumbnail failed", 503)
                return
            self._send_bytes(dst.read_bytes(), "image/jpeg", cache="public, max-age=86400")

        def _send_bytes(self, body: bytes, ctype: str, cache="no-store"):
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", cache)
            self._cors()
            self.end_headers()
            self.wfile.write(body)

        def _send_file(self, d: Path, name: str):
            """Serve an episode file, honouring Range so <video> can seek.

            stdlib's handler ignores Range entirely, which is why the review
            dashboard could only ever play an episode from frame 0 — a 35 MB
            scan-cam file streamed from the start on every scrub.
            """
            if not MEDIA_RE.match(name):
                self._fail("bad file")
                return
            f = d / name
            if not f.is_file():
                self._fail("no such file")
                return
            size = f.stat().st_size
            ctype = _MIME.get(f.suffix, "application/octet-stream")
            rng = self.headers.get("Range")
            start, end = 0, size - 1
            partial = False
            if rng:
                m = RANGE_RE.fullmatch(rng.strip())
                if m:
                    lo, hi = m.group(1), m.group(2)
                    if lo:
                        start = int(lo)
                        end = int(hi) if hi else size - 1
                    else:
                        start = max(0, size - int(hi or 0))
                    end = min(end, size - 1)
                    if start > end or start >= size:
                        self.send_response(416)
                        self.send_header("Content-Range", f"bytes */{size}")
                        self.send_header("Content-Length", "0")
                        self._cors()
                        self.end_headers()
                        return
                    partial = True

            length = end - start + 1
            self.send_response(206 if partial else 200)
            self.send_header("Content-Type", ctype)
            self.send_header("Accept-Ranges", "bytes")
            if partial:
                self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
            self.send_header("Content-Length", str(length))
            self._cors()
            self.end_headers()
            try:
                with f.open("rb") as fh:
                    fh.seek(start)
                    left = length
                    while left > 0:
                        chunk = fh.read(min(256 * 1024, left))
                        if not chunk:
                            break
                        self.wfile.write(chunk)
                        left -= len(chunk)
            except (BrokenPipeError, ConnectionResetError):
                return   # the browser seeked away; normal, not an error

        # -- POST -----------------------------------------------------------
        def do_POST(self):
            u = urlparse(self.path)
            parts = [p for p in u.path.split("/") if p]
            if len(parts) == 4 and parts[0] == "episodes":
                date, ep_id, action = parts[1], parts[2], parts[3]
                if action == "delete":
                    r = archive.trash(date, ep_id)
                    self._json(r, 200 if r.get("ok") else 400)
                    return
                if action == "restore":
                    r = archive.restore(date, ep_id)
                    self._json(r, 200 if r.get("ok") else 400)
                    return
            self._fail("not found")

    return Handler


class EpisodeServer:
    def __init__(self, root: Path, arm: str = "left",
                 host: str = "127.0.0.1", port: int = 8793):
        self.archive = Archive(root, arm)
        self.host, self.port = host, port
        self._httpd: ThreadingHTTPServer | None = None

    def start_background(self) -> bool:
        try:
            self._httpd = ThreadingHTTPServer((self.host, self.port),
                                              make_handler(self.archive))
        except OSError:
            return False
        self._httpd.daemon_threads = True
        threading.Thread(target=self._httpd.serve_forever,
                         name="episode-archive", daemon=True).start()
        return True

    def shutdown(self):
        if self._httpd is not None:
            try:
                self._httpd.shutdown()
                self._httpd.server_close()
            except Exception:
                pass


def main(argv=None):
    ap = argparse.ArgumentParser(description="Serve the recorded-episode archive to the cockpit.")
    ap.add_argument("--save-root", default="recordings", help="rr-session recordings root")
    ap.add_argument("--arm", default="left")
    ap.add_argument("--host", default="127.0.0.1",
                    help="bind address (0.0.0.0 to reach it over Tailscale/LAN)")
    ap.add_argument("--port", type=int, default=8793)
    ap.add_argument("--prewarm", action="store_true",
                    help="generate missing thumbnails in the background at startup")
    args = ap.parse_args(argv)

    root = Path(args.save_root).resolve()
    srv = EpisodeServer(root, arm=args.arm, host=args.host, port=args.port)
    if not srv.start_background():
        print(f"[archive] port {args.port} busy — is another episode_server running?")
        return 1
    n = len(srv.archive.list()["episodes"])
    print(f"[archive] http://{args.host}:{args.port}  root={root}  ({n} episodes)")

    if args.prewarm:
        def _warm():
            for e in srv.archive.list()["episodes"]:
                for c in e["cameras"]:
                    dst = thumb_path(root, e["date"], e["id"], c["file"])
                    if dst.exists():
                        continue
                    src = root / e["date"] / e["id"] / c["file"]
                    if src.exists():
                        make_thumb(src, dst)
            print("[archive] thumbnails warm")
        threading.Thread(target=_warm, daemon=True).start()

    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        srv.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
