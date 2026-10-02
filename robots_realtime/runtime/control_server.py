"""HTTP control surface for a running session — the cockpit's hands.

Why this exists
---------------
Before this, the cockpit and the recorder were two processes that never
spoke. The cockpit's REC button flipped a local JavaScript boolean and
POSTed ``/seed`` to the *labelling* backend (which only freezes the kit
as the scan currently sees it). Nothing ever reached ``rr-session``, so
the operator had to press ``r`` in the terminal, then click REC in the
cockpit, then hit "Refresh feeds" — three actions for one intent, and
the cockpit's idea of "recording" could silently disagree with the
recorder's.

This server runs *inside* the rr-session process, so it holds a direct
reference to the live ``Session``. It calls exactly the same methods the
TUI key handler calls, under the same lock, which makes the cockpit and
the keyboard genuine peers: press ``r`` in the terminal and the cockpit
sees it on its next poll; click REC in the cockpit and the TUI shows it.
Neither is the master.

Deliberately NOT a websocket: the cockpit already polls ``/state`` on the
labelling backend, and a 4 Hz poll of a local HTTP endpoint is free. One
fewer moving part to fail at 7 a.m.

Endpoints
---------
    GET  /status            current recorder state (the poll target)
    GET  /health            liveness, no session touch
    POST /record/start      begin a take            (TUI: r)
    POST /record/save       end + keep              (TUI: 1)
    POST /record/discard    end + throw away        (TUI: d)
    POST /record/advance    idle→start, rec→save    (TUI: RIGHT arrow)
    POST /record/rerecord   discard + restart       (TUI: LEFT arrow)
    POST /pause             toggle hold             (TUI: space)
    POST /flag  {"tag":...} tag current episode     (TUI: g/x/s)

All POSTs return ``{"ok": true, "accepted": bool, ...}``. ``accepted`` is
false when another session action is already in flight — the caller
should re-poll ``/status`` rather than assume the action landed.
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

# The TUI owns the canonical action lock and the advance/rerecord
# semantics; importing them here is what keeps the two control surfaces
# from drifting apart or racing each other.
from robots_realtime.runtime.tui import (
    ACTION_LOCK,
    _advance,
    _default_instruction,
    _rerecord,
    _run_session_action_async,
)

_FLAG_TAGS = {"re_grasp", "bad", "slow"}


def _status_payload(session) -> dict:
    """A snapshot the cockpit can render without further calls."""
    import time

    started = getattr(session, "episode_start_time", None)
    try:
        recording = bool(session.is_recording)
    except Exception:
        recording = False
    try:
        paused = bool(session.is_paused)
    except Exception:
        paused = False

    return {
        "recording": recording,
        "paused": paused,
        "episode_started_at": started,
        "elapsed_s": (time.time() - started) if (recording and started) else 0.0,
        "instruction": str(getattr(session, "instruction", "") or ""),
        "instruction_mappings": dict(getattr(session, "instruction_mappings", {}) or {}),
        "save_root": str(getattr(session, "save_root", "") or ""),
        "busy": ACTION_LOCK.locked(),
    }


def make_handler(session):
    class Handler(BaseHTTPRequestHandler):
        # keep the TUI readable — the default handler logs every request to stderr
        def log_message(self, *_args) -> None:
            return

        def _cors(self) -> None:
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Headers", "Content-Type")
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")

        def _json(self, payload: dict, code: int = 200) -> None:
            body = json.dumps(payload).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self._cors()
            self.end_headers()
            self.wfile.write(body)

        def _body(self) -> dict:
            try:
                n = int(self.headers.get("Content-Length") or 0)
                return json.loads(self.rfile.read(n) or b"{}") if n else {}
            except Exception:
                return {}

        def do_OPTIONS(self) -> None:
            self.send_response(204)
            self._cors()
            self.end_headers()

        def do_GET(self) -> None:
            path = self.path.split("?", 1)[0]
            if path == "/health":
                self._json({"ok": True, "service": "rr-session control"})
            elif path == "/status":
                self._json({"ok": True, **_status_payload(session)})
            else:
                self._json({"ok": False, "error": "not found"}, 404)

        def do_POST(self) -> None:
            path = self.path.split("?", 1)[0]

            # A session action is already running. Say so rather than
            # silently dropping it — the cockpit shows this as "busy"
            # instead of a button that appears to do nothing.
            def fire(fn, *a, **kw) -> None:
                if ACTION_LOCK.locked():
                    self._json({"ok": True, "accepted": False, "reason": "busy",
                                **_status_payload(session)})
                    return
                _run_session_action_async(ACTION_LOCK, fn, *a, **kw)
                self._json({"ok": True, "accepted": True, **_status_payload(session)})

            if path == "/record/start":
                if session.is_recording:
                    self._json({"ok": True, "accepted": False, "reason": "already recording",
                                **_status_payload(session)})
                    return
                fire(session.start_episode)

            elif path == "/record/save":
                if not session.is_recording:
                    self._json({"ok": True, "accepted": False, "reason": "not recording",
                                **_status_payload(session)})
                    return
                instruction = self._body().get("instruction") or _default_instruction(session)
                fire(session.end_episode, save=True, instruction=instruction)

            elif path == "/record/discard":
                if not session.is_recording:
                    self._json({"ok": True, "accepted": False, "reason": "not recording",
                                **_status_payload(session)})
                    return
                fire(session.end_episode, save=False)

            elif path == "/record/advance":
                # identical semantics to the RIGHT arrow, by construction
                if ACTION_LOCK.locked():
                    self._json({"ok": True, "accepted": False, "reason": "busy",
                                **_status_payload(session)})
                    return
                _advance(session, ACTION_LOCK)
                self._json({"ok": True, "accepted": True, **_status_payload(session)})

            elif path == "/record/next":
                # save-then-start, atomically. This is what one-episode-per-
                # grasp needs and what /record/advance canNOT give it:
                # advance TOGGLES, so a placement would save the take and leave
                # the recorder idle, losing everything until the next placement.
                # Here the next take begins the instant the previous one closes.
                if ACTION_LOCK.locked():
                    self._json({"ok": True, "accepted": False, "reason": "busy",
                                **_status_payload(session)})
                    return
                instruction = self._body().get("instruction") or _default_instruction(session)

                def _next() -> None:
                    if session.is_recording:
                        session.end_episode(save=True, instruction=instruction)
                    session.start_episode()

                _run_session_action_async(ACTION_LOCK, _next)
                self._json({"ok": True, "accepted": True, **_status_payload(session)})

            elif path == "/record/rerecord":
                if ACTION_LOCK.locked():
                    self._json({"ok": True, "accepted": False, "reason": "busy",
                                **_status_payload(session)})
                    return
                _rerecord(session, ACTION_LOCK)
                self._json({"ok": True, "accepted": True, **_status_payload(session)})

            elif path == "/pause":
                # toggle_pause is cheap and non-blocking; call it inline so
                # the reply already carries the new state
                session.toggle_pause()
                self._json({"ok": True, "accepted": True, **_status_payload(session)})

            elif path == "/flag":
                tag = str(self._body().get("tag") or "")
                if tag not in _FLAG_TAGS:
                    self._json({"ok": False, "error": f"tag must be one of {sorted(_FLAG_TAGS)}"}, 400)
                    return
                fire(session.flag_episode, tag)

            else:
                self._json({"ok": False, "error": "not found"}, 404)

    return Handler


class ControlServer:
    """Owns the HTTP thread. Failure to bind is never fatal to a session."""

    def __init__(self, session, host: str = "127.0.0.1", port: int = 8792):
        self.session = session
        self.host = host
        self.port = port
        self.httpd: ThreadingHTTPServer | None = None
        self.thread: threading.Thread | None = None

    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.port}"

    def start(self) -> bool:
        """Bind and serve in a daemon thread. Returns False if the port is
        taken — a recording session must still run without the cockpit."""
        try:
            self.httpd = ThreadingHTTPServer((self.host, self.port), make_handler(self.session))
        except OSError:
            return False
        self.httpd.daemon_threads = True
        self.thread = threading.Thread(target=self.httpd.serve_forever,
                                       name="rr-control", daemon=True)
        self.thread.start()
        return True

    def stop(self) -> None:
        if self.httpd is not None:
            try:
                self.httpd.shutdown()
                self.httpd.server_close()
            except Exception:
                pass
