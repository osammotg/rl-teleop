#!/usr/bin/env python3
"""Child process for tools/cold_start_check.py — starts a cameras-only session.

Separate file rather than an inline -c program so it can be read, and so the
safety guard runs in the child too: whatever kills this, it must never have been
able to start a config with a RobotNode in it.

Prints one machine-readable line per event on stdout, then blocks until killed.
"""

from __future__ import annotations

import argparse
import signal
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from robots_realtime.runtime.config import load_session          # noqa: E402
from robots_realtime.runtime.safety_guard import assert_safe_to_soak  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--pub-port", type=int, required=True)
    ap.add_argument("--sub-port", type=int, required=True)
    ap.add_argument("--save-root", required=True)
    ap.add_argument("--warmup", type=float, default=12.0)
    ap.add_argument("--record", action="store_true")
    args = ap.parse_args()

    assert_safe_to_soak(args.config, args.pub_port, args.sub_port, allow_live_ports=True)

    session = load_session(args.config, pub_port=args.pub_port, sub_port=args.sub_port)
    session._save_root = Path(args.save_root)

    # Stop the session on SIGTERM. Python runs `finally` on KeyboardInterrupt but
    # NOT on a default SIGTERM, so `kill <runner pid>` left every camera node
    # orphaned onto init, still holding the bus ports — and the next start then
    # failed with "Address already in use" and a message blaming a running
    # rr-session. A wrong diagnosis manufactured by our own cleanup path.
    def _stop(signum, _frame):
        print(f"SIGNAL {signum} — stopping the session", flush=True)
        try:
            session.stop()
        finally:
            sys.exit(0)

    for _sig in (signal.SIGTERM, signal.SIGINT):
        try:
            signal.signal(_sig, _stop)
        except (OSError, ValueError):
            pass

    session.start()
    print("STARTED", flush=True)
    time.sleep(args.warmup)

    for st in session.node_statuses():
        print(f"NODE {st.name} alive={st.alive} state={st.camera_state}", flush=True)

    if args.record:
        session.start_episode()
        print(f"EPISODE {session._episode_dir}", flush=True)

    print("READY", flush=True)
    while True:          # wait to be killed — that is the point of this program
        time.sleep(0.5)


if __name__ == "__main__":
    sys.exit(main())
