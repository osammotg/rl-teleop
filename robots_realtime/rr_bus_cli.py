"""Standalone MessageBus broker — the one process that outlives everything else.

Normally a Session spawns its own broker (Session.start -> MessageBus.start),
which binds 5555/5556. That couples bus lifetime to session lifetime: the
second session to start finds the ports taken and dies, so cameras, arm and
every dev tool are forced into a single process that must be restarted as a
unit.

Running the broker here instead inverts that. The bus becomes infrastructure:
sessions attach to it with `rr-session --attach-bus` and come and go freely,
while the camera daemon keeps publishing across every restart. Nothing in the
transport had to change for this -- publishers and subscribers already
`connect()` rather than `bind()` (transport/publisher.py, subscriber.py), so
they neither know nor care who owns the ports.

    rr-bus                          # 5555/5556, the defaults everything expects
    rr-bus --pub-port 5565 --sub-port 5566

Ctrl-C stops it. Any attached session loses its transport when it does, so
this is the process to start first and stop last.
"""

from __future__ import annotations

import argparse
import signal
import sys
import threading

from robots_realtime.runtime.transport.message_bus import (
    DEFAULT_PUB_PORT,
    DEFAULT_SUB_PORT,
    MessageBus,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="rr-bus",
        description="Run the XPUB/XSUB message broker on its own, outliving sessions.",
    )
    parser.add_argument("--pub-port", type=int, default=DEFAULT_PUB_PORT,
                        help=f"XSUB frontend; publishers connect here. Default {DEFAULT_PUB_PORT}.")
    parser.add_argument("--sub-port", type=int, default=DEFAULT_SUB_PORT,
                        help=f"XPUB backend; subscribers connect here. Default {DEFAULT_SUB_PORT}.")
    args = parser.parse_args()

    bus = MessageBus(pub_port=args.pub_port, sub_port=args.sub_port)
    try:
        bus.start()
    except RuntimeError as e:
        # MessageBus.start already explains "Address already in use" and how to
        # find the owner, so pass its message through rather than re-wording it.
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"message bus up  —  publishers → tcp://127.0.0.1:{args.pub_port}   "
          f"subscribers → tcp://127.0.0.1:{args.sub_port}")
    print("attach sessions with:  rr-session <config> --attach-bus")
    print("Ctrl-C to stop (this drops the transport for every attached session).")

    stop = threading.Event()
    signal.signal(signal.SIGINT, lambda *_: stop.set())
    signal.signal(signal.SIGTERM, lambda *_: stop.set())
    try:
        stop.wait()
    finally:
        bus.stop()
        print("\nmessage bus stopped")


if __name__ == "__main__":
    main()
