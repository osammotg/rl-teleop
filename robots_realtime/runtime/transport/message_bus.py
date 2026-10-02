"""XPUB/XSUB message broker.

Runs in its own subprocess so its GIL and GC pauses are isolated from all
node processes.  Publishers connect to the XSUB frontend; subscribers connect
to the XPUB backend.

Usage (typically via Session):
    bus = MessageBus()
    bus.start()           # spawns subprocess
    ...
    bus.stop()
"""

from __future__ import annotations

import multiprocessing as mp
import time


DEFAULT_PUB_PORT = 5555   # nodes publish  → connect here
DEFAULT_SUB_PORT = 5556   # nodes subscribe → connect here


def _broker_worker(pub_port: int, sub_port: int, ready_event: mp.Event,
                   err: "mp.Queue") -> None:
    import zmq

    ctx = zmq.Context()
    try:
        xsub = ctx.socket(zmq.XSUB)   # receives from publishers
        xsub.bind(f"tcp://*:{pub_port}")

        xpub = ctx.socket(zmq.XPUB)   # sends to subscribers
        xpub.bind(f"tcp://*:{sub_port}")
    except Exception as e:
        # Report WHY before dying. Without this the parent only sees the
        # ready_event never fire and reports a bare "failed to start within
        # timeout" — which reads like a slow machine but is almost always
        # "another rr-session already owns these ports".
        err.put(f"{type(e).__name__}: {e}")
        ready_event.set()
        return

    ready_event.set()
    zmq.proxy(xsub, xpub)         # blocks forever; proxy handles all routing


class MessageBus:
    """XPUB/XSUB broker running in a dedicated subprocess."""

    def __init__(
        self,
        pub_port: int = DEFAULT_PUB_PORT,
        sub_port: int = DEFAULT_SUB_PORT,
    ) -> None:
        self.pub_port = pub_port
        self.sub_port = sub_port
        self._proc: mp.Process | None = None

    def start(self, timeout: float = 5.0) -> None:
        ready = mp.Event()
        err: mp.Queue = mp.Queue()
        self._proc = mp.Process(
            target=_broker_worker,
            args=(self.pub_port, self.sub_port, ready, err),
            daemon=True,
            name="MessageBus",
        )
        self._proc.start()
        started = ready.wait(timeout)

        # The broker signals `ready` on BOTH paths — bound successfully, or
        # failed and queued the reason. So check the queue before trusting it.
        try:
            reason = err.get_nowait()
        except Exception:
            reason = None

        if reason is not None:
            self.stop()
            hint = ""
            if "Address already in use" in reason:
                hint = (
                    f"\n\nPorts {self.pub_port}/{self.sub_port} are already bound — "
                    "another rr-session is almost certainly still running.\n"
                    "  Find it:  ss -tlnp | grep -E ':(5555|5556)'\n"
                    "  Stop it:  kill <pid>   (or quit that session's TUI with [q])\n"
                    "Or run this one on different ports: "
                    "rr-session <config> --pub-port 5565 --sub-port 5566"
                )
            raise RuntimeError(f"MessageBus failed to start — {reason}{hint}")

        if not started:
            raise RuntimeError(
                f"MessageBus failed to start within {timeout:.0f}s "
                f"(ports {self.pub_port}/{self.sub_port}, no error reported)"
            )

    def stop(self) -> None:
        if self._proc is not None and self._proc.is_alive():
            self._proc.terminate()
            self._proc.join(timeout=2.0)
            self._proc = None

    def __enter__(self) -> "MessageBus":
        self.start()
        return self

    def __exit__(self, *_) -> None:
        self.stop()
