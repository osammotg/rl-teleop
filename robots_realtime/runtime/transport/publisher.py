"""ZMQ publisher with per-topic publish-rate throttling and optional writer recording."""

from __future__ import annotations

import time

import zmq

from robots_realtime.runtime.transport.message_bus import DEFAULT_PUB_PORT
from robots_realtime.runtime.transport.serialization import pack


class Publisher:
    """Publishes msgpack-encoded messages to the XPUB/XSUB broker.

    Also records every message to an injected Writer at the full call rate,
    independent of the ZMQ publish throttle.

    Args:
        node_name:    Prepended to every topic as ``"{node_name}/{topic}"``.
        writer:       Optional Writer instance.  If provided and open, every
                      publish() call writes to it before throttle check.
        publish_freq: If set, caps how often each topic is actually sent on the
                      bus.  None sends on every call.
        host:         Broker host (default localhost).
        port:         Broker XSUB port (publishers connect here).
    """

    def __init__(
        self,
        node_name: str,
        writer=None,
        publish_freq: float | None = None,
        host: str = "127.0.0.1",
        port: int = DEFAULT_PUB_PORT,
    ) -> None:
        self._node_name = node_name
        self._writer = writer  # Writer | None
        self._publish_freq = publish_freq
        self._min_interval = (1.0 / publish_freq) if publish_freq else 0.0
        self._last_sent: dict[str, float] = {}

        self._ctx = zmq.Context.instance()
        self._sock = self._ctx.socket(zmq.PUB)
        # BOUND THE QUEUE, DELIBERATELY. No HWM was set anywhere in this repo, so
        # every PUB socket ran on ZMQ's default of 1000 queued messages per peer.
        # For a camera that is 1000 x ~2.7 MB = ~2.7 GB of publisher-side buffer
        # before a single message is dropped, and the drops are silent either way
        # (PUB always discards rather than blocking). A small HWM is strictly
        # better on both counts: memory is bounded, and a slow subscriber gets
        # FRESH frames instead of a long queue of stale ones. Delivery loss is
        # still detected off the bus by tools/check_streams.py and shows up as
        # decaying pub_hz in the TUI — it is not, and cannot be, made invisible.
        self._sock.setsockopt(zmq.SNDHWM, 8)
        self._sock.connect(f"tcp://{host}:{port}")

        # SEPARATE SOCKET FOR TELEMETRY. Health and _step_hz are tiny and matter
        # most exactly when the frame path is congested. Sharing one socket with
        # multi-megabyte frames means a backed-up peer queue drops the health
        # message that would have explained the backlog — the monitoring channel
        # failing in sympathy with the thing it monitors. Its own socket, with its
        # own HWM, keeps that channel open.
        self._ctrl_sock = self._ctx.socket(zmq.PUB)
        self._ctrl_sock.setsockopt(zmq.SNDHWM, 200)
        self._ctrl_sock.connect(f"tcp://{host}:{port}")
        # Give the slow-joiner a moment to let subscriptions propagate
        time.sleep(0.01)

    #: Topics carried on the telemetry socket rather than the data socket.
    _TELEMETRY_TOPICS = ("health",)

    def _socket_for(self, topic_suffix: str):
        if topic_suffix.startswith("_") or topic_suffix in self._TELEMETRY_TOPICS:
            return self._ctrl_sock
        return self._sock

    def publish(
        self,
        topic_suffix: str,
        data: dict,
        ts: float | None = None,
        record: bool = True,
        record_data: dict | None = None,
    ) -> bool:
        """Send ``data`` on ``"{node_name}/{topic_suffix}"``.

        Records to the writer (if open) at the full call rate, unless ``record``
        is False or the topic is internal (prefixed with ``_``).

        ``record_data`` lets a node split bus and disk payloads — write the
        full-fidelity copy while shipping a smaller one over the wire (e.g.
        CameraNode resizes frames for the bus but keeps full-res on disk).
        Defaults to ``data`` when None.

        Returns True if the message was sent on the bus, False if throttled.
        """
        ts_val = ts if ts is not None else time.time()

        if (
            record
            and self._writer is not None
            and self._writer.is_open
            and not topic_suffix.startswith("_")
        ):
            self._writer.write(topic_suffix, ts_val, data if record_data is None else record_data)

        # Throttle ZMQ bus sends.
        #
        # MONOTONIC, NOT time.time(). This used to throttle on the wall clock,
        # which meant a backwards clock step (NTP correction, a manual `date`, a
        # VM resume) left `_last_sent` holding a timestamp in the future: every
        # subsequent `now - last` came out negative, compared as "< min_interval",
        # and the topic was throttled off the bus for the whole length of the step
        # — silently, with the writer still recording and every rate display
        # still showing the last healthy number. time.monotonic() cannot go
        # backwards, so the wedge cannot happen. The `ts` field in the envelope
        # stays on the wall clock, because consumers align it against recorded
        # timestamps.
        now = time.monotonic()
        if self._min_interval:
            last = self._last_sent.get(topic_suffix)
            if last is not None and now - last < self._min_interval:
                return False

        self._last_sent[topic_suffix] = now

        topic = f"{self._node_name}/{topic_suffix}"
        envelope = {"ts": ts_val, "src": self._node_name, "data": data}
        self._socket_for(topic_suffix).send_multipart([topic.encode(), pack(envelope)])
        return True

    def close(self) -> None:
        self._sock.close(linger=0)
        self._ctrl_sock.close(linger=0)
