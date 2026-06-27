"""In-memory progress hub keyed by ``tripId`` (Task 19.3).

The planning graph (:mod:`app.orchestration.graph`) emits a
:class:`~app.orchestration.graph.ProgressEvent` at the start and completion of
every node through a :class:`~app.orchestration.graph.ProgressReporter`
callback. Those events must reach the frontend over the
``WS /api/trips/{tripId}/progress`` stream while the run executes asynchronously.

This module is the bridge between the (possibly background-thread) producer and
the async WebSocket consumer:

* The plan route (Task 19.1) kicks off the graph with a reporter built by
  :meth:`ProgressHub.reporter_for`, which publishes each event to the hub and
  marks the channel complete when the run ends.
* The WebSocket handler subscribes via :meth:`ProgressHub.stream` and relays
  every event to the client until the channel is closed.

Design notes:

* **Replay buffer.** Each channel retains the events emitted so far, so a client
  that connects slightly after the run starts still receives the earlier
  stages. Late subscribers are caught up immediately on connect.
* **Thread-safe publish.** The graph runs synchronously and is typically driven
  from a worker thread, so :meth:`publish` / :meth:`complete` may be called off
  the event loop. They hand events to each subscriber's :class:`asyncio.Queue`
  using ``loop.call_soon_threadsafe`` when invoked from another thread.
* **Prices are never carried.** A :class:`ProgressEvent` only holds a stage
  name, a phase, and a human-readable message — never a live/real-time price
  (Requirement 13.3 / Property 26). The hub forwards events verbatim and adds
  nothing, so the no-prices guarantee established upstream is preserved.

Requirements: 13.1, 13.2, 13.3.
"""

from __future__ import annotations

import asyncio
import threading
from collections.abc import AsyncIterator
from typing import Optional

from app.orchestration.graph import ProgressEvent, ProgressReporter

__all__ = ["ProgressHub", "get_progress_hub"]


# Sentinel pushed onto subscriber queues to signal "no more events".
_DONE = object()


class _TripChannel:
    """Per-trip fan-out state: buffered events plus live subscriber queues."""

    def __init__(self) -> None:
        self.buffer: list[ProgressEvent] = []
        self.subscribers: set[asyncio.Queue] = set()
        self.closed: bool = False
        # The event loop the WebSocket handler(s) run on, captured on subscribe
        # so off-loop producers can schedule queue puts safely.
        self.loop: Optional[asyncio.AbstractEventLoop] = None


class ProgressHub:
    """Fan-out hub mapping ``tripId`` to its progress event stream.

    A single instance is shared process-wide (see :func:`get_progress_hub`).
    Producers call :meth:`publish` / :meth:`complete` (or use
    :meth:`reporter_for`); consumers iterate :meth:`stream`.
    """

    def __init__(self) -> None:
        self._channels: dict[str, _TripChannel] = {}
        self._lock = threading.Lock()

    # -- internal --------------------------------------------------------- #
    def _channel(self, trip_id: str) -> _TripChannel:
        with self._lock:
            channel = self._channels.get(trip_id)
            if channel is None:
                channel = _TripChannel()
                self._channels[trip_id] = channel
            return channel

    @staticmethod
    def _put(
        loop: Optional[asyncio.AbstractEventLoop],
        queue: "asyncio.Queue",
        item: object,
    ) -> None:
        """Place ``item`` on ``queue`` safely from any thread."""
        try:
            running = asyncio.get_running_loop()
        except RuntimeError:
            running = None
        if loop is not None and running is not loop:
            loop.call_soon_threadsafe(queue.put_nowait, item)
        else:
            queue.put_nowait(item)

    # -- producer side ---------------------------------------------------- #
    def publish(self, trip_id: str, event: ProgressEvent) -> None:
        """Record ``event`` for ``trip_id`` and fan it out to subscribers."""
        channel = self._channel(trip_id)
        with self._lock:
            if channel.closed:
                return
            channel.buffer.append(event)
            subscribers = list(channel.subscribers)
            loop = channel.loop
        for queue in subscribers:
            self._put(loop, queue, event)

    def complete(self, trip_id: str) -> None:
        """Mark ``trip_id``'s run finished; end every subscriber's stream."""
        channel = self._channel(trip_id)
        with self._lock:
            if channel.closed:
                return
            channel.closed = True
            subscribers = list(channel.subscribers)
            loop = channel.loop
        for queue in subscribers:
            self._put(loop, queue, _DONE)

    def reporter_for(self, trip_id: str) -> ProgressReporter:
        """Build a :class:`ProgressReporter` that publishes to this hub.

        Hand this to :func:`app.orchestration.graph.run_plan` /
        :func:`~app.orchestration.graph.build_graph` so each node's
        start/complete event is streamed to ``trip_id``'s subscribers. Call
        :meth:`complete` once the run returns to close the stream.
        """
        return ProgressReporter(lambda event: self.publish(trip_id, event))

    def reset(self, trip_id: str) -> None:
        """Drop any buffered state for ``trip_id`` (used between runs/tests)."""
        with self._lock:
            self._channels.pop(trip_id, None)

    # -- consumer side ---------------------------------------------------- #
    async def stream(self, trip_id: str) -> AsyncIterator[ProgressEvent]:
        """Yield progress events for ``trip_id`` until the run completes.

        Replays any events already buffered for the trip, then yields new ones
        as they are published. Returns when the channel is marked complete.
        """
        channel = self._channel(trip_id)
        queue: asyncio.Queue = asyncio.Queue()
        with self._lock:
            channel.loop = asyncio.get_running_loop()
            for event in channel.buffer:
                queue.put_nowait(event)
            if channel.closed:
                queue.put_nowait(_DONE)
            channel.subscribers.add(queue)
        try:
            while True:
                item = await queue.get()
                if item is _DONE:
                    return
                yield item  # type: ignore[misc]
        finally:
            with self._lock:
                channel.subscribers.discard(queue)


# Process-wide singleton shared by the plan route (producer) and the WS handler
# (consumer).
_HUB = ProgressHub()


def get_progress_hub() -> ProgressHub:
    """Return the shared :class:`ProgressHub` (FastAPI dependency-friendly)."""
    return _HUB
