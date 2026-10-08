"""Heap-backed deterministic discrete-event queue."""

from __future__ import annotations

import heapq
from collections.abc import Iterator
from typing import Any

from .events import EVENT_PRIORITY, SPECULATIVE_DMA_REQUEST_PRIORITY, Event, EventType


class EventQueue:
    """Own a heap and a monotonically increasing tie-break sequence."""

    def __init__(self) -> None:
        self._heap: list[Event] = []
        self._next_sequence_number = 0

    def push(
        self,
        cycle: int,
        event_type: EventType,
        payload: dict[str, Any] | None = None,
        *,
        priority: int | None = None,
    ) -> Event:
        """Insert an event and return it."""
        if cycle < 0:
            raise ValueError("event cycle must be non-negative")
        resolved_priority = EVENT_PRIORITY[event_type] if priority is None else priority
        if event_type is EventType.DMA_REQUEST and payload and payload.get("request_type") == "speculative":
            resolved_priority = SPECULATIVE_DMA_REQUEST_PRIORITY
        elif event_type is EventType.DMA_REQUEST and payload and payload.get("request_type") == "background":
            resolved_priority = SPECULATIVE_DMA_REQUEST_PRIORITY - 1
        event = Event(cycle, resolved_priority, self._next_sequence_number, event_type, payload or {})
        self._next_sequence_number += 1
        heapq.heappush(self._heap, event)
        return event

    def pop(self) -> Event:
        """Remove and return the earliest event."""
        if not self._heap:
            raise IndexError("pop from empty event queue")
        return heapq.heappop(self._heap)

    def __bool__(self) -> bool:
        return bool(self._heap)

    def __len__(self) -> int:
        return len(self._heap)

    def __iter__(self) -> Iterator[Event]:
        """Yield a sorted snapshot without consuming the queue."""
        return iter(sorted(self._heap))
