"""Heap event ordering tests."""

from edge_moe_sim.event_queue import EventQueue
from edge_moe_sim.events import EventType


def test_same_cycle_priority_and_sequence_are_deterministic() -> None:
    """Completions precede demand/spec requests and sequence breaks ties."""
    queue = EventQueue()
    queue.push(10, EventType.DMA_REQUEST, {"request_type": "speculative", "id": 1})
    queue.push(10, EventType.DMA_REQUEST, {"request_type": "demand", "id": 2})
    queue.push(10, EventType.DMA_DONE, {"id": 3})
    queue.push(10, EventType.DMA_REQUEST, {"request_type": "demand", "id": 4})
    assert [queue.pop().payload["id"] for _ in range(4)] == [3, 2, 4, 1]
