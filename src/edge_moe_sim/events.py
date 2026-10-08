"""Discrete-event types and deterministic same-cycle priorities."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any


class EventType(Enum):
    """Events understood by the unified simulation core."""

    LAYER_START = auto()
    PREDICTOR_READY = auto()
    ROUTER_START = auto()
    ROUTER_DONE = auto()
    TOPK_DONE = auto()
    DISPATCH_DONE = auto()
    DMA_REQUEST = auto()
    DMA_START = auto()
    DMA_DONE = auto()
    DMA_CANCEL = auto()
    DMA_COMMIT = auto()
    AIE_OPERATION_START = auto()
    AIE_OPERATION_DONE = auto()
    NONLINEAR_START = auto()
    NONLINEAR_DONE = auto()
    EXPERT_DONE = auto()
    LAYER_DONE = auto()
    SCHEDULER_INVOKE = auto()
    PREDICTION_TRIGGER = auto()
    INTER_LAYER_START = auto()
    INTER_LAYER_FIXED_DONE = auto()
    INTER_LAYER_DONE = auto()
    LAYER_COMBINE_DONE = auto()
    CONTROL_CREDIT_READY = auto()
    OCM_PAYLOAD_READY = auto()
    INGRESS_DONE = auto()
    INTER_LAYER_RETIRE_CHECK = auto()
    PREFETCH_ALLOCATION_READY = auto()


EVENT_PRIORITY: dict[EventType, int] = {
    EventType.PREFETCH_ALLOCATION_READY: 12,
    EventType.DMA_DONE: 1,
    EventType.AIE_OPERATION_DONE: 2,
    EventType.NONLINEAR_DONE: 3,
    EventType.ROUTER_DONE: 4,
    EventType.TOPK_DONE: 5,
    EventType.DISPATCH_DONE: 6,
    EventType.PREDICTOR_READY: 7,
    EventType.DMA_COMMIT: 8,
    EventType.DMA_CANCEL: 9,
    EventType.DMA_REQUEST: 10,
    EventType.SCHEDULER_INVOKE: 12,
    EventType.DMA_START: 13,
    EventType.AIE_OPERATION_START: 14,
    EventType.NONLINEAR_START: 15,
    EventType.EXPERT_DONE: 16,
    EventType.LAYER_DONE: 17,
    EventType.LAYER_START: 18,
    EventType.ROUTER_START: 14,
    EventType.PREDICTION_TRIGGER: 6,
    EventType.INTER_LAYER_FIXED_DONE: 4,
    EventType.INTER_LAYER_DONE: 17,
    EventType.INTER_LAYER_START: 18,
    EventType.LAYER_COMBINE_DONE: 17,
    EventType.CONTROL_CREDIT_READY: 10,
    EventType.OCM_PAYLOAD_READY: 1,
    EventType.INGRESS_DONE: 10,
    EventType.INTER_LAYER_RETIRE_CHECK: 18,
}

SPECULATIVE_DMA_REQUEST_PRIORITY = 11


@dataclass(order=True)
class Event:
    """One ordered event in integer-cycle time."""

    cycle: int
    priority: int
    sequence_number: int
    event_type: EventType = field(compare=False)
    payload: dict[str, Any] = field(compare=False)
