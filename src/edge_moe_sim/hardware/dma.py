"""Single non-preemptive DMA engine for Milestone 1/2 baselines."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from ..performance_model import dma_runtime_cycles
from ..schema import DMAConfig

DMARequestType = Literal["demand", "background", "speculative"]


@dataclass(frozen=True)
class DMATask:
    """One whole-matrix DMA transfer."""

    transfer_bytes: int
    request_type: DMARequestType
    layer_id: int
    expert_id: int
    matrix_type: Literal["gate_up", "down"]


@dataclass
class DMARequest:
    """A mutable queued request and its commit/cancel lifecycle."""

    layer_id: int
    expert_id: int
    matrix_type: Literal["gate_up", "down"]
    size_bytes: int
    request_type: DMARequestType
    score: float
    issue_cycle: int
    tile_id: int | None = None
    candidate_rank: int | None = None
    started: bool = False
    canceled: bool = False
    committed: bool = False
    useful: bool | None = None
    start_cycle: int | None = None
    end_cycle: int | None = None
    speculative_scope: Literal["", "current_layer", "future_layer"] = ""
    source_target_prediction_id: str | None = None
    target_request_id: int | None = None
    target_token_position: int | None = None
    target_layer_id: int | None = None
    prediction_stage: str | None = None
    prediction_ready_cycle: int | None = None
    source_layer_id: int | None = None
    source_token_position: int | None = None

    @property
    def key(self) -> tuple[int, int, Literal["gate_up", "down"]] | tuple[int, int, Literal["gate_up", "down"], int]:
        """Return the OCM identity of this matrix."""
        if self.tile_id is None:
            return (self.layer_id, self.expert_id, self.matrix_type)
        return (self.layer_id, self.expert_id, self.matrix_type, self.tile_id)


@dataclass(frozen=True)
class DMAInterval:
    """A completed half-open DMA interval."""

    task: DMATask
    start_cycle: int
    end_cycle: int


class DMAEngine:
    """Deterministic max-inflight-one DMA server."""

    def __init__(self, config: DMAConfig) -> None:
        self.config = config
        self.available_cycle = 0
        self.intervals: list[DMAInterval] = []
        self.demand_queue: list[DMARequest] = []
        self.background_queue: list[DMARequest] = []
        self.speculative_queue: list[DMARequest] = []
        self.active_request: DMARequest | None = None
        self.busy_cycles = 0
        self.demand_bytes = 0
        self.speculative_bytes = 0
        self.useful_speculative_bytes = 0
        self.wasted_speculative_bytes = 0
        self.canceled_speculative_bytes = 0
        self.demand_queue_wait_cycles = 0
        self.demand_blocked_by_speculative_cycles = 0
        self.speculative_requests_issued = 0
        self.speculative_requests_committed = 0
        self.speculative_requests_canceled = 0

    def enqueue(self, request: DMARequest) -> None:
        """Append a request to its logical queue."""
        if request.request_type == "demand":
            self.demand_queue.append(request)
        elif request.request_type == "background":
            self.background_queue.append(request)
        else:
            self.speculative_queue.append(request)
            self.speculative_requests_issued += 1

    def pending_requests(self) -> list[DMARequest]:
        """Return all nonstarted, noncanceled requests."""
        return [
            request
            for request in self.demand_queue + self.background_queue + self.speculative_queue
            if not request.started and not request.canceled
        ]

    def start_request(self, request: DMARequest, start_cycle: int) -> int:
        """Start one selected request and return its completion cycle."""
        if self.active_request is not None:
            raise ValueError("DMA already has an active request")
        task = DMATask(
            request.size_bytes, request.request_type, request.layer_id,
            request.expert_id, request.matrix_type,
        )
        end_cycle = self.reserve(task, start_cycle)
        request.started = True
        request.start_cycle = start_cycle
        request.end_cycle = end_cycle
        self.active_request = request
        return end_cycle

    def complete_request(self, request: DMARequest) -> None:
        """Finish the active request and account transferred bytes."""
        if self.active_request is not request:
            raise ValueError("completed request is not the active DMA request")
        self.active_request = None
        if request.request_type in {"demand", "background"}:
            self.demand_bytes += request.size_bytes
        else:
            self.speculative_bytes += request.size_bytes
            if request.useful is True:
                self.useful_speculative_bytes += request.size_bytes
            elif request.useful is False:
                self.wasted_speculative_bytes += request.size_bytes

    def earliest_start(self, task: DMATask, current_cycle: int) -> int:
        """Return the first non-overlapping DMA start cycle."""
        return max(current_cycle, self.available_cycle)

    def reserve(self, task: DMATask, start_cycle: int) -> int:
        """Reserve one non-preemptive transfer and return its finish cycle."""
        if start_cycle < self.available_cycle:
            raise ValueError("DMA reservation overlaps an existing transfer")
        end_cycle = start_cycle + dma_runtime_cycles(task.transfer_bytes, self.config)
        self.intervals.append(DMAInterval(task, start_cycle, end_cycle))
        self.available_cycle = end_cycle
        self.busy_cycles += end_cycle - start_cycle
        return end_cycle
