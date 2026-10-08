"""Pluggable deterministic DMA request-selection policies."""

from __future__ import annotations

from typing import Protocol, Sequence

from .hardware.dma import DMARequest
from .scheduler import SimulatorState


class DMARequestScheduler(Protocol):
    """Interface for choosing the next request on an idle DMA engine."""

    def choose_next_request(
        self,
        pending_requests: Sequence[DMARequest],
        simulator_state: SimulatorState,
        current_cycle: int,
    ) -> DMARequest | None:
        """Choose one request, or return ``None`` if none are eligible."""
        ...


def _eligible(pending_requests: Sequence[DMARequest]) -> list[DMARequest]:
    requests = [request for request in pending_requests if not request.started and not request.canceled]
    demands = [
        request for request in requests
        if request.request_type == "demand" or request.committed
    ]
    return demands if demands else requests


def _fallback_key(request: DMARequest) -> tuple[int, float, int, int, int]:
    return (
        0 if request.request_type == "demand" else 1,
        -request.score,
        request.issue_cycle,
        request.candidate_rank if request.candidate_rank is not None else 2**31,
        request.expert_id,
    )


class AllGateUpThenDownScheduler:
    """Select all Gate-Up matrices before any Down matrices."""

    def choose_next_request(
        self, pending_requests: Sequence[DMARequest], simulator_state: SimulatorState, current_cycle: int
    ) -> DMARequest | None:
        """Choose by matrix type and router order, preserving demand priority."""
        requests = _eligible(pending_requests)
        if not requests:
            return None
        if not simulator_state.expert_tasks:
            return min(requests, key=_fallback_key)
        rank = simulator_state.expert_rank or {}
        return min(requests, key=lambda request: (
            0 if request.matrix_type == "gate_up" else 1,
            rank.get(
                request.expert_id,
                request.candidate_rank if request.candidate_rank is not None else 2**30,
            ),
            *_fallback_key(request),
        ))


class ExpertContiguousScheduler:
    """Select Gate-Up then Down for each expert in router order."""

    def choose_next_request(
        self, pending_requests: Sequence[DMARequest], simulator_state: SimulatorState, current_cycle: int
    ) -> DMARequest | None:
        """Choose by expert rank then stage, preserving demand priority."""
        requests = _eligible(pending_requests)
        if not requests:
            return None
        rank = simulator_state.expert_rank or {}
        return min(requests, key=lambda request: (
            rank.get(
                request.expert_id,
                request.candidate_rank if request.candidate_rank is not None else 2**30,
            ),
            0 if request.matrix_type == "gate_up" else 1,
            *_fallback_key(request),
        ))


class CriticalPathAwareScheduler:
    """Prioritize the current and next expert's critical weight stages."""

    def choose_next_request(
        self, pending_requests: Sequence[DMARequest], simulator_state: SimulatorState, current_cycle: int
    ) -> DMARequest | None:
        """Apply critical-path rules A-E with deterministic demand priority."""
        requests = _eligible(pending_requests)
        if not requests:
            return None
        if not simulator_state.expert_tasks:
            return min(requests, key=_fallback_key)

        def find(expert_id: int | None, matrix_type: str) -> DMARequest | None:
            if expert_id is None:
                return None
            matches = [
                request for request in requests
                if request.expert_id == expert_id and request.matrix_type == matrix_type
            ]
            return min(matches, key=_fallback_key) if matches else None

        current = simulator_state.current_expert_id
        phase = simulator_state.current_expert_phase
        gate = find(current, "gate_up")
        if gate is not None and phase in {None, "waiting_gate"}:
            return gate
        down = find(current, "down")
        if down is not None and phase in {"gate_up", "waiting_nonlinear", "waiting_down", "down"}:
            return down
        next_expert = simulator_state.next_expert_id
        next_gate = find(next_expert, "gate_up")
        if next_gate is not None:
            return next_gate
        next_down = find(next_expert, "down")
        if next_down is not None:
            return next_down
        task_by_id = {task.expert_id: task for task in simulator_state.expert_tasks}
        rank = simulator_state.expert_rank or {}
        return min(requests, key=lambda request: (
            -task_by_id.get(request.expert_id, _ZERO_TASK).token_count,
            -task_by_id.get(request.expert_id, _ZERO_TASK).gating_weight,
            request.expert_id,
            0 if request.matrix_type == "gate_up" else 1,
            rank.get(request.expert_id, 2**30),
            *_fallback_key(request),
        ))


class TileCriticalPathAwareScheduler:
    """Prioritize the earliest missing tile on the active matrix path."""

    def choose_next_request(
        self, pending_requests: Sequence[DMARequest], simulator_state: SimulatorState, current_cycle: int
    ) -> DMARequest | None:
        """Select current-matrix prefix, current Down, then next Expert tiles."""
        requests = _eligible(pending_requests)
        if not requests:
            return None
        if not simulator_state.expert_tasks:
            return min(requests, key=lambda request: (
                -request.score,
                request.candidate_rank if request.candidate_rank is not None else 2**30,
                request.tile_id if request.tile_id is not None else -1,
                request.expert_id,
            ))

        def earliest(expert_id: int | None, matrix_type: str) -> DMARequest | None:
            matches = [
                request for request in requests
                if request.expert_id == expert_id and request.matrix_type == matrix_type
            ]
            return min(matches, key=lambda request: (
                request.tile_id if request.tile_id is not None else -1,
                *_fallback_key(request),
            )) if matches else None

        current = simulator_state.current_expert_id
        phase = simulator_state.current_expert_phase or ""
        current_matrix = "down" if "down" in phase else "gate_up"
        request = earliest(current, current_matrix)
        if request is not None:
            return request
        if current_matrix == "gate_up":
            request = earliest(current, "down")
            if request is not None:
                return request
        next_expert = simulator_state.next_expert_id
        request = earliest(next_expert, "gate_up")
        if request is not None:
            return request
        request = earliest(next_expert, "down")
        if request is not None:
            return request
        task_by_id = {task.expert_id: task for task in simulator_state.expert_tasks}
        return min(requests, key=lambda request: (
            0 if request.request_type == "demand" or request.committed else 1,
            0 if request.matrix_type == current_matrix else 1,
            request.tile_id if request.tile_id is not None else -1,
            -task_by_id.get(request.expert_id, _ZERO_TASK).token_count,
            -task_by_id.get(request.expert_id, _ZERO_TASK).gating_weight,
            -request.score,
            request.expert_id,
        ))


from .schema import ExpertTask  # noqa: E402  (constant after class definitions)

_ZERO_TASK = ExpertTask(0, 0, 0, 0.0)
