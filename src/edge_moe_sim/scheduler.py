"""Pluggable, deterministic expert scheduling policies."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, Protocol, Sequence

from .schema import ExpertTask

if TYPE_CHECKING:
    from .hardware.ocm import OnChipMemory


@dataclass(frozen=True)
class SimulatorState:
    """Read-only scheduling view of Gate-Up weight readiness."""

    gate_up_ready_cycles: dict[int, int | None]
    ocm: "OnChipMemory | None" = None
    layer_id: int = 0
    current_expert_id: int | None = None
    current_expert_phase: str | None = None
    next_expert_id: int | None = None
    expert_tasks: tuple[ExpertTask, ...] = ()
    expert_rank: dict[int, int] | None = None
    current_tile_id: int | None = None


class ExpertScheduler(Protocol):
    """Interface for selecting the next complete expert to execute."""

    def choose_next(
        self,
        pending_experts: Sequence[ExpertTask],
        simulator_state: SimulatorState,
        current_cycle: int,
    ) -> ExpertTask | None:
        """Choose one pending task, or return ``None`` when empty."""
        ...


class RouterOrderScheduler:
    """Preserve the true-expert order recorded by the router."""

    def choose_next(
        self,
        pending_experts: Sequence[ExpertTask],
        simulator_state: SimulatorState,
        current_cycle: int,
    ) -> ExpertTask | None:
        """Return the first pending expert."""
        return pending_experts[0] if pending_experts else None


class ReadyFirstScheduler:
    """Prioritize resident, then earliest-ready, larger, heavier experts."""

    def choose_next(
        self,
        pending_experts: Sequence[ExpertTask],
        simulator_state: SimulatorState,
        current_cycle: int,
    ) -> ExpertTask | None:
        """Return the deterministic minimum under the ready-first key."""
        if not pending_experts:
            return None

        def key(task: ExpertTask) -> tuple[int, int, int, float, int]:
            ready_cycle = simulator_state.gate_up_ready_cycles.get(task.expert_id)
            resolved_ready = ready_cycle if ready_cycle is not None else 2**63 - 1
            return (
                0 if ready_cycle is not None and ready_cycle <= current_cycle else 1,
                resolved_ready,
                -task.token_count,
                -task.gating_weight,
                task.expert_id,
            )

        return min(pending_experts, key=key)
