"""Single, mutually exclusive logical AIE GEMM engine."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from ..performance_model import aie_payload_cycles
from ..schema import AIEConfig

AIEOperationType = Literal["router", "gate_up", "down", "inter_layer"]


@dataclass(frozen=True)
class AIETask:
    """An AIE operation with an integer operation count."""

    operation_type: AIEOperationType
    operation_count: int
    tile_mode: bool = False
    matrix_start: bool = True


@dataclass(frozen=True)
class AIEInterval:
    """A reserved half-open AIE interval and its accounting split."""

    operation_type: AIEOperationType
    start_cycle: int
    end_cycle: int
    payload_cycles: int
    control_cycles: int


class AIEEngine:
    """A deterministic single-server AIE engine."""

    def __init__(self, config: AIEConfig) -> None:
        self.config = config
        self.available_cycle = 0
        self.last_operation_type: AIEOperationType | None = None
        self.intervals: list[AIEInterval] = []
        self.busy_cycles = 0
        self.memory_stall_cycles = 0
        self.control_transition_cycles = 0

    def earliest_start(self, task: AIETask, current_cycle: int) -> int:
        """Return the first cycle permitted by mutual exclusion."""
        return max(current_cycle, self.available_cycle)

    def reserve(self, task: AIETask, start_cycle: int) -> int:
        """Reserve one GEMM and return its finish cycle."""
        if start_cycle < self.available_cycle:
            raise ValueError("AIE reservation overlaps an existing operation")
        rate = {
            "router": self.config.router_ops_per_cycle,
            "gate_up": self.config.gate_up_ops_per_cycle,
            "down": self.config.down_ops_per_cycle,
        }[task.operation_type]
        payload_cycles = aie_payload_cycles(task.operation_count, rate)
        pays_switch = (
            self.config.first_operation_pays_switch_cycles
            if self.last_operation_type is None
            else self.last_operation_type != task.operation_type
        )
        startup_cycles = (
            self.config.effective_matrix_startup_cycles
            if task.tile_mode and task.matrix_start
            else self.config.tile_pipeline_cycles
            if task.tile_mode
            else self.config.startup_cycles
        )
        control_cycles = startup_cycles + (
            self.config.operation_switch_cycles if pays_switch else 0
        )
        end_cycle = start_cycle + control_cycles + payload_cycles
        self.intervals.append(AIEInterval(
            task.operation_type, start_cycle, end_cycle, payload_cycles, control_cycles
        ))
        self.available_cycle = end_cycle
        self.last_operation_type = task.operation_type
        self.busy_cycles += payload_cycles
        self.control_transition_cycles += control_cycles
        return end_cycle

    def reserve_fixed(self, cycles: int, start_cycle: int) -> int:
        """Reserve fixed synthetic inter-layer AIE work without GEMM overheads."""
        if cycles < 0:
            raise ValueError("fixed AIE cycles must be non-negative")
        if start_cycle < self.available_cycle:
            raise ValueError("AIE reservation overlaps an existing operation")
        end_cycle = start_cycle + cycles
        self.intervals.append(AIEInterval(
            "inter_layer", start_cycle, end_cycle, cycles, 0
        ))
        self.available_cycle = end_cycle
        self.last_operation_type = "inter_layer"
        self.busy_cycles += cycles
        return end_cycle
