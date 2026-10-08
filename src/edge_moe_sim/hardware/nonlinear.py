"""Independent, mutually exclusive nonlinear engine."""

from __future__ import annotations

from dataclasses import dataclass

from ..performance_model import nonlinear_runtime_cycles
from ..schema import NonlinearConfig


@dataclass(frozen=True)
class NonlinearTask:
    """A nonlinear operation over an integer element count."""

    element_count: int


class NonlinearEngine:
    """Single-server DSP/PL nonlinear resource."""

    def __init__(self, config: NonlinearConfig) -> None:
        self.config = config
        self.available_cycle = 0
        self.busy_cycles = 0
        self.intervals: list[tuple[int, int]] = []

    def earliest_start(self, task: NonlinearTask, current_cycle: int) -> int:
        """Return the first non-overlapping start cycle."""
        return max(current_cycle, self.available_cycle)

    def reserve(self, task: NonlinearTask, start_cycle: int) -> int:
        """Reserve one nonlinear operation and return its finish cycle."""
        if start_cycle < self.available_cycle:
            raise ValueError("nonlinear reservation overlaps an existing operation")
        end_cycle = start_cycle + nonlinear_runtime_cycles(task.element_count, self.config)
        self.intervals.append((start_cycle, end_cycle))
        self.available_cycle = end_cycle
        self.busy_cycles += end_cycle - start_cycle
        return end_cycle

    def reserve_fixed(self, cycles: int, start_cycle: int) -> int:
        """Reserve fixed synthetic inter-layer nonlinear work."""
        if cycles < 0:
            raise ValueError("fixed nonlinear cycles must be non-negative")
        if start_cycle < self.available_cycle:
            raise ValueError("nonlinear reservation overlaps an existing operation")
        end_cycle = start_cycle + cycles
        self.intervals.append((start_cycle, end_cycle))
        self.available_cycle = end_cycle
        self.busy_cycles += cycles
        return end_cycle
