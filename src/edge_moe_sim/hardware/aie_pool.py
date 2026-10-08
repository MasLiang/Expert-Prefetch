"""Capacity-based AIE pool supporting concurrent expert tile tasks."""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil

from ..hardware_profile import HardwareProfile
from ..schema import ValidationError


@dataclass(frozen=True)
class AIEPoolTask:
    """One compute task requesting a fixed AIE subset."""

    operation_type: str
    operation_count: int
    requested_aie_units: int
    layer_id: int = 0
    expert_id: int | None = None
    tile_id: int | None = None


@dataclass(frozen=True)
class AIEPoolInterval:
    """One capacity reservation with payload/startup accounting."""

    task: AIEPoolTask
    start_cycle: int
    end_cycle: int
    payload_cycles: int
    startup_cycles: int


class AIEPool:
    """Deterministic cumulative resource for multiple concurrent AIE tasks."""

    def __init__(self, profile: HardwareProfile) -> None:
        self.profile = profile
        self.intervals: list[AIEPoolInterval] = []

    def task_duration(self, task: AIEPoolTask) -> tuple[int, int]:
        """Return payload and total cycles at the requested allocation."""
        if task.operation_count < 0:
            raise ValidationError("AIE operation_count must be non-negative")
        if not 1 <= task.requested_aie_units <= self.profile.total_aie_units:
            raise ValidationError("requested AIE units exceed pool capacity")
        rate = task.requested_aie_units * self.profile.compute_rate_per_cycle(task.operation_type)
        payload = ceil(task.operation_count / rate) if task.operation_count else 0
        return payload, payload + self.profile.aie_task_startup_cycles

    def earliest_start(self, task: AIEPoolTask, current_cycle: int) -> int:
        """Find the first interval with enough AIE capacity for the full task."""
        _, duration = self.task_duration(task)
        candidate = current_cycle
        while True:
            overlapping = [
                interval for interval in self.intervals
                if interval.start_cycle < candidate + duration and interval.end_cycle > candidate
            ]
            boundaries = sorted({candidate, candidate + duration} | {
                point for interval in overlapping
                for point in (max(candidate, interval.start_cycle), min(candidate + duration, interval.end_cycle))
            })
            feasible = True
            for left, right in zip(boundaries, boundaries[1:]):
                if left == right:
                    continue
                active = sum(
                    interval.task.requested_aie_units for interval in overlapping
                    if interval.start_cycle < right and interval.end_cycle > left
                )
                if active + task.requested_aie_units > self.profile.total_aie_units:
                    feasible = False
                    candidate = min(
                        interval.end_cycle for interval in overlapping
                        if interval.start_cycle < right and interval.end_cycle > left
                    )
                    break
            if feasible:
                return candidate

    def reserve(self, task: AIEPoolTask, current_cycle: int) -> tuple[int, int]:
        """Reserve at the earliest feasible cycle and return start/end."""
        start = self.earliest_start(task, current_cycle)
        payload, duration = self.task_duration(task)
        end = start + duration
        self.intervals.append(AIEPoolInterval(
            task, start, end, payload, self.profile.aie_task_startup_cycles
        ))
        return start, end

    def reserve_fixed(self, cycles: int, requested_aie_units: int, current_cycle: int,
                      layer_id: int = 0) -> tuple[int, int]:
        """Reserve calibrated/synthetic fixed-duration work on the shared pool."""
        if cycles < 0 or not 1 <= requested_aie_units <= self.profile.total_aie_units:
            raise ValidationError("invalid fixed AIE-pool reservation")
        task = AIEPoolTask("inter_layer", 0, requested_aie_units, layer_id)
        if cycles == 0:
            return current_cycle, current_cycle
        candidate = current_cycle
        while True:
            conflicts = [i for i in self.intervals if i.start_cycle < candidate + cycles
                         and i.end_cycle > candidate and
                         i.task.requested_aie_units + requested_aie_units > self.profile.total_aie_units]
            if not conflicts:
                break
            candidate = min(i.end_cycle for i in conflicts)
        end = candidate + cycles
        self.intervals.append(AIEPoolInterval(task, candidate, end, 0, cycles))
        return candidate, end

    def metrics(self, makespan_cycles: int) -> dict[str, float | int]:
        """Return allocated/useful unit-cycles and board utilization."""
        if makespan_cycles <= 0:
            raise ValidationError("AIE metrics require positive makespan")
        allocated = sum(
            interval.task.requested_aie_units * (interval.end_cycle - interval.start_cycle)
            for interval in self.intervals
        )
        useful = sum(
            interval.task.requested_aie_units * interval.payload_cycles
            for interval in self.intervals
        )
        peak = max((
            sum(
                interval.task.requested_aie_units for interval in self.intervals
                if interval.start_cycle <= cycle < interval.end_cycle
            )
            for cycle in {point for interval in self.intervals for point in (interval.start_cycle,)}
        ), default=0)
        capacity = self.profile.total_aie_units * makespan_cycles
        return {
            "allocated_aie_cycles": allocated,
            "useful_aie_cycles": useful,
            "aie_idle_cycles": max(0, capacity - allocated),
            "average_active_aie_units": allocated / makespan_cycles,
            "peak_active_aie_units": peak,
            "board_aie_utilization": allocated / capacity,
        }
