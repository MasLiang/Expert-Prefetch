"""Finite tile-granular OCM admission and speculative eviction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Hashable

from ..schema import ValidationError
from ..execution_rules import ExecutionRules


@dataclass
class OCMObject:
    """One reserved or resident tile buffer object."""

    key: Hashable
    size_bytes: int
    state: str
    priority: float
    admitted_cycle: int
    resident: bool = False
    running: bool = False
    useful: bool | None = None
    ready_cycle: int | None = None
    allocated_bytes: int = 0
    bank_id: int = 0


@dataclass(frozen=True)
class OCMAdmission:
    """Admission result including evicted speculative objects."""

    admitted: bool
    evicted_keys: tuple[Hashable, ...] = ()


class CapacityOnChipMemory:
    """Finite byte-capacity OCM with running-object protection."""

    def __init__(self, capacity_bytes: int, rules: ExecutionRules | None = None) -> None:
        if capacity_bytes <= 0:
            raise ValidationError("OCM capacity must be positive")
        self.capacity_bytes = capacity_bytes
        self.rules = rules or ExecutionRules()
        if self.rules.ocm_slot_bytes:
            unit = self.rules.ocm_slot_bytes * self.rules.ocm_banks
            if capacity_bytes % unit or self.rules.reserved_demand_slots > capacity_bytes // self.rules.ocm_slot_bytes:
                raise ValidationError('fixed-slot capacity 必须均分到 bank，demand 保留不能超过总 slots')
        self.objects: dict[Hashable, OCMObject] = {}
        self.current_bytes = 0
        self.peak_ocm_bytes = 0
        self.rejected_prefetch_bytes = 0
        self.evicted_speculative_bytes = 0
        self._last_cycle = 0
        self._byte_cycle_area = 0

    def _advance(self, cycle: int) -> None:
        if cycle < self._last_cycle:
            raise ValueError("OCM accounting cycle moved backwards")
        self._byte_cycle_area += self.current_bytes * (cycle - self._last_cycle)
        self._last_cycle = cycle

    def admit(
        self, key: Hashable, size_bytes: int, state: str, priority: float,
        cycle: int,
    ) -> OCMAdmission:
        """Reserve capacity, evicting false then lowest-priority speculation."""
        self._advance(cycle)
        if key in self.objects:
            return OCMAdmission(True)
        if self.rules.ocm_slot_bytes:
            return self.admit_group(((key, size_bytes),), state, priority, cycle)
        if size_bytes <= 0 or size_bytes > self.capacity_bytes:
            if state == "speculative":
                self.rejected_prefetch_bytes += max(0, size_bytes)
                return OCMAdmission(False)
            raise ValidationError("required demand tile exceeds OCM capacity")
        evicted: list[Hashable] = []
        candidates = sorted(
            (
                obj for obj in self.objects.values()
                if obj.state == "speculative" and not obj.running
            ),
            key=lambda obj: (
                0 if obj.useful is False else 1,
                obj.priority,
                obj.admitted_cycle,
                repr(obj.key),
            ),
        )
        while self.current_bytes + size_bytes > self.capacity_bytes and candidates:
            victim = candidates.pop(0)
            self.current_bytes -= victim.size_bytes
            self.evicted_speculative_bytes += victim.size_bytes
            evicted.append(victim.key)
            del self.objects[victim.key]
        if self.current_bytes + size_bytes > self.capacity_bytes:
            if state == "speculative":
                self.rejected_prefetch_bytes += size_bytes
                return OCMAdmission(False, tuple(evicted))
            return OCMAdmission(False, tuple(evicted))
        self.objects[key] = OCMObject(key, size_bytes, state, priority, cycle, allocated_bytes=size_bytes)
        self.current_bytes += size_bytes
        self.peak_ocm_bytes = max(self.peak_ocm_bytes, self.current_bytes)
        return OCMAdmission(True, tuple(evicted))

    def bank_for(self, key: Hashable) -> int:
        """映射 layer/expert/matrix/tile 到 bank；epoch=epoch_base+layer。"""
        if not isinstance(key, tuple) or len(key) != 4:
            raise ValidationError('fixed-slot key 必须为 (layer, expert, matrix, tile)')
        layer, expert, matrix, tile = key
        if matrix not in ('gate_up', 'down') or any(not isinstance(v, int) or v < 0 for v in (layer, expert, tile)):
            raise ValidationError('fixed-slot key 包含非法索引或 matrix')
        return ((self.rules.epoch_base + layer) ^ layer ^ expert ^ tile ^
                (0 if matrix == 'gate_up' else 1)) & (self.rules.ocm_banks - 1)

    def admit_group(self, items: tuple[tuple[Hashable, int], ...], state: str,
                    priority: float, cycle: int) -> OCMAdmission:
        """fixed-slot 原子预留：失败不改变任何对象，不驱逐 speculative 数据。"""
        self._advance(cycle)
        slot = self.rules.ocm_slot_bytes
        if not slot:
            raise ValidationError('admit_group requires fixed-slot OCM')
        if len({key for key, _ in items}) != len(items):
            raise ValidationError('atomic admission 不允许重复 key')
        missing = [(key, size) for key, size in items if key not in self.objects]
        if any(size <= 0 or size > slot for _, size in items):
            raise ValidationError('payload 必须为正且不超过一个 OCM slot')
        limit = self.capacity_bytes - (self.rules.reserved_demand_slots * slot if state == 'speculative' else 0)
        per_bank = self.capacity_bytes // self.rules.ocm_banks
        fits = self.current_bytes + len(missing) * slot <= limit
        for bank in range(self.rules.ocm_banks):
            used = sum(obj.allocated_bytes for obj in self.objects.values() if obj.bank_id == bank)
            needed = sum(slot for key, _ in missing if self.bank_for(key) == bank)
            fits = fits and used + needed <= per_bank
        if not fits:
            return OCMAdmission(False)
        for key, size in missing:
            self.objects[key] = OCMObject(key, size, state, priority, cycle,
                                         allocated_bytes=slot, bank_id=self.bank_for(key))
        self.current_bytes += len(missing) * slot
        self.peak_ocm_bytes = max(self.peak_ocm_bytes, self.current_bytes)
        return OCMAdmission(True)

    def mark_resident(self, key: Hashable, ready_cycle: int) -> None:
        """Mark an admitted object resident at a known cycle."""
        self._advance(ready_cycle)
        obj = self.objects[key]
        obj.resident = True
        obj.ready_cycle = ready_cycle

    def commit(self, key: Hashable) -> None:
        """Convert a speculative tile into authoritative useful data."""
        obj = self.objects[key]
        obj.state = "committed"
        obj.useful = True

    def mark_false(self, key: Hashable) -> None:
        """Mark speculative data false without evicting an active burst."""
        if key in self.objects:
            self.objects[key].useful = False

    def set_running(self, key: Hashable, running: bool) -> None:
        """Protect or unprotect a tile participating in compute/active DMA."""
        if key in self.objects:
            self.objects[key].running = running

    def release(self, key: Hashable, cycle: int) -> None:
        """Release one tile buffer after consumption or cancellation."""
        self._advance(cycle)
        obj = self.objects.pop(key, None)
        if obj is not None:
            self.current_bytes -= obj.allocated_bytes

    def is_resident(self, key: Hashable, cycle: int | None = None) -> bool:
        """Return whether an object is resident by an optional cycle."""
        obj = self.objects.get(key)
        return bool(
            obj and obj.resident
            and (cycle is None or (obj.ready_cycle is not None and obj.ready_cycle <= cycle))
        )

    def average_ocm_bytes(self, makespan_cycle: int) -> float:
        """Return time-weighted average reserved bytes."""
        self._advance(makespan_cycle)
        return self._byte_cycle_area / makespan_cycle if makespan_cycle else 0.0
