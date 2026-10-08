"""Unlimited-capacity on-chip-memory state abstraction."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import Literal, Union

MatrixType = Literal["gate_up", "down"]
OCMKey = tuple[int, int, MatrixType]
TileKey = tuple[int, int, MatrixType, int]
MemoryKey = Union[OCMKey, TileKey]


class WeightState(Enum):
    """Lifecycle states retained for future predictive-prefetch work."""

    ABSENT = auto()
    QUEUED_SPECULATIVE = auto()
    QUEUED_DEMAND = auto()
    IN_FLIGHT_SPECULATIVE = auto()
    IN_FLIGHT_DEMAND = auto()
    RESIDENT_SPECULATIVE = auto()
    RESIDENT_DEMAND = auto()


@dataclass
class WeightRecord:
    """OCM state and known readiness for one matrix."""

    state: WeightState = WeightState.ABSENT
    ready_cycle: int | None = None
    original_request_type: str | None = None
    speculative_useful: bool | None = None


class OnChipMemory:
    """Unlimited OCM implementation with a capacity-ready API."""

    def __init__(self) -> None:
        self._records: dict[MemoryKey, WeightRecord] = {}
        self._matrix_tile_counts: dict[OCMKey, int] = {}

    def register_tiled_matrix(self, key: OCMKey, num_tiles: int) -> None:
        """Register the expected tile count for matrix-level queries."""
        if num_tiles <= 0:
            raise ValueError("num_tiles must be positive")
        existing = self._matrix_tile_counts.get(key)
        if existing is not None and existing != num_tiles:
            raise ValueError(f"matrix {key} already registered with {existing} tiles")
        self._matrix_tile_counts[key] = num_tiles

    def get_tile_state(self, key: TileKey) -> WeightState:
        """Return one tile's lifecycle state."""
        return self.state(key)

    def get_tile_ready_cycle(self, key: TileKey) -> int | None:
        """Return one tile's ready cycle."""
        return self.ready_cycle(key)

    def is_tile_resident(self, key: TileKey) -> bool:
        """Return whether one tile is resident."""
        return self.is_resident(key)

    def mark_tile_queued(self, key: TileKey, request_type: str) -> None:
        """Mark a tile queued."""
        self.mark_queued(key, request_type)

    def mark_tile_in_flight(self, key: TileKey, request_type: str, ready_cycle: int) -> None:
        """Mark a tile in flight."""
        self.mark_in_flight(key, request_type, ready_cycle)

    def mark_tile_resident(self, key: TileKey, request_type: str, ready_cycle: int) -> None:
        """Mark a tile resident."""
        self.mark_resident(key, request_type, ready_cycle)

    def commit_tile(self, key: TileKey) -> None:
        """Commit a speculative tile."""
        self.commit(key)

    def cancel_tile_if_queued(self, key: TileKey) -> bool:
        """Cancel a queued speculative tile."""
        return self.cancel_if_queued(key)

    def resident_tile_count(self, layer_id: int, expert_id: int, matrix_type: MatrixType) -> int:
        """Count resident tiles for a registered matrix."""
        matrix_key = (layer_id, expert_id, matrix_type)
        count = self._matrix_tile_counts.get(matrix_key, 0)
        return sum(
            self.is_tile_resident((layer_id, expert_id, matrix_type, tile_id))
            for tile_id in range(count)
        )

    def resident_fraction(self, layer_id: int, expert_id: int, matrix_type: MatrixType) -> float:
        """Return resident tile fraction for a registered matrix."""
        count = self._matrix_tile_counts.get((layer_id, expert_id, matrix_type), 0)
        return self.resident_tile_count(layer_id, expert_id, matrix_type) / count if count else 0.0

    def first_tile_ready_cycle(self, layer_id: int, expert_id: int, matrix_type: MatrixType) -> int | None:
        """Return tile zero's ready cycle."""
        return self.get_tile_ready_cycle((layer_id, expert_id, matrix_type, 0))

    def contiguous_ready_prefix(self, layer_id: int, expert_id: int, matrix_type: MatrixType) -> int:
        """Count the resident prefix beginning at tile zero."""
        count = self._matrix_tile_counts.get((layer_id, expert_id, matrix_type), 0)
        prefix = 0
        for tile_id in range(count):
            if not self.is_tile_resident((layer_id, expert_id, matrix_type, tile_id)):
                break
            prefix += 1
        return prefix

    def record_ready(self, key: OCMKey, ready_cycle: int, request_type: str) -> None:
        """Record the known cycle at which a matrix becomes resident."""
        state = WeightState.RESIDENT_DEMAND if request_type == "demand" else WeightState.RESIDENT_SPECULATIVE
        self._records[key] = WeightRecord(state, ready_cycle, request_type, None)

    def get_state(self, key: OCMKey) -> WeightState:
        """Return the current state of a matrix."""
        return self.state(key)

    def get_ready_cycle(self, key: OCMKey) -> int | None:
        """Return the known completion cycle of a matrix transfer."""
        return self.ready_cycle(key)

    def mark_queued(self, key: OCMKey, request_type: str) -> None:
        """Mark a unique matrix request as queued."""
        if self.state(key) is not WeightState.ABSENT:
            raise ValueError(f"matrix {key} is already requested")
        state = WeightState.QUEUED_DEMAND if request_type == "demand" else WeightState.QUEUED_SPECULATIVE
        self._records[key] = WeightRecord(state, None, request_type, None)

    def mark_in_flight(self, key: OCMKey, request_type: str, ready_cycle: int) -> None:
        """Mark a queued matrix in flight with its scheduled ready cycle."""
        expected = WeightState.QUEUED_DEMAND if request_type == "demand" else WeightState.QUEUED_SPECULATIVE
        if self.state(key) is not expected:
            raise ValueError(f"matrix {key} is not in expected queued state")
        state = WeightState.IN_FLIGHT_DEMAND if request_type == "demand" else WeightState.IN_FLIGHT_SPECULATIVE
        record = self._records[key]
        record.state = state
        record.ready_cycle = ready_cycle

    def mark_resident(self, key: OCMKey, request_type: str, ready_cycle: int) -> None:
        """Mark an in-flight matrix resident."""
        state = WeightState.RESIDENT_DEMAND if request_type == "demand" else WeightState.RESIDENT_SPECULATIVE
        record = self._records.get(key, WeightRecord())
        record.state = state
        record.ready_cycle = ready_cycle
        record.original_request_type = record.original_request_type or request_type
        self._records[key] = record

    def commit(self, key: OCMKey) -> None:
        """Mark speculative data useful without changing its transfer origin."""
        record = self._records.get(key)
        if record is None or record.state not in {
            WeightState.QUEUED_SPECULATIVE, WeightState.IN_FLIGHT_SPECULATIVE,
            WeightState.RESIDENT_SPECULATIVE,
        }:
            raise ValueError(f"matrix {key} has no speculative request to commit")
        record.speculative_useful = True

    def mark_wasted(self, key: OCMKey) -> None:
        """Mark an in-flight or resident speculative matrix as wasted."""
        record = self._records.get(key)
        if record is None:
            raise ValueError(f"matrix {key} is absent")
        record.speculative_useful = False

    def cancel_if_queued(self, key: OCMKey) -> bool:
        """Cancel a queued speculative request and return whether canceled."""
        if self.state(key) is not WeightState.QUEUED_SPECULATIVE:
            return False
        self._records[key] = WeightRecord()
        return True

    def is_resident(self, key: OCMKey, current_cycle: int | None = None) -> bool:
        """Return whether the matrix is resident at ``current_cycle``."""
        record = self._records.get(key)
        if not record or record.state not in {WeightState.RESIDENT_DEMAND, WeightState.RESIDENT_SPECULATIVE}:
            return False
        return current_cycle is None or bool(record.ready_cycle is not None and record.ready_cycle <= current_cycle)

    def is_queued(self, key: OCMKey) -> bool:
        """Return whether the matrix is queued but not started."""
        return self.state(key) in {WeightState.QUEUED_DEMAND, WeightState.QUEUED_SPECULATIVE}

    def is_in_flight(self, key: OCMKey) -> bool:
        """Return whether the matrix is currently marked in flight."""
        return self.state(key) in {WeightState.IN_FLIGHT_DEMAND, WeightState.IN_FLIGHT_SPECULATIVE}

    def ready_cycle(self, key: OCMKey) -> int | None:
        """Return the known ready cycle, if any."""
        record = self._records.get(key)
        return record.ready_cycle if record else None

    def state(self, key: OCMKey) -> WeightState:
        """Return the matrix lifecycle state."""
        return self._records.get(key, WeightRecord()).state

    def original_request_type(self, key: OCMKey) -> str | None:
        """Return the request type that first fetched the matrix."""
        record = self._records.get(key)
        return record.original_request_type if record else None

    def speculative_useful(self, key: OCMKey) -> bool | None:
        """Return future speculative usefulness bookkeeping."""
        record = self._records.get(key)
        return record.speculative_useful if record else None

    def target_tiles_resident_at_cycle(
        self,
        layer_id: int,
        expert_ids: tuple[int, ...],
        current_cycle: int,
        matrix_type: MatrixType = "gate_up",
        tile_id: int = 0,
    ) -> tuple[int, ...]:
        """Return target experts whose selected tile is resident by a milestone."""
        return tuple(
            expert_id
            for expert_id in expert_ids
            if self.is_resident((layer_id, expert_id, matrix_type, tile_id), current_cycle)
        )
