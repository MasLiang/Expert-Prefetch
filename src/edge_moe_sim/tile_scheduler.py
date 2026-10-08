"""Prediction-aware, byte-budgeted Expert tile allocation policies.

The schedulers in this module are hardware-profile agnostic.  They allocate
logical F-axis tiles; each logical tile contains the Gate-Up and Down weight
fragments required to execute that slice of an Expert.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from itertools import product
from math import ceil, log2
from typing import Callable, Mapping, Protocol, Sequence

from .schema import ValidationError


class TileResidentState(str, Enum):
    """Observable lifetime state of one logical Expert tile."""

    ABSENT = "absent"
    PLANNED = "planned"
    IN_FLIGHT = "in_flight"
    RESIDENT = "resident"
    CONSUMED = "consumed"


@dataclass(frozen=True)
class ExpertDescriptor:
    """Tile geometry of one candidate Expert."""

    expert_id: int
    num_tiles: int
    tile_size_bytes: int

    def __post_init__(self) -> None:
        if self.expert_id < 0 or self.num_tiles <= 0 or self.tile_size_bytes <= 0:
            raise ValidationError("invalid Expert tile descriptor")


@dataclass(frozen=True)
class ExpertTile:
    """One independently planned logical F-axis Expert tile."""

    expert_id: int
    tile_id: int
    size_bytes: int
    gate_up_bytes: int
    down_bytes: int
    predicted_rank: int
    confidence: float
    dma_ready_cycle: int | None = None
    resident_state: TileResidentState = TileResidentState.ABSENT

    def __post_init__(self) -> None:
        if self.expert_id < 0 or self.tile_id < 0:
            raise ValidationError("Expert and tile IDs must be non-negative")
        if self.gate_up_bytes <= 0 or self.down_bytes <= 0:
            raise ValidationError("both matrix fragments must have positive byte size")
        if self.size_bytes != self.gate_up_bytes + self.down_bytes:
            raise ValidationError("logical tile size must equal Gate-Up plus Down bytes")
        if self.predicted_rank < 0 or self.confidence < 0.0:
            raise ValidationError("rank and confidence must be non-negative")


@dataclass(frozen=True)
class TilePredictionResult:
    """A deterministic ranked candidate list supplied to a tile scheduler."""

    candidates: tuple[tuple[int, float], ...]

    def __post_init__(self) -> None:
        ids = [expert_id for expert_id, _ in self.candidates]
        if len(ids) != len(set(ids)) or any(expert_id < 0 for expert_id in ids):
            raise ValidationError("tile prediction candidates must be unique and non-negative")
        if any(score < 0.0 for _, score in self.candidates):
            raise ValidationError("tile prediction confidence must be non-negative")


@dataclass(frozen=True)
class TileSchedulingHardware:
    """Hardware limits visible to a pure tile allocator."""

    ocm_capacity_bytes: int
    bandwidth_bytes_per_cycle_per_channel: float
    dma_channels: int

    def __post_init__(self) -> None:
        if (
            self.ocm_capacity_bytes <= 0
            or self.bandwidth_bytes_per_cycle_per_channel <= 0
            or self.dma_channels <= 0
        ):
            raise ValidationError("tile scheduling hardware limits must be positive")


@dataclass(frozen=True)
class CurrentTimelineState:
    """Resource state at the instant a prediction becomes available."""

    current_cycle: int
    occupied_ocm_bytes: int = 0
    explicit_prefetch_budget_bytes: int | None = None
    prefetch_deadline_cycle: int | None = None

    def __post_init__(self) -> None:
        if self.current_cycle < 0 or self.occupied_ocm_bytes < 0:
            raise ValidationError("timeline cycles and occupancy must be non-negative")
        if self.explicit_prefetch_budget_bytes is not None and self.explicit_prefetch_budget_bytes < 0:
            raise ValidationError("explicit tile budget must be non-negative")
        if self.prefetch_deadline_cycle is not None and self.prefetch_deadline_cycle < self.current_cycle:
            raise ValidationError("prefetch deadline cannot precede the current cycle")


@dataclass(frozen=True)
class PlannedTileTransfer:
    """One logical tile in deterministic DMA issue order."""

    issue_order: int
    earliest_issue_cycle: int
    tile: ExpertTile

    @property
    def priority(self) -> int:
        """Return deterministic hardware queue priority; lower values issue first."""
        return self.issue_order


@dataclass(frozen=True)
class PrefetchPlan:
    """Byte-feasible allocation and deterministic logical-tile request order."""

    scheduler_name: str
    transfers: tuple[PlannedTileTransfer, ...]
    budget_bytes: int

    @property
    def allocated_bytes(self) -> int:
        """Return total planned Gate-Up plus Down payload bytes."""
        return sum(item.tile.size_bytes for item in self.transfers)

    @property
    def allocated_tiles(self) -> int:
        """Return number of planned logical tiles."""
        return len(self.transfers)

    @property
    def depths_by_expert(self) -> dict[int, int]:
        """Return allocated prefix depth for each candidate Expert."""
        result: dict[int, int] = {}
        for item in self.transfers:
            result[item.tile.expert_id] = max(
                result.get(item.tile.expert_id, 0), item.tile.tile_id + 1
            )
        return dict(sorted(result.items()))


class TileAllocationScheduler(Protocol):
    """Pure interface for prediction-aware logical-tile allocation."""

    def schedule(
        self,
        prediction: TilePredictionResult,
        expert_tiles: Mapping[int, Sequence[ExpertTile]],
        hardware: TileSchedulingHardware,
        timeline: CurrentTimelineState,
    ) -> PrefetchPlan:
        """Return a capacity-feasible deterministic prefetch plan."""
        ...


# Backward-compatible M8 name. New work should use TileAllocationScheduler.
TilePrefetchScheduler = TileAllocationScheduler


def _budget(hardware: TileSchedulingHardware, timeline: CurrentTimelineState) -> int:
    available = max(0, hardware.ocm_capacity_bytes - timeline.occupied_ocm_bytes)
    if timeline.explicit_prefetch_budget_bytes is not None:
        available = min(available, timeline.explicit_prefetch_budget_bytes)
    if timeline.prefetch_deadline_cycle is not None:
        window_cycles = timeline.prefetch_deadline_cycle - timeline.current_cycle
        bandwidth_envelope = int(
            window_cycles * hardware.bandwidth_bytes_per_cycle_per_channel
            * hardware.dma_channels
        )
        available = min(available, bandwidth_envelope)
    return available


def _ranked_geometry(
    prediction: TilePredictionResult,
    expert_tiles: Mapping[int, Sequence[ExpertTile]],
) -> list[tuple[int, float, tuple[ExpertTile, ...]]]:
    result = []
    for rank, (expert_id, confidence) in enumerate(prediction.candidates):
        if expert_id not in expert_tiles:
            raise ValidationError(f"missing tile geometry for candidate Expert {expert_id}")
        tiles = tuple(expert_tiles[expert_id])
        if not tiles or any(tile.tile_id != index for index, tile in enumerate(tiles)):
            raise ValidationError("candidate tiles must form a non-empty contiguous prefix geometry")
        if any(tile.predicted_rank != rank or tile.confidence != confidence for tile in tiles):
            raise ValidationError("tile rank/confidence must agree with prediction")
        result.append((expert_id, confidence, tiles))
    return result


def _plan(name: str, tiles: list[ExpertTile], budget: int, cycle: int) -> PrefetchPlan:
    plan = PrefetchPlan(
        name,
        tuple(PlannedTileTransfer(index, cycle, tile) for index, tile in enumerate(tiles)),
        budget,
    )
    if plan.allocated_bytes > budget:
        raise AssertionError("tile scheduler exceeded byte budget")
    depths: dict[int, set[int]] = {}
    for item in plan.transfers:
        depths.setdefault(item.tile.expert_id, set()).add(item.tile.tile_id)
    if any(ids != set(range(max(ids) + 1)) for ids in depths.values() if ids):
        raise AssertionError("sequential tile compute requires prefix-closed allocations")
    return plan


class NoPrefetchScheduler:
    """Issue no speculative tiles; all weight tiles are fetched on demand."""

    def schedule(self, prediction: TilePredictionResult,
                 expert_tiles: Mapping[int, Sequence[ExpertTile]],
                 hardware: TileSchedulingHardware,
                 timeline: CurrentTimelineState) -> PrefetchPlan:
        """Return an empty plan."""
        _ranked_geometry(prediction, expert_tiles)
        return _plan(type(self).__name__, [], _budget(hardware, timeline), timeline.current_cycle)


class FullExpertPrefetchScheduler:
    """Select rank-ordered Experts only when their complete tile set fits."""

    def schedule(self, prediction: TilePredictionResult,
                 expert_tiles: Mapping[int, Sequence[ExpertTile]],
                 hardware: TileSchedulingHardware,
                 timeline: CurrentTimelineState) -> PrefetchPlan:
        """Allocate complete Experts without partially admitting the next Expert."""
        budget = _budget(hardware, timeline)
        selected: list[ExpertTile] = []
        used = 0
        for _, _, tiles in _ranked_geometry(prediction, expert_tiles):
            expert_bytes = sum(tile.size_bytes for tile in tiles)
            if used + expert_bytes <= budget:
                selected.extend(tiles)
                used += expert_bytes
        return _plan(type(self).__name__, selected, budget, timeline.current_cycle)


class UniformTilePrefetchScheduler:
    """Allocate equal prefix depth using breadth-first tile rounds."""

    def schedule(self, prediction: TilePredictionResult,
                 expert_tiles: Mapping[int, Sequence[ExpertTile]],
                 hardware: TileSchedulingHardware,
                 timeline: CurrentTimelineState) -> PrefetchPlan:
        """Give every candidate tile zero before considering tile one."""
        budget = _budget(hardware, timeline)
        geometry = _ranked_geometry(prediction, expert_tiles)
        selected: list[ExpertTile] = []
        used = 0
        depth = 0
        while True:
            available = False
            added = False
            for _, _, tiles in geometry:
                if depth >= len(tiles):
                    continue
                available = True
                tile = tiles[depth]
                if used + tile.size_bytes <= budget:
                    selected.append(tile)
                    used += tile.size_bytes
                    added = True
            if not available or not added:
                break
            depth += 1
        return _plan(type(self).__name__, selected, budget, timeline.current_cycle)


class RankedTilePrefetchScheduler:
    """Greedily allocate prefix tiles by confidence-weighted marginal value/byte."""

    def __init__(self, depth_decay: float = 0.5) -> None:
        if not 0.0 < depth_decay <= 1.0:
            raise ValidationError("ranked tile depth_decay must be in (0, 1]")
        self.depth_decay = depth_decay

    def schedule(self, prediction: TilePredictionResult,
                 expert_tiles: Mapping[int, Sequence[ExpertTile]],
                 hardware: TileSchedulingHardware,
                 timeline: CurrentTimelineState) -> PrefetchPlan:
        """Choose the feasible next prefix tile with maximum marginal value/byte."""
        budget = _budget(hardware, timeline)
        geometry = _ranked_geometry(prediction, expert_tiles)
        depths = [0] * len(geometry)
        selected: list[ExpertTile] = []
        used = 0
        while True:
            choices: list[tuple[float, int, int, ExpertTile]] = []
            for rank, (_, confidence, tiles) in enumerate(geometry):
                depth = depths[rank]
                if depth >= len(tiles):
                    continue
                tile = tiles[depth]
                if used + tile.size_bytes > budget:
                    continue
                marginal = confidence * (self.depth_decay ** depth) / tile.size_bytes
                choices.append((-marginal, rank, tile.tile_id, tile))
            if not choices:
                break
            _, rank, _, tile = min(choices)
            selected.append(tile)
            depths[rank] += 1
            used += tile.size_bytes
        return _plan(type(self).__name__, selected, budget, timeline.current_cycle)


class BudgetProportionalScheduler:
    """Allocate per-Expert byte shares using quantized confidence and prefixes."""

    def __init__(self, confidence_bits: int = 12) -> None:
        if not 1 <= confidence_bits <= 24:
            raise ValidationError("confidence_bits must be in [1, 24]")
        self.confidence_bits = confidence_bits

    def schedule(self, prediction: TilePredictionResult,
                 expert_tiles: Mapping[int, Sequence[ExpertTile]],
                 hardware: TileSchedulingHardware,
                 timeline: CurrentTimelineState) -> PrefetchPlan:
        """Use integer proportional shares, then spend remainders by rank."""
        budget = _budget(hardware, timeline)
        geometry = _ranked_geometry(prediction, expert_tiles)
        scale = (1 << self.confidence_bits) - 1
        quantized = [max(1, round(confidence * scale)) for _, confidence, _ in geometry]
        total = sum(quantized)
        shares = [budget * confidence // total for confidence in quantized]
        depths = [0] * len(geometry)
        spent_by_expert = [0] * len(geometry)
        selected: list[ExpertTile] = []
        used = 0
        for rank, (_, _, tiles) in enumerate(geometry):
            for tile in tiles:
                if spent_by_expert[rank] + tile.size_bytes > shares[rank]:
                    break
                selected.append(tile)
                depths[rank] += 1
                spent_by_expert[rank] += tile.size_bytes
                used += tile.size_bytes
        # A rank-ordered residual pass avoids stranding capacity because of
        # whole-tile rounding while preserving every Expert's prefix.
        while True:
            added = False
            for rank, (_, _, tiles) in enumerate(geometry):
                if depths[rank] >= len(tiles):
                    continue
                tile = tiles[depths[rank]]
                if used + tile.size_bytes <= budget:
                    selected.append(tile)
                    depths[rank] += 1
                    spent_by_expert[rank] += tile.size_bytes
                    used += tile.size_bytes
                    added = True
            if not added:
                break
        return _plan(type(self).__name__, selected, budget, timeline.current_cycle)


class HardwareAwareGreedyScheduler:
    """PL-friendly fixed-point greedy scheduler with channel-load awareness."""

    def __init__(self, confidence_bits: int = 12, priority_shift_per_depth: int = 1) -> None:
        if not 1 <= confidence_bits <= 24 or priority_shift_per_depth < 0:
            raise ValidationError("invalid hardware-greedy quantization")
        self.confidence_bits = confidence_bits
        self.priority_shift_per_depth = priority_shift_per_depth

    def schedule(self, prediction: TilePredictionResult,
                 expert_tiles: Mapping[int, Sequence[ExpertTile]],
                 hardware: TileSchedulingHardware,
                 timeline: CurrentTimelineState) -> PrefetchPlan:
        """Greedily compare integer confidence*priority/cost without division."""
        geometry = _ranked_geometry(prediction, expert_tiles)
        # The one-tile compatibility path is exactly the original ranked rule.
        if geometry and all(len(tiles) == 1 for _, _, tiles in geometry):
            return RankedTilePrefetchScheduler().schedule(
                prediction, expert_tiles, hardware, timeline
            )
        budget = _budget(hardware, timeline)
        scale = (1 << self.confidence_bits) - 1
        confidence = [max(1, round(score * scale)) for _, score, _ in geometry]
        depths = [0] * len(geometry)
        channel_load = [0] * hardware.dma_channels
        selected: list[ExpertTile] = []
        used = 0
        while True:
            best: tuple[int, int, ExpertTile] | None = None
            best_numerator = 0
            best_cost = 1
            for rank, (expert_id, _, tiles) in enumerate(geometry):
                depth = depths[rank]
                if depth >= len(tiles):
                    continue
                tile = tiles[depth]
                if used + tile.size_bytes > budget:
                    continue
                priority = max(1, scale >> (depth * self.priority_shift_per_depth))
                numerator = confidence[rank] * priority
                # Round-robin Expert placement is the current M8 execution
                # mapping. The load factor is integer and requires no division.
                channel = expert_id % hardware.dma_channels
                cost = tile.size_bytes * (1 + channel_load[channel])
                if (
                    best is None
                    or numerator * best_cost > best_numerator * cost
                    or (
                        numerator * best_cost == best_numerator * cost
                        and (rank, tile.tile_id, expert_id) <
                        (best[0], best[2].tile_id, best[2].expert_id)
                    )
                ):
                    best = (rank, channel, tile)
                    best_numerator, best_cost = numerator, cost
            if best is None:
                break
            rank, channel, tile = best
            selected.append(tile)
            depths[rank] += 1
            channel_load[channel] += 1
            used += tile.size_bytes
        return _plan(type(self).__name__, selected, budget, timeline.current_cycle)


class CriticalityAwareTileScheduler:
    """PL-friendly greedy allocator weighted by estimated critical-path urgency.

    The scheduler uses only prediction, tile geometry, aggregate DMA state, and
    the prediction-to-Router deadline.  It never observes routing truth.  For a
    feasible next prefix tile, its integer score is proportional to::

        confidence * depth_priority * execution_urgency * transfer_cycles
        -----------------------------------------------------------------
          tile_bytes * remaining_slack * (1 + channel_load_tiles)

    Candidate scores are compared by cross multiplication, so the online path
    needs no divider or floating-point unit.
    """

    hardware_implementable = True

    def __init__(
        self, confidence_bits: int = 12, priority_shift_per_depth: int = 1,
    ) -> None:
        if not 1 <= confidence_bits <= 24 or priority_shift_per_depth < 0:
            raise ValidationError("invalid criticality-aware quantization")
        self.confidence_bits = confidence_bits
        self.priority_shift_per_depth = priority_shift_per_depth

    def schedule(
        self,
        prediction: TilePredictionResult,
        expert_tiles: Mapping[int, Sequence[ExpertTile]],
        hardware: TileSchedulingHardware,
        timeline: CurrentTimelineState,
    ) -> PrefetchPlan:
        """Allocate prefix tiles using a deterministic criticality estimate."""
        geometry = _ranked_geometry(prediction, expert_tiles)
        budget = _budget(hardware, timeline)
        scale = (1 << self.confidence_bits) - 1
        confidence = [max(1, round(score * scale)) for _, score, _ in geometry]
        depths = [0] * len(geometry)
        channel_load_tiles = [0] * hardware.dma_channels
        channel_load_bytes = [0] * hardware.dma_channels
        selected: list[ExpertTile] = []
        used = 0
        lead_cycles = max(
            1,
            (timeline.prefetch_deadline_cycle or timeline.current_cycle + 1)
            - timeline.current_cycle,
        )

        while True:
            best: tuple[int, int, ExpertTile] | None = None
            best_numerator = 0
            best_denominator = 1
            for rank, (expert_id, _, tiles) in enumerate(geometry):
                depth = depths[rank]
                if depth >= len(tiles):
                    continue
                tile = tiles[depth]
                if used + tile.size_bytes > budget:
                    continue
                channel = expert_id % hardware.dma_channels
                transfer_cycles = max(
                    1,
                    ceil(
                        tile.size_bytes
                        / hardware.bandwidth_bytes_per_cycle_per_channel
                    ),
                )
                projected_finish = ceil(
                    (channel_load_bytes[channel] + tile.size_bytes)
                    / hardware.bandwidth_bytes_per_cycle_per_channel
                )
                remaining_slack = max(1, lead_cycles - projected_finish + 1)
                execution_urgency = len(tiles) - depth
                depth_priority = max(
                    1, scale >> (depth * self.priority_shift_per_depth)
                )
                numerator = (
                    confidence[rank]
                    * depth_priority
                    * execution_urgency
                    * transfer_cycles
                )
                denominator = (
                    tile.size_bytes
                    * remaining_slack
                    * (1 + channel_load_tiles[channel])
                )
                if (
                    best is None
                    or numerator * best_denominator
                    > best_numerator * denominator
                    or (
                        numerator * best_denominator
                        == best_numerator * denominator
                        and (rank, tile.tile_id, expert_id)
                        < (best[0], best[2].tile_id, best[2].expert_id)
                    )
                ):
                    best = (rank, channel, tile)
                    best_numerator = numerator
                    best_denominator = denominator
            if best is None:
                break
            rank, channel, tile = best
            selected.append(tile)
            depths[rank] += 1
            channel_load_tiles[channel] += 1
            channel_load_bytes[channel] += tile.size_bytes
            used += tile.size_bytes
        return _plan(type(self).__name__, selected, budget, timeline.current_cycle)


class FixedPrefetchPlanScheduler:
    """Replay a prefix-closed plan identity through the normal event engine."""

    def __init__(self, tile_order: Sequence[tuple[int, int]], name: str = "fixed_plan") -> None:
        self.tile_order = tuple(tile_order)
        self.name = name

    def schedule(self, prediction: TilePredictionResult,
                 expert_tiles: Mapping[int, Sequence[ExpertTile]],
                 hardware: TileSchedulingHardware,
                 timeline: CurrentTimelineState) -> PrefetchPlan:
        """Rebuild a stored Expert/tile order against current geometry."""
        _ranked_geometry(prediction, expert_tiles)
        selected: list[ExpertTile] = []
        seen: dict[int, set[int]] = {}
        for expert_id, tile_id in self.tile_order:
            if expert_id not in expert_tiles or not 0 <= tile_id < len(expert_tiles[expert_id]):
                raise ValidationError("fixed plan references missing Expert tile")
            if tile_id in seen.setdefault(expert_id, set()):
                raise ValidationError("fixed plan contains a duplicate tile")
            seen[expert_id].add(tile_id)
            selected.append(expert_tiles[expert_id][tile_id])
        budget = _budget(hardware, timeline)
        return _plan(self.name, selected, budget, timeline.current_cycle)


@dataclass(frozen=True)
class OracleSearchResult:
    """Offline-only exhaustive search outcome."""

    plan: PrefetchPlan
    latency_cycles: int
    evaluated_plans: int


class OracleTileAllocationScheduler:
    """Offline latency oracle; never a hardware implementation candidate."""

    hardware_implementable = False

    def __init__(self, true_expert_ids: Sequence[int]) -> None:
        if not true_expert_ids or len(set(true_expert_ids)) != len(true_expert_ids):
            raise ValidationError("oracle requires unique true Expert IDs")
        self.true_expert_ids = tuple(true_expert_ids)

    def candidate_plans(
        self, prediction: TilePredictionResult,
        expert_tiles: Mapping[int, Sequence[ExpertTile]],
        hardware: TileSchedulingHardware,
        timeline: CurrentTimelineState,
    ) -> tuple[PrefetchPlan, ...]:
        """Enumerate feasible true-Expert prefix depths and two issue orders."""
        _ranked_geometry(prediction, expert_tiles)
        if any(expert not in expert_tiles for expert in self.true_expert_ids):
            raise ValidationError("oracle truth must occur in candidate geometry")
        budget = _budget(hardware, timeline)
        ranges = [range(len(expert_tiles[expert]) + 1) for expert in self.true_expert_ids]
        unique: dict[tuple[tuple[int, int], ...], PrefetchPlan] = {}
        for depths in product(*ranges):
            selected_bytes = sum(
                tile.size_bytes
                for expert, depth in zip(self.true_expert_ids, depths)
                for tile in expert_tiles[expert][:depth]
            )
            if selected_bytes > budget:
                continue
            orders: list[list[ExpertTile]] = []
            # Breadth-first order exposes startup tiles of both true Experts.
            breadth = [
                expert_tiles[expert][tile_id]
                for tile_id in range(max(depths, default=0))
                for expert, depth in zip(self.true_expert_ids, depths)
                if tile_id < depth
            ]
            orders.append(breadth)
            # Expert-major order can be better with one DMA channel.
            orders.append([
                tile for expert, depth in zip(self.true_expert_ids, depths)
                for tile in expert_tiles[expert][:depth]
            ])
            for order in orders:
                signature = tuple((tile.expert_id, tile.tile_id) for tile in order)
                unique.setdefault(
                    signature,
                    _plan(type(self).__name__, order, budget, timeline.current_cycle),
                )
        return tuple(unique[key] for key in sorted(unique))

    def optimize(
        self, prediction: TilePredictionResult,
        expert_tiles: Mapping[int, Sequence[ExpertTile]],
        hardware: TileSchedulingHardware,
        timeline: CurrentTimelineState,
        evaluate_latency: Callable[[PrefetchPlan], int],
        extra_plans: Sequence[PrefetchPlan] = (),
    ) -> OracleSearchResult:
        """Evaluate every candidate with the event simulator and select latency minimum."""
        candidates = list(self.candidate_plans(prediction, expert_tiles, hardware, timeline))
        signatures = {
            tuple((item.tile.expert_id, item.tile.tile_id) for item in plan.transfers)
            for plan in candidates
        }
        for plan in extra_plans:
            signature = tuple((item.tile.expert_id, item.tile.tile_id) for item in plan.transfers)
            if plan.allocated_bytes <= _budget(hardware, timeline) and signature not in signatures:
                candidates.append(plan); signatures.add(signature)
        scored = [(evaluate_latency(plan), plan) for plan in candidates]
        latency, plan = min(scored, key=lambda item: (
            item[0], item[1].allocated_bytes,
            tuple((entry.tile.expert_id, entry.tile.tile_id) for entry in item[1].transfers),
        ))
        return OracleSearchResult(plan, latency, len(scored))


class FixedPredictionOracleScheduler:
    """Offline allocation oracle constrained to an unchanged predictor output.

    The search never receives routing truth.  It retains the supplied candidate
    set and rank order while enumerating rectangular breadth/depth allocations
    in breadth-major and Expert-major DMA order.  Callers may add existing
    online plans to the candidate set so the bound is never worse than a
    reported online scheduler.
    """

    hardware_implementable = False

    def candidate_plans(
        self,
        prediction: TilePredictionResult,
        expert_tiles: Mapping[int, Sequence[ExpertTile]],
        hardware: TileSchedulingHardware,
        timeline: CurrentTimelineState,
    ) -> tuple[PrefetchPlan, ...]:
        """Enumerate rank-preserving breadth/depth allocations and DMA orders."""
        geometry = _ranked_geometry(prediction, expert_tiles)
        budget = _budget(hardware, timeline)
        unique: dict[tuple[tuple[int, int], ...], PrefetchPlan] = {}

        def add(order: list[ExpertTile]) -> None:
            if sum(tile.size_bytes for tile in order) > budget:
                return
            signature = tuple((tile.expert_id, tile.tile_id) for tile in order)
            unique.setdefault(
                signature,
                _plan(type(self).__name__, order, budget, timeline.current_cycle),
            )

        add([])
        for breadth in range(1, len(geometry) + 1):
            active = geometry[:breadth]
            max_depth = max(len(tiles) for _, _, tiles in active)
            for depth in range(1, max_depth + 1):
                breadth_major = [
                    tiles[tile_id]
                    for tile_id in range(depth)
                    for _, _, tiles in active
                    if tile_id < len(tiles)
                ]
                expert_major = [
                    tile
                    for _, _, tiles in active
                    for tile in tiles[:depth]
                ]
                add(breadth_major)
                add(expert_major)
        return tuple(unique[key] for key in sorted(unique))

    def optimize(
        self,
        prediction: TilePredictionResult,
        expert_tiles: Mapping[int, Sequence[ExpertTile]],
        hardware: TileSchedulingHardware,
        timeline: CurrentTimelineState,
        evaluate_latency: Callable[[PrefetchPlan], int],
        extra_plans: Sequence[PrefetchPlan] = (),
    ) -> OracleSearchResult:
        """Select minimum event-simulated latency without inspecting truth IDs."""
        candidates = list(
            self.candidate_plans(prediction, expert_tiles, hardware, timeline)
        )
        signatures = {
            tuple((item.tile.expert_id, item.tile.tile_id) for item in plan.transfers)
            for plan in candidates
        }
        budget = _budget(hardware, timeline)
        for plan in extra_plans:
            signature = tuple(
                (item.tile.expert_id, item.tile.tile_id) for item in plan.transfers
            )
            if plan.allocated_bytes <= budget and signature not in signatures:
                candidates.append(plan)
                signatures.add(signature)
        scored = [(evaluate_latency(plan), plan) for plan in candidates]
        latency, plan = min(
            scored,
            key=lambda item: (
                item[0],
                item[1].allocated_bytes,
                tuple(
                    (entry.tile.expert_id, entry.tile.tile_id)
                    for entry in item[1].transfers
                ),
            ),
        )
        return OracleSearchResult(plan, latency, len(scored))


@dataclass(frozen=True)
class SchedulerComplexityEstimate:
    """Qualitative operation/state estimate for one online allocation decision."""

    scheduler: str
    comparisons: int | str
    multiplications: int | str
    divisions: int | str
    additions: int | str
    state_entries: int | str
    estimated_control_cycles: int | str
    hardware_friendly: bool
    notes: str


def estimate_scheduler_complexity(
    scheduler: str, candidate_count: int, tiles_per_expert: int,
    allocated_tiles: int,
) -> SchedulerComplexityEstimate:
    """Estimate primitive operations; values are not synthesis results."""
    c = candidate_count
    t = tiles_per_expert
    s = allocated_tiles
    if scheduler in {"oracle", "fixed_prediction_oracle"}:
        return SchedulerComplexityEstimate(
            scheduler, "exhaustive", "exhaustive", "offline", "exhaustive",
            "offline", "offline", False, "analysis upper bound only",
        )
    if scheduler == "criticality_aware":
        compare = s * max(0, c - 1)
        cycles = c + s * (max(1, ceil(log2(max(2, c)))) + 5)
        return SchedulerComplexityEstimate(
            scheduler, compare, 4 * compare, 0, 3 * c + 2 * s,
            4 * c, cycles, True,
            "fixed-point criticality; ratios compared by cross multiplication",
        )
    if scheduler == "hardware_greedy":
        compare = s * max(0, c - 1)
        cycles = c + s * (max(1, ceil(log2(max(2, c)))) + 2)
        return SchedulerComplexityEstimate(
            scheduler, compare, 2 * compare, 0, c + s, c + max(1, c),
            cycles, True,
            "fixed-point score; ratio compared by cross multiplication; channel-load counters",
        )
    if scheduler == "budget_proportional":
        return SchedulerComplexityEstimate(
            scheduler, c + s, c, c, 2 * c + s, 3 * c,
            10 * c + s, True,
            "integer confidence accumulator and fixed-latency divider/reciprocal",
        )
    if scheduler == "ranked_tile":
        compare = s * max(0, c - 1)
        return SchedulerComplexityEstimate(
            scheduler, compare, s * c, s * c, s * c, 2 * c,
            c + s * (max(1, ceil(log2(max(2, c)))) + 3), False,
            "software floating-point heuristic retained as M8 reference",
        )
    if scheduler == "uniform_tile":
        return SchedulerComplexityEstimate(
            scheduler, c * t, 0, 0, c * t, c,
            c + s, True, "nested rank/depth counters",
        )
    if scheduler == "full_expert":
        return SchedulerComplexityEstimate(
            scheduler, c, 0, 0, c, c,
            2 * c, True, "rank scan plus capacity accumulator",
        )
    return SchedulerComplexityEstimate(
        scheduler, 0, 0, 0, 0, 1, 1, True, "no speculative allocation",
    )


def scheduler_by_name(name: str) -> TilePrefetchScheduler:
    """Construct one of the four canonical M8 schedulers."""
    schedulers: dict[str, TilePrefetchScheduler] = {
        "no_prefetch": NoPrefetchScheduler(),
        "full_expert": FullExpertPrefetchScheduler(),
        "uniform_tile": UniformTilePrefetchScheduler(),
        "ranked_tile": RankedTilePrefetchScheduler(),
        "budget_proportional": BudgetProportionalScheduler(),
        "hardware_greedy": HardwareAwareGreedyScheduler(),
        "criticality_aware": CriticalityAwareTileScheduler(),
    }
    try:
        return schedulers[name]
    except KeyError as exc:
        raise ValidationError(f"unknown tile scheduler: {name}") from exc
