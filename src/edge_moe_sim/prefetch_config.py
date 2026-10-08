"""Independent allocation and physical DMA ordering controls."""
from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Literal, Mapping, Sequence

from .schema import ValidationError
from .hardware_greedy_v1 import HardwareGreedyV1Candidate, HardwareGreedyV1Config, hardware_greedy_v1
from .tile_scheduler import (
    CurrentTimelineState, ExpertTile, FullExpertPrefetchScheduler, PlannedTileTransfer,
    PrefetchPlan, RankedTilePrefetchScheduler, TileAllocationScheduler,
    TilePredictionResult, TileSchedulingHardware, UniformTilePrefetchScheduler,
)


@dataclass(frozen=True)
class PrefetchConfig:
    """Allocator selection; credits are payload bytes, not allocated OCM slots.

    Legacy preserves the historical prediction-spec path. Atomic RTL admission
    supports pair-first only; up-first is the independent-request research path.
    """

    allocator: Literal['legacy', 'full_expert', 'uniform', 'ranked', 'hardware_greedy_v1'] = 'legacy'
    dma_order: Literal['legacy', 'up_first', 'pair_first'] = 'legacy'
    ranked_depth_decay: float = 0.5
    confidence_encoding: Literal['probability', 'q12'] = 'probability'
    priority_shift_per_depth: int = 1
    channel_credit_bytes: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if self.allocator not in {'legacy', 'full_expert', 'uniform', 'ranked', 'hardware_greedy_v1'}:
            raise ValidationError('unknown prefetch allocator')
        if self.dma_order not in {'legacy', 'up_first', 'pair_first'}:
            raise ValidationError('unknown prefetch DMA order')
        if self.confidence_encoding not in {'probability', 'q12'}:
            raise ValidationError('unknown confidence encoding')
        if not isfinite(self.ranked_depth_decay) or not 0 < self.ranked_depth_decay <= 1:
            raise ValidationError('ranked_depth_decay must be in (0, 1]')
        if type(self.priority_shift_per_depth) is not int or self.priority_shift_per_depth < 0:
            raise ValidationError('priority_shift_per_depth must be a non-negative integer')
        if any(type(value) is not int or value < 0 for value in self.channel_credit_bytes):
            raise ValidationError('channel credits must be non-negative integer bytes')
        if self.channel_credit_bytes and self.allocator != 'hardware_greedy_v1':
            raise ValidationError('channel credits require hardware_greedy_v1')


@dataclass(frozen=True)
class HardwareGreedyV1Scheduler:
    """Adapt the frozen integer golden allocator, without floating comparisons."""

    config: PrefetchConfig
    logical_channels: int

    def schedule(self, prediction: TilePredictionResult,
                 expert_tiles: Mapping[int, Sequence[ExpertTile]],
                 hardware: TileSchedulingHardware,
                 state: CurrentTimelineState) -> PrefetchPlan:
        """Allocate complete logical prefixes using Q12 confidence and byte credits."""
        budget = max(0, hardware.ocm_capacity_bytes - state.occupied_ocm_bytes)
        if state.explicit_prefetch_budget_bytes is not None:
            budget = min(budget, state.explicit_prefetch_budget_bytes)
        if state.prefetch_deadline_cycle is not None:
            raise ValidationError('v1 adapter does not infer credits from a future deadline')
        candidates = []
        for rank, (expert, score) in enumerate(prediction.candidates):
            if not isfinite(score):
                raise ValidationError('confidence must be finite')
            if self.config.confidence_encoding == 'q12':
                if score != int(score) or not 1 <= score <= 4095:
                    raise ValidationError('q12 confidence must be integer in [1,4095]')
                encoded = int(score)
            else:
                if not 0 <= score <= 1:
                    raise ValidationError('probability confidence must be in [0,1]')
                encoded = max(1, round(score * 4095))
            tiles = tuple(expert_tiles[expert])
            if not tiles or any(t.tile_id != i or t.expert_id != expert for i, t in enumerate(tiles)):
                raise ValidationError('v1 requires contiguous tile geometry')
            candidates.append(HardwareGreedyV1Candidate(expert, rank, encoded,
                              tuple(tile.size_bytes for tile in tiles)))
        credits = self.config.channel_credit_bytes or (budget,) * self.logical_channels
        if len(credits) != self.logical_channels:
            raise ValidationError('channel credits must match logical allocation channels')
        allocations = hardware_greedy_v1(candidates, HardwareGreedyV1Config(
            self.logical_channels, budget, credits, self.config.priority_shift_per_depth))
        return PrefetchPlan('HardwareGreedy-v1', tuple(
            PlannedTileTransfer(i, state.current_cycle, expert_tiles[a.expert_id][a.tile_id])
            for i, a in enumerate(allocations)), budget)


def make_prefetch_scheduler(config: PrefetchConfig, placement: str,
                            physical_channels: int) -> TileAllocationScheduler | None:
    """Create an allocator; a striped logical tile uses one aggregate channel."""
    if config.allocator == 'legacy':
        return None
    if config.allocator == 'full_expert':
        return FullExpertPrefetchScheduler()
    if config.allocator == 'uniform':
        return UniformTilePrefetchScheduler()
    if config.allocator == 'ranked':
        return RankedTilePrefetchScheduler(config.ranked_depth_decay)
    return HardwareGreedyV1Scheduler(config,
        physical_channels if placement == 'round_robin_expert' else 1)
