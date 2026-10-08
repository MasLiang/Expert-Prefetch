"""Small-case exhaustive Oracle Gate-Up prefetch-depth search."""

from __future__ import annotations

import itertools
from dataclasses import dataclass

from .dma_scheduler import DMARequestScheduler, TileCriticalPathAwareScheduler
from .prefetch_policy import PredictorSpecifiedPolicy
from .scheduler import ExpertScheduler, ReadyFirstScheduler
from .schema import (
    CandidatePrefetch, ExpertTask, PredictorOutput, PrefetchConfig, RoutingEvent,
    SimulationConfig, TileConfig, ValidationError,
)
from .simulator import Scenario, SimulationOutput, Simulator
from .tiling import FixedByteTilingPolicy, WeightTilingPolicy


@dataclass(frozen=True)
class OracleOptimalDepthResult:
    """Best feasible depth vector and its simulation output."""

    optimal_depths: dict[int, int]
    optimal_cycles: int
    evaluated_combinations: int
    output: SimulationOutput


class _DepthOracle:
    def __init__(self, candidates: tuple[CandidatePrefetch, ...], ready_offset_cycles: int) -> None:
        self.candidates = candidates
        self.ready_offset_cycles = ready_offset_cycles

    def predict(self, event: RoutingEvent, layer_start_cycle: int) -> PredictorOutput:
        return PredictorOutput(
            event.request_id, event.token_position, event.layer_id,
            ready_offset_cycles=self.ready_offset_cycles,
            candidates=self.candidates,
        )


def optimize_oracle_depth(
    simulator: Simulator,
    event: RoutingEvent,
    *,
    tile_config: TileConfig,
    byte_budget: int,
    ready_offset_cycles: int = 0,
    tiling_policy: WeightTilingPolicy | None = None,
    dma_scheduler: DMARequestScheduler | None = None,
    expert_scheduler: ExpertScheduler | None = None,
) -> OracleOptimalDepthResult:
    """Exhaustively choose true-Expert Gate-Up prefix depths under a byte budget."""
    if len(event.true_expert_ids) > 4:
        raise ValidationError("OracleOptimalDepth supports K <= 4")
    if byte_budget <= 0:
        raise ValidationError("OracleOptimalDepth byte_budget must be positive")
    policy = tiling_policy or FixedByteTilingPolicy()
    gate_tiles = {}
    for expert_id, score in zip(event.true_expert_ids, event.true_gating_weights):
        task = ExpertTask(
            event.layer_id, expert_id, event.expert_token_counts[expert_id], score
        )
        gate, _ = policy.build_tiles(task, simulator.model, tile_config)
        if len(gate) > 8:
            raise ValidationError("OracleOptimalDepth supports at most 8 Gate-Up tiles per Expert")
        gate_tiles[expert_id] = gate
    best: tuple[int, tuple[int, ...], SimulationOutput] | None = None
    evaluated = 0
    ranges = [range(len(gate_tiles[expert_id]) + 1) for expert_id in event.true_expert_ids]
    for depths in itertools.product(*ranges):
        used_bytes = sum(
            sum(tile.size_bytes for tile in gate_tiles[expert_id][:depth])
            for expert_id, depth in zip(event.true_expert_ids, depths)
        )
        if used_bytes > byte_budget:
            continue
        evaluated += 1
        candidates = tuple(
            CandidatePrefetch(expert_id, score, depth, 0)
            for expert_id, score, depth in zip(
                event.true_expert_ids, event.true_gating_weights, depths
            )
        )
        output = simulator.simulate(
            event,
            Scenario.ORACLE,
            predictor=_DepthOracle(candidates, ready_offset_cycles),
            simulation=SimulationConfig("tile"),
            tile_config=tile_config,
            tiling_policy=policy,
            prefetch_policy=PredictorSpecifiedPolicy(None, None),
            prefetch=PrefetchConfig(
                policy="predictor_specified", max_candidates=len(candidates)
            ),
            dma_scheduler=dma_scheduler or TileCriticalPathAwareScheduler(),
            expert_scheduler=expert_scheduler or ReadyFirstScheduler(),
        )
        candidate = (output.result.total_cycles, tuple(depths), output)
        if best is None or candidate[:2] < best[:2]:
            best = candidate
    if best is None:
        raise RuntimeError("OracleOptimalDepth found no feasible combination")
    return OracleOptimalDepthResult(
        dict(zip(event.true_expert_ids, best[1])), best[0], evaluated, best[2]
    )
