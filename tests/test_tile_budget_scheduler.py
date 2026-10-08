"""M8 resource-aware tile allocation and execution invariants."""
from __future__ import annotations

from collections import defaultdict

import pytest

from edge_moe_sim.advanced_schema import AdvancedExecutionConfig, AdvancedScenario
from edge_moe_sim.advanced_simulator import AdvancedSimulator
from edge_moe_sim.batch_predictor import (
    BatchPredictorOutput, PrecomputedBatchPredictor, TokenPrediction,
)
from edge_moe_sim.batch_routing import BatchRoutingEvent
from edge_moe_sim.hardware_profile import HardwareProfile
from edge_moe_sim.schema import CandidatePrefetch, ModelConfig
from edge_moe_sim.tile_scheduler import (
    BudgetProportionalScheduler, CurrentTimelineState, ExpertTile,
    FullExpertPrefetchScheduler, HardwareAwareGreedyScheduler,
    NoPrefetchScheduler, OracleTileAllocationScheduler,
    RankedTilePrefetchScheduler, TilePredictionResult,
    TileSchedulingHardware, UniformTilePrefetchScheduler,
    estimate_scheduler_complexity,
)


def geometry(candidates: tuple[tuple[int, float], ...], tiles: int = 4) -> dict[int, tuple[ExpertTile, ...]]:
    """Build equal 12-byte logical tile geometry for pure allocator tests."""
    return {
        expert: tuple(
            ExpertTile(expert, tile, 12, 8, 4, rank, confidence)
            for tile in range(tiles)
        )
        for rank, (expert, confidence) in enumerate(candidates)
    }


def test_tile_accounting_conservation_and_capacity() -> None:
    """All allocators are deterministic, prefix-closed, and byte feasible."""
    candidates = ((0, 0.95), (1, 0.8), (2, 0.5), (3, 0.2))
    prediction = TilePredictionResult(candidates)
    tiles = geometry(candidates)
    hardware = TileSchedulingHardware(96, 8.0, 2)
    timeline = CurrentTimelineState(7)
    schedulers = (
        NoPrefetchScheduler(), FullExpertPrefetchScheduler(),
        UniformTilePrefetchScheduler(), RankedTilePrefetchScheduler(),
    )
    for scheduler in schedulers:
        first = scheduler.schedule(prediction, tiles, hardware, timeline)
        second = scheduler.schedule(prediction, tiles, hardware, timeline)
        assert first == second
        assert first.allocated_bytes == sum(item.tile.size_bytes for item in first.transfers)
        assert first.allocated_bytes <= first.budget_bytes <= hardware.ocm_capacity_bytes
        for expert, depth in first.depths_by_expert.items():
            assert {item.tile.tile_id for item in first.transfers if item.tile.expert_id == expert} == set(range(depth))
    uniform = UniformTilePrefetchScheduler().schedule(prediction, tiles, hardware, timeline)
    assert uniform.depths_by_expert == {0: 2, 1: 2, 2: 2, 3: 2}
    ranked = RankedTilePrefetchScheduler().schedule(prediction, tiles, hardware, timeline)
    depths = [ranked.depths_by_expert.get(expert, 0) for expert, _ in candidates]
    assert depths == sorted(depths, reverse=True)
    assert depths[0] > depths[-1]


def test_bandwidth_window_caps_allocator_budget() -> None:
    """The pure allocation budget includes aggregate bandwidth times lead window."""
    candidates = ((0, 1.0), (1, 0.8))
    plan = UniformTilePrefetchScheduler().schedule(
        TilePredictionResult(candidates), geometry(candidates),
        TileSchedulingHardware(4096, 4.0, 2),
        CurrentTimelineState(10, prefetch_deadline_cycle=16),
    )
    assert plan.budget_bytes == 48
    assert plan.allocated_bytes == 48


def test_pl_friendly_schedulers_are_deterministic_and_feasible() -> None:
    """Proportional and integer-greedy policies obey the same plan constraints."""
    candidates = ((0, 0.9), (1, 0.6), (2, 0.3), (3, 0.1))
    prediction = TilePredictionResult(candidates); tiles = geometry(candidates)
    hardware = TileSchedulingHardware(96, 8.0, 2); timeline = CurrentTimelineState(0)
    proportional = BudgetProportionalScheduler().schedule(prediction, tiles, hardware, timeline)
    greedy = HardwareAwareGreedyScheduler().schedule(prediction, tiles, hardware, timeline)
    for plan in (proportional, greedy):
        assert plan.allocated_bytes <= 96
    assert proportional == BudgetProportionalScheduler().schedule(prediction, tiles, hardware, timeline)
    assert greedy == HardwareAwareGreedyScheduler().schedule(prediction, tiles, hardware, timeline)
    assert proportional.depths_by_expert.get(0, 0) >= proportional.depths_by_expert.get(3, 0)
    assert greedy.depths_by_expert.get(0, 0) >= greedy.depths_by_expert.get(3, 0)


def test_hardware_greedy_one_tile_matches_ranked() -> None:
    """The required one-tile compatibility path exactly reuses M8 ranked behavior."""
    candidates = ((0, 0.9), (1, 0.7), (2, 0.4))
    prediction = TilePredictionResult(candidates); tiles = geometry(candidates, tiles=1)
    hardware = TileSchedulingHardware(24, 8.0, 2); timeline = CurrentTimelineState(0)
    old = RankedTilePrefetchScheduler().schedule(prediction, tiles, hardware, timeline)
    new = HardwareAwareGreedyScheduler().schedule(prediction, tiles, hardware, timeline)
    assert [(x.tile.expert_id, x.tile.tile_id) for x in new.transfers] == [(x.tile.expert_id, x.tile.tile_id) for x in old.transfers]
    assert new.allocated_bytes == old.allocated_bytes


def test_oracle_exhaustive_selection_and_complexity_labels() -> None:
    """Offline oracle evaluates all feasible prefixes and is never marked hardware friendly."""
    candidates = ((0, 1.0), (1, 0.8), (2, 0.2))
    prediction = TilePredictionResult(candidates); tiles = geometry(candidates, tiles=2)
    hardware = TileSchedulingHardware(48, 8.0, 1); timeline = CurrentTimelineState(0)
    oracle = OracleTileAllocationScheduler((0, 1))
    result = oracle.optimize(
        prediction, tiles, hardware, timeline,
        lambda plan: 100 - 10 * plan.depths_by_expert.get(0, 0) - 5 * plan.depths_by_expert.get(1, 0),
    )
    assert result.latency_cycles == 70
    assert result.plan.depths_by_expert == {0: 2, 1: 2}
    assert result.evaluated_plans > 1
    assert estimate_scheduler_complexity("hardware_greedy", 16, 16, 16).divisions == 0
    assert estimate_scheduler_complexity("oracle", 16, 16, 16).hardware_friendly is False


def profile(*, channels: int = 2, ocm: int = 2048) -> HardwareProfile:
    """Return a small generic synthetic profile."""
    return HardwareProfile(
        "m8_test_generic", 1000, channels, 8_000, 1.0, 2, 1, 64,
        True, 4, 8, 1.0, 1, ocm, hardware_tag="synthetic_placeholder",
    )


def predictor(candidates: tuple[CandidatePrefetch, ...]) -> PrecomputedBatchPredictor:
    """Build one ready-at-layer-start single-token predictor."""
    output = BatchPredictorOutput(1, 0, 0, 0, (TokenPrediction(0, candidates),))
    return PrecomputedBatchPredictor({(1, 0, 0): output})


def run(scheduler, *, channels: int = 2, f_tile_size: int = 2, model_f: int = 8):
    """Execute one single-token layer with an explicit tile scheduler."""
    model = ModelConfig("m8_test", 4, model_f, 8, 1, 8, 8)
    event = BatchRoutingEvent(1, 0, 0, ((0,),))
    config = AdvancedExecutionConfig(
        f_tile_size, aie_units_per_expert=1, max_parallel_experts=1,
        candidate_count=4, startup_gu_tiles=1,
    )
    candidates = (
        CandidatePrefetch(0, 0.9), CandidatePrefetch(1, 0.8),
        CandidatePrefetch(2, 0.7), CandidatePrefetch(3, 0.6),
    )
    return AdvancedSimulator(model, profile(channels=channels)).simulate(
        event, AdvancedScenario.PREDICTIVE_DSE, config,
        predictor=predictor(candidates), tile_prefetch_scheduler=scheduler,
    )


def test_executed_tile_conservation_ocm_and_dma_channels() -> None:
    """Physical speculative tiles conserve labels and respect OCM/channel exclusion."""
    result = run(RankedTilePrefetchScheduler(), channels=2)
    metrics = result.tile_metrics
    assert metrics["prefetched_tiles"] == metrics["useful_prefetched_tiles"] + metrics["wasted_prefetched_tiles"]
    assert metrics["prefetched_bytes"] == metrics["useful_prefetched_bytes"] + metrics["wasted_prefetched_bytes"]
    assert metrics["allocated_tile_transfers"] == metrics["useful_tile_transfers"] + metrics["wasted_tile_transfers"] + metrics["cancelled_tile_transfers"]
    assert metrics["allocated_transfer_bytes"] == metrics["useful_prefetched_bytes"] + metrics["wasted_prefetched_bytes"] + metrics["cancelled_transfer_bytes"]
    assert result.peak_ocm_bytes <= 2048
    by_channel: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for interval in (entry for entry in result.timeline if entry.resource == "MEMORY"):
        channels = interval.operation.rsplit("ch", 1)[1]
        for channel in map(int, channels.split("-")):
            by_channel[channel].append((interval.start_cycle, interval.end_cycle))
    assert set(by_channel) <= {0, 1}
    for intervals in by_channel.values():
        ordered = sorted(intervals)
        assert all(left[1] <= right[0] for left, right in zip(ordered, ordered[1:]))


def test_tile_compute_dependency_invariant() -> None:
    """Every tile compute reservation begins no earlier than its weight-ready cycle."""
    result = run(UniformTilePrefetchScheduler())
    compute = [entry for entry in result.timeline if entry.resource == "AIE_POOL" and entry.expert_id is not None]
    assert compute
    assert all(entry.ready_cycle is not None and entry.start_cycle >= entry.ready_cycle for entry in compute)
    per_expert_tile = defaultdict(dict)
    for entry in compute:
        matrix = "gate_up" if "gate_up" in entry.operation else "down"
        per_expert_tile[(entry.expert_id, entry.tile_id)][matrix] = entry
    assert all(pair["down"].start_cycle >= pair["gate_up"].end_cycle for pair in per_expert_tile.values())


def test_one_tile_full_expert_matches_legacy_policy() -> None:
    """Explicit full-Expert scheduling matches the old policy for one logical tile."""
    model = ModelConfig("one_tile", 4, 4, 4, 1, 8, 8)
    event = BatchRoutingEvent(1, 0, 0, ((0,),))
    config = AdvancedExecutionConfig(
        4, aie_units_per_expert=1, max_parallel_experts=1, candidate_count=1,
    )
    pred = predictor((CandidatePrefetch(0, 1.0),))
    simulator = AdvancedSimulator(model, profile(ocm=4096))
    legacy = simulator.simulate(
        event, AdvancedScenario.PREDICTIVE_FULL_EXPERT, config, predictor=pred
    )
    explicit = simulator.simulate(
        event, AdvancedScenario.PREDICTIVE_FULL_EXPERT, config, predictor=pred,
        tile_prefetch_scheduler=FullExpertPrefetchScheduler(),
    )
    assert explicit.layer_makespan_cycles == legacy.layer_makespan_cycles
    assert explicit.memory_metrics == legacy.memory_metrics
    assert explicit.aie_metrics == legacy.aie_metrics


def test_invalid_tile_size_accounting_rejected() -> None:
    """A logical tile cannot hide bytes outside its two matrix fragments."""
    with pytest.raises(ValueError):
        ExpertTile(0, 0, 13, 8, 4, 0, 1.0)
