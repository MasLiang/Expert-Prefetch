"""Unit tests for M9.2 allocation-only scheduler extensions."""

from __future__ import annotations

from collections import defaultdict

from edge_moe_sim.advanced_schema import AdvancedExecutionConfig, AdvancedScenario
from edge_moe_sim.advanced_simulator import AdvancedSimulator
from edge_moe_sim.batch_predictor import (
    BatchPredictorOutput,
    PrecomputedBatchPredictor,
    TokenPrediction,
)
from edge_moe_sim.batch_routing import BatchRoutingEvent
from edge_moe_sim.hardware_profile import HardwareProfile
from edge_moe_sim.schema import CandidatePrefetch, ModelConfig
from edge_moe_sim.tile_scheduler import (
    BudgetProportionalScheduler,
    CriticalityAwareTileScheduler,
    CurrentTimelineState,
    ExpertTile,
    FixedPredictionOracleScheduler,
    HardwareAwareGreedyScheduler,
    TilePredictionResult,
    TileSchedulingHardware,
    estimate_scheduler_complexity,
)


def geometry(
    candidates: tuple[tuple[int, float], ...], tiles: int = 4,
) -> dict[int, tuple[ExpertTile, ...]]:
    """Create equal-size prefix geometry for pure scheduler tests."""
    return {
        expert_id: tuple(
            ExpertTile(expert_id, tile_id, 12, 8, 4, rank, confidence)
            for tile_id in range(tiles)
        )
        for rank, (expert_id, confidence) in enumerate(candidates)
    }


def test_criticality_scheduler_is_deterministic_prefix_closed_and_feasible() -> None:
    """Criticality allocation obeys OCM/bandwidth budgets deterministically."""
    candidates = ((0, 0.95), (1, 0.8), (2, 0.55), (3, 0.2))
    prediction = TilePredictionResult(candidates)
    tiles = geometry(candidates)
    hardware = TileSchedulingHardware(120, 4.0, 2)
    timeline = CurrentTimelineState(10, prefetch_deadline_cycle=22)
    scheduler = CriticalityAwareTileScheduler()
    first = scheduler.schedule(prediction, tiles, hardware, timeline)
    second = scheduler.schedule(prediction, tiles, hardware, timeline)

    assert first == second
    assert first.allocated_bytes <= first.budget_bytes <= 96
    for expert_id, depth in first.depths_by_expert.items():
        selected = {
            item.tile.tile_id
            for item in first.transfers if item.tile.expert_id == expert_id
        }
        assert selected == set(range(depth))
    complexity = estimate_scheduler_complexity(
        "criticality_aware", len(candidates), 4, first.allocated_tiles
    )
    assert complexity.hardware_friendly is True
    assert complexity.divisions == 0


def test_fixed_prediction_oracle_preserves_candidate_set_and_rank_order() -> None:
    """The fixed oracle only emits tiles belonging to the unchanged prediction."""
    candidates = ((7, 0.91), (3, 0.72), (9, 0.31))
    prediction = TilePredictionResult(candidates)
    tiles = geometry(candidates, tiles=3)
    hardware = TileSchedulingHardware(72, 8.0, 1)
    timeline = CurrentTimelineState(0)
    plans = FixedPredictionOracleScheduler().candidate_plans(
        prediction, tiles, hardware, timeline
    )
    candidate_ids = tuple(expert_id for expert_id, _ in candidates)

    assert plans
    for plan in plans:
        assert {item.tile.expert_id for item in plan.transfers} <= set(candidate_ids)
        # Every admitted Expert is in a rank prefix; the search cannot skip a
        # higher-ranked candidate to expose a lower-ranked one.
        admitted = {item.tile.expert_id for item in plan.transfers}
        if admitted:
            deepest_rank = max(candidate_ids.index(expert_id) for expert_id in admitted)
            assert set(candidate_ids[:deepest_rank + 1]) <= admitted


def test_fixed_prediction_oracle_includes_online_plans_and_dominates_them() -> None:
    """Extra online candidates make the fixed-prediction bound no worse."""
    candidates = ((0, 0.9), (1, 0.7), (2, 0.4))
    prediction = TilePredictionResult(candidates)
    tiles = geometry(candidates, tiles=3)
    hardware = TileSchedulingHardware(72, 8.0, 2)
    timeline = CurrentTimelineState(0)
    online = [
        scheduler.schedule(prediction, tiles, hardware, timeline)
        for scheduler in (
            BudgetProportionalScheduler(),
            HardwareAwareGreedyScheduler(),
            CriticalityAwareTileScheduler(),
        )
    ]

    def latency(plan: object) -> int:
        allocated = getattr(plan, "allocated_tiles")
        return 1000 - 13 * allocated

    result = FixedPredictionOracleScheduler().optimize(
        prediction, tiles, hardware, timeline, latency, extra_plans=online
    )
    assert result.latency_cycles <= min(latency(plan) for plan in online)
    assert result.evaluated_plans >= len(online)


def test_criticality_scheduler_obeys_event_dma_and_ocm_constraints() -> None:
    """An executed Criticality plan respects physical channel exclusion and OCM."""
    model = ModelConfig("m92_test", 4, 8, 8, 1, 8, 8)
    hardware = HardwareProfile(
        "m92_test_hardware", 1000, 2, 8_000, 1.0, 2, 1, 64,
        True, 4, 8, 1.0, 1, 2048, hardware_tag="synthetic_placeholder",
    )
    event = BatchRoutingEvent(92, 0, 0, ((0,),))
    candidates = tuple(
        CandidatePrefetch(expert_id, score)
        for expert_id, score in ((0, .9), (1, .7), (2, .5), (3, .3))
    )
    output = BatchPredictorOutput(
        92, 0, 0, 0, (TokenPrediction(0, candidates),)
    )
    predictor = PrecomputedBatchPredictor({(92, 0, 0): output})
    config = AdvancedExecutionConfig(
        2, aie_units_per_expert=1, max_parallel_experts=1,
        candidate_count=4, startup_gu_tiles=1,
    )
    result = AdvancedSimulator(model, hardware).simulate(
        event,
        AdvancedScenario.PREDICTIVE_DSE,
        config,
        predictor=predictor,
        tile_prefetch_scheduler=CriticalityAwareTileScheduler(),
    )

    metrics = result.tile_metrics
    assert metrics["allocated_transfer_bytes"] == (
        metrics["useful_prefetched_bytes"]
        + metrics["wasted_prefetched_bytes"]
        + metrics["cancelled_transfer_bytes"]
    )
    assert result.peak_ocm_bytes <= hardware.ocm_bytes
    by_channel: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for interval in (
        item for item in result.timeline if item.resource == "MEMORY"
    ):
        channel_text = interval.operation.rsplit("ch", 1)[1]
        for channel in map(int, channel_text.split("-")):
            by_channel[channel].append((interval.start_cycle, interval.end_cycle))
    assert set(by_channel) <= {0, 1}
    for intervals in by_channel.values():
        ordered = sorted(intervals)
        assert all(
            left[1] <= right[0]
            for left, right in zip(ordered, ordered[1:])
        )
