"""Event-DAG streaming, prefix startability, and wrong-traffic tests."""

from __future__ import annotations

from edge_moe_sim.advanced_schema import AdvancedExecutionConfig
from edge_moe_sim.advanced_simulator import AdvancedSimulator
from edge_moe_sim.batch_predictor import (
    BatchPredictorOutput, PrecomputedBatchPredictor, TokenPrediction,
)
from edge_moe_sim.batch_routing import BatchRoutingEvent
from edge_moe_sim.hardware_profile import HardwareProfile
from edge_moe_sim.schema import CandidatePrefetch, ModelConfig


def _model(experts: int = 8) -> ModelConfig:
    return ModelConfig("stream", 4, 8, experts, 1, 8, 8)


def _profile(bytes_per_cycle: int, ops_per_cycle: int = 4) -> HardwareProfile:
    return HardwareProfile(
        "stream", 1000, 1, bytes_per_cycle * 1000, 1.0, 1, 1, 32,
        False, 4, ops_per_cycle, 1.0, 0, 4096,
    )


def _predictor(candidates: tuple[CandidatePrefetch, ...], ready: int = 0) -> PrecomputedBatchPredictor:
    output = BatchPredictorOutput(0, 0, 0, ready, (TokenPrediction(0, candidates),))
    return PrecomputedBatchPredictor({(0, 0, 0): output})


def test_exact_tile_streaming_overlaps_dma_and_aie() -> None:
    event = BatchRoutingEvent(0, 0, 0, ((0,),))
    config = AdvancedExecutionConfig(4, aie_units_per_expert=1, max_parallel_experts=1)
    result = AdvancedSimulator(_model(), _profile(4)).simulate(
        event, "exact_tile_streaming", config
    )
    assert result.dma_compute_overlap_cycles > 0
    serialized = (
        result.router_resolve_cycle + result.memory_metrics["memory_busy_cycles"]
        + sum(record.compute_ops for record in result.expert_records) // 4
    )
    assert result.layer_makespan_cycles < serialized


def test_streaming_starvation_when_aie_outruns_dma() -> None:
    event = BatchRoutingEvent(0, 0, 0, ((0,),))
    result = AdvancedSimulator(_model(), _profile(1, 64)).simulate(
        event, "exact_tile_streaming", AdvancedExecutionConfig(4)
    )
    assert result.expert_records[0].inter_tile_starvation_cycles > 0


def test_prefetched_startup_prefix_eliminates_initial_stall() -> None:
    event = BatchRoutingEvent(0, 0, 0, ((0,),))
    predictor = _predictor((CandidatePrefetch(0, 1.0),))
    result = AdvancedSimulator(_model(), _profile(16)).simulate(
        event, "predictive_uniform_prefix",
        AdvancedExecutionConfig(4, candidate_count=1, uniform_gu_prefetch_depth=1),
        predictor=predictor,
    )
    record = result.expert_records[0]
    assert result.startable_recall == 1.0
    assert record.initial_memory_stall == 0
    assert record.compute_start_time == result.router_resolve_cycle


def test_active_false_burst_finishes_queued_false_cancels() -> None:
    event = BatchRoutingEvent(0, 0, 0, ((0,),))
    candidates = (
        CandidatePrefetch(1, 1.0), CandidatePrefetch(2, 0.9), CandidatePrefetch(0, 0.8)
    )
    result = AdvancedSimulator(_model(), _profile(1)).simulate(
        event, "predictive_uniform_prefix",
        AdvancedExecutionConfig(4, candidate_count=3, uniform_gu_prefetch_depth=1),
        predictor=_predictor(candidates),
    )
    assert result.memory_metrics["wasted_speculative_bytes"] > 0
    assert result.memory_metrics["speculative_bytes"] > result.memory_metrics["useful_speculative_bytes"]
    assert result.demand_blocked_cycles > 0
    assert result.memory_metrics["transferred_bytes"] > 0
