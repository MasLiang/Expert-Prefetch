"""Breadth/depth accounting, predictor reuse, DSE Pareto, and AIE saturation."""

from __future__ import annotations

from edge_moe_sim.advanced_schema import AdvancedExecutionConfig
from edge_moe_sim.advanced_simulator import AdvancedSimulator
from edge_moe_sim.batch_predictor import (
    BatchPredictorOutput, PrecomputedBatchPredictor, TokenPrediction,
)
from edge_moe_sim.batch_routing import BatchRoutingEvent
from edge_moe_sim.hardware_profile import HardwareProfile
from edge_moe_sim.roofline_prefetch_dse import (
    allocate_uniform_prefix_under_budget, run_roofline_prefetch_dse,
)
from edge_moe_sim.schema import CandidatePrefetch, ModelConfig


def _profile() -> HardwareProfile:
    return HardwareProfile(
        "dse", 1000, 1, 1000, 1.0, 1, 1, 64, False,
        8, 8, 1.0, 0, 100_000,
    )


def _predictor(event: BatchRoutingEvent) -> PrecomputedBatchPredictor:
    candidates = tuple(CandidatePrefetch(expert, 1.0 - expert * 0.01) for expert in range(8))
    output = BatchPredictorOutput(
        event.request_id, 0, 0, 0,
        tuple(TokenPrediction(token, candidates) for token in range(event.batch_size)),
    )
    return PrecomputedBatchPredictor({(event.request_id, 0, 0): output})


def test_fixed_budget_breadth_depth_conservation() -> None:
    deep, deep_bytes = allocate_uniform_prefix_under_budget(2, 8, 100, 1600)
    broad, broad_bytes = allocate_uniform_prefix_under_budget(8, 8, 100, 1600)
    assert (deep, broad) == (8, 2)
    assert deep_bytes == broad_bytes == 1600


def test_same_predictor_compares_full_and_prefix_prefetch() -> None:
    model = ModelConfig("compare", 4, 16, 8, 1, 8, 8)
    event = BatchRoutingEvent(1, 0, 0, ((0,),))
    predictor = _predictor(event)
    simulator = AdvancedSimulator(model, _profile())
    full = simulator.simulate(
        event, "predictive_full_expert",
        AdvancedExecutionConfig(4, candidate_count=8), predictor=predictor,
    )
    prefix = simulator.simulate(
        event, "predictive_uniform_prefix",
        AdvancedExecutionConfig(4, candidate_count=8, uniform_gu_prefetch_depth=1),
        predictor=predictor,
    )
    assert full.recall_at_m == prefix.recall_at_m
    assert full.true_expert_rank_statistics == prefix.true_expert_rank_statistics
    assert full.memory_metrics["speculative_bytes"] > prefix.memory_metrics["speculative_bytes"]


def test_aie_overprovisioning_plateaus_under_fixed_bandwidth() -> None:
    model = ModelConfig("overprovision", 4, 32, 8, 2, 8, 8)
    event = BatchRoutingEvent(0, 0, 0, ((0, 1),))
    results = [AdvancedSimulator(model, _profile()).simulate(
        event, "exact_tile_streaming",
        AdvancedExecutionConfig(4, aie_units_per_expert=units),
    ) for units in (1, 2, 4, 8)]
    cycles = [result.layer_makespan_cycles for result in results]
    assert cycles == sorted(cycles, reverse=True)
    assert cycles[-2] - cycles[-1] <= cycles[0] - cycles[1]
    assert results[-1].aie_starvation_cycles >= results[0].aie_starvation_cycles


def test_small_dse_marks_latency_byte_pareto_points() -> None:
    model = ModelConfig("pareto", 4, 16, 8, 1, 8, 8)
    event = BatchRoutingEvent(2, 0, 0, ((0,), (0,)))
    rows = run_roofline_prefetch_dse(
        model, _profile(), event, _predictor(event),
        candidate_counts=(1, 2), prefetch_policies=("uniform",),
        depths=(1, 2), aie_units_per_expert=(1, 2),
        max_parallel_experts=(1,), f_tile_sizes=(4,),
        placement_policies=("single_channel",),
        effective_bandwidth_gbps=(0.001,), speculative_byte_budget=1024,
    )
    assert rows
    assert any(row.pareto_latency_bytes for row in rows)
    assert all(row.speculative_bytes <= 1024 for row in rows if row.feasible)
