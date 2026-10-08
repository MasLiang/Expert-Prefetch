"""Causal acceptance tests for delayed cross-layer prefetch admission."""
from __future__ import annotations

from dataclasses import replace

from edge_moe_sim.advanced_schema import AdvancedExecutionConfig
from edge_moe_sim.batch_routing import BatchRoutingEvent
from edge_moe_sim.hardware_profile import HardwareProfile
from edge_moe_sim.predictor_sensitivity import build_sensitivity_trace
from edge_moe_sim.schema import ModelConfig
from edge_moe_sim.unified_schema import UnifiedLayerSequence, UnifiedPredictionSpec
from edge_moe_sim.unified_sequence_simulator import UnifiedSequenceSimulator


def test_two_layer_prediction_waits_for_all_predecessor_weights() -> None:
    """Early prediction does not reserve OCM or start DMA before the barrier."""
    model = ModelConfig("barrier", 4, 8, 8, 2, 8, 8)
    hardware = HardwareProfile("test", 1000, 2, 8000, 1, 1, 1, 32,
                               True, 2, 4, 1, 10, 4096)
    config = AdvancedExecutionConfig(4, max_parallel_experts=2, candidate_count=2)
    layers = tuple(BatchRoutingEvent(10, 0, i, ((0, 1),)) for i in range(3))
    trace = build_sensitivity_trace(request_id=10, token_position=0, layer_id=2,
        num_experts=8, top_k=2, candidate_count=2, quality=1,
        calibration_mode="well_calibrated")
    simulator = UnifiedSequenceSimulator(model, hardware)
    results = []
    for source in (0, 1):
        spec = UnifiedPredictionSpec(source, 2, "source_layer_start",
            candidate_count=2, wait_for_previous_layer_weights=True)
        result = simulator.simulate(UnifiedLayerSequence(layers), config,
            predictions=(spec,), predictors={(source, 2): trace.predictor}, trace_enabled=True)
        record = result.predictions[0]
        weight_ends = [int(item["cycle"]) for item in result.trace
            if item.get("event_type") == "dma_done" and item.get("layer_id") == 1]
        assert record.prefetch_release_cycle == max(weight_ends)
        assert record.predictor_ready_cycle < record.prefetch_release_cycle
        assert record.prefetch_release_cycle < result.layer_results[1].layer_done_cycle
        assert result.peak_ocm_bytes <= hardware.ocm_bytes
        for item in result.speculative_request_records:
            assert item["queue_cycle"] >= record.prefetch_release_cycle
            if item["issue_cycle"] is not None:
                assert item["issue_cycle"] >= record.prefetch_release_cycle
        results.append(result)
    assert results[0].makespan_cycles == results[1].makespan_cycles
    assert results[0].predictions[0].prefetch_release_cycle == results[1].predictions[0].prefetch_release_cycle


def test_late_prediction_cannot_retroactively_prefetch() -> None:
    """A prediction ready after target resolve creates no speculative traffic."""
    model = ModelConfig("barrier", 4, 8, 8, 2, 8, 8)
    hardware = HardwareProfile("test", 1000, 2, 8000, 1, 1, 1, 32,
                               True, 2, 4, 1, 10, 4096)
    config = AdvancedExecutionConfig(4, max_parallel_experts=2, candidate_count=2)
    layers = tuple(BatchRoutingEvent(10, 0, i, ((0, 1),)) for i in range(3))
    trace = build_sensitivity_trace(request_id=10, token_position=0, layer_id=2,
        num_experts=8, top_k=2, candidate_count=2, quality=1,
        calibration_mode="well_calibrated")
    spec = UnifiedPredictionSpec(0, 2, "source_layer_start", 100000,
        candidate_count=2, wait_for_previous_layer_weights=True)
    result = UnifiedSequenceSimulator(model, hardware).simulate(
        UnifiedLayerSequence(layers), config, predictions=(spec,),
        predictors={(0, 2): trace.predictor})
    assert result.predictions[0].prefetch_release_cycle is None
    assert result.predictions[0].speculative_bytes == 0


def test_default_resource_aware_release_precedes_weight_barrier() -> None:
    """Ready prediction can load future tiles while preceding Expert still streams weights."""
    model = ModelConfig("flexible", 4, 8, 8, 2, 8, 8)
    hardware = HardwareProfile("test", 1000, 2, 8000, 1, 1, 1, 32,
                               True, 2, 4, 1, 10, 4096)
    config = AdvancedExecutionConfig(4, max_parallel_experts=2, candidate_count=2)
    layers = tuple(BatchRoutingEvent(10, 0, i, ((0, 1),)) for i in range(3))
    trace = build_sensitivity_trace(request_id=10, token_position=0, layer_id=2,
        num_experts=8, top_k=2, candidate_count=2, quality=1,
        calibration_mode="well_calibrated")
    spec = UnifiedPredictionSpec(1, 2, "source_router_done", candidate_count=2)
    assert not spec.wait_for_previous_layer_weights
    simulator = UnifiedSequenceSimulator(model, hardware)
    flexible = simulator.simulate(UnifiedLayerSequence(layers), config,
        predictions=(spec,), predictors={(1, 2): trace.predictor}, trace_enabled=True)
    conservative = simulator.simulate(UnifiedLayerSequence(layers), config,
        predictions=(replace(spec, wait_for_previous_layer_weights=True),),
        predictors={(1, 2): trace.predictor}, trace_enabled=True)
    record = flexible.predictions[0]
    assert record.prefetch_release_cycle == record.predictor_ready_cycle
    weight_end = max(int(item["cycle"]) for item in flexible.trace
                     if item.get("event_type") == "dma_done" and item.get("layer_id") == 1)
    starts = [item["issue_cycle"] for item in flexible.speculative_request_records
              if item["issue_cycle"] is not None]
    assert starts and min(starts) < weight_end
    assert record.prefetch_release_cycle < conservative.predictions[0].prefetch_release_cycle
    assert flexible.peak_ocm_bytes <= hardware.ocm_bytes
    assert all(r.true_expert_ids == (0, 1) for r in flexible.layer_results.values())
    assert flexible == simulator.simulate(UnifiedLayerSequence(layers), config,
        predictions=(spec,), predictors={(1, 2): trace.predictor}, trace_enabled=True)
