"""弱 baseline 与 demand tile streaming 的相同主模型对照。"""
from dataclasses import replace

import pytest

from edge_moe_sim.advanced_schema import AdvancedExecutionConfig
from edge_moe_sim.batch_routing import BatchRoutingEvent
from edge_moe_sim.batch_predictor import BatchPredictorOutput, PrecomputedBatchPredictor, TokenPrediction
from edge_moe_sim.hardware_profile import HardwareProfile
from edge_moe_sim.schema import CandidatePrefetch, ModelConfig, ValidationError
from edge_moe_sim.unified_schema import UnifiedLayerSequence, UnifiedPredictionSpec
from edge_moe_sim.unified_sequence_simulator import UnifiedSequenceSimulator


@pytest.mark.parametrize("mode", ["full_expert_load", "tile_serial"])
@pytest.mark.parametrize("experts", [1, 2])
def test_weak_baselines_exact_cycles_and_no_dma_compute_overlap(mode: str, experts: int) -> None:
    """手算：Router=2 cycles，每 Expert DMA=16、compute=4、ReLU=2。"""
    model = ModelConfig("tiny", 2, 4, 4, experts, 8, 8, expert_architecture="relu_ffn")
    hardware = HardwareProfile("synthetic", 1000, 1, 1000, 1, 0, 1, 4, True, 1, 8, 1, 0, 64)
    config = AdvancedExecutionConfig(2, nonlinear_elements_per_cycle=2, demand_loading=mode)
    sequence = UnifiedLayerSequence(tuple(BatchRoutingEvent(1, 0, layer, (tuple(range(experts)),)) for layer in (0, 1)))
    simulator = UnifiedSequenceSimulator(model, hardware)
    result = simulator.simulate(sequence, config, trace_enabled=True)
    assert result.inference_makespan_cycles == 2 * (2 + experts * 22)
    assert all(layer.demand_bytes == experts * 16 for layer in result.layer_results.values())
    assert result.memory_metrics["speculative_bytes"] == 0
    assert result.peak_ocm_bytes == (16 if mode == "full_expert_load" else 4)
    assert result == simulator.simulate(sequence, config, trace_enabled=True)
    for request in result.speculative_request_records:
        pytest.fail(f"unexpected speculative request: {request}")
    # 重建每个 demand DMA interval，并检查不与任何 Expert GEMM 相交。
    starts = {(r["layer_id"], r["expert_id"], r["matrix_type"], r["tile_id"]): r["cycle"]
              for r in result.trace if r.get("event_type") == "dma_start"}
    intervals = [(starts[(r["layer_id"], r["expert_id"], r["matrix_type"], r["tile_id"])], r["cycle"])
                 for r in result.trace if r.get("event_type") == "dma_done"]
    compute = [(r["cycle"], r["end_cycle"]) for r in result.trace
               if r.get("event_type") == "aie_start" and r.get("expert_id") is not None]
    assert all(end <= start_compute or end_compute <= start
               for start, end in intervals for start_compute, end_compute in compute)
    if mode == "full_expert_load":
        for layer in (0, 1):
            for expert in range(experts):
                weight_done = max(r["cycle"] for r in result.trace if r.get("event_type") == "dma_done"
                                  and r["layer_id"] == layer and r["expert_id"] == expert)
                first_compute = min(r["cycle"] for r in result.trace if r.get("event_type") == "aie_start"
                                    and r.get("layer_id") == layer and r.get("expert_id") == expert)
                assert weight_done <= first_compute
    streaming = simulator.simulate(sequence, replace(config, demand_loading="tile_streaming"))
    assert streaming.inference_makespan_cycles <= result.inference_makespan_cycles


def test_full_load_requires_capacity_and_rejects_prediction() -> None:
    """整 Expert 无法驻留时明确报错，不静默变成 partial-load。"""
    model = ModelConfig("tiny", 2, 4, 4, 1, 8, 8, expert_architecture="relu_ffn")
    hardware = HardwareProfile("synthetic", 1000, 1, 1000, 1, 0, 1, 4, True, 1, 8, 1, 0, 8)
    sequence = UnifiedLayerSequence((BatchRoutingEvent(1, 0, 0, ((0,),)), BatchRoutingEvent(1, 0, 1, ((1,),))))
    config = AdvancedExecutionConfig(2, demand_loading="full_expert_load")
    with pytest.raises(ValidationError, match="fit one complete Expert"):
        UnifiedSequenceSimulator(model, hardware).simulate(sequence, config)
    with pytest.raises(ValidationError, match="no prediction"):
        UnifiedSequenceSimulator(model, replace(hardware, ocm_bytes=64)).simulate(
            sequence, replace(config, demand_loading="tile_serial"), predictions=(UnifiedPredictionSpec(0, 1),))


def test_nonpipeline_full_prefetch_keeps_full_readiness_and_cross_layer_overlap() -> None:
    """允许跨层预取，但真实 Expert 的任何 GEMM 都不能早于本 Expert 全权重完成。"""
    model = ModelConfig("tiny", 2, 4, 4, 1, 8, 8, expert_architecture="relu_ffn")
    hardware = HardwareProfile("synthetic", 1000, 1, 1000, 1, 0, 1, 4, True, 1, 1, 1, 0, 128)
    config = AdvancedExecutionConfig(2, nonlinear_elements_per_cycle=2, demand_loading="full_expert_load")
    truth = (0, 1, 0)
    sequence = UnifiedLayerSequence(tuple(BatchRoutingEvent(1, 0, layer, ((expert,),)) for layer, expert in enumerate(truth)))
    specs, predictors = [], {}
    for source, target in ((0, 1), (1, 2)):
        specs.append(UnifiedPredictionSpec(source, target, candidate_count=2, prefetch_policy="full_expert_prefetch"))
        output = BatchPredictorOutput(1, 0, target, 0, (TokenPrediction(0, (
            CandidatePrefetch(truth[target], 0.9), CandidatePrefetch(2, 0.1))),))
        predictors[(source, target)] = PrecomputedBatchPredictor({(1, 0, target): output})
    simulator = UnifiedSequenceSimulator(model, hardware)
    result = simulator.simulate(sequence, config, predictions=tuple(specs), predictors=predictors, trace_enabled=True)
    assert result == simulator.simulate(sequence, config, predictions=tuple(specs), predictors=predictors, trace_enabled=True)
    assert result.inference_makespan_cycles <= simulator.simulate(sequence, config).inference_makespan_cycles
    assert result.peak_ocm_bytes <= hardware.ocm_bytes
    overlap = False
    for layer, expert in enumerate(truth):
        starts = [r for r in result.trace if r.get("event_type") == "aie_start"
                  and r.get("layer_id") == layer and r.get("expert_id") == expert]
        weights = [r for r in result.trace if r.get("event_type") == "dma_done"
                   and r.get("layer_id") == layer and r.get("expert_id") == expert]
        assert len(weights) == 4
        assert max(r["cycle"] for r in weights) <= min(r["cycle"] for r in starts)
        if layer < 2:
            future = [r for r in result.trace if r.get("event_type") == "dma_start" and r.get("layer_id") == layer + 1]
            overlap |= any(dma["cycle"] < compute["end_cycle"] and compute["cycle"] < dma["end_cycle"]
                           for dma in future for compute in starts)
    assert overlap
