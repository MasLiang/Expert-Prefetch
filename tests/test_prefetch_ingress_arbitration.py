"""有界 allocator cadence 和 demand 游标独占 ingress 的因果回归。"""
from __future__ import annotations

from dataclasses import replace

import pytest

from edge_moe_sim.advanced_schema import AdvancedExecutionConfig
from edge_moe_sim.batch_predictor import BatchPredictorOutput, PrecomputedBatchPredictor, TokenPrediction
from edge_moe_sim.batch_routing import BatchRoutingEvent
from edge_moe_sim.execution_rules import ExecutionRules
from edge_moe_sim.hardware_profile import HardwareProfile
from edge_moe_sim.hardware.multi_channel_memory import MultiChannelMemory
from edge_moe_sim.schema import CandidatePrefetch, ModelConfig, ValidationError
from edge_moe_sim.unified_schema import UnifiedLayerSequence, UnifiedPredictionSpec, UnifiedSequenceResult
from edge_moe_sim.unified_sequence_simulator import UnifiedSequenceSimulator


def run_fixture(interval_cycles: int = 6) -> UnifiedSequenceResult:
    """四 tiles、12 slots：demand bank 满时未来 allocator 不得绕过游标。"""
    model = ModelConfig('ingress', 4, 16, 8, 1, 8, 8,
                        expert_architecture='relu_ffn', fuse_gate_up=False)
    rules = ExecutionRules(ocm_slot_bytes=16, ocm_banks=4, atomic_tile_pair=True,
        compute_policy='tile_ready_first', retry_future_prefetch=True, request_order='pair_fifo',
        ingress_pair_cycles=3, prefetch_pair_interval_cycles=interval_cycles,
        demand_ingress_blocks_prefetch=True)
    profile = HardwareProfile('ingress', 1000, 4, 1000, 1, 0, 1, 16, True, 1, 8, 1, 0, 192)
    config = AdvancedExecutionConfig(4, placement_policy='striped_tile',
        speculative_byte_budget=128, execution_rules=rules)
    sequence = UnifiedLayerSequence(tuple(BatchRoutingEvent(0, 0, l, ((l,),)) for l in range(2)))
    output = BatchPredictorOutput(0, 0, 1, 0, (TokenPrediction(0, (CandidatePrefetch(1, .9),)),))
    return UnifiedSequenceSimulator(model, profile).simulate(sequence, config,
        predictions=(UnifiedPredictionSpec(0, 1, 'source_layer_start', 0,
            prefetch_policy='full_expert_prefetch'),),
        predictors={(0, 1): PrecomputedBatchPredictor({(0, 0, 1): output})}, trace_enabled=True)


def test_cadence_and_ingress_priority() -> None:
    """成功 admission 消耗 allocator 周期；不等待当前层全部 DMA 完成。"""
    result = run_fixture()
    assert result == run_fixture()
    admissions = [v for v in result.trace if v['event_type'] == 'pair_admit']
    spec = [v for v in admissions if v['request_state'] == 'speculative']
    assert len(spec) == 4
    assert all(b['cycle']-a['cycle'] >= 6 for a, b in zip(spec, spec[1:]))
    last_demand = max(v['cycle'] for v in admissions if v['request_state'] == 'demand')
    resolve = result.layer_results[0].router_resolve_cycle
    assert not any(resolve <= v['cycle'] < last_demand for v in spec)
    resumed = next(v['cycle'] for v in spec if v['cycle'] > last_demand)
    last_payload = max(v['cycle'] for v in result.trace if v['event_type'] == 'dma_done' and v['layer_id'] == 0)
    assert resumed < last_payload
    assert result.peak_ocm_bytes <= 192
    assert len({(v['layer_id'], v['expert_id'], v['tile_id']) for v in admissions}) == len(admissions)


@pytest.mark.parametrize('interval', [0, 1, 6, 50, 100])
def test_cadence_progress_and_payload(interval: int) -> None:
    """即使下一个 selection 晚于 target start，也须正常取消并完成所有层。"""
    result = run_fixture(interval)
    assert len(result.layer_results) == 2
    memory = result.memory_metrics
    assert memory['demand_bytes']+memory['useful_speculative_bytes'] == 2*2*4*16
    assert memory['speculative_bytes'] == memory['useful_speculative_bytes']+memory['wasted_speculative_bytes']


@pytest.mark.parametrize('changes', [dict(prefetch_pair_interval_cycles=-1),
    dict(prefetch_pair_interval_cycles=True), dict(demand_ingress_blocks_prefetch=1),
    dict(prefetch_pair_interval_cycles=6), dict(demand_ingress_blocks_prefetch=True)])
def test_invalid_ingress_rules(changes: dict[str, object]) -> None:
    """错误类型/负周期和非 pair 配置不能静默接受。"""
    with pytest.raises(ValidationError):
        replace(ExecutionRules(), **changes)


def test_cadence_requires_retry() -> None:
    """不能让 direct/non-retry 路径静默绕过 allocator cadence。"""
    with pytest.raises(ValidationError, match='重试'):
        ExecutionRules(ocm_slot_bytes=16, atomic_tile_pair=True, prefetch_pair_interval_cycles=6)


def test_reserved_demand_credit_does_not_preempt_before_enqueue() -> None:
    """burst 完成前预留 demand，实际 token 尚未入 FIFO，不应提前让出。"""
    rules = ExecutionRules(burst_arbitration=True, dma_request_latency_cycles=3)
    profile = HardwareProfile('fifo', 1000, 1, 1000, 1, 0, 1, 4, True, 1, 8, 1, 0, 128)
    memory = MultiChannelMemory(profile, rules=rules)
    spec = memory.new_request(layer_id=1, expert_id=1, matrix_type='gate_up', tile_ids=(0,),
        object_keys=((1, 1, 'gate_up', 0),), transfer_bytes=16, request_type='speculative', issue_cycle=0)
    memory.enqueue(spec)
    assert memory.start_ready(3) == [spec]
    demand = memory.new_request(layer_id=0, expert_id=0, matrix_type='gate_up', tile_ids=(0,),
        object_keys=((0, 0, 'gate_up', 0),), transfer_bytes=4, request_type='demand', issue_cycle=6)
    memory.enqueue(demand)
    assert demand.enqueue_ready_cycle == 9
    assert not memory.complete(spec, 7)
    assert spec in memory.continuations
    assert memory.start_ready(7) == [spec]
    assert not memory.complete(spec, 11)
    assert spec not in memory.continuations
    assert memory.start_ready(11) == [demand]
