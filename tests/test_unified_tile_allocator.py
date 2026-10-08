"""宽候选 allocator 主模型接入：容量、顺序、守恒和确定性。"""
from __future__ import annotations

from pathlib import Path
import sys

import pytest

from edge_moe_sim.advanced_schema import AdvancedExecutionConfig
from edge_moe_sim.batch_predictor import BatchPredictorOutput, PrecomputedBatchPredictor, TokenPrediction
from edge_moe_sim.batch_routing import BatchRoutingEvent
from edge_moe_sim.execution_rules import ExecutionRules
from edge_moe_sim.hardware_profile import HardwareProfile
from edge_moe_sim.schema import CandidatePrefetch, ModelConfig, ValidationError
from edge_moe_sim.tile_scheduler import (
    FullExpertPrefetchScheduler, RankedTilePrefetchScheduler, UniformTilePrefetchScheduler,
    TileAllocationScheduler,
)
from edge_moe_sim.unified_schema import UnifiedLayerSequence, UnifiedPredictionSpec, UnifiedSequenceResult
from edge_moe_sim.unified_sequence_simulator import UnifiedSequenceSimulator


def run(scheduler: TileAllocationScheduler, *, managed: bool = False,
        capacity_bytes: int = 1024) -> UnifiedSequenceResult:
    """两完整 Expert 预取预算，八候选，每个 Expert 四个逻辑 tile。"""
    model = ModelConfig('wide', 4, 16, 8, 1, 8, 8)
    profile = HardwareProfile('synthetic', 1000, 2, 8000, 1, 1, 1, 32, True, 1, 4, 1, 1, capacity_bytes)
    rules = ExecutionRules(burst_arbitration=managed)
    config = AdvancedExecutionConfig(4, candidate_count=8, speculative_byte_budget=384, execution_rules=rules)
    sequence = UnifiedLayerSequence((BatchRoutingEvent(1, 0, 0, ((0,),)), BatchRoutingEvent(1, 0, 1, ((7,),))))
    output = BatchPredictorOutput(1, 0, 1, 0, (TokenPrediction(0, tuple(
        CandidatePrefetch(e, 1-.1*e) for e in range(8))),))
    return UnifiedSequenceSimulator(model, profile).simulate(sequence, config,
        predictions=(UnifiedPredictionSpec(0, 1, candidate_count=8),),
        predictors={(0, 1): PrecomputedBatchPredictor({(1, 0, 1): output})},
        tile_prefetch_scheduler=scheduler, trace_enabled=True)


@pytest.mark.parametrize('managed', [False, True])
def test_same_byte_budget_widens_coverage(managed: bool) -> None:
    """同预算 Full 覆盖2个，Uniform覆盖8个，不能只比较预取深度。"""
    full = run(FullExpertPrefetchScheduler(), managed=managed)
    uniform = run(UniformTilePrefetchScheduler(), managed=managed)
    assert full.predictions[0].planned_candidate_depths == (4, 4, 0, 0, 0, 0, 0, 0)
    assert uniform.predictions[0].planned_candidate_depths == (1,)*8
    assert full.predictions[0].planned_speculative_bytes == uniform.predictions[0].planned_speculative_bytes == 384
    # Both matrices are really requested, not GU-only charged as a logical pair.
    assert {r['matrix_type'] for r in uniform.speculative_request_records} == {'gate_up', 'down'}
    queued = sorted(uniform.speculative_request_records, key=lambda r: r['request_id'])
    assert [r['expert_id'] for r in queued[:8]] == list(range(8))
    assert all(r['matrix_type'] == 'gate_up' for r in queued[:8])


@pytest.mark.parametrize('scheduler', [FullExpertPrefetchScheduler(), UniformTilePrefetchScheduler(), RankedTilePrefetchScheduler()])
def test_determinism_and_accounting(scheduler: TileAllocationScheduler) -> None:
    """主模型重复执行完全相同，真实路由、OCM 与 byte 守恒不变。"""
    first, second = run(scheduler), run(scheduler)
    assert first == second
    assert first.layer_results[1].true_expert_ids == (7,)
    assert first.peak_ocm_bytes <= 1024
    metrics = first.memory_metrics
    assert metrics['speculative_bytes'] == metrics['useful_speculative_bytes'] + metrics['wasted_speculative_bytes']
    assert sum(first.predictions[0].planned_candidate_depths) == 8


def test_fixed_slot_adapter_rejected() -> None:
    """尚未支持的 finite-slot 接入不能静默执行。"""
    model = ModelConfig('wide', 4, 16, 8, 1, 8, 8)
    profile = HardwareProfile('synthetic', 1000, 2, 8000, 1, 1, 1, 32, True, 1, 4, 1, 1, 1024)
    config = AdvancedExecutionConfig(4, execution_rules=ExecutionRules(ocm_slot_bytes=64))
    with pytest.raises(ValidationError, match='byte OCM'):
        UnifiedSequenceSimulator(model, profile).simulate(
            UnifiedLayerSequence((BatchRoutingEvent(1, 0, 0, ((0,),)),)), config,
            tile_prefetch_scheduler=UniformTilePrefetchScheduler())


def test_allocator_budget_uses_actual_free_ocm() -> None:
    """配置预算不代表空闲容量；当前层占用必须从 OCM 扣除。"""
    result = run(UniformTilePrefetchScheduler(), capacity_bytes=320)
    record = result.predictions[0]
    assert 0 < record.allocation_budget_bytes < 320 < 384
    assert record.planned_speculative_bytes <= record.allocation_budget_bytes
    assert result.peak_ocm_bytes <= 320


def test_wide_trace_preserves_top_two_and_ranking() -> None:
    """宽候选排名与策略无关，保留准确 Top-1/2 次数、尾部与真值匹配。"""
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root/'examples'))
    try:
        import switch_wide_tile_prefetch as example
        model = ModelConfig('trace', 4, 16, 8, 1, 8, 8, moe_layer_indices=(0, 1, 2, 3, 4, 5))
        settings = dict(samples=100, seed=20261005, coverage=.844, top1_coverage=.758)
        records = example.make_wide_traces(model, settings)
        assert records == example.make_wide_traces(model, settings)
        predictions = [p for r in records for p in r['predictions']]
        assert sum(p['true_rank'] == 0 for p in predictions) == 379
        assert sum(p['true_rank'] < 2 for p in predictions) == 422
        assert all(sorted(p['candidate_ids']) == list(range(8)) for p in predictions)
        assert all(p['scores'] == sorted(p['scores'], reverse=True) for p in predictions)
    finally:
        sys.path.pop(0)
