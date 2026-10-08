"""Independent allocator/order and finite RTL admission regressions."""
from __future__ import annotations

from dataclasses import replace

import pytest

from edge_moe_sim.advanced_schema import AdvancedExecutionConfig
from edge_moe_sim.batch_predictor import BatchPredictorOutput, PrecomputedBatchPredictor, TokenPrediction
from edge_moe_sim.batch_routing import BatchRoutingEvent
from edge_moe_sim.execution_rules import ExecutionRules
from edge_moe_sim.hardware_profile import HardwareProfile
from edge_moe_sim.prefetch_config import PrefetchConfig, HardwareGreedyV1Scheduler
from edge_moe_sim.hardware_greedy_v1 import HardwareGreedyV1Candidate, HardwareGreedyV1Config, hardware_greedy_v1
from edge_moe_sim.schema import CandidatePrefetch, ModelConfig, ValidationError
from edge_moe_sim.tile_scheduler import ExpertTile, TilePredictionResult, TileSchedulingHardware, CurrentTimelineState
from edge_moe_sim.unified_schema import UnifiedLayerSequence, UnifiedPredictionSpec, UnifiedSequenceResult
from edge_moe_sim.unified_sequence_simulator import UnifiedSequenceSimulator


def run_mode(allocator: str, order: str, finite: bool = False,
             ready_first: bool = False) -> UnifiedSequenceResult:
    """Run two causal layers with eight candidates and 384 payload-byte credit."""
    model = ModelConfig('synthetic', 4, 16, 8, 1, 8, 8)
    profile = HardwareProfile('synthetic_not_measured', 1000, 2, 8000, 1, 1, 1,
                              32, True, 1, 4, 1, 1, 1024)
    rules = (ExecutionRules(ocm_slot_bytes=64, atomic_tile_pair=True,
                           reserved_demand_slots=2, descriptor_entries=8,
                           dma_fifo_depth=4, retry_future_prefetch=True,
                           compute_policy='tile_ready_first' if ready_first else 'expert_locked',
                           request_order='pair_fifo', burst_arbitration=True)
             if finite else ExecutionRules())
    config = AdvancedExecutionConfig(4, candidate_count=8, speculative_byte_budget=384,
        execution_rules=rules, prefetch=PrefetchConfig(allocator=allocator, dma_order=order))
    sequence = UnifiedLayerSequence((BatchRoutingEvent(1, 0, 0, ((0,),)),
                                     BatchRoutingEvent(1, 0, 1, ((7,),))))
    output = BatchPredictorOutput(1, 0, 1, 0, (TokenPrediction(0, tuple(
        CandidatePrefetch(e, 1-.1*e) for e in range(8))),))
    return UnifiedSequenceSimulator(model, profile).simulate(sequence, config,
        predictions=(UnifiedPredictionSpec(0, 1, candidate_count=8),),
        predictors={(0, 1): PrecomputedBatchPredictor({(1, 0, 1): output})}, trace_enabled=True)


@pytest.mark.parametrize('allocator', ['full_expert', 'uniform', 'ranked', 'hardware_greedy_v1'])
@pytest.mark.parametrize('order', ['up_first', 'pair_first'])
def test_research_combinations(allocator: str, order: str) -> None:
    """All six combinations conserve bytes, preserve truth and are deterministic."""
    result = run_mode(allocator, order)
    assert result == run_mode(allocator, order)
    assert result.predictions[0].dma_order == order
    assert result.predictions[0].planned_speculative_bytes == 384
    assert result.layer_results[1].true_expert_ids == (7,)
    requests = sorted(result.speculative_request_records, key=lambda r: r['request_id'])
    assert len(requests) == 16
    matrices = [r['matrix_type'] for r in requests]
    assert matrices == (['gate_up']*8 + ['down']*8 if order == 'up_first'
                        else ['gate_up', 'down']*8)
    if order == 'pair_first':
        assert all((a['expert_id'], a['tile_id']) == (b['expert_id'], b['tile_id'])
                   for a, b in zip(requests[::2], requests[1::2]))
    metrics = result.memory_metrics
    assert metrics['speculative_bytes'] == metrics['useful_speculative_bytes'] + metrics['wasted_speculative_bytes']
    assert result.peak_ocm_bytes <= 1024


@pytest.mark.parametrize('allocator', ['full_expert', 'uniform', 'ranked', 'hardware_greedy_v1'])
def test_finite_pair_retry(allocator: str) -> None:
    """Backpressure must not convert breadth-first plans into rank-major plans."""
    result = run_mode(allocator, 'pair_first', True)
    assert result == run_mode(allocator, 'pair_first', True)
    assert len(result.layer_results) == 2
    record = result.predictions[0]
    assert record.initial_allocated_speculative_bytes <= record.planned_speculative_bytes == 384
    assert result.peak_ocm_bytes <= 1024
    requests = sorted(result.speculative_request_records, key=lambda r: r['request_id'])
    assert [r['matrix_type'] for r in requests] == ['gate_up', 'down'] * (len(requests)//2)
    research = run_mode(allocator, 'pair_first')
    expected = sorted(research.speculative_request_records, key=lambda r: r['request_id'])
    assert [(r['expert_id'], r['tile_id']) for r in requests] == [
        (r['expert_id'], r['tile_id']) for r in expected[:len(requests)]]


def test_atomic_up_first_is_explicitly_rejected() -> None:
    """Do not silently remove RTL resource constraints to enable Up-first."""
    with pytest.raises(ValidationError, match='pair_first'):
        run_mode('uniform', 'up_first', True)


@pytest.mark.parametrize('allocator', ['uniform', 'ranked', 'hardware_greedy_v1'])
def test_finite_ready_first(allocator: str) -> None:
    """New allocators also integrate with the existing RTL tile-ready-first engine."""
    result = run_mode(allocator, 'pair_first', True, True)
    assert result == run_mode(allocator, 'pair_first', True, True)
    assert len(result.layer_results) == 2
    assert result.peak_ocm_bytes <= 1024


@pytest.mark.parametrize('channels', [1, 4])
def test_integer_golden_and_channel_credit(channels: int) -> None:
    """Exact raw Q12 and cross-product choices agree with the frozen golden."""
    prediction = TilePredictionResult(((0, 3000.), (1, 700.), (2, 200.)))
    tiles = {e: tuple(ExpertTile(e, t, 48, 32, 16, rank, score) for t in range(4))
             for rank, (e, score) in enumerate(prediction.candidates)}
    credits = (144,)*channels
    config = PrefetchConfig(allocator='hardware_greedy_v1', confidence_encoding='q12',
                            channel_credit_bytes=credits)
    plan = HardwareGreedyV1Scheduler(config, channels).schedule(prediction, tiles,
        TileSchedulingHardware(1024, 8, channels), CurrentTimelineState(0, 0, 288))
    golden = hardware_greedy_v1(tuple(HardwareGreedyV1Candidate(e, rank, int(score), (48,)*4)
        for rank, (e, score) in enumerate(prediction.candidates)),
        HardwareGreedyV1Config(channels, 288, credits))
    assert [(t.tile.expert_id, t.tile.tile_id) for t in plan.transfers] == [
        (a.expert_id, a.tile_id) for a in golden]


@pytest.mark.parametrize('changes', [dict(allocator='bad'), dict(dma_order='bad'),
    dict(confidence_encoding='bad'), dict(ranked_depth_decay=float('nan')),
    dict(priority_shift_per_depth=1.5), dict(channel_credit_bytes=(-1,))])
def test_bad_config(changes: dict[str, object]) -> None:
    """Invalid configuration fails clearly rather than being silently ignored."""
    with pytest.raises(ValidationError):
        replace(PrefetchConfig(), **changes)
