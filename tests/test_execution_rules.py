"""固定 slot、原子 pair、重试与单 Engine tile 调度的定向验证。"""
from __future__ import annotations

from dataclasses import replace

import pytest

from edge_moe_sim.advanced_schema import AdvancedExecutionConfig
from edge_moe_sim.batch_predictor import BatchPredictorOutput, PrecomputedBatchPredictor, TokenPrediction
from edge_moe_sim.batch_routing import BatchRoutingEvent
from edge_moe_sim.execution_rules import ExecutionRules
from edge_moe_sim.hardware.capacity_ocm import CapacityOnChipMemory
from edge_moe_sim.hardware.multi_channel_memory import MultiChannelMemory
from edge_moe_sim.hardware_profile import HardwareProfile
from edge_moe_sim.schema import CandidatePrefetch, ModelConfig, ValidationError
from edge_moe_sim.unified_schema import UnifiedLayerSequence, UnifiedPredictionSpec
from edge_moe_sim.unified_schema import UnifiedSequenceResult
from edge_moe_sim.unified_sequence_simulator import UnifiedSequenceSimulator


def rules() -> ExecutionRules:
    """测试使用 32-byte slot；不依赖 U200 的 512-byte 常量。"""
    return ExecutionRules(32, 2, 2, 1, True, 'tile_ready_first', True, 'pair_fifo')


def profile(capacity: int = 128) -> HardwareProfile:
    """单 MAC、单 memory channel 的可手算配置。"""
    return HardwareProfile('rules_test', 1000, 1, 1000, 1, 0, 1, 16,
                           True, 1, 16, 1, 0, capacity)


@pytest.mark.parametrize('entries,expected_spec', [(2, False), (4, True)])
def test_descriptor_demand_reserve(entries: int, expected_spec: bool) -> None:
    """OCM 保留不能代替 table 保留：当前 demand 始终保有一对 credit。"""
    memory = MultiChannelMemory(profile(), rules=replace(rules(),
        descriptor_entries=entries, reserved_demand_descriptors=2))
    assert memory.can_admit('demand', 2)
    assert memory.can_admit('speculative', 2) is expected_spec


def test_descriptor_reserve_rejects_invalid_capacity() -> None:
    """不能在无限 table 或小于 reserve 的 table 上声明保留。"""
    with pytest.raises(ValidationError, match='table 容量'):
        replace(rules(), descriptor_entries=2, reserved_demand_descriptors=3)


def run_sequence(*, retry: bool = True, capacity: int = 128,
                 barrier: bool = False, order: tuple[int, int] = (3, 2)) -> UnifiedSequenceResult:
    """独立事先定义路由；候选只覆盖 target 的第二个 Expert 首 tile。"""
    model = ModelConfig('rules', 4, 8, 8, 2, 8, 8)
    config = AdvancedExecutionConfig(4, placement_policy='striped_tile',
        speculative_byte_budget=48, nonlinear_elements_per_cycle=1,
        execution_rules=replace(rules(), retry_future_prefetch=retry))
    source = BatchRoutingEvent(0, 0, 0, ((0,),))
    target = BatchRoutingEvent(0, 0, 1, (order,))
    output = BatchPredictorOutput(0, 0, 1, 1,
        (TokenPrediction(0, (CandidatePrefetch(2, 0.9),)),))
    return UnifiedSequenceSimulator(model, profile(capacity)).simulate(
        UnifiedLayerSequence((source, target)), config,
        predictions=(UnifiedPredictionSpec(0, 1, 'source_router_done', 1,
            prefetch_policy='uniform_prefix', prefix_depth=1,
            wait_for_previous_layer_weights=barrier),),
        predictors={(0, 1): PrecomputedBatchPredictor({(0, 0, 1): output})}, trace_enabled=True)


def test_fixed_slots_charge_down_padding_not_dma_bytes() -> None:
    """GU32/DOWN16 实际传输48 byte，OCM 占64 byte。"""
    ocm = CapacityOnChipMemory(128, rules())
    keys = ((0, 0, 'gate_up', 0), (0, 0, 'down', 0))
    assert ocm.admit_group(((keys[0], 32), (keys[1], 16)), 'demand', 1, 0).admitted
    assert ocm.current_bytes == 64
    assert sum(v.size_bytes for v in ocm.objects.values()) == 48
    assert {v.bank_id for v in ocm.objects.values()} == {0, 1}
    ocm.release(keys[0], 1)
    assert ocm.current_bytes == 32 and keys[1] in ocm.objects
    ocm.release(keys[1], 2)
    assert ocm.current_bytes == 0


def test_atomic_rollback_and_reserved_demand_capacity() -> None:
    """投机只能占两 slot，第二 pair 被背压，demand 能占保留空间。"""
    ocm = CapacityOnChipMemory(128, rules())
    first = (((0, 0, 'gate_up', 0), 32), ((0, 0, 'down', 0), 16))
    second = (((1, 2, 'gate_up', 0), 32), ((1, 2, 'down', 0), 16))
    assert ocm.admit_group(first, 'speculative', .9, 0).admitted
    before = dict(ocm.objects)
    assert not ocm.admit_group(second, 'speculative', .8, 1).admitted
    assert ocm.objects == before and ocm.current_bytes == 64
    assert ocm.admit_group(second, 'demand', .8, 1).admitted
    assert ocm.current_bytes == 128 and ocm.evicted_speculative_bytes == 0


def test_bank_full_rejects_despite_global_space() -> None:
    """bank 冲突不是总 byte 容量检查能够替代的。"""
    ocm = CapacityOnChipMemory(128, rules())
    for expert in (0, 2):
        assert ocm.admit((0, expert, 'gate_up', 0), 32, 'demand', 1, 0).admitted
    pair = (((0, 4, 'gate_up', 0), 32), ((0, 4, 'down', 0), 16))
    assert not ocm.admit_group(pair, 'demand', 1, 1).admitted
    assert ocm.current_bytes == 64 and pair[1][0] not in ocm.objects


@pytest.mark.parametrize('bad', [dict(ocm_banks=3), dict(reserved_demand_slots=2),
    dict(atomic_tile_pair=True), dict(compute_policy='tile_ready_first'), dict(request_order='unknown')])
def test_invalid_rules_rejected(bad: dict[str, object]) -> None:
    """不支持的配置不得静默忽略。"""
    with pytest.raises(ValidationError):
        ExecutionRules(**bad)


def test_oversize_and_duplicate_atomic_objects_rejected() -> None:
    """错误输入不能导致超占或重复计费。"""
    ocm = CapacityOnChipMemory(128, rules())
    key = (0, 0, 'gate_up', 0)
    with pytest.raises(ValidationError):
        ocm.admit_group(((key, 33),), 'demand', 0, 0)
    with pytest.raises(ValidationError):
        ocm.admit_group(((key, 32), (key, 32)), 'demand', 0, 0)
    assert ocm.current_bytes == 0


def test_priority_demand_then_promoted_then_fifo_speculative() -> None:
    """score 不改变已 admission 后的 FIFO 顺序，promotion 不超越 demand。"""
    memory = MultiChannelMemory(profile(), pair_fifo=True)
    requests = [memory.new_request(layer_id=0, expert_id=e, matrix_type='gate_up',
        tile_ids=(0,), object_keys=((0, e, 'gate_up', 0),), transfer_bytes=16,
        request_type='demand' if e == 2 else 'speculative', issue_cycle=0,
        score=1 if e == 3 else .1) for e in range(4)]
    requests[1].committed = True
    for request in requests:
        memory.enqueue(request)
    observed = []
    for cycle in (0, 16, 32, 48):
        started = memory.start_ready(cycle)
        assert len(started) == 1
        observed.append(started[0].expert_id)
        memory.complete(started[0], cycle + 16)
    assert observed == [2, 1, 0, 3]


def test_retry_after_capacity_release_without_duplicates() -> None:
    """预测到达时 OCM 满；释放后重试，而不是直接丢弃。"""
    result = run_sequence()
    admissions = [v for v in result.trace if v['event_type'] == 'pair_admit']
    spec = [v for v in admissions if v['request_state'] == 'speculative']
    assert spec and spec[0]['cycle'] > result.predictions[0].predictor_ready_cycle
    assert not [v for v in run_sequence(retry=False).trace
                if v['event_type'] == 'pair_admit' and v['request_state'] == 'speculative']
    keys = [(v['layer_id'], v['expert_id'], v['tile_id']) for v in admissions]
    assert len(keys) == len(set(keys))
    assert all(v['occupied_bytes'] <= 128 and max(v['occupied_bytes_per_bank']) <= 64 for v in admissions)
    assert all(v['occupied_bytes'] <= 64 for v in spec)


def test_pair_dma_order_and_consumed_payload_conservation() -> None:
    """没有 slot padding DMA；每个 stage 只传一次且成对排队。"""
    result = run_sequence()
    source = [v for v in result.trace if v['event_type'] == 'dma_start' and v['layer_id'] == 0]
    assert [(v['tile_id'], v['matrix_type']) for v in source] == [
        (0, 'gate_up'), (0, 'down'), (1, 'gate_up'), (1, 'down')]
    assert result.layer_results[0].demand_bytes == 96
    assert result.layer_results[1].demand_bytes + result.layer_results[1].useful_speculative_bytes == 192


def test_gu_starts_before_down_payload_is_ready() -> None:
    """预留 DOWN ownership 不等于等待 DOWN payload；GU 的 overlap 保留。"""
    result = run_sequence()
    gu = next(v for v in result.trace if v['event_type'] == 'aie_start' and
              v['layer_id'] == 1 and v['expert_id'] == 2 and v['matrix_type'] == 'gate_up')
    down_done = next(v for v in result.trace if v['event_type'] == 'dma_done' and
                    v['layer_id'] == 1 and v['expert_id'] == 2 and v['tile_id'] == 0 and v['matrix_type'] == 'down')
    assert gu['cycle'] < down_done['cycle']


def test_single_engine_interleaves_at_tile_boundary() -> None:
    """有 ready prefix 的后序 Expert 可先执行，但不能锁整个 Expert。"""
    result = run_sequence(capacity=128)
    starts = [v for v in result.trace if v['event_type'] == 'aie_start' and v['layer_id'] == 1]
    gu = [v for v in starts if v['matrix_type'] == 'gate_up']
    assert gu[0]['expert_id'] == 2
    assert [v['expert_id'] for v in gu] == [2, 3, 3, 2]
    for gu_start, down_start in zip(starts[::2], starts[1::2]):
        assert gu_start['matrix_type'] == 'gate_up' and down_start['matrix_type'] == 'down'
        assert (gu_start['expert_id'], gu_start['tile_id']) == (down_start['expert_id'], down_start['tile_id'])
    intervals = sorted((v['cycle'], v['end_cycle']) for v in result.trace if v['event_type'] == 'aie_start')
    assert all(left[1] <= right[0] for left, right in zip(intervals, intervals[1:]))


def test_previous_weights_barrier_and_repeat_determinism() -> None:
    """不提前越过前层真实权重完成 barrier，重复执行结果一致。"""
    result = run_sequence(barrier=True)
    ready = max(v['cycle'] for v in result.trace if v['event_type'] == 'dma_done' and v['layer_id'] == 0)
    spec = [v for v in result.trace if v['event_type'] == 'pair_admit' and v['request_state'] == 'speculative']
    assert spec and min(v['cycle'] for v in spec) >= ready
    assert result == run_sequence(barrier=True)


def test_legacy_byte_mode_remains_payload_based() -> None:
    """通用模式不应无意变成 FPGA slot 模式。"""
    ocm = CapacityOnChipMemory(128)
    assert ocm.admit((0, 0, 'down', 0), 16, 'demand', 1, 0).admitted
    assert ocm.current_bytes == 16


def test_unadmitted_future_cancels_at_target_start() -> None:
    """所有 slots 保留给 demand 时，未 admission 的计划退休且不伪计 DMA。"""
    result = run_sequence(capacity=64)
    record = result.predictions[0]
    assert record.planned_speculative_bytes == record.canceled_unadmitted_bytes == 48
    assert record.initial_allocated_speculative_bytes == record.speculative_bytes == 0
    assert record.pending_unissued_speculative_bytes == 0
    stops = [v for v in result.trace if v['event_type'] == 'prefetch_stop']
    assert len(stops) == 1 and stops[0]['cycle'] == result.layer_results[1].layer_start_cycle


@pytest.mark.parametrize('capacity', [64, 128, 256])
@pytest.mark.parametrize('order', [(3, 2), (2, 3), (3, 4)])
def test_small_ocm_hit_and_wrong_prediction_progress(capacity: int, order: tuple[int, int]) -> None:
    """不同容量、真值顺序和零命中均完成；payload 不丢失、不重复执行。"""
    result = run_sequence(capacity=capacity, order=order)
    starts = [v for v in result.trace if v['event_type'] == 'aie_start' and v['layer_id'] == 1]
    expected = {(e, t, matrix) for e in order for t in (0, 1) for matrix in ('gate_up', 'down')}
    observed = [(v['expert_id'], v['tile_id'], v['matrix_type']) for v in starts]
    assert len(observed) == len(expected) and set(observed) == expected
    assert result.peak_ocm_bytes <= capacity
    record = result.predictions[0]
    assert record.speculative_bytes == record.useful_bytes + record.wasted_bytes
    assert record.initial_allocated_speculative_bytes == record.speculative_bytes + record.canceled_bytes


def test_non_unified_engine_rejects_execution_rules() -> None:
    """历史 AdvancedSimulator 不支持新规则，必须明确拒绝而非忽略。"""
    from edge_moe_sim.advanced_simulator import AdvancedSimulator
    from edge_moe_sim.advanced_schema import AdvancedScenario
    with pytest.raises(ValidationError, match='UnifiedSequenceSimulator'):
        AdvancedSimulator(ModelConfig('rules', 4, 8, 8, 2, 8, 8), profile()).simulate(
            BatchRoutingEvent(0, 0, 0, ((0,),)), AdvancedScenario.EXACT_TILE_STREAMING,
            AdvancedExecutionConfig(4, execution_rules=rules()))
