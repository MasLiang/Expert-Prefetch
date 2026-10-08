"""有限控制 credits 与非抢占 burst 边界的定向验证。"""
from __future__ import annotations

from dataclasses import replace

import pytest

from edge_moe_sim.execution_rules import ExecutionRules
from edge_moe_sim.hardware.multi_channel_memory import MemoryRequest, MultiChannelMemory
from test_execution_rules import profile, rules


def server(*, descriptors: int = 4, depth: int = 4) -> MultiChannelMemory:
    """每 burst 16 byte，1 byte/cycle、2-cycle 初次 setup。"""
    config = replace(rules(), descriptor_entries=descriptors, dma_fifo_depth=depth, burst_arbitration=True)
    return MultiChannelMemory(replace(profile(), dma_setup_cycles=2), pair_fifo=True, rules=config)


def enqueue(memory: MultiChannelMemory, expert: int, kind: str, size: int = 32) -> MemoryRequest:
    """创建一个保持原始 request class 的物理描述符。"""
    request = memory.new_request(layer_id=0, expert_id=expert, matrix_type='gate_up',
        tile_ids=(0,), object_keys=((0, expert, 'gate_up', 0),), transfer_bytes=size,
        request_type=kind, issue_cycle=0)
    memory.enqueue(request)
    return request


def test_speculative_yields_only_after_completed_burst() -> None:
    """demand 不能打断已开始的 burst，但不必等完整 descriptor。"""
    memory = server()
    spec = enqueue(memory, 0, 'speculative')
    assert memory.start_ready(0) == [spec]
    demand = enqueue(memory, 1, 'demand', 16)
    assert memory.start_ready(4) == []
    assert not memory.complete(spec, 18)
    assert spec.transferred_bytes == 16 and spec.end_cycle is None
    assert memory.start_ready(18) == [demand]
    assert memory.complete(demand, 36)
    assert memory.start_ready(36) == [spec]
    assert memory.complete(spec, 54)
    assert spec.start_cycle == 0 and spec.end_cycle == 54
    assert memory.metrics(54)['transferred_bytes'] == 48


def test_descriptor_retirement_delay_starts_after_truth_arrives() -> None:
    """早完成、晚 truth 的回收延迟从 truth cycle 而非旧 DMA cycle 起算。"""
    memory = server(descriptors=2, depth=2)
    memory.rules = replace(memory.rules, descriptor_retirement_cycles=2)
    spec = enqueue(memory, 0, 'speculative', 16)
    memory.start_ready(0)
    memory.complete(spec, 18)
    assert spec.request_id in memory.live_descriptors
    spec.useful = False
    memory.retire_descriptors(30)
    assert memory.retirement_deadlines[spec.request_id] == 32
    memory.start_ready(31)
    assert spec.request_id in memory.live_descriptors
    memory.start_ready(32)
    assert spec.request_id not in memory.live_descriptors
    assert memory.final_retirement_cycle == 32


def test_serial_issue_cadence_and_return_tail() -> None:
    """issue 保持 credits，payload 节拍与 completion 返回尾部独立。"""
    memory = server()
    memory.rules = replace(memory.rules, dma_request_latency_cycles=3,
        dma_request_interval_cycles=1, dma_payload_cycle_multiplier=2, dma_return_cycles=2)
    first = enqueue(memory, 0, 'demand', 16)
    second = enqueue(memory, 1, 'demand', 16)
    assert (first.enqueue_ready_cycle, second.enqueue_ready_cycle) == (3, 4)
    assert memory.start_ready(2) == []
    assert memory.start_ready(3) == [first]
    assert first.burst_history == [(3, 37, 16, 5)]
    assert first.burst_end_cycle == 39
    assert memory.complete(first, 39)
    assert memory.start_ready(39) == [second]


@pytest.mark.parametrize('field', ['dma_request_latency_cycles', 'dma_request_interval_cycles',
                                  'dma_return_cycles', 'router_truth_emit_cycles'])
def test_negative_interface_time_rejected(field: str) -> None:
    """接口时间不能静默接受负 cycle。"""
    with pytest.raises(ValueError):
        replace(rules(), **{field: -1})


def test_demand_descriptor_does_not_yield_to_later_demand() -> None:
    """边界仲裁只允许低优先级 spec 让出，不把 demand descriptor 切碎重排。"""
    memory = server()
    first = enqueue(memory, 0, 'demand')
    second = enqueue(memory, 1, 'demand', 16)
    memory.start_ready(0)
    assert not memory.complete(first, 18)
    assert memory.start_ready(18) == [first]
    assert memory.complete(first, 34)
    assert memory.start_ready(34) == [second]


def test_canceled_fifo_token_must_retire_before_descriptor_reuse() -> None:
    """cancel 不立即回收 table：FIFO token 仍引用该 generation。"""
    memory = server(descriptors=2, depth=2)
    first = enqueue(memory, 0, 'speculative', 16)
    second = enqueue(memory, 1, 'speculative', 16)
    first.useful = second.useful = False
    assert memory.cancel(first) and memory.cancel(second)
    memory.retire_descriptors()
    assert len(memory.live_descriptors) == 2 and not memory.can_admit('demand', 2)
    assert memory.start_ready(0) == []
    assert memory.can_admit('demand', 2) and memory.tombstones_retired == 2


def test_completed_spec_waits_for_truth_and_stale_token() -> None:
    """completion 前后未知 truth 不提前释放 table；promotion token 不替代旧引用。"""
    memory = server()
    spec = enqueue(memory, 0, 'speculative', 16)
    assert memory.promote(spec)
    assert memory.start_ready(0) == [spec]
    assert memory.complete(spec, 18)
    assert spec.request_id in memory.live_descriptors
    assert memory.start_ready(18) == []
    assert spec.request_id not in memory.live_descriptors
    unknown = enqueue(memory, 1, 'speculative', 16)
    memory.start_ready(18)
    memory.complete(unknown, 36)
    assert unknown.request_id in memory.live_descriptors
    memory.promote(unknown)
    assert unknown.request_id not in memory.live_descriptors


def test_promoted_fifo_has_independent_capacity() -> None:
    """多个 hit 不得越过 promoted credit；dequeue 后可继续提交。"""
    memory = server(descriptors=6, depth=2)
    a = enqueue(memory, 0, 'speculative', 16)
    b = enqueue(memory, 1, 'speculative', 16)
    assert memory.promote(a) and memory.promote(b)
    memory.start_ready(0)
    # 原 speculative FIFO token 尚未退休，所以 spec admission 仍背压。
    assert not memory.can_admit('speculative', 1)
    assert len(memory.tokens['promoted']) == 1


def test_fifo_pair_credit_and_speculative_return_reservation() -> None:
    """active spec 需要保留一个回队 credit。"""
    memory = server(descriptors=8, depth=2)
    request = enqueue(memory, 0, 'speculative')
    memory.start_ready(0)
    assert not memory.can_admit('speculative', 2)
    assert memory.can_admit('demand', 2)
    assert memory.fifo_peaks['speculative'] <= 2


def test_payload_and_startup_bandwidth_conservation() -> None:
    """active interval 的 setup 不能产生 payload；每个 burst 统计实际 bytes。"""
    memory = server()
    request = enqueue(memory, 0, 'demand', 33)
    cycle = 0
    while not request.completed:
        assert memory.start_ready(cycle) == [request]
        cycle = request.burst_end_cycle
        assert cycle is not None
        memory.complete(request, cycle)
    assert [v.bytes_on_channel for v in memory.channels[0].intervals] == [16, 16, 1]
    for start, end, size, payload_start in request.burst_history:
        assert size <= (end - payload_start) and payload_start >= start
    assert memory.metrics(cycle)['transferred_bytes'] == 33


@pytest.mark.parametrize('field', ['descriptor_entries', 'dma_fifo_depth', 'router_control_cycles', 'dma_burst_gap_cycles'])
def test_negative_control_parameters_rejected(field: str) -> None:
    """所有 cycle 与 credit 数量必须有效。"""
    from edge_moe_sim.schema import ValidationError
    with pytest.raises(ValidationError):
        replace(rules(), **{field: -1})
