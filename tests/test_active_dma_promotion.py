"""命中提升影响服务优先级，不改变原 speculative byte 归属。"""
from __future__ import annotations

from dataclasses import replace

import pytest

from edge_moe_sim.hardware.multi_channel_memory import MemoryRequest, MultiChannelMemory
from test_execution_rules import profile, rules


@pytest.mark.parametrize('promote_after_requeue', [False, True])
def test_started_hit_does_not_fall_behind_future_speculation(promote_after_requeue: bool) -> None:
    """覆盖 active 命中和 burst 已让出后命中；保留 payload 与 token 守恒。"""
    memory = MultiChannelMemory(profile(), pair_fifo=True, rules=replace(rules(),
        descriptor_entries=8, dma_fifo_depth=8, burst_arbitration=True))

    def request(expert: int, size: int, kind: str) -> MemoryRequest:
        """创建并提交有限 credit 请求，所有大小为 byte。"""
        item = memory.new_request(layer_id=expert, expert_id=expert, matrix_type='gate_up',
            tile_ids=(0,), object_keys=((expert, expert, 'gate_up', 0),), transfer_bytes=size,
            request_type=kind, issue_cycle=0)
        memory.enqueue(item)
        return item

    hit = request(0, 64, 'speculative')
    assert memory.start_ready(0) == [hit]
    future = request(1, 16, 'speculative')
    demand = request(2, 16, 'demand')
    if not promote_after_requeue:
        assert memory.promote(hit, 4)
    assert not memory.complete(hit, 16)
    if promote_after_requeue:
        assert hit in memory.tokens['speculative']
        assert memory.promote(hit, 16)
        assert hit.promotion_token_created
    cycle = 16
    finished = []
    for _ in range(10):
        started = memory.start_ready(cycle)
        if not started:
            break
        item = started[0]
        cycle = item.burst_end_cycle
        assert cycle is not None
        if memory.complete(item, cycle):
            finished.append(item.expert_id)
    assert finished.index(0) < finished.index(1)
    assert finished.index(2) < finished.index(1)
    assert hit.transferred_bytes == 64 and demand.transferred_bytes == future.transferred_bytes == 16
    assert hit.request_type == 'speculative' and hit.useful is True
    assert sum(channel.transferred_bytes for channel in memory.channels) == 96
    assert memory.promote(future, cycle)  # Future truth arrives only after its DMA completes.
    memory.retire_descriptors(cycle)
    assert not memory.live_descriptors
