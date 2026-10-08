"""共享ingress与FIFO credit/occupancy的因果及守恒检查。"""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
import yaml

from edge_moe_sim.hardware.multi_channel_memory import MultiChannelMemory, MemoryRequest
from edge_moe_sim.schema import ValidationError
from test_execution_rules import profile, rules
from test_unified_rtl_comparison import load_runner


def memory() -> MultiChannelMemory:
    """单通道小模型：入队latency=3、每拍一个token、push优先。"""
    return MultiChannelMemory(profile(), pair_fifo=True, rules=replace(rules(),
        descriptor_entries=8, dma_fifo_depth=4, dma_request_latency_cycles=3,
        dma_request_interval_cycles=1, ingress_pair_cycles=3, fifo_push_blocks_dequeue=True))


def request(server: MultiChannelMemory, expert: int, kind: str='speculative') -> MemoryRequest:
    """构造一个16-byte请求，不使用任何RTL结果。"""
    return server.new_request(layer_id=0, expert_id=expert, matrix_type='gate_up',
        tile_ids=(0,),object_keys=((0,expert,'gate_up',0),),transfer_bytes=16,
        request_type=kind,issue_cycle=0)


def test_reserved_credit_is_not_fifo_occupancy() -> None:
    """延迟入队、push优先pop和active spec回队保留不能混成一个数字。"""
    server=memory()
    for expert in (0,1):
        server.enqueue(request(server,expert))
    counts=server.sample_fifo_metrics(0)
    assert counts['reserved_credits']['speculative']==2
    assert counts['occupancy']['speculative']==0
    assert counts['reserved_not_enqueued']['speculative']==2
    server.start_ready(3)
    assert server.sample_fifo_metrics(4)['occupancy']['speculative']==2
    counts=server.sample_fifo_metrics(5)
    assert counts['occupancy']['speculative']==1
    assert counts['reserved_credits']['speculative']==2
    assert counts['reserved_not_enqueued']['speculative']==1
    for snapshot in server.fifo_snapshots:
        for kind in ('demand','promoted','speculative'):
            assert snapshot['reserved_credits'][kind]==snapshot['occupancy'][kind]+snapshot['reserved_not_enqueued'][kind]


def test_ingress_is_a_mutually_exclusive_resource() -> None:
    """pair尾部未退休时不能接受下一pair；不要求等待DMA payload结束。"""
    server=memory()
    assert server.reserve_ingress_pair(0)==3
    with pytest.raises(ValidationError,match='ingress'):
        server.reserve_ingress_pair(2)
    assert server.reserve_ingress_pair(3)==6
    assert server.ingress_intervals==[(0,3),(3,6)]


def test_fifo_pop_reference_prevents_early_descriptor_recycle() -> None:
    """逻辑选走stale token后，物理pop仍未可见时不能回收descriptor。"""
    server=memory();item=request(server,0,'demand')
    server.enqueue(item);assert server.cancel(item)
    assert server.start_ready(3)==[]
    assert item.request_id in server.live_descriptors
    server.retire_descriptors(4)
    assert item.request_id not in server.live_descriptors


def test_layer_transition_waits_actual_ingress_only(tmp_path: Path) -> None:
    """临界pair尾部导致一拍交接等待，但不等未来DMA整体排空。"""
    runner=load_runner()
    cfg=yaml.safe_load((runner.PROJECT/'configs/hardware/u200_ingress_alignment.yaml').read_text())
    result=runner.simulate_fixture(cfg,dict(cfg['cases'][1],predictor_delay_cycles=446),tmp_path)
    waits=[v for v in result['trace'] if v['event_type']=='retirement_resource_wait']
    assert len(waits)==1
    assert waits[0]['cycle']==458 and waits[0]['end_cycle']==459
    assert waits[0]['ingress_pending'] and not waits[0]['resolver_pending']
    assert result['layer_results'][1]['layer_start_cycle']==472
    assert result['makespan_cycles']==1323
    assert result['memory_metrics']['fifo_occupancy_peaks']['demand']==3
    assert result['memory_metrics']['fifo_reserved_credit_peaks']['demand']==4
    assert result['memory_metrics']['fifo_occupancy_peaks']['speculative']==2
    assert result['memory_metrics']['descriptor_live_at_end']==0


def test_ingress_trace_determinism(tmp_path: Path) -> None:
    """资源退休与FIFO计数不依赖运行历史。"""
    runner=load_runner()
    cfg=yaml.safe_load((runner.PROJECT/'configs/hardware/u200_ingress_alignment.yaml').read_text())
    first,second=tmp_path/'first',tmp_path/'second';first.mkdir();second.mkdir()
    assert runner.simulate_fixture(cfg,cfg['cases'][1],first)==runner.simulate_fixture(cfg,cfg['cases'][1],second)
