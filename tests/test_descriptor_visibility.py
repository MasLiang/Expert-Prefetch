"""逐 descriptor 的控制可见时序；不以 RTL 总周期作为模型输入。"""
from __future__ import annotations

from pathlib import Path
from copy import deepcopy

import pytest
import yaml

from edge_moe_sim.execution_rules import ExecutionRules
from edge_moe_sim.schema import ValidationError
from edge_moe_sim.hardware.multi_channel_memory import MultiChannelMemory, MemoryRequest
from test_unified_rtl_comparison import load_runner


def configuration() -> dict:
    """读取固定、显式声明的 RTL 结构配置。"""
    runner = load_runner()
    return yaml.safe_load((runner.PROJECT/'configs/hardware/u200_descriptor_alignment.yaml').read_text())


def test_commit_and_payload_are_independent(tmp_path: Path) -> None:
    """hit GU 可被选择，但 GEMM 必须晚于逐项 commit 的可见时间。"""
    runner, cfg = load_runner(), configuration()
    result = runner.simulate_fixture(cfg, cfg['cases'][1], tmp_path)
    trace = result['trace']
    for layer in (1, 2):
        truth = result['layer_results'][layer]['router_resolve_cycle']
        commits = [v for v in trace if v['event_type']=='descriptor_commit_visible' and v['layer_id']==layer]
        assert len(commits)==2
        assert [v['cycle']-truth for v in commits]==[7, 8]
        for commit in commits:
            key = tuple(commit[k] for k in ('layer_id','expert_id','tile_id','matrix_type'))
            start = next(v for v in trace if v['event_type']=='aie_start'
                and tuple(v.get(k) for k in ('layer_id','expert_id','tile_id','matrix_type'))==key)
            assert start['cycle']>=commit['cycle']+1
    assert result['memory_metrics']['descriptor_live_at_end']==0


def test_visibility_change_not_bandwidth_compensation(tmp_path: Path) -> None:
    """OCM 可见延迟不改变 payload bytes 或单个 DMA 服务区间。"""
    runner, cfg = load_runner(), configuration()
    delayed, immediate = tmp_path/'delayed', tmp_path/'immediate'
    delayed.mkdir(); immediate.mkdir()
    a = runner.simulate_fixture(cfg, cfg['cases'][0], delayed)
    cfg = deepcopy(cfg); cfg['execution_rules']['ocm_payload_visibility_cycles']=0
    b = runner.simulate_fixture(cfg, cfg['cases'][0], immediate)
    for result in (a,b):
        for layer in result['layer_results'].values():
            assert layer['demand_bytes']==3072
    first_a=next(v for v in a['trace'] if v['event_type']=='dma_done')
    first_b=next(v for v in b['trace'] if v['event_type']=='dma_done')
    assert first_a['cycle']==first_b['cycle']
    payload=next(v for v in a['trace'] if v['event_type']=='payload_ready')
    assert payload['cycle']==first_a['cycle']+1


@pytest.mark.parametrize('delay,expected_bytes', [(440,1536),(441,1536),(446,1536),(447,768),(448,768)])
def test_admission_boundary_from_structure(tmp_path: Path, delay: int, expected_bytes: int) -> None:
    """候选树大小与 source 生命周期共同决定截止点，不读取 RTL log。"""
    runner, cfg = load_runner(), configuration()
    case = dict(cfg['cases'][1], predictor_delay_cycles=delay)
    result = runner.simulate_fixture(cfg, case, tmp_path)
    assert sum(v['request_bytes'] for v in result['speculative_request_records'])==expected_bytes
    gaps=[(v['layer_done_cycle'],result['layer_results'][i+1]['layer_start_cycle'])
        for i,v in result['layer_results'].items() if i+1 in result['layer_results']]
    # 同cycle的PREDICTOR_READY在LAYER_COMBINE_DONE之前：已接受的pair不撤销。
    # 逐事件检查截止，而不是将合法的同拍reserve误作retirement新请求。
    assert not any(begin<v['cycle']<end for begin,end in gaps for v in result['trace']
        if v['event_type']=='pair_admit' and v['request_state']=='speculative')
    for begin,_ in gaps:
        end_index=next(i for i,v in enumerate(result['trace'])
            if v['event_type']=='LAYER_COMBINE_DONE' and v['cycle']==begin)
        assert not any(v['event_type']=='pair_admit' and v['request_state']=='speculative'
                       and v['cycle']==begin for v in result['trace'][end_index+1:])


def test_descriptor_timing_requires_capacity() -> None:
    """无限 descriptor 表不能隐式指定逐项有限 resolver。"""
    with pytest.raises(ValidationError, match='descriptor_entries'):
        ExecutionRules(route_commit_visibility_cycles=3)


def test_promotion_backpressure_cannot_overtake(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """单元级注入一次promotion backpressure，验证resolver不会跳过受阻token。"""
    original = MultiChannelMemory.promote
    rejected: set[int] = set()

    def stall_first(memory: MultiChannelMemory, request: MemoryRequest, cycle: int) -> bool:
        """仅延迟第一个hit；仍使用真实promotion实现，不修改请求内容。"""
        if request.layer_id == 1 and request.matrix_type == 'gate_up' and request.request_id not in rejected:
            rejected.add(request.request_id)
            return False
        return original(memory, request, cycle)

    monkeypatch.setattr(MultiChannelMemory, 'promote', stall_first)
    runner, cfg = load_runner(), configuration()
    result = runner.simulate_fixture(cfg, cfg['cases'][1], tmp_path)
    commits = [v for v in result['trace'] if v['event_type']=='descriptor_commit_visible' and v['layer_id']==1]
    assert [v['matrix_type'] for v in commits]==['gate_up', 'down']
    assert commits[1]['cycle']>commits[0]['cycle']
    truth=result['layer_results'][1]['router_resolve_cycle']
    assert commits[0]['cycle']==truth+8
    for commit in commits:
        start=next(v for v in result['trace'] if v['event_type']=='aie_start'
            and all(v.get(k)==commit[k] for k in ('layer_id','expert_id','tile_id','matrix_type')))
        assert start['cycle']>commit['cycle']


def test_false_tokens_retire_after_resolution(tmp_path: Path) -> None:
    """错误预测的resolver entry也需按扫描时序退休，最终无payload/credit泄漏。"""
    runner, cfg = load_runner(), configuration()
    result=runner.simulate_fixture(cfg, cfg['cases'][2], tmp_path)
    cancel_events=[v for v in result['trace'] if v['event_type']=='descriptor_cancel_visible']
    assert len(cancel_events)==4
    for layer in (1,2):
        truth=result['layer_results'][layer]['router_resolve_cycle']
        assert [v['cycle']-truth for v in cancel_events if v['layer_id']==layer]==[7,8]
    assert result['memory_metrics']['descriptor_live_at_end']==0


def test_descriptor_mode_determinism(tmp_path: Path) -> None:
    """同一逐descriptor配置必须产生完全相同的结果和事件顺序。"""
    runner,cfg=load_runner(),configuration()
    first,second=tmp_path/'first',tmp_path/'second'
    first.mkdir();second.mkdir()
    assert runner.simulate_fixture(cfg,cfg['cases'][1],first)==runner.simulate_fixture(cfg,cfg['cases'][1],second)
