"""控制时间假设和 burst 进度的主模型验证，不读取 RTL 结果拟合。"""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import yaml

from edge_moe_sim.hardware.multi_channel_memory import MemoryRequest
from edge_moe_sim.schema import ModelConfig
from edge_moe_sim.unified_sequence_simulator import UnifiedSequenceSimulator
from test_execution_rules import profile
from test_unified_rtl_comparison import load_runner


def config() -> dict:
    """获取事先声明的结构参数。"""
    runner = load_runner()
    return yaml.safe_load((runner.PROJECT / 'configs/hardware/u200_control_alignment.yaml').read_text())


def test_burst_progress_excludes_setup_and_preemption_gap() -> None:
    """不能以 start→end 整段时间乘带宽，暂停窗口不传输。"""
    sim = UnifiedSequenceSimulator(ModelConfig('progress', 4, 8, 8, 2, 8, 8), profile())
    request = MemoryRequest(0, 0, 0, 'gate_up', (0,), ((0, 0, 'gate_up', 0),), 32, 'speculative', 0)
    request.channel_ids = (0,)
    request.burst_history = [(0, 18, 16, 2), (40, 58, 16, 42)]
    assert [sim._transferred_bytes_at(request, t) for t in (0, 2, 3, 18, 30, 42, 43, 58)] == [0, 0, 1, 16, 16, 16, 17, 32]


def test_control_config_deterministic_and_critical_experts_preserved(tmp_path: Path) -> None:
    """combine 不是 Expert 计算；不能因尾部控制把 critical Expert 清空。"""
    runner, cfg = load_runner(), config()
    a, b = tmp_path / 'a', tmp_path / 'b'
    a.mkdir(); b.mkdir()
    result = runner.simulate_fixture(cfg, cfg['cases'][1], a)
    assert result == runner.simulate_fixture(cfg, cfg['cases'][1], b)
    for layer in result['layer_results'].values():
        assert layer['critical_completion_expert_ids']
        finish = max(v['finish_cycle'] for v in layer['expert_timings'])
        assert layer['layer_done_cycle'] == finish + cfg['execution_rules']['layer_combine_cycles']
    assert result['memory_metrics']['descriptor_peak'] <= 8
    assert max(result['memory_metrics']['fifo_peaks'].values()) <= 4
    assert result['memory_metrics']['descriptor_live_at_end'] == 0


def test_truth_ready_and_demand_release_are_separate(tmp_path: Path) -> None:
    """resolver 扫描阻塞新 demand，但不禁止已 resident 的 speculative GU。"""
    runner, cfg = load_runner(), config()
    result = runner.simulate_fixture(cfg, cfg['cases'][1], tmp_path)
    layer = result['layer_results'][1]
    release = next(v for v in result['trace'] if v['event_type'] == 'demand_dispatch' and v['layer_id'] == 1)
    gu = next(v for v in result['trace'] if v['event_type'] == 'aie_start' and v['layer_id'] == 1)
    assert release['end_cycle'] - release['cycle'] == 13
    assert layer['router_resolve_cycle'] <= gu['cycle'] < release['end_cycle']


def test_predictor_transaction_stride_from_protocol(tmp_path: Path) -> None:
    """每个事务含 begin/candidate/teardown；后续不是错误的 2*layer 估计。"""
    runner, cfg = load_runner(), config()
    case = dict(cfg['cases'][3], predictor_delay_cycles=64)
    result = runner.simulate_fixture(cfg, case, tmp_path)
    assert [r['predictor_ready_cycle'] for r in result['predictions']] == [66, 69, 72, 75, 78, 81, 84]


def test_router_loading_has_explicit_time_and_does_not_change_truth(tmp_path: Path) -> None:
    """host port loading 是独立控制阶段，不误计为 Expert DDR payload。"""
    runner, cfg = load_runner(), config()
    cfg = deepcopy(cfg)
    cfg['execution_rules']['router_load_cycles'] = 7
    result = runner.simulate_fixture(cfg, cfg['cases'][0], tmp_path)
    assert len([v for v in result['trace'] if v['event_type'] == 'router_weight_load']) == 3
    for layer_id, layer in result['layer_results'].items():
        assert layer['true_expert_ids'] == (layer_id % 4 + 1, layer_id % 4 + 2)
        assert layer['demand_bytes'] == 3072


def test_completed_wrong_spec_releases_table_after_truth(tmp_path: Path) -> None:
    """全错且 DMA 早于 truth 完成时，不能因残留 table ownership 阻塞 demand。"""
    runner, cfg = load_runner(), config()
    case = dict(cfg['cases'][2], descriptor_entries=4, fifo_depth=2)
    result = runner.simulate_fixture(cfg, case, tmp_path)
    assert result['memory_metrics']['descriptor_peak'] <= 4
    assert result['memory_metrics']['descriptor_live_at_end'] == 0
    assert len(result['layer_results']) == 3


def test_minimal_table_keeps_demand_progress_without_weight_barrier(tmp_path: Path) -> None:
    """两项 table 全部保留给 demand；未来层无法占满而形成循环等待。"""
    runner, cfg = load_runner(), config()
    result = runner.simulate_fixture(cfg, dict(cfg['cases'][2], descriptor_entries=2, fifo_depth=2), tmp_path)
    assert len(result['layer_results']) == 3
    assert result['memory_metrics']['speculative_bytes'] == 0
    assert result['memory_metrics']['descriptor_live_at_end'] == 0


def test_interface_trace_determinism_and_retirement_guard(tmp_path: Path) -> None:
    """晚预测计划不能跨层清理期间偷偷 admission；不是删除晚预测激励。"""
    runner = load_runner()
    cfg = yaml.safe_load((runner.PROJECT / 'configs/hardware/u200_interface_alignment.yaml').read_text())
    a, b = tmp_path / 'a', tmp_path / 'b'
    a.mkdir(); b.mkdir()
    case = next(v for v in cfg['cases'] if v['name'] == 'predictor_late')
    result = runner.simulate_fixture(cfg, case, a)
    assert result == runner.simulate_fixture(cfg, case, b)
    gaps = [(layer['layer_done_cycle'], result['layer_results'][lid+1]['layer_start_cycle'])
            for lid, layer in result['layer_results'].items() if lid+1 in result['layer_results']]
    admits = [v for v in result['trace'] if v['event_type']=='pair_admit' and v['request_state']=='speculative']
    assert not any(start <= v['cycle'] < end for start, end in gaps for v in admits)
    assert not any(v['layer_id'] == 1 for v in admits)
    assert result['memory_metrics']['descriptor_live_at_end'] == 0


def test_interface_progress_excludes_return_tail() -> None:
    """较慢 payload 服务和 ready 返回不能提前累计字节。"""
    sim = UnifiedSequenceSimulator(ModelConfig('progress', 4, 8, 8, 2, 8, 8), profile())
    request = MemoryRequest(0, 0, 0, 'gate_up', (0,), ((0, 0, 'gate_up', 0),), 16, 'demand', 0)
    request.channel_ids = (0,)
    request.payload_cycle_multiplier = 2
    request.burst_history = [(3, 37, 16, 5)]
    request.burst_end_cycle = 39
    assert [sim._transferred_bytes_at(request, t) for t in (5, 7, 37, 38, 39)] == [0, 1, 16, 16, 16]
