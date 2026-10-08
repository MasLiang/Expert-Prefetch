"""DSP 扩宽扫描的真实性能参数、预算、配对和离散饱和点验证。"""
from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any

import pytest
import yaml


def driver(monkeypatch: pytest.MonkeyPatch) -> Any:
    """导入驱动而不执行模拟矩阵。"""
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]/'examples'))
    return importlib.import_module('u200_dsp_lane_sweep')


def settings() -> dict[str, Any]:
    """读取实际实验配置，确保测试与运行采用同一资源映射。"""
    return yaml.safe_load((Path(__file__).resolve().parents[1]/'configs/experiments/u200_dsp_lane_sweep.yaml').read_text())


@pytest.mark.parametrize('family',['switch','moe'])
def test_rate_label_matches_actual_config(family: str,monkeypatch: pytest.MonkeyPatch) -> None:
    """Switch 的真实速率来自 config，而其他模型来自 rate；两者必须一起更新。"""
    m = driver(monkeypatch)
    original = {'ops_per_cycle':32}
    job = (family,None,original,{}, {},16,32,'stream_demand',{})
    result = m.scale_job(job,256,settings())
    assert result[2]['ops_per_cycle']==512
    assert result[6]==512
    assert original['ops_per_cycle']==32


def test_dsp_estimate_not_peak_performance(monkeypatch: pytest.MonkeyPatch) -> None:
    """预算按一 lane 一 DSP 估计，保留控制与预留，不作综合结果声明。"""
    r = driver(monkeypatch).dsp_estimate(512,settings())
    assert r['estimated_mac_dsp']==512
    assert r['estimated_total_dsp']==572
    assert r['ops_per_cycle']==1024
    assert r['study_dsp_budget']==5472
    assert r['fits_study_dsp_budget']


@pytest.mark.parametrize('fault',['duplicate','tail','no_reference','over_budget','multiple_bandwidths','bool_dsp'])
def test_invalid_configuration(fault: str,monkeypatch: pytest.MonkeyPatch) -> None:
    """拒绝重复档位、非 16 倍数、缺少 baseline、超预算和多带宽混入。"""
    cfg = settings()
    if fault=='duplicate':cfg['mac_lanes']=[16,32,32]
    elif fault=='tail':cfg['mac_lanes']=[16,31]
    elif fault=='no_reference':cfg['mac_lanes']=[32,64]
    elif fault=='over_budget':cfg['mac_lanes']=[16,6000]
    elif fault=='multiple_bandwidths':cfg['bytes_per_cycle_per_channel']=[8,16]
    else:cfg['dsp_per_mac_lane']=True
    with pytest.raises(ValueError):driver(monkeypatch).validate_settings(cfg)


def sample(lanes: int,policy: str,index: int,cycles: int) -> dict[str, Any]:
    """构造含相同工作量的周期样本。"""
    return dict(model='test',family='switch',ops_per_cycle=lanes*2,bytes_per_cycle_per_channel=16,
        ocm_bytes=1024,policy=policy,sample=index,clock_hz=200000000,num_memory_channels=4,
        nominal_total_bandwidth_gbps=12.8,bandwidth_efficiency=1.,total_cycles=cycles,
        peak_ocm_bytes=512,memory_wait_cycles=10,demand_bytes=100,speculative_bytes=0,
        useful_speculative_bytes=0,wasted_speculative_bytes=0,payload_ops=3200,
        ideal_payload_compute_lower_bound_cycles=(3200+lanes*2-1)//(lanes*2))


def test_aggregate_keeps_two_speedup_denominators(monkeypatch: pytest.MonkeyPatch) -> None:
    """扩宽收益和预取收益不能共用分母；总周期配对优于逐样本比的均值。"""
    m = driver(monkeypatch)
    cfg = {**settings(),'mac_lanes':[16,32]}
    raw = [sample(l,'stream_demand',i,c) for l,cs in [(16,[100,300]),(32,[100,100])]
        for i,c in enumerate(cs)]
    raw += [sample(l,'stream_uniform',i,c) for l,cs in [(16,[100,200]),(32,[50,100])]
        for i,c in enumerate(cs)]
    rows = m.aggregate(raw,cfg)
    point = next(r for r in rows if r['policy']=='stream_uniform' and r['mac_lanes']==32)
    assert point['speedup_vs_16_lanes_same_policy']==2
    assert point['speedup_vs_stream_demand']==pytest.approx(200/150)
    assert point['speedup_vs_previous_lane_point']==2
    assert 'absolute_speedup_vs_lowest_bandwidth_same_policy' not in point
    assert point['payload_fraction_lower_bound']==pytest.approx(100/150)


def test_discrete_knee_does_not_assume_monotonicity(monkeypatch: pytest.MonkeyPatch) -> None:
    """最宽点不必最快；距最快 1% 内最小档位不等于连续拐点。"""
    rows = [dict(model='x',policy='stream_demand',mac_lanes=l,cycles_per_sequence=c,
        speedup_vs_16_lanes_same_policy=200/c,payload_fraction_lower_bound=.2,
        speedup_vs_stream_demand=1) for l,c in [(16,200),(32,101),(64,100),(128,105)]]
    p = driver(monkeypatch).knees(rows)[0]
    assert p['fastest_tested_lanes']==64
    assert p['smallest_tested_lanes_within_1percent_fastest']==32
    assert p['largest_lanes_speedup_vs_16']==pytest.approx(200/105)


def test_reject_different_16_lane_samples(monkeypatch: pytest.MonkeyPatch) -> None:
    """不同算力档位必须用同样本，不只核对相同档位内的预取 baseline。"""
    raw = [sample(16,'stream_demand',0,200),sample(32,'stream_demand',1,100)]
    with pytest.raises(ValueError):driver(monkeypatch).aggregate(raw,{**settings(),'mac_lanes':[16,32]})
