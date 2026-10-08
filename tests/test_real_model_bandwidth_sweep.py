"""带宽扫描的配对、阈值与输入检查；不修改模拟器语义。"""
from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any

import pytest


def driver(monkeypatch: pytest.MonkeyPatch) -> Any:
    """导入实验驱动，不启动进程或真实模型矩阵。"""
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]/'examples'))
    return importlib.import_module('real_model_bandwidth_sweep')


def sample(bw: int, policy: str, index: int, cycles: int) -> dict[str, Any]:
    """构造相同模型/硬件的配对样本，时间为整数 cycle。"""
    return dict(model='test', family='switch', ops_per_cycle=32,
        bytes_per_cycle_per_channel=bw, ocm_bytes=1024, policy=policy, sample=index,
        clock_hz=200000000, num_memory_channels=4, nominal_total_bandwidth_gbps=.8*bw,
        bandwidth_efficiency=1., total_cycles=cycles, peak_ocm_bytes=512,
        memory_wait_cycles=20, demand_bytes=1024, speculative_bytes=0,
        useful_speculative_bytes=0, wasted_speculative_bytes=0)


def test_aggregate_uses_paired_total_not_mean_speedup(monkeypatch: pytest.MonkeyPatch) -> None:
    """必须 sum(baseline cycles)/sum(policy cycles)，并明确绝对提速的分母。"""
    m = driver(monkeypatch)
    rows = []
    for bw in (8, 16):
        for policy, cycles in [('stream_demand', (100, 300)), ('stream_uniform', (100, 100)),
                               ('whole_demand', (200, 600))]:
            rows += [sample(bw, policy, i, c//(bw//8)) for i, c in enumerate(cycles)]
    out = m.aggregate(rows)
    p = next(r for r in out if r['policy']=='stream_uniform' and r['bytes_per_cycle_per_channel']==16)
    assert p['speedup_vs_stream_demand'] == 2
    assert p['speedup_vs_whole_demand'] == 4
    assert p['absolute_speedup_vs_lowest_bandwidth_same_policy'] == 2
    assert p['cycles_per_sequence'] == 50
    assert p['saved_cycles_vs_stream_demand'] == 50


@pytest.mark.parametrize('fault', ['duplicate', 'unpaired'])
def test_reject_bad_baseline_pairing(fault: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """不允许重复样本或跨样本配对 baseline。"""
    m = driver(monkeypatch)
    rows = [sample(8, 'stream_demand', 0, 100), sample(8, 'stream_uniform', 0, 90)]
    if fault == 'duplicate':
        rows.append(dict(rows[1]))
    else:
        rows[1]['sample'] = 1
    with pytest.raises(ValueError):
        m.aggregate(rows)


def test_mixtral_has_no_fabricated_whole_baseline(monkeypatch: pytest.MonkeyPatch) -> None:
    """容量不可行的 whole-residency baseline 输出 null，不伪造性能。"""
    m = driver(monkeypatch)
    rows = [sample(16, 'no_prediction', 0, 100), sample(16, 'uniform_prefix', 0, 90)]
    for r in rows:
        r['family'] = 'moe'
    out = m.aggregate(rows)
    assert all(r['speedup_vs_whole_demand'] is None for r in out)


def test_nonmonotonic_bandwidth_threshold(monkeypatch: pytest.MonkeyPatch) -> None:
    """带宽最大不必收益最大；阈值仅指测试点，不能作单调插值。"""
    m = driver(monkeypatch)
    rows = [dict(model='x', ops_per_cycle=32, policy='stream_uniform',
        bytes_per_cycle_per_channel=bw, speedup_vs_stream_demand=s)
        for bw,s in [(8,.9), (16,1.005), (32,1.02), (64,1.001)]]
    result = m.crossover(rows)[0]
    assert result['first_tested_bw_positive'] == 16
    assert result['first_tested_bw_above_1percent'] == 32
    assert result['best_tested_bw'] == 32
    assert result['highest_bw_speedup_vs_stream_demand'] == 1.001


@pytest.mark.parametrize('bandwidths', [[0], [8,8], [16,8], [True], [8.5], []])
def test_invalid_bandwidths(bandwidths: list[Any], monkeypatch: pytest.MonkeyPatch) -> None:
    """byte/cycle 必须正整数、唯一且递增。"""
    with pytest.raises(ValueError):
        driver(monkeypatch).validate_settings(dict(samples_per_model=20, ocm_bytes=1024,
            bytes_per_cycle_per_channel=bandwidths))
