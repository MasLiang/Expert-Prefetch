"""内存利用率口径、相图分类、因果perfect及配对样本验证。"""
from __future__ import annotations
import importlib.util
import json
from pathlib import Path
from typing import Any
import pytest
import yaml


@pytest.fixture
def experiment(monkeypatch: pytest.MonkeyPatch) -> tuple[Any, dict[str, Any], dict[str, Any], dict[str, Any]]:
    """加载真实配置与保存的第一条trace。"""
    root=Path(__file__).resolve().parents[1]; monkeypatch.syspath_prepend(str(root/"examples"))
    spec=importlib.util.spec_from_file_location("headroom",root/"examples/prefetch_memory_headroom.py")
    assert spec and spec.loader
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    config=yaml.safe_load((root/"configs/experiments/prefetch_memory_headroom.yaml").read_text())
    base=yaml.safe_load((root/config["base_config"]).read_text())
    record=json.loads((root/config["historical_directory"]/"inputs.json").read_text())["20261005"][0]
    return module,config,base,record


def test_utilization_channel_time_not_any_channel(experiment: tuple[Any,...]) -> None:
    """单通道100cycles不等于四通道100%利用率；setup与payload分开。"""
    m,*_=experiment
    assert m.measure_utilization(100,160,4,8,100)==(.25,.05)
    with pytest.raises(AssertionError):m.measure_utilization(401,0,4,8,100)


def test_phase_threshold_and_cost_tie(experiment: tuple[Any,...]) -> None:
    """均值正收益不能替代所有seed阈值；near-tie优先LIGHT。"""
    m,*_=experiment
    light=dict(policy="LIGHT",speedup_min=1.02,total_cycles=100,speculative_bytes=16)
    full=dict(policy="PREFETCH",speedup_min=1.021,total_cycles=99.8,speculative_bytes=96)
    assert m.recommend([light,full],1.01,.005)=="LIGHT"
    assert m.recommend([light,{**full,"total_cycles":90}],1.01,.005)=="PREFETCH"
    assert m.recommend([{**light,"speedup_min":1.009}],1.01,.005)=="OFF"


@pytest.mark.parametrize("policy",["OFF","LIGHT","PREFETCH"])
def test_actual_perfect_determinism(experiment: tuple[Any,...],policy: str) -> None:
    """真实运行不绕过资源；perfect无错字节，baseline利用率分母无尾部。"""
    m,c,b,r=experiment
    job=(b,c,.9,1.,policy,20261005,r)
    a=m.run_sample(job)
    assert a==m.run_sample(job)
    assert a["wasted_speculative_bytes"]==0
    assert 0<=a["payload_utilization"]<=a["occupied_utilization"]<=1
    if policy=="OFF":assert a["speculative_bytes"]==0
    else:assert a["hit_decisions"]==4


def test_paired_aggregation_rejects_missing(experiment: tuple[Any,...]) -> None:
    """配对分母不能混用硬件点或不同样本。"""
    m,c,b,r=experiment
    baseline=m.run_sample((b,c,.9,0.,"OFF",20261005,r))
    light=m.run_sample((b,c,.9,1.,"LIGHT",20261005,r))
    with pytest.raises(ValueError,match="paired sample"):
        m.aggregate([baseline,{**light,"sample":99}],4)
