"""FLAME条件对照的质量、放行屏障、完整驻留诊断及配对指标。"""
from __future__ import annotations

from dataclasses import replace
import importlib.util
from pathlib import Path
import sys
from typing import Any

import pytest
import yaml


@pytest.fixture
def study(monkeypatch: pytest.MonkeyPatch) -> tuple[Any,dict[str,Any],dict[str,Any],dict[str,Any]]:
    """加载受控实验，不使用RTL或外部predictor。"""
    root = Path(__file__).resolve().parents[1]
    monkeypatch.setattr(sys,"path",[str(root/"examples"),*sys.path])
    spec = importlib.util.spec_from_file_location("flame_condition_study",root/"examples/flame_condition_comparison.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    geometry = yaml.safe_load((root/"configs/experiments/flame_condition_comparison.yaml").read_text())
    common = yaml.safe_load((root/geometry["common_config"]).read_text())
    base = yaml.safe_load((root/geometry["base_config"]).read_text())
    return module,geometry,common,base


def test_exact_flame_inspired_candidate_quality(study: tuple[Any,...]) -> None:
    """每seed250个预测目标中211命中；解除屏障是主比较，不是后验选择。"""
    module,geometry,_,base = study
    assert geometry["primary_wait_for_previous_layer_weights"] is False
    for model_path in geometry["models"]:
        model = module.load_model_config(module.ROOT/model_path)
        assert model.top_k == 1 and model.expert_architecture == "relu_ffn"
        for seed in geometry["seeds"]:
            records = module.make_inputs(model,{**base,**geometry,"seed":seed})
            assert sum(p["true_rank"]>=0 for r in records for p in r["predictions"]) == 211


def test_reported_head_quality_sensitivity(study: tuple[Any,...]) -> None:
    """独立控制Top-1/Top-2数量，不改truth或candidate集合，不称为实际CEPR分布。"""
    module,geometry,_,base = study
    model = module.load_model_config(module.ROOT/geometry["models"][0])
    settings = {**base,"samples":100,"coverage":.844,"candidate_count":2,"top1_coverage":.758}
    original = module.make_inputs(model,settings)
    traces = module.make_traces(model,settings)
    assert traces == module.make_traces(model,settings)
    predictions = [p for r in traces for p in r["predictions"]]
    assert sum(p["true_rank"]==0 for p in predictions) == 379
    assert sum(p["true_rank"]>=0 for p in predictions) == 422
    for old,new in zip(original,traces):
        assert [[expert] for expert in old["truth"]] == new["truth"]
        for a,b in zip(old["predictions"],new["predictions"]):
            assert set(a["candidate_ids"]) == set(b["candidate_ids"])
    with pytest.raises(ValueError,match="top1_coverage"):
        module.make_traces(model,{**settings,"top1_coverage":.9})


@pytest.mark.parametrize('experts', [8, 16])
def test_real_switch_experiment_configuration(study: tuple[Any,...], experts: int) -> None:
    """真实尺寸、可装下整 Expert 的容量和准确候选统计不得被小型夹具替代。"""
    module,_,_,base = study
    geometry = yaml.safe_load((module.ROOT/'configs/experiments/switch_real_model_four_cases.yaml').read_text())
    model = module.load_model_config(module.ROOT/f'configs/models/switch_base_{experts}.yaml')
    assert (model.hidden_size,model.intermediate_size,model.num_experts,model.top_k) == (768,3072,experts,1)
    assert model.input_projection_count == 1 and model.expert_architecture == 'relu_ffn'
    assert len(model.moe_layer_indices) == 6
    assert geometry['ocm_bytes'] >= 2*2*768*3072
    assert geometry['mac_groups'] == [1] and geometry['ops_per_cycle_per_group'] == 32
    assert geometry['primary_wait_for_previous_layer_weights'] is False
    records = module.make_traces(model,{**base,**geometry,'seed':geometry['seeds'][0]})
    predictions = [p for r in records for p in r['predictions']]
    assert len(records) == 100 and len(predictions) == 500
    assert sum(p['true_rank']==0 for p in predictions) == 379
    assert sum(p['true_rank']>=0 for p in predictions) == 422


@pytest.mark.parametrize("case,barrier",[("C_whole_prefetch",False),("C_whole_prefetch",True),
    ("D_stream_tile_prefetch",False),("D_stream_tile_prefetch",True)])
def test_release_and_residency_diagnostics(study: tuple[Any,...],case: str,barrier: bool) -> None:
    """诊断可重复；完整resident和candidate hit不同，D不冒充完整Expert就绪。"""
    module,geometry,common,base = study
    original = module.load_model_config(module.ROOT/geometry["models"][0])
    model = replace(original,hidden_size=32,intermediate_size=128,moe_layer_indices=(0,1,2))
    record = module.make_inputs(model,{**base,"samples":1,"coverage":1,"candidate_count":2})[0]
    record["truth"] = [[expert] for expert in record["truth"]]
    point = dict(model=model.name,mac_groups=1,bytes_per_cycle_per_channel=16,
        ocm_bytes=32768,case=case,barrier=barrier,seed=base["seed"])
    job = (model,base,geometry,common,{"f_tile_size":32},point,record)
    row,details = module.simulate(job)
    assert (row,details) == module.simulate(job)
    assert row["candidate_hits"] == row["prediction_targets"] == 2
    for item in details:
        assert item["prefetch_release_cycle"] is not None
        assert item["prefetch_release_cycle"] >= item["predictor_ready_cycle"]
        if case=="C_whole_prefetch":
            assert item["full_true_resident_at_router"] in {0,1}
            assert item["startup_resident_fraction_at_router"] <= 1
        else:
            assert item["full_true_resident_at_router"] is None


def test_barrier_ablation_denominators(study: tuple[Any,...]) -> None:
    """屏障关闭收益与相对弱baseline的收益必须分别计算。"""
    module,_,_,_ = study
    rows = []
    for (case,barrier),cycles in zip(module.CASES,[200,120,150,100,180,125]):
        rows.append(dict(model="test",mac_groups=1,bytes_per_cycle_per_channel=16,ocm_bytes=1024,
            case=case,barrier=barrier,seed=1,samples=1,expert_bytes=128,peak_ocm_bytes=128,
            total_cycles=cycles,memory_wait_sum_cycles=cycles-80,prediction_targets=2,
            candidate_hits=2,full_true_resident_hits=1,full_true_transferred_hits=1,
            useful_physical_bytes_before_router=128,release_delay_sum_cycles=4,
            speculative_bytes=256,useful_speculative_bytes=128,wasted_speculative_bytes=128))
    result = module.aggregate(rows)
    d = next(r for r in result if r["case"]=="D_stream_tile_prefetch" and not r["barrier"])
    assert d["speedup_vs_A"] == 2.0 and d["speedup_vs_B"] == 1.2
    assert d["speedup_vs_barrier_on"] == 1.25
    c = next(r for r in result if r["case"]=="C_whole_prefetch" and not r["barrier"])
    assert c["full_true_ready_recall"] == .5 and c["candidate_recall"] == 1.0
