"""真实结构比例映射的预算、路由公平性与确定性。"""
from __future__ import annotations

from dataclasses import replace
import importlib.util
from pathlib import Path
import sys
from typing import Any

import pytest
import yaml


@pytest.fixture
def mapping(monkeypatch: pytest.MonkeyPatch) -> tuple[Any,dict[str,Any]]:
    """加载实验入口与配置，不执行完整矩阵。"""
    root = Path(__file__).resolve().parents[1]
    monkeypatch.setattr(sys,"path",[str(root/"examples"),*sys.path])
    spec = importlib.util.spec_from_file_location("real_mapping",root/"examples/real_ratio_mapping.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module,yaml.safe_load((root/"configs/experiments/real_ratio_mapping.yaml").read_text())


def test_equal_gu_budget_and_mixtral_scope(mapping: tuple[Any,dict[str,Any]]) -> None:
    """细tile等byte对照不改变GU预算；Mixtral明确为超预算理论配置。"""
    module,config = mapping
    models = [module.load_model_config(module.ROOT/p) for p in config["switch_models"]]
    models += [p.model for p in module.load_real_moe_profiles(module.ROOT/config["catalogue"])]
    for model in models:
        choices = {name:(tile,depth) for name,tile,depth in module.layouts(model,config)}
        if model.name.startswith("mixtral"):
            assert set(choices)=={"mapped_2_groups","mapped_4_groups"}
            assert module.expert_weight_bytes(model)*4>config["current_ocm_budget_bytes"]
        else:
            assert choices["mapped_2_groups"][0]*choices["mapped_2_groups"][1] == (
                choices["fine_equal_half_gu"][0]*choices["fine_equal_half_gu"][1])
            assert module.expert_weight_bytes(model)*4<=config["current_ocm_budget_bytes"]


def test_hits_first_only_changes_ranking(mapping: tuple[Any,dict[str,Any]]) -> None:
    """truth辅助排序必须保持候选集合、recall和truth不变。"""
    module,config = mapping
    record = {"truth":[[3,4],[0,2]]}
    prediction = {"target":1,"candidate_ids":[3,0,4,2]}
    random_ids = module.ordered_candidates(record,prediction,"uniform_random")
    sorted_ids = module.ordered_candidates(record,prediction,"uniform_hits_first")
    assert sorted_ids==(0,2,3,4)
    assert set(random_ids)==set(sorted_ids)
    assert prediction["candidate_ids"]==[3,0,4,2]
    assert module.ordered_candidates(record,prediction,"oracle_all_gu")== (0,2)


@pytest.mark.parametrize("policy",["no_prediction","uniform_random","uniform_hits_first","oracle_all_gu"])
def test_topk_and_deterministic_execution(mapping: tuple[Any,dict[str,Any]],policy: str) -> None:
    """保留真实OLMoE的Top-8，全部PE分配的结果可重复且byte守恒。"""
    module,config = mapping
    model = next(p.model for p in module.load_real_moe_profiles(module.ROOT/config["catalogue"])
                 if p.model.name=="olmoe_1b_7b")
    model = replace(model,moe_layer_indices=tuple(range(3)))
    settings = yaml.safe_load((module.ROOT/config["base_config"]).read_text())
    settings.update(samples=5,coverage=.9,candidate_count=16)
    record = module.make_batch_inputs(model,settings)[0]
    point = dict(model=model.name,layout="mapped_2_groups",f_tile_size=512,prefix_depth=1,
        pe_count=4,ops_per_cycle_per_pe=32,ratio=.9,bytes_per_cycle_per_channel=14.4,
        ocm_bytes=25165824,within_current_ocm_budget=True,seed=settings["seed"],policy=policy)
    row,timing = module.simulate((model,settings,point,record))
    assert (row,timing)==module.simulate((model,settings,point,record))
    assert row["top_k"]==8
    assert row["candidate_count"]==(8 if policy=="oracle_all_gu" else 16)
    assert row["peak_ocm_bytes"]<=row["ocm_bytes"]
    assert row["speculative_bytes"]==row["useful_speculative_bytes"]+row["wasted_speculative_bytes"]
    if policy=="oracle_all_gu":
        assert row["wasted_speculative_bytes"]==0


def test_cross_layout_baseline_is_paired(mapping: tuple[Any,dict[str,Any]]) -> None:
    """同layout收益不替代原始细tile速度比；缺少Mixtral分母保持空值。"""
    import analyze_real_ratio_mapping as analysis
    rows = [dict(model="switch",layout=layout,pe_count="1",ratio="0.9",seed="1",policy=policy,
        total_cycles=str(cycles),speedup=str(speed),within_current_ocm_budget="True")
        for layout,policy,cycles,speed in [("fine_original","no_prediction",100,1),
            ("mapped_2_groups","no_prediction",200,1),("mapped_2_groups","uniform_random",160,1.25)]]
    result = analysis.compare_layouts(rows)
    point = next(r for r in result if r["policy"]=="uniform_random")
    assert point["speedup_vs_same_layout_mean"]==1.25
    assert point["speedup_vs_fine_original_mean"]==.625
    missing = {**rows[2],"model":"mixtral"}
    assert analysis.compare_layouts([missing])[0]["speedup_vs_fine_original_mean"] is None
