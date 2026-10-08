"""四case共享参数、完整加载语义配置和配对分母验证。"""
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
    """加载实验入口及冻结配置。"""
    root = Path(__file__).resolve().parents[1]
    monkeypatch.setattr(sys,"path",[str(root/"examples"),*sys.path])
    spec = importlib.util.spec_from_file_location("four_cases_study",root/"examples/mac_four_cases.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    config = yaml.safe_load((root/"configs/experiments/mac_four_cases.yaml").read_text())
    geometry = yaml.safe_load((root/config["geometry_config"]).read_text())
    base = yaml.safe_load((root/geometry["base_config"]).read_text())
    return module,config,geometry,base


@pytest.mark.parametrize("case",["A_whole_demand","B_stream_demand","C_whole_prefetch","D_stream_tile_prefetch"])
@pytest.mark.parametrize("groups",[1,4])
def test_shared_hardware_and_conservation(study: tuple[Any,...],case: str,groups: int) -> None:
    """四case使用同一Engine和供数参数，序列完成、容量及byte守恒。"""
    module,config,geometry,base = study
    model,entry = module.build_models(geometry)[0]
    model = replace(model,moe_layer_indices=(0,1,2))
    traces = __import__("olmoe_mixtral_cases").make_batch_inputs(model,
        {**base,"samples":1,"coverage":.9,"candidate_count":4})
    point = dict(model=model.name,case=case,mac_groups=groups,bytes_per_cycle_per_channel=16,
        seed=base["seed"],ocm_bytes=786432)
    rules = module.common_rules(model,geometry,config)
    assert not rules.atomic_tile_pair and not rules.ocm_slot_bytes and not rules.descriptor_entries
    assert rules.compute_policy == "expert_locked" and rules.dma_payload_cycle_multiplier == 2
    job = (model,base,geometry,config,entry,point,traces[0])
    row = module.run_case(job)
    assert row == module.run_case(job)
    assert row["total_cycles"] > 0 and row["peak_ocm_bytes"] <= point["ocm_bytes"]
    assert row["speculative_bytes"] == row["useful_speculative_bytes"] + row["wasted_speculative_bytes"]
    if case in {"A_whole_demand","B_stream_demand"}:
        assert row["speculative_bytes"] == 0


def test_paired_denominators(study: tuple[Any,...]) -> None:
    """相同PE的A/B/C分母与相同case的16-lane分母不得混淆。"""
    module,_,_,_ = study
    rows = []
    for seed in [1,2,3]:
        for groups in [1,4]:
            for case,cycles in zip(module.CASES,[200,100,160,80]):
                rows.append(dict(model="test",bytes_per_cycle_per_channel=16,mac_groups=groups,
                    seed=seed,case=case,ocm_bytes=1024,expert_bytes=128,samples=1,
                    total_cycles=cycles//groups,peak_ocm_bytes=128,
                    speculative_bytes=0,useful_speculative_bytes=0,wasted_speculative_bytes=0))
    output = module.aggregate(rows)
    d = next(r for r in output if r["case"]==module.CASES[3] and r["mac_groups"]==4)
    assert d["speedup_vs_A"] == 2.5
    assert d["speedup_vs_B"] == 1.25
    assert d["speedup_vs_C"] == 2.0
    assert d["speedup_vs_16_lanes"] == 4.0
