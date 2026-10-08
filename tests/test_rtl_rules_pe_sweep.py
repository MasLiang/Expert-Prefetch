"""RTL规则研究配置与单Engine MAC扩宽语义验证，不运行RTL。"""
from __future__ import annotations

from dataclasses import replace
import importlib.util
from pathlib import Path
import sys
from typing import Any

import pytest
import yaml


@pytest.fixture
def sweep(monkeypatch: pytest.MonkeyPatch) -> tuple[Any,dict[str,Any]]:
    """加载独立扫描入口和配置。"""
    root = Path(__file__).resolve().parents[1]
    monkeypatch.setattr(sys,"path",[str(root/"examples"),*sys.path])
    spec = importlib.util.spec_from_file_location("rtl_rules_scan",root/"examples/rtl_rules_pe_sweep.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module,yaml.safe_load((root/"configs/experiments/rtl_rules_pe_sweep.yaml").read_text())


def test_architecture_rule_mapping(sweep: tuple[Any,dict[str,Any]]) -> None:
    """按slot预留并保护demand，保持tile ready-first和burst服务。"""
    module,config = sweep
    model,entry = module.build_models(config)[0]
    rules = module.rules_for("rtl_resources",model,entry["f_tile_size"],config)
    assert rules.ocm_slot_bytes == 16384
    assert rules.ocm_banks == 4 and rules.reserved_demand_slots == 2
    assert rules.atomic_tile_pair and rules.compute_policy == "tile_ready_first"
    assert rules.request_order == "pair_fifo" and rules.retry_future_prefetch
    assert rules.burst_arbitration and rules.descriptor_entries == 64 and rules.dma_fifo_depth == 16
    assert rules.layer_retirement_wait_for_ingress and rules.fifo_push_blocks_dequeue
    assert module.rules_for("legacy",model,entry["f_tile_size"],config).ocm_slot_bytes == 0
    with pytest.raises(ValueError,match="unknown rule"):
        module.rules_for("invalid",model,entry["f_tile_size"],config)


@pytest.mark.parametrize("mode",["legacy","tile_scheduler","burst_fifo","rtl_resources"])
@pytest.mark.parametrize("groups",[1,4,16])
def test_single_engine_widening_and_resource_limits(
        sweep: tuple[Any,dict[str,Any]],mode: str,groups: int) -> None:
    """增加MAC宽度不增加Engine/Expert并发，结果确定且资源不溢出。"""
    module,config = sweep
    model,entry = module.build_models(config)[0]
    model = replace(model,moe_layer_indices=tuple(range(3)))
    settings = yaml.safe_load((module.ROOT/config["base_config"]).read_text())
    settings.update(samples=2,coverage=.9,candidate_count=4)
    record = module.make_batch_inputs(model,settings)[0]
    point = dict(model=model.name,mode=mode,mac_groups=groups,bytes_per_cycle_per_channel=16,
        policy="uniform_prefix",seed=settings["seed"],f_tile_size=64,ocm_bytes=786432)
    result = module.simulate((model,settings,config,point,record))
    assert result == module.simulate((model,settings,config,point,record))
    assert result["logical_engines"] == 1 and result["hypothetical_mac_lanes"] == groups*16
    assert result["peak_ocm_bytes"] <= result["ocm_bytes"]
    assert result["speculative_bytes"] == result["useful_speculative_bytes"] + result["wasted_speculative_bytes"]
    assert result["wait_for_previous_weights"] is False
    if mode in {"burst_fifo","rtl_resources"}:
        assert result["descriptor_peak"] <= 64
        assert result["fifo_reserved_credit_peak"] <= 16
        assert result["fifo_occupancy_peak"] <= 16


@pytest.mark.parametrize("mode",["tile_scheduler","burst_fifo","rtl_resources"])
def test_top8_partial_residency_forward_progress(sweep: tuple[Any,dict[str,Any]],mode: str) -> None:
    """Top-8、饱和pair容量和非Router顺序预测必须完成，不锁定整个Expert。"""
    module,config = sweep
    original,entry = module.build_models(config)[1]
    model = replace(original,hidden_size=128,intermediate_size=1024)
    settings = yaml.safe_load((module.ROOT/config["base_config"]).read_text())
    settings.update(samples=1,coverage=.9,candidate_count=16,seed=config["seeds"][0])
    record = module.make_batch_inputs(model,settings)[0]
    slot = module.rules_for(mode,model,128,config).ocm_slot_bytes
    point = dict(model=model.name,mode=mode,mac_groups=1,bytes_per_cycle_per_channel=16,
        policy="uniform_prefix",seed=settings["seed"],f_tile_size=128,ocm_bytes=slot*48)
    result = module.simulate((model,settings,config,point,record))
    assert result["total_cycles"] > 0
    assert result["peak_ocm_bytes"] <= slot*48
