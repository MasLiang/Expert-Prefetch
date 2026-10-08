"""固定模型粒度、只读测量、slack区间与不同最佳粒度的单元验证。"""
from __future__ import annotations
import importlib.util
import json
from pathlib import Path
from typing import Any
import pytest
import yaml


@pytest.fixture
def experiment(monkeypatch: pytest.MonkeyPatch) -> tuple[Any,dict[str,Any],dict[str,Any],dict[str,Any]]:
    """加载驱动和原保存激励。"""
    root=Path(__file__).resolve().parents[1];monkeypatch.syspath_prepend(str(root/"examples"))
    spec=importlib.util.spec_from_file_location("tile_sweep",root/"examples/prefetch_tile_size.py")
    assert spec and spec.loader
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    c=yaml.safe_load((root/"configs/experiments/prefetch_tile_size.yaml").read_text())
    b=yaml.safe_load((root/c["base_config"]).read_text())
    r=json.loads((root/c["historical_directory"]/"inputs.json").read_text())["20261005"][0]
    return m,c,b,r


def test_slack_union_and_deadline(experiment: tuple[Any,...]) -> None:
    """重叠区间合并，窗口裁剪，等长可fit，无窗口不除零。"""
    m,*_=experiment
    gaps=m.idle_windows([(0,12),(10,20),(30,40)],5,35)
    assert gaps==[(20,30)]
    assert m.slack_metrics(gaps,10)["safe_slack_fit_ratio"]==1
    assert m.slack_metrics(gaps,11)["safe_slack_fit_ratio"]==0
    assert m.slack_metrics([],5)["safe_slack_fit_ratio"]==0


def test_fixed_geometry_and_budget(experiment: tuple[Any,...]) -> None:
    """F不随tile改变；32KiB GU预算各粒度精确表示。"""
    m,c,b,_=experiment;m.validate(c,b)
    for tile in c["f_tile_sizes"]:
        assert (256//tile)*(3*128*tile)==98304
        assert c["fixed_gu_budget_bytes"]%(2*128*tile)==0
    with pytest.raises(ValueError):m.validate({**c,"f_tile_sizes":[24]},b)


@pytest.mark.parametrize("tile",[8,64,128])
def test_actual_request_conservation(experiment: tuple[Any,...],tile: int) -> None:
    """baseline完整权重传输与request数，完美预测字节守恒。"""
    m,c,b,r=experiment
    off=m.worker((b,c,tile,.9,0.,"OFF",20261005,r))
    assert off["request_count"]==5*2*(256//tile)
    assert off["transferred_bytes"]==5*3*128*256
    assert 0<=off["dma_setup_fraction"]<=1
    assert 0<=off["payload_utilization"]<=off["memory_occupancy"]<=1
    assert 0<=off["aie_payload_utilization"]<=off["aie_utilization"]<=1
    perfect=m.worker((b,c,tile,.9,1.,"GU32",20261005,r))
    assert perfect["wasted_speculative_bytes"]==0
    assert perfect==m.worker((b,c,tile,.9,1.,"GU32",20261005,r))


def test_original_headroom_equivalence(experiment: tuple[Any,...]) -> None:
    """64宽OFF/LIGHT/FULL与旧headroom完全相同，观测不改变调度。"""
    import prefetch_memory_headroom as original
    m,c,b,r=experiment
    old={"tile_groups":4,"ocm_expert_equivalents":4}
    for policy,old_policy in (("OFF","OFF"),("LIGHT","LIGHT"),("FULL","PREFETCH")):
        a=m.worker((b,c,64,.9,1.,policy,20261005,r))
        z=original.run_sample((b,old,.9,1.,old_policy,20261005,r))
        assert a["total_cycles"]==z["total_cycles"]
        assert a["transferred_bytes"]==z["transferred_bytes"]


def test_optimum_speedup_not_optimum_latency(experiment: tuple[Any,...]) -> None:
    """增量speedup分母随tile变；不能把最高比值当最短延迟。"""
    m,*_=experiment
    rows=[dict(ratio=1.,accuracy=0.,policy="OFF",f_tile_size=t,mean_cycles=c,speedup=1.) for t,c in ((8,100),(64,80))]
    rows += [dict(ratio=1.,accuracy=.9,policy="LIGHT",f_tile_size=t,mean_cycles=c,speedup=s) for t,c,s in ((8,90,100/90),(64,75,80/75))]
    result=m.optima(rows)[0]
    assert result["incremental_optimum_differs"]
    assert not result["overall_optimum_differs"]
