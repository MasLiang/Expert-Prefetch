"""Top-K 合成覆盖率与真实模型 case 选择的回归测试。"""
import importlib.util
import sys
from pathlib import Path
from typing import Any

import pytest
import yaml

from edge_moe_sim.schema import ModelConfig
from edge_moe_sim.real_model_profile import load_real_moe_profiles

ROOT = Path(__file__).resolve().parents[1]


def driver(monkeypatch: pytest.MonkeyPatch) -> Any:
    """加载实验模块而不运行矩阵。"""
    monkeypatch.setattr(sys, "path", [str(ROOT / "examples"), *sys.path])
    spec = importlib.util.spec_from_file_location("model_cases", ROOT / "examples/olmoe_mixtral_cases.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("index,hits,total", [(0, 1688, 2000), (1, 422, 500), (2, 422, 500)])
def test_topk_recall_and_randomized_ranking(monkeypatch: pytest.MonkeyPatch, index: int, hits: int, total: int) -> None:
    """覆盖率基于真实 Expert 数，不将 any-hit 当 recall，命中不强制排在最前。"""
    module = driver(monkeypatch)
    from dataclasses import replace
    profile = load_real_moe_profiles(ROOT / "configs/models/real_moe_catalogue.yaml")[index]
    model = replace(profile.model, moe_layer_indices=tuple(range(6)))
    settings = {"samples": 50, "seed": 20261005, "coverage": 0.844, "candidate_count": 2 * model.top_k}
    records = module.make_batch_inputs(model, settings)
    assert records == module.make_batch_inputs(model, settings)
    ranks = [rank for r in records for p in r["predictions"] for rank in p["true_ranks"]]
    assert len(ranks) == total and sum(rank >= 0 for rank in ranks) == hits
    assert any(rank >= model.top_k for rank in ranks)
    for record in records:
        for truth, prediction in zip(record["truth"][1:], record["predictions"]):
            ids = prediction["candidate_ids"]
            assert len(set(ids)) == settings["candidate_count"]
            assert len(set(truth) & set(ids)) == sum(rank >= 0 for rank in prediction["true_ranks"])


@pytest.mark.parametrize("policy", ["no_prediction", "full_expert_prefetch", "uniform_prefix", "ranked_prefix",
                                    "demand_full_expert", "demand_tile_serial", "nonpipeline_full_expert_prefetch"])
def test_multiexpert_policy_small_capacity(monkeypatch: pytest.MonkeyPatch, policy: str) -> None:
    """容量只够一个 Expert 时，多真实 Expert 仍可顺序完成，不丢失被驱逐的命中。"""
    module = driver(monkeypatch)
    settings = yaml.safe_load((ROOT / "configs/experiments/switch_prefetch.yaml").read_text())
    settings.update(samples=2, candidate_count=4, f_tile_size=2)
    model = ModelConfig("tiny", 2, 8, 8, 2, 8, 8, moe_layer_indices=(0, 1, 2))
    record = module.make_batch_inputs(model, settings)[0]
    row = module.run_sample(model, settings, record, 64, 32, policy)
    assert row == module.run_sample(model, settings, record, 64, 32, policy)
    assert row["peak_ocm_bytes"] <= 64
    if policy.startswith("demand_") or policy == "no_prediction":
        assert row["demand_bytes"] == 3 * 2 * 48


def test_worker_checkpoint_reexecution(monkeypatch: pytest.MonkeyPatch) -> None:
    """并行 worker 必须产生与单进程相同的记录，旧样本不一致时明确报错。"""
    module = driver(monkeypatch)
    settings = yaml.safe_load((ROOT / "configs/experiments/switch_prefetch.yaml").read_text())
    settings.update(samples=2, candidate_count=4, f_tile_size=2)
    model = ModelConfig("tiny", 2, 8, 8, 2, 8, 8, moe_layer_indices=(0, 1, 2))
    record = module.make_batch_inputs(model, settings)[0]
    args = (model, settings, record, 64, 32, "uniform_prefix")
    row, timing = module.run_worker((*args, None))
    assert (row, timing) == module.run_worker((*args, row))
    assert len(timing) == 2
    with pytest.raises(AssertionError, match="checkpoint"):
        module.run_worker((*args, {**row, "total_cycles": row["total_cycles"] + 1}))
