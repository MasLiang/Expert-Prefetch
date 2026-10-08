"""共享双候选覆盖率激励及无预取/完整/tile 策略对照。"""
import importlib.util
from pathlib import Path
from typing import Any

import pytest
import yaml

from edge_moe_sim.config import load_model_config
from edge_moe_sim.schema import ModelConfig

ROOT = Path(__file__).resolve().parents[1]


def driver() -> Any:
    """读取实验脚本而不执行矩阵。"""
    spec = importlib.util.spec_from_file_location("switch_comparison", ROOT / "examples/switch_prefetch_comparison.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_exact_coverage_and_balanced_hit_ranks() -> None:
    """250 个目标中 211 命中，候选排名 106/105；输入可复现。"""
    module = driver()
    settings = yaml.safe_load((ROOT / "configs/experiments/switch_prefetch.yaml").read_text())
    model = load_model_config(ROOT / settings["models"][0])
    records = module.make_inputs(model, settings)
    assert records == module.make_inputs(model, settings)
    ranks = [p["true_rank"] for r in records for p in r["predictions"]]
    assert ranks.count(0) == 106 and ranks.count(1) == 105 and ranks.count(-1) == 39
    for record in records:
        for truth, prediction in zip(record["truth"][1:], record["predictions"]):
            assert len(set(prediction["candidate_ids"])) == 2
            assert (truth in prediction["candidate_ids"]) == (prediction["true_rank"] >= 0)


@pytest.mark.parametrize("policy", ["no_prediction", "full_expert_prefetch", "uniform_prefix", "ranked_prefix"])
def test_main_engine_policy_determinism(policy: str) -> None:
    """相同输入、真实 byte OCM，在各策略重复执行完全一致。"""
    module = driver()
    settings = yaml.safe_load((ROOT / "configs/experiments/switch_prefetch.yaml").read_text())
    settings.update(samples=2, f_tile_size=3)
    model = ModelConfig("tiny", 4, 8, 8, 1, 8, 8, expert_architecture="relu_ffn", moe_layer_indices=(1, 3, 5))
    record = module.make_inputs(model, settings)[0]
    first = module.run_sample(model, settings, record, 2048, 32, policy)
    assert first == module.run_sample(model, settings, record, 2048, 32, policy)
    assert first["peak_ocm_bytes"] <= 2048
    if policy == "no_prediction":
        assert first["speculative_bytes"] == 0
        assert first["demand_bytes"] == 3 * 64
