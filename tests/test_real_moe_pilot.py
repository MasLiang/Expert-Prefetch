"""真实模型配置提取、激励一致性及主模型执行的快速回归。"""
from __future__ import annotations

import importlib.util
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
import yaml

from edge_moe_sim.real_model_profile import load_real_moe_profiles
from edge_moe_sim.schema import ModelConfig, ValidationError

ROOT = Path(__file__).resolve().parents[1]


def driver() -> Any:
    """加载实验驱动，不启动 CLI 或下载模型。"""
    spec = importlib.util.spec_from_file_location("real_pilot", ROOT / "examples/real_moe_pilot.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("index,shape,expert_params", [
    (0, (2048, 1024, 64, 8, 16), 6291456),
    (1, (4096, 14336, 8, 2, 32), 176160768),
    (2, (6144, 16384, 8, 2, 56), 301989888),
])
def test_official_dimensions(index: int, shape: tuple[int, ...], expert_params: int) -> None:
    """验证真实中间尺寸而非 dense FFN 尺寸；Expert 数量为每层值。"""
    profile = load_real_moe_profiles(ROOT / "configs/models/real_moe_catalogue.yaml")[index]
    model = profile.model
    assert (model.hidden_size, model.intermediate_size, model.num_experts, model.top_k,
            profile.num_moe_layers) == shape
    assert profile.expert_parameters == expert_params
    assert profile.geometry(128)["expert_weight_bytes"] == expert_params
    assert profile.geometry(128)["tiles_per_expert"] == shape[1] // 128
    assert profile.revision in profile.config_url


def test_quantization_is_an_assumption() -> None:
    """源 BF16 与假设 INT8 区分；16bit 部署 byte 翻倍，不改变参数量。"""
    profile = load_real_moe_profiles(ROOT / "configs/models/real_moe_catalogue.yaml", weight_bits=16)[0]
    assert profile.checkpoint_dtype == "bfloat16"
    assert profile.geometry(128)["expert_weight_bytes"] == 2 * profile.expert_parameters


def test_trace_recall_and_truth_stability() -> None:
    """同一 seed 下，quality 不改变 truth；两种候选 recall 实际为 1 和 0.5。"""
    profile = load_real_moe_profiles(ROOT / "configs/models/real_moe_catalogue.yaml")[0]
    module = driver()
    perfect, _, p = module.synthetic_inputs(profile, 4, 17, 16, "perfect")
    partial, _, q = module.synthetic_inputs(profile, 4, 17, 16, "partial")
    assert perfect == partial
    assert module.synthetic_inputs(profile, 4, 17, 16, "perfect")[2] == p
    for event, prediction in zip(perfect.layers[1:], p["predictions"]):
        assert set(event.routes[0]) <= set(prediction["candidate_ids"])
    for event, prediction in zip(partial.layers[1:], q["predictions"]):
        assert len(set(event.routes[0]) & set(prediction["candidate_ids"])) == 4


def test_main_simulator_smoke_and_determinism() -> None:
    """实际执行主模型，检查峰值容量和确定性；不依赖真实模型推理。"""
    module = driver()
    profile = load_real_moe_profiles(ROOT / "configs/models/real_moe_catalogue.yaml")[1]
    profile = replace(profile, model=ModelConfig("tiny", 16, 32, 8, 2, 8, 8), num_moe_layers=2)
    settings = yaml.safe_load((ROOT / "configs/experiments/real_moe_pilot.yaml").read_text())
    settings["f_tile_size"] = 16
    settings["burst_bytes"] = 256
    base = yaml.safe_load((ROOT / settings["alignment_base"]).read_text())
    kwargs = dict(capacity_bytes=2048, ops_per_cycle=32, quality="partial", policy="uniform_prefix", layers=2)
    first = module.run_case(profile, settings, base, **kwargs)
    assert first == module.run_case(profile, settings, base, **kwargs)
    assert first[0]["peak_ocm_bytes"] <= 2048
    assert len(first[1]) == 2


def test_unsupported_architecture_is_rejected(tmp_path: Path) -> None:
    """不允许将 shared Expert 或未知架构静默当成 Mixtral。"""
    data = yaml.safe_load((ROOT / "configs/models/real_moe_catalogue.yaml").read_text())
    data["models"][0]["upstream_config"]["shared_expert_intermediate_size"] = 2048
    path = tmp_path / "unsupported.yaml"
    path.write_text(yaml.safe_dump(data))
    with pytest.raises(ValidationError, match="shared experts"):
        load_real_moe_profiles(path)


def test_full_target_does_not_claim_complete_expert() -> None:
    """预算被截断时，完整目标不能记作全 Expert 传输完成。"""
    module = driver()
    profile = load_real_moe_profiles(ROOT / "configs/models/real_moe_catalogue.yaml")[1]
    profile = replace(profile, model=ModelConfig("tiny", 16, 32, 8, 2, 8, 8), num_moe_layers=2)
    settings = yaml.safe_load((ROOT / "configs/experiments/real_moe_pilot.yaml").read_text())
    settings["f_tile_size"] = 16
    settings["burst_bytes"] = 256
    base = yaml.safe_load((ROOT / settings["alignment_base"]).read_text())
    row, _ = module.run_case(profile, settings, base, capacity_bytes=2048,
        ops_per_cycle=32, quality="perfect", policy="full_expert_prefetch", layers=2, diagnostics=True)
    assert row["planned_speculative_bytes"] <= row["speculative_payload_budget_per_target_bytes"]
    assert not row["whole_expert_fits_ocm_with_demand_reserve"]
    assert row["experts_all_spec_transfers_completed_by_router"] == 0


def test_byte_capacity_is_not_fixed_slot_capacity() -> None:
    """OLMoE 两 Expert 实际 byte 能放进 16 MiB，但 slot 加 demand reserve 需要 17 MiB。"""
    profile = load_real_moe_profiles(ROOT / "configs/models/real_moe_catalogue.yaml")[0]
    geometry = profile.geometry(128)
    slot_bytes = 2 * profile.model.hidden_size * 128
    two_expert_reserved_bytes = 2 * geometry["tiles_per_expert"] * 2 * slot_bytes
    assert 2 * geometry["expert_weight_bytes"] == 12 * 1024**2
    assert two_expert_reserved_bytes == 16 * 1024**2
    assert two_expert_reserved_bytes + 2 * slot_bytes == 17 * 1024**2
