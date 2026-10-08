"""官方 MoE 结构摘录适配；不下载权重，不推断真实路由或量化精度。"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from .schema import ModelConfig, ValidationError


@dataclass(frozen=True)
class RealMoEProfile:
    """真实模型维度及来源；总参数量为发布近似值，Expert 参数量为精确公式。"""

    model: ModelConfig
    num_moe_layers: int
    repository: str
    revision: str
    published_total_parameters_approx: int
    checkpoint_dtype: str
    upstream_config: dict[str, Any]

    @property
    def config_url(self) -> str:
        """返回固定 revision 的官方配置来源。"""
        return f"https://huggingface.co/{self.repository}/blob/{self.revision}/config.json"

    @property
    def expert_parameters(self) -> int:
        """返回配置架构的单层单个 Expert 权重参数数量（不含 Router）。"""
        return (self.model.input_projection_count + 1) * self.model.hidden_size * self.model.intermediate_size

    def geometry(self, f_tile_size: int) -> dict[str, Any]:
        """返回参数量和部署 byte；量化 scale/zero-point 元数据不计入。"""
        if f_tile_size <= 0:
            raise ValidationError("f_tile_size must be positive")
        count = (self.model.intermediate_size + f_tile_size - 1) // f_tile_size
        return {
            "model": self.model.name, "hidden_size": self.model.hidden_size,
            "expert_architecture": self.model.expert_architecture,
            "nonlinear_operation": self.model.nonlinear_operation,
            "intermediate_size": self.model.intermediate_size,
            "num_experts_per_layer": self.model.num_experts,
            "top_k": self.model.top_k, "num_moe_layers": self.num_moe_layers,
            "published_total_parameters_approx": self.published_total_parameters_approx,
            "expert_parameters": self.expert_parameters,
            "routed_expert_parameters_all_layers": self.expert_parameters * self.model.num_experts * self.num_moe_layers,
            "active_expert_parameters_per_layer": self.expert_parameters * self.model.top_k,
            "expert_weight_bytes": (self.expert_parameters * self.model.weight_bits + 7) // 8,
            "checkpoint_dtype": self.checkpoint_dtype,
            "deployment_weight_bits_assumed": self.model.weight_bits,
            "f_tile_size": f_tile_size, "tiles_per_expert": count,
            "config_url": self.config_url,
        }


def load_real_moe_profiles(path: str | Path, *, weight_bits: int = 8,
                           activation_bits: int = 8) -> tuple[RealMoEProfile, ...]:
    """将 mixtral/olmoe 配置摘录转为主模拟器模型，拒绝未知或 shared-expert 架构。"""
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    profiles: list[RealMoEProfile] = []
    for row in data["models"]:
        config = row["upstream_config"]
        if config["model_type"] not in {"mixtral", "olmoe"} or config.get("hidden_act") != "silu":
            raise ValidationError("unsupported architecture; do not silently omit shared experts")
        if any(config.get(key, 0) for key in ("num_shared_experts", "shared_expert_intermediate_size")):
            raise ValidationError("shared experts are not covered by this adapter")
        layers = config["num_hidden_layers"]
        if not isinstance(layers, int) or isinstance(layers, bool) or layers <= 0:
            raise ValidationError("num_hidden_layers must be a positive integer")
        experts = config["num_local_experts"] if config["model_type"] == "mixtral" else config["num_experts"]
        model = ModelConfig(row["name"], config["hidden_size"], config["intermediate_size"],
                            experts, config["num_experts_per_tok"], weight_bits, activation_bits,
                            moe_layer_indices=tuple(range(layers)))
        profiles.append(RealMoEProfile(model, layers, row["repository"], row["revision"],
                                       row["published_total_parameters_approx"],
                                       config["torch_dtype"], dict(config)))
    if not profiles or len({profile.model.name for profile in profiles}) != len(profiles):
        raise ValidationError("model catalogue must contain unique model names")
    return tuple(profiles)
