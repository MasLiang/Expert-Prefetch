"""Validated realistic-scale MoE characterization workload profiles."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import yaml

from .schema import ModelConfig, ValidationError
from .work_model import expert_weight_bytes


@dataclass(frozen=True)
class WorkloadProfile:
    """One model/tile/prediction profile for single-token characterization."""

    name: str
    description: str
    provenance: str
    model: ModelConfig
    num_moe_layers: int
    f_tile_size: int
    candidate_count: int
    true_ranks_by_quality: Mapping[str, tuple[int, ...]]

    def __post_init__(self) -> None:
        if self.num_moe_layers <= 0 or self.f_tile_size <= 0:
            raise ValidationError("workload layers and F tile size must be positive")
        if self.model.intermediate_size % self.f_tile_size:
            raise ValidationError("M9.1 workloads require exact F tile partitioning")
        if not self.model.top_k <= self.candidate_count <= self.model.num_experts:
            raise ValidationError("candidate count must fit Top-K and Expert count")
        if set(self.true_ranks_by_quality) != {"perfect", "high", "medium", "low"}:
            raise ValidationError("workload must define perfect/high/medium/low ranks")
        for quality, ranks in self.true_ranks_by_quality.items():
            if len(ranks) != self.model.top_k or len(set(ranks)) != len(ranks):
                raise ValidationError(f"{quality} ranks must contain Top-K unique entries")
            if any(not 1 <= rank <= self.candidate_count for rank in ranks):
                raise ValidationError(f"{quality} rank exceeds candidate count")

    @property
    def tiles_per_expert(self) -> int:
        """Return exact number of F-axis logical tiles per Expert."""
        return self.model.intermediate_size // self.f_tile_size

    @property
    def gate_up_bytes_per_tile(self) -> int:
        """Return INT-weight Gate-Up bytes in one logical F tile."""
        bits = self.model.input_projection_count * self.model.hidden_size * self.f_tile_size * self.model.weight_bits
        return (bits + 7) // 8

    @property
    def down_bytes_per_tile(self) -> int:
        """Return INT-weight Down bytes in one logical F tile."""
        bits = self.model.hidden_size * self.f_tile_size * self.model.weight_bits
        return (bits + 7) // 8

    @property
    def bytes_per_logical_tile(self) -> int:
        """Return Gate-Up plus Down bytes per logical tile."""
        return self.gate_up_bytes_per_tile + self.down_bytes_per_tile

    @property
    def gate_up_ops_per_tile(self) -> int:
        """Return single-token Gate-Up operations per logical tile."""
        return 2 * self.model.input_projection_count * self.model.hidden_size * self.f_tile_size

    @property
    def down_ops_per_tile(self) -> int:
        """Return single-token Down operations per logical tile."""
        return 2 * self.model.hidden_size * self.f_tile_size

    @property
    def expert_bytes(self) -> int:
        """Return exact full-Expert weight bytes."""
        return expert_weight_bytes(self.model)


def load_workload_profile(path: str | Path) -> WorkloadProfile:
    """Load one strict M9.1 workload YAML profile."""
    source = Path(path)
    try:
        raw = yaml.safe_load(source.read_text(encoding="utf-8"))
        workload = raw["workload"]
        model_raw = workload["model"]
        tile = workload["tile"]
        routing = workload["routing"]
        model = ModelConfig(
            str(workload["name"]), int(model_raw["hidden_size"]),
            int(model_raw["intermediate_size"]), int(model_raw["num_experts"]),
            int(model_raw["top_k"]), int(model_raw["weight_bits"]),
            int(model_raw["activation_bits"]),
            expert_architecture=model_raw.get("expert_architecture", "swiglu"),
        )
        ranks = {
            str(quality): tuple(int(rank) for rank in values)
            for quality, values in routing["true_ranks_by_quality"].items()
        }
        return WorkloadProfile(
            str(workload["name"]), str(workload["description"]),
            str(workload["provenance"]), model, int(model_raw["num_moe_layers"]),
            int(tile["f_tile_size"]), int(routing["candidate_count"]), ranks,
        )
    except (OSError, KeyError, TypeError, ValueError, yaml.YAMLError) as exc:
        raise ValidationError(f"invalid workload profile {source}: {exc}") from exc
