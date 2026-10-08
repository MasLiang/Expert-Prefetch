"""Strict YAML configuration loading."""

from __future__ import annotations

from pathlib import Path
from typing import Any, TypeVar

import yaml

from .schema import (
    AIEConfig, DMAConfig, HardwareConfig, ModelConfig, NonlinearConfig,
    PLControlConfig, PrefetchConfig, SimulationConfig, TileConfig, ValidationError,
    InterLayerConfig, InterLayerSegment,
)
from .sequence_predictor import ScoreModelConfig, SyntheticPredictorConfig
from .hardware_profile import HardwareProfile

T = TypeVar("T")


def _mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValidationError(f"{name} must be a YAML mapping")
    return value


def _construct(cls: type[T], values: dict[str, Any], name: str) -> T:
    try:
        return cls(**values)
    except TypeError as exc:
        raise ValidationError(f"invalid {name} fields: {exc}") from exc


def _load_yaml(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    try:
        data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ValidationError(f"cannot read configuration {config_path}: {exc}") from exc
    except yaml.YAMLError as exc:
        raise ValidationError(f"invalid YAML in {config_path}: {exc}") from exc
    return _mapping(data, str(config_path))


def load_model_config(path: str | Path) -> ModelConfig:
    """Load and validate a ``model:`` YAML document."""
    root = _load_yaml(path)
    if set(root) != {"model"}:
        raise ValidationError("model YAML must contain exactly the top-level key 'model'")
    return _construct(ModelConfig, _mapping(root["model"], "model"), "model")


def load_hardware_config(path: str | Path) -> HardwareConfig:
    """Load and validate a ``hardware:`` YAML document."""
    root = _load_yaml(path)
    if set(root) != {"hardware"}:
        raise ValidationError("hardware YAML must contain exactly the top-level key 'hardware'")
    values = dict(_mapping(root["hardware"], "hardware"))
    for key, cls in (
        ("aie", AIEConfig), ("dma", DMAConfig),
        ("nonlinear", NonlinearConfig), ("pl_control", PLControlConfig),
    ):
        if key not in values:
            raise ValidationError(f"hardware.{key} is required")
        values[key] = _construct(cls, _mapping(values[key], f"hardware.{key}"), f"hardware.{key}")
    return _construct(HardwareConfig, values, "hardware")


def load_prefetch_config(path: str | Path) -> PrefetchConfig:
    """Load and validate a standalone ``prefetch:`` YAML document."""
    root = _load_yaml(path)
    if set(root) != {"prefetch"}:
        raise ValidationError("prefetch YAML must contain exactly the top-level key 'prefetch'")
    return _construct(PrefetchConfig, _mapping(root["prefetch"], "prefetch"), "prefetch")


def load_simulation_config(path: str | Path) -> SimulationConfig:
    """Load and validate a standalone ``simulation:`` YAML document."""
    root = _load_yaml(path)
    if set(root) != {"simulation"}:
        raise ValidationError("simulation YAML must contain exactly the top-level key 'simulation'")
    return _construct(
        SimulationConfig, _mapping(root["simulation"], "simulation"), "simulation"
    )


def load_tile_config(path: str | Path) -> TileConfig:
    """Load and validate a standalone ``tile:`` YAML document."""
    root = _load_yaml(path)
    if set(root) != {"tile"}:
        raise ValidationError("tile YAML must contain exactly the top-level key 'tile'")
    return _construct(TileConfig, _mapping(root["tile"], "tile"), "tile")


def load_inter_layer_config(path: str | Path) -> InterLayerConfig:
    """Load default and edge-specific inter-layer synthetic work."""
    root = _load_yaml(path)
    if set(root) != {"inter_layer"}:
        raise ValidationError("inter-layer YAML must contain exactly 'inter_layer'")
    values = _mapping(root["inter_layer"], "inter_layer")
    if set(values) - {"default", "overrides"}:
        raise ValidationError(
            f"unknown inter_layer fields: {sorted(set(values) - {'default', 'overrides'})}"
        )
    default = _mapping(values.get("default", {}), "inter_layer.default")
    allowed = {"aie_cycles", "nonlinear_cycles", "controller_cycles", "background_dma_bytes"}
    unknown = set(default) - allowed
    if unknown:
        raise ValidationError(f"unknown inter_layer.default fields: {sorted(unknown)}")
    overrides: dict[str, InterLayerSegment] = {}
    for key, raw in _mapping(values.get("overrides", {}), "inter_layer.overrides").items():
        try:
            source_text, target_text = key.split("->", 1)
            source_id, target_id = int(source_text), int(target_text)
        except (AttributeError, ValueError) as exc:
            raise ValidationError(f"invalid inter-layer override key {key!r}") from exc
        segment_values = dict(_mapping(raw, f"inter_layer.overrides.{key}"))
        overrides[key] = _construct(
            InterLayerSegment,
            {"source_layer_id": source_id, "target_layer_id": target_id, **segment_values},
            f"inter_layer.overrides.{key}",
        )
    return InterLayerConfig(
        default_aie_cycles=default.get("aie_cycles", 0),
        default_nonlinear_cycles=default.get("nonlinear_cycles", 0),
        default_controller_cycles=default.get("controller_cycles", 0),
        default_background_dma_bytes=default.get("background_dma_bytes", 0),
        overrides=overrides,
    )


def load_synthetic_predictor_config(path: str | Path, model: ModelConfig) -> SyntheticPredictorConfig:
    """Load and validate an oracle-informed synthetic predictor configuration."""
    root = _load_yaml(path)
    if set(root) != {"synthetic_predictor"}:
        raise ValidationError("synthetic predictor YAML must contain exactly 'synthetic_predictor'")
    values = dict(_mapping(root["synthetic_predictor"], "synthetic_predictor"))
    score = values.pop("score_model", {})
    prefetch = _mapping(values.pop("prefetch", {}), "synthetic_predictor.prefetch")
    allowed_prefetch = {"policy", "gate_up_tiles_per_candidate", "down_tiles_per_candidate"}
    if set(prefetch) - allowed_prefetch:
        raise ValidationError(
            f"unknown synthetic_predictor.prefetch fields: {sorted(set(prefetch) - allowed_prefetch)}"
        )
    if prefetch.get("policy", "fixed_depth") != "fixed_depth":
        raise ValidationError("synthetic predictor config currently supports prefetch.policy=fixed_depth")
    distribution = values.get("hit_distribution")
    if distribution is not None:
        values["hit_distribution"] = {
            int(key): float(value)
            for key, value in _mapping(distribution, "synthetic_predictor.hit_distribution").items()
        }
    config = _construct(
        SyntheticPredictorConfig,
        {
            **values,
            "score_model": _construct(
                ScoreModelConfig, _mapping(score, "synthetic_predictor.score_model"),
                "synthetic_predictor.score_model",
            ),
            "gate_up_tiles_per_candidate": prefetch.get("gate_up_tiles_per_candidate", 1),
            "down_tiles_per_candidate": prefetch.get("down_tiles_per_candidate", 0),
        },
        "synthetic_predictor",
    )
    config.validate(model)
    return config


def load_hardware_profile(path: str | Path) -> HardwareProfile:
    """Load a Milestone 6 ``hardware_profile:`` YAML document."""
    root = _load_yaml(path)
    if set(root) != {"hardware_profile"}:
        raise ValidationError("profile YAML must contain exactly 'hardware_profile'")
    return _construct(
        HardwareProfile, _mapping(root["hardware_profile"], "hardware_profile"),
        "hardware_profile",
    )
