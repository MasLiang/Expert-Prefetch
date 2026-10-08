"""Aggregate statistics for synthetic cross-layer sequence datasets."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Sequence

from .prefetch_policy import PrefetchPolicy
from .schema import (
    HardwareConfig, InterLayerConfig, LayerSequence, ModelConfig, SimulationConfig, TileConfig,
)
from .sequence_predictor import SyntheticCrossLayerPredictor
from .sequence_simulator import SequenceScenario, SequenceSimulator, SequenceSimulationOutput


@dataclass(frozen=True)
class SyntheticDatasetSummary:
    """Observed quality and timing statistics across predictions."""

    num_sequences: int
    num_predictions: int
    mean_static_recall: float
    full_hit_rate: float
    any_hit_rate: float
    hit_count_histogram: dict[int, int]
    mean_candidate_count: float
    mean_prediction_lead_to_resolve: float
    mean_sequence_cycles: float
    mean_target_layer_cycles: float
    mean_target_memory_stall_cycles: float

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable representation."""
        return asdict(self)


def aggregate_sequence_outputs(
    outputs: Sequence[SequenceSimulationOutput],
) -> SyntheticDatasetSummary:
    """Aggregate observed metrics without assuming configured probabilities are exact."""
    predictions = [metric for output in outputs for metric in output.result.prediction_metrics]
    count = len(predictions)
    histogram: dict[int, int] = {}
    for metric in predictions:
        histogram[metric.sampled_hit_count] = histogram.get(metric.sampled_hit_count, 0) + 1

    def mean(values: list[float]) -> float:
        return sum(values) / len(values) if values else 0.0

    return SyntheticDatasetSummary(
        len(outputs), count,
        mean([metric.static_recall for metric in predictions]),
        mean([float(metric.full_hit) for metric in predictions]),
        mean([float(metric.any_hit) for metric in predictions]),
        dict(sorted(histogram.items())),
        mean([float(metric.configured_candidate_count) for metric in predictions]),
        mean([float(metric.prediction_lead_to_resolve) for metric in predictions]),
        mean([float(output.result.total_cycles) for output in outputs]),
        mean([float(output.result.target_layer_cycles) for output in outputs]),
        mean([float(output.result.target_layer_memory_stall_cycles) for output in outputs]),
    )


def simulate_synthetic_dataset(
    sequences: Sequence[LayerSequence],
    model_config: ModelConfig,
    hardware_config: HardwareConfig,
    simulation_config: SimulationConfig,
    inter_layer_config: InterLayerConfig,
    predictor: SyntheticCrossLayerPredictor,
    prefetch_policy: PrefetchPolicy,
    *,
    tile_config: TileConfig | None = None,
    seed: int | None = None,
) -> tuple[list[SequenceSimulationOutput], SyntheticDatasetSummary]:
    """Simulate a dataset with deterministic per-sequence RNG streams."""
    base_seed = predictor.config.seed if seed is None else seed
    simulator = SequenceSimulator(model_config, hardware_config)
    outputs = [
        simulator.simulate_layer_sequence(
            sequence, SequenceScenario.SYNTHETIC, simulation=simulation_config,
            tile_config=tile_config, inter_layer_config=inter_layer_config,
            predictor=predictor, prefetch_policy=prefetch_policy,
            seed=base_seed + index,
        )
        for index, sequence in enumerate(sequences)
    ]
    return outputs, aggregate_sequence_outputs(outputs)
