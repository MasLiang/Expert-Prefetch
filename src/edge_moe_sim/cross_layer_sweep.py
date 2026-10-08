"""Cross-layer synthetic predictor design-space sweeps."""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass, replace
from itertools import product
from pathlib import Path
from typing import Sequence

from .prefetch_policy import FixedDepthPolicy
from .schema import HardwareConfig, InterLayerConfig, LayerSequence, ModelConfig, SimulationConfig, TileConfig
from .sequence_metrics import aggregate_sequence_outputs
from .sequence_predictor import SyntheticCrossLayerPredictor, SyntheticPredictorConfig
from .sequence_simulator import SequenceScenario, SequenceSimulator


@dataclass(frozen=True)
class CrossLayerSweepRow:
    """One fully specified synthetic cross-layer sweep point."""

    candidate_count: int
    target_recall: float
    observed_recall: float
    full_hit_rate: float
    source_point: str
    predictor_latency_cycles: int
    mean_lead_to_resolve: float
    prefetch_depth: int
    inter_layer_cycles: int
    seed: int
    sequence_cycles: float
    target_layer_cycles: float
    target_memory_stall: float
    speedup_vs_sequence_exact: float
    future_speculative_bytes: int
    useful_future_speculative_bytes: int
    wasted_future_speculative_bytes: int
    current_demand_blocked_cycles: int


def run_cross_layer_sweep(
    sequences: Sequence[LayerSequence], model: ModelConfig, hardware: HardwareConfig,
    simulation: SimulationConfig, tile: TileConfig,
    base_predictor: SyntheticPredictorConfig,
    *, candidate_counts: Sequence[int], recalls: Sequence[float],
    predictor_latencies: Sequence[int], source_points: Sequence[str],
    prefetch_depths: Sequence[int], inter_layer_cycles: Sequence[int],
    seeds: Sequence[int],
) -> list[CrossLayerSweepRow]:
    """Run a deterministic Cartesian grid and return flat CSV-ready rows."""
    simulator = SequenceSimulator(model, hardware)
    exact_means = {}
    for gap in set(inter_layer_cycles):
        exact_outputs = [
            simulator.simulate_layer_sequence(
                sequence, SequenceScenario.EXACT, simulation=simulation,
                tile_config=tile, inter_layer_config=InterLayerConfig(default_aie_cycles=gap),
            ) for sequence in sequences
        ]
        exact_means[gap] = sum(
            output.result.total_cycles for output in exact_outputs
        ) / len(exact_outputs)
    rows: list[CrossLayerSweepRow] = []
    for candidate_count, recall, latency, source, depth, gap, seed in product(
        candidate_counts, recalls, predictor_latencies, source_points,
        prefetch_depths, inter_layer_cycles, seeds,
    ):
        config = replace(
            base_predictor, generation_mode="independent_recall", target_recall=recall,
            candidate_count=candidate_count, predictor_latency_cycles=latency,
            source_point=source, seed=seed, gate_up_tiles_per_candidate=depth,
        )
        predictor = SyntheticCrossLayerPredictor(config, model)
        outputs = [
            simulator.simulate_layer_sequence(
                sequence, SequenceScenario.SYNTHETIC, simulation=simulation,
                tile_config=tile,
                inter_layer_config=InterLayerConfig(default_aie_cycles=gap),
                predictor=predictor, prefetch_policy=FixedDepthPolicy(depth, 0),
                seed=seed + index,
            ) for index, sequence in enumerate(sequences)
        ]
        aggregate = aggregate_sequence_outputs(outputs)
        mean_cycles = aggregate.mean_sequence_cycles
        rows.append(CrossLayerSweepRow(
            candidate_count, recall, aggregate.mean_static_recall,
            aggregate.full_hit_rate, source, latency,
            aggregate.mean_prediction_lead_to_resolve, depth, gap, seed,
            mean_cycles, aggregate.mean_target_layer_cycles,
            aggregate.mean_target_memory_stall_cycles, exact_means[gap] / mean_cycles,
            sum(output.result.future_speculative_bytes for output in outputs),
            sum(output.result.useful_future_speculative_bytes for output in outputs),
            sum(output.result.wasted_future_speculative_bytes for output in outputs),
            sum(output.result.total_demand_blocked_by_future_speculative_cycles for output in outputs),
        ))
    return rows


def write_cross_layer_sweep_csv(rows: Sequence[CrossLayerSweepRow], path: str | Path) -> None:
    """Write sweep points using stable dataclass field order."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fields = list(CrossLayerSweepRow.__dataclass_fields__)
    with output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))
