"""Synthetic generation, late output, and cross-layer oracle tests."""

from __future__ import annotations

from random import Random

from edge_moe_sim.prefetch_policy import FixedDepthPolicy
from edge_moe_sim.schema import (
    AIEConfig, DMAConfig, HardwareConfig, InterLayerConfig, LayerSequence, ModelConfig,
    NonlinearConfig, PLControlConfig, RoutingEvent, SimulationConfig, TileConfig,
)
from edge_moe_sim.sequence_oracle import optimize_cross_layer_oracle_depth
from edge_moe_sim.sequence_predictor import (
    CrossLayerOracleSetPredictor, SyntheticCrossLayerPredictor, SyntheticPredictorConfig,
)
from edge_moe_sim.sequence_simulator import SequenceSimulator


def _inputs():
    model = ModelConfig("stats", 4, 8, 8, 2, 8, 8, moe_layer_indices=(3, 7))
    hardware = HardwareConfig(
        1_000_000_000, AIEConfig(16, 16, 16, 2, 1), DMAConfig(8, 1),
        NonlinearConfig(8, 1), PLControlConfig(1, 1, 0, 0, 2),
    )
    source = RoutingEvent(5, 2, 3, "decode", (0, 1), (0.6, 0.4), {0: 1, 1: 1})
    target = RoutingEvent(5, 2, 7, "decode", (2, 3), (0.7, 0.3), {2: 1, 3: 1})
    return model, hardware, source, target, LayerSequence(5, 2, (source, target))


def test_hit_distribution_determinism() -> None:
    model, _, source, target, sequence = _inputs()
    predictor = SyntheticCrossLayerPredictor(SyntheticPredictorConfig(
        candidate_count=4, hit_distribution={0: 0.2, 1: 0.4, 2: 0.4},
        ranking_mode="noisy_score",
    ), model)
    first = predictor.create_prediction(sequence, model, source, target, 10, Random(77))
    second = predictor.create_prediction(sequence, model, source, target, 10, Random(77))
    assert first == second


def test_independent_recall_statistics() -> None:
    model, _, source, target, sequence = _inputs()
    predictor = SyntheticCrossLayerPredictor(SyntheticPredictorConfig(
        generation_mode="independent_recall", target_recall=0.75,
        candidate_count=4,
    ), model)
    recalls = []
    true = set(target.true_expert_ids)
    for seed in range(5000):
        prediction = predictor.create_prediction(sequence, model, source, target, 0, Random(seed))
        recalls.append(len(true & {candidate.expert_id for candidate in prediction.candidates}) / 2)
    assert abs(sum(recalls) / len(recalls) - 0.75) < 0.02


def test_prediction_too_late_degrades_to_exact() -> None:
    model, hardware, _, _, sequence = _inputs()
    simulator = SequenceSimulator(model, hardware)
    exact = simulator.simulate_layer_sequence(
        sequence, "sequence_exact", tile_config=TileConfig(32)
    )
    config = SyntheticPredictorConfig(
        candidate_count=4, hit_distribution={2: 1.0}, predictor_latency_cycles=10_000,
    )
    late = simulator.simulate_layer_sequence(
        sequence, "synthetic_cross_layer", tile_config=TileConfig(32),
        predictor=SyntheticCrossLayerPredictor(config, model),
        prefetch_policy=FixedDepthPolicy(1, 0),
    )
    assert late.result.total_cycles == exact.result.total_cycles
    assert late.result.prediction_metrics[0].prediction_lead_to_resolve < 0
    assert late.result.future_speculative_bytes == 0


def test_cross_layer_oracle_bound() -> None:
    model, hardware, _, _, sequence = _inputs()
    simulator = SequenceSimulator(model, hardware)
    config = SyntheticPredictorConfig(candidate_count=4, hit_distribution={2: 1.0})
    oracle_set = simulator.simulate_layer_sequence(
        sequence, "cross_layer_oracle_set", simulation=SimulationConfig("tile"),
        tile_config=TileConfig(32),
        predictor=CrossLayerOracleSetPredictor(config, model),
        prefetch_policy=FixedDepthPolicy(1, 0),
    )
    optimal = optimize_cross_layer_oracle_depth(
        sequence, model, hardware, SimulationConfig("tile"), TileConfig(32),
        InterLayerConfig(), config, 64,
    ).output
    ideal = simulator.simulate_layer_sequence(
        sequence, "sequence_ideal", simulation=SimulationConfig("tile"),
        tile_config=TileConfig(32),
    )
    assert ideal.result.total_cycles <= optimal.result.total_cycles
    assert optimal.result.total_cycles <= oracle_set.result.total_cycles
