"""Small-sequence exhaustive OracleOptimalDepth integration."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from random import Random

from .prefetch_policy import PredictorSpecifiedPolicy
from .schema import (
    CandidatePrefetch, ExpertTask, HardwareConfig, InterLayerConfig, LayerSequence, ModelConfig,
    PredictionEvent, RoutingEvent, SimulationConfig, TileConfig, ValidationError,
)
from .sequence_predictor import CrossLayerOracleSetPredictor, SyntheticPredictorConfig
from .sequence_simulator import SequenceScenario, SequenceSimulationOutput, SequenceSimulator
from .tiling import FixedByteTilingPolicy


class _DepthOraclePredictor(CrossLayerOracleSetPredictor):
    def __init__(
        self, config: SyntheticPredictorConfig, model: ModelConfig,
        depths: dict[int, tuple[int, ...]],
    ) -> None:
        super().__init__(config, model)
        self.depths = depths

    def create_prediction(
        self, layer_sequence: LayerSequence, model_config: ModelConfig,
        source_event: RoutingEvent, target_event: RoutingEvent,
        feature_ready_cycle: int, rng: Random,
    ) -> PredictionEvent:
        base = super().create_prediction(
            layer_sequence, model_config, source_event, target_event, feature_ready_cycle, rng
        )
        target_depths = self.depths[target_event.layer_id]
        candidates = tuple(
            CandidatePrefetch(candidate.expert_id, candidate.score, target_depths[index], 0)
            for index, candidate in enumerate(base.candidates)
        )
        return PredictionEvent(
            base.request_id, base.source_token_position, base.source_layer_id,
            base.target_token_position, base.target_layer_id,
            "cross_layer_oracle_optimal", base.feature_type, base.feature_ready_cycle,
            base.predictor_latency_cycles, base.prediction_ready_cycle, candidates,
            base.prediction_id.replace("oracle_", "oracle_optimal_"),
            len(candidates), "oracle_optimal_depth", 1.0, len(candidates),
        )


@dataclass(frozen=True)
class OracleOptimalSequenceResult:
    """Best feasible prefix-depth allocation and its sequence output."""

    output: SequenceSimulationOutput
    gate_up_depths_by_target: dict[int, tuple[int, ...]]
    evaluated_allocations: int


def optimize_cross_layer_oracle_depth(
    layer_sequence: LayerSequence,
    model: ModelConfig,
    hardware: HardwareConfig,
    simulation: SimulationConfig,
    tile_config: TileConfig,
    inter_layer: InterLayerConfig,
    predictor_config: SyntheticPredictorConfig,
    byte_budget: int,
    *,
    seed: int = 0,
    max_allocations: int = 100_000,
) -> OracleOptimalSequenceResult:
    """Enumerate feasible true-expert GU prefixes under one budget per target."""
    if byte_budget < 0:
        raise ValidationError("oracle byte_budget must be non-negative")
    policy = FixedByteTilingPolicy()
    target_options: list[tuple[int, list[tuple[int, ...]]]] = []
    for event in layer_sequence.routing_events[1:]:
        counts: list[int] = []
        sizes: list[list[int]] = []
        for expert_id in event.true_expert_ids:
            # Tile bytes are token-count independent; counts retain true compute geometry.
            task = ExpertTask(
                event.layer_id, expert_id, event.expert_token_counts[expert_id], 0.0
            )
            gate, _ = policy.build_tiles(task, model, tile_config)
            counts.append(len(gate))
            sizes.append([tile.size_bytes for tile in gate])
        feasible: list[tuple[int, ...]] = []
        for depths in product(*(range(count + 1) for count in counts)):
            used = sum(sum(expert_sizes[:depth]) for expert_sizes, depth in zip(sizes, depths))
            if used <= byte_budget:
                feasible.append(tuple(depths))
        target_options.append((event.layer_id, feasible))
    allocation_count = 1
    for _, options in target_options:
        allocation_count *= len(options)
    if allocation_count > max_allocations:
        raise ValidationError(
            f"oracle depth search has {allocation_count} allocations; limit is {max_allocations}"
        )
    simulator = SequenceSimulator(model, hardware)
    best: SequenceSimulationOutput | None = None
    best_depths: dict[int, tuple[int, ...]] = {}
    evaluated = 0
    for selected in product(*(options for _, options in target_options)):
        depths = {
            target_options[index][0]: tuple(value)
            for index, value in enumerate(selected)
        }
        predictor = _DepthOraclePredictor(predictor_config, model, depths)
        output = simulator.simulate_layer_sequence(
            layer_sequence, SequenceScenario.ORACLE_OPTIMAL, simulation=simulation,
            tile_config=tile_config, inter_layer_config=inter_layer,
            predictor=predictor, prefetch_policy=PredictorSpecifiedPolicy(None, None), seed=seed,
        )
        evaluated += 1
        ordering = tuple(depth for target in sorted(depths) for depth in depths[target])
        best_ordering = tuple(
            depth for target in sorted(best_depths) for depth in best_depths[target]
        )
        if best is None or (output.result.total_cycles, ordering) < (
            best.result.total_cycles, best_ordering
        ):
            best, best_depths = output, depths
    if best is None:
        raise RuntimeError("oracle depth search produced no feasible allocation")
    return OracleOptimalSequenceResult(best, best_depths, evaluated)
