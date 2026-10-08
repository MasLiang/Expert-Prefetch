"""Oracle-informed synthetic predictors for cross-layer sensitivity studies."""

from __future__ import annotations

from dataclasses import dataclass, field
from random import Random
from typing import Mapping, Protocol

from .schema import (
    CandidatePrefetch, LayerSequence, LayerTimingRecord, ModelConfig, PredictionEvent,
    RoutingEvent, ValidationError,
)


SUPPORTED_SOURCE_POINTS = {
    "source_layer_start", "source_router_done", "source_resolve",
    "source_expert_start", "source_gate_up_done", "source_expert_done", "custom_offset",
}


@dataclass(frozen=True)
class ScoreModelConfig:
    """Clipped normal score distributions for true and false candidates."""

    true_mean: float = 0.8
    true_std: float = 0.1
    false_mean: float = 0.3
    false_std: float = 0.1
    min_score: float = 0.0
    max_score: float = 1.0

    def __post_init__(self) -> None:
        if self.true_std < 0 or self.false_std < 0:
            raise ValidationError("synthetic score standard deviations must be non-negative")
        if self.min_score > self.max_score:
            raise ValidationError("synthetic score min_score cannot exceed max_score")


@dataclass(frozen=True)
class SyntheticPredictorConfig:
    """Controlled oracle-informed candidate-generation parameters."""

    enabled: bool = True
    mapping: str = "next_moe_layer"
    source_point: str = "source_router_done"
    predictor_latency_cycles: int = 20
    custom_offset_cycles_from_source_start: int | None = None
    generation_mode: str = "hit_distribution"
    candidate_count: int = 4
    hit_distribution: Mapping[int, float] = field(default_factory=lambda: {0: 0.0, 1: 0.0, 2: 1.0})
    target_recall: float | None = None
    true_expert_selection: str = "random"
    false_expert_selection: str = "uniform"
    ranking_mode: str = "noisy_score"
    score_model: ScoreModelConfig = field(default_factory=ScoreModelConfig)
    seed: int = 1234
    gate_up_tiles_per_candidate: int = 1
    down_tiles_per_candidate: int = 0

    def validate(self, model: ModelConfig) -> None:
        """Validate this generator against model dimensions."""
        if self.mapping != "next_moe_layer":
            raise ValidationError("only synthetic predictor mapping=next_moe_layer is supported")
        if self.source_point not in SUPPORTED_SOURCE_POINTS:
            raise ValidationError(f"unsupported synthetic source_point {self.source_point!r}")
        if self.predictor_latency_cycles < 0:
            raise ValidationError("predictor_latency_cycles must be non-negative")
        if self.source_point == "custom_offset":
            if self.custom_offset_cycles_from_source_start is None or self.custom_offset_cycles_from_source_start < 0:
                raise ValidationError("custom_offset requires a non-negative custom offset")
        if not 1 <= self.candidate_count <= model.num_experts:
            raise ValidationError("candidate_count must be in [1, num_experts]")
        if self.generation_mode not in {"hit_distribution", "independent_recall"}:
            raise ValidationError("generation_mode must be hit_distribution or independent_recall")
        if self.true_expert_selection not in {
            "random", "router_order", "highest_gating_weight", "lowest_gating_weight",
        }:
            raise ValidationError("unsupported true_expert_selection")
        if self.false_expert_selection != "uniform":
            raise ValidationError("Milestone 5 supports false_expert_selection=uniform only")
        if self.ranking_mode not in {"true_first", "random", "noisy_score", "perfect_router_order"}:
            raise ValidationError("unsupported ranking_mode")
        if self.gate_up_tiles_per_candidate < 0 or self.down_tiles_per_candidate < 0:
            raise ValidationError("synthetic prefetch depths must be non-negative")
        if self.generation_mode == "hit_distribution":
            if not self.hit_distribution:
                raise ValidationError("hit_distribution must not be empty")
            if any(not isinstance(hit, int) or not 0 <= hit <= model.top_k
                   for hit in self.hit_distribution):
                raise ValidationError("hit_distribution keys must be integers in [0, top_k]")
            if any(probability < 0 for probability in self.hit_distribution.values()):
                raise ValidationError("hit_distribution probabilities must be non-negative")
            if abs(sum(self.hit_distribution.values()) - 1.0) > 1e-9:
                raise ValidationError("hit_distribution probabilities must sum to 1")
            for hit_count, probability in self.hit_distribution.items():
                if probability <= 0:
                    continue
                if hit_count > self.candidate_count:
                    raise ValidationError("candidate_count cannot be smaller than sampled hit_count")
                if self.candidate_count - hit_count > model.num_experts - model.top_k:
                    raise ValidationError("not enough false experts to fill candidate_count")
        else:
            if self.target_recall is None or not 0.0 <= self.target_recall <= 1.0:
                raise ValidationError("independent_recall requires target_recall in [0, 1]")
            guaranteed_hits = model.top_k if self.target_recall == 1.0 else 0
            if self.candidate_count - guaranteed_hits > model.num_experts - model.top_k:
                raise ValidationError(
                    "independent_recall can sample too few hits to fill candidate_count "
                    "from non-true experts"
                )


class SequencePredictor(Protocol):
    """Interface for source-event-triggered future-layer prediction."""

    @property
    def source_point(self) -> str:
        """Return the source timing feature that triggers prediction."""
        ...

    def create_prediction(
        self,
        layer_sequence: LayerSequence,
        model_config: ModelConfig,
        source_event: RoutingEvent,
        target_event: RoutingEvent,
        feature_ready_cycle: int,
        rng: Random,
    ) -> PredictionEvent:
        """Generate one prediction only after its real source feature occurs."""
        ...

    def schedule_predictions(
        self,
        layer_sequence: LayerSequence,
        model_config: ModelConfig,
        source_timing: Mapping[int, LayerTimingRecord],
        rng: Random,
    ) -> list[PredictionEvent]:
        """Build predictions for source records whose trigger timing is known."""
        ...


class SyntheticCrossLayerPredictor:
    """Oracle-informed synthetic candidate generator for sensitivity analysis only."""

    def __init__(self, config: SyntheticPredictorConfig, model: ModelConfig) -> None:
        config.validate(model)
        self.config = config
        self.model = model

    @property
    def source_point(self) -> str:
        """Return the configured source trigger."""
        return self.config.source_point

    def _selected_true(self, target: RoutingEvent, count: int, rng: Random) -> list[int]:
        ids = list(target.true_expert_ids)
        if self.config.true_expert_selection == "random":
            return rng.sample(ids, count)
        if self.config.true_expert_selection == "highest_gating_weight":
            ids = [item[0] for item in sorted(
                zip(ids, target.true_gating_weights), key=lambda item: (-item[1], item[0])
            )]
        elif self.config.true_expert_selection == "lowest_gating_weight":
            ids = [item[0] for item in sorted(
                zip(ids, target.true_gating_weights), key=lambda item: (item[1], item[0])
            )]
        return ids[:count]

    def _sample_hit_count(self, target: RoutingEvent, rng: Random) -> int:
        if self.config.generation_mode == "hit_distribution":
            draw = rng.random()
            cumulative = 0.0
            for hit_count, probability in sorted(self.config.hit_distribution.items()):
                cumulative += probability
                if draw <= cumulative:
                    return hit_count
            return max(self.config.hit_distribution)
        selected = [expert for expert in target.true_expert_ids if rng.random() < float(self.config.target_recall)]
        return min(len(selected), self.config.candidate_count)

    def _candidate_ids(self, target: RoutingEvent, rng: Random) -> tuple[list[int], set[int]]:
        if self.config.generation_mode == "independent_recall":
            chosen = [expert for expert in target.true_expert_ids if rng.random() < float(self.config.target_recall)]
            if len(chosen) > self.config.candidate_count:
                selected_order = self._selected_true(target, len(target.true_expert_ids), rng)
                chosen = [expert for expert in selected_order if expert in chosen][:self.config.candidate_count]
        else:
            chosen = self._selected_true(target, self._sample_hit_count(target, rng), rng)
        true_set = set(chosen)
        false_pool = [expert for expert in range(self.model.num_experts) if expert not in target.true_expert_ids]
        false_count = self.config.candidate_count - len(chosen)
        if false_count > len(false_pool):
            raise ValidationError(
                f"cannot choose {false_count} false candidates from {len(false_pool)} non-true experts"
            )
        return chosen + rng.sample(false_pool, false_count), true_set

    def _rank(self, ids: list[int], selected_true: set[int], target: RoutingEvent, rng: Random) -> tuple[CandidatePrefetch, ...]:
        score_config = self.config.score_model

        def noisy(expert_id: int) -> float:
            mean, std = (
                (score_config.true_mean, score_config.true_std)
                if expert_id in selected_true
                else (score_config.false_mean, score_config.false_std)
            )
            return min(score_config.max_score, max(score_config.min_score, rng.gauss(mean, std)))

        if self.config.ranking_mode == "random":
            ordered = list(ids)
            rng.shuffle(ordered)
            scored = [(expert, float(len(ordered) - rank)) for rank, expert in enumerate(ordered)]
        elif self.config.ranking_mode == "true_first":
            ordered = [expert for expert in ids if expert in selected_true] + [
                expert for expert in ids if expert not in selected_true
            ]
            scored = [(expert, float(len(ordered) - rank)) for rank, expert in enumerate(ordered)]
        elif self.config.ranking_mode == "perfect_router_order":
            ordered = [expert for expert in target.true_expert_ids if expert in selected_true]
            ordered.extend(sorted(expert for expert in ids if expert not in selected_true))
            scored = [(expert, float(len(ordered) - rank)) for rank, expert in enumerate(ordered)]
        else:
            scored = sorted(((expert, noisy(expert)) for expert in ids), key=lambda item: (-item[1], item[0]))
        return tuple(CandidatePrefetch(
            expert, score,
            self.config.gate_up_tiles_per_candidate,
            self.config.down_tiles_per_candidate,
        ) for expert, score in scored)

    def create_prediction(
        self, layer_sequence: LayerSequence, model_config: ModelConfig,
        source_event: RoutingEvent, target_event: RoutingEvent,
        feature_ready_cycle: int, rng: Random,
    ) -> PredictionEvent:
        """Generate a deterministic controlled prediction from target routing truth."""
        ids, selected_true = self._candidate_ids(target_event, rng)
        candidates = self._rank(ids, selected_true, target_event, rng)
        ready = feature_ready_cycle + self.config.predictor_latency_cycles
        prediction_id = (
            f"p{source_event.request_id}_t{source_event.token_position}_"
            f"l{source_event.layer_id}_to_l{target_event.layer_id}"
        )
        return PredictionEvent(
            source_event.request_id, source_event.token_position, source_event.layer_id,
            target_event.token_position, target_event.layer_id,
            "cross_layer_synthetic", self.source_point, feature_ready_cycle,
            self.config.predictor_latency_cycles, ready, candidates, prediction_id,
            self.config.candidate_count, self.config.generation_mode,
            self.config.target_recall,
            len(selected_true),
        )

    def schedule_predictions(
        self, layer_sequence: LayerSequence, model_config: ModelConfig,
        source_timing: Mapping[int, LayerTimingRecord], rng: Random,
    ) -> list[PredictionEvent]:
        """Generate predictions only for source features already present in timing records."""
        by_layer = {event.layer_id: event for event in layer_sequence.routing_events}
        results: list[PredictionEvent] = []
        for source in layer_sequence.routing_events:
            target_id = model_config.next_moe_layer(source.layer_id)
            if target_id is None or target_id not in by_layer or source.layer_id not in source_timing:
                continue
            timing = source_timing[source.layer_id]
            feature = {
                "source_layer_start": timing.layer_start_cycle,
                "source_router_done": timing.router_done_cycle,
                "source_resolve": timing.resolve_cycle,
                "source_expert_start": timing.first_expert_start_cycle,
                "source_gate_up_done": timing.gate_up_done_cycle,
                "source_expert_done": timing.expert_phase_done_cycle,
                "custom_offset": timing.layer_start_cycle + int(self.config.custom_offset_cycles_from_source_start or 0),
            }[self.source_point]
            if feature is not None:
                results.append(self.create_prediction(
                    layer_sequence, model_config, source, by_layer[target_id], feature, rng
                ))
        return results


class CrossLayerOracleSetPredictor(SyntheticCrossLayerPredictor):
    """Oracle true-set predictor at the same source point and latency as synthetic."""

    def create_prediction(
        self, layer_sequence: LayerSequence, model_config: ModelConfig,
        source_event: RoutingEvent, target_event: RoutingEvent,
        feature_ready_cycle: int, rng: Random,
    ) -> PredictionEvent:
        """Return every true target expert in exact router order."""
        candidates = tuple(CandidatePrefetch(
            expert_id, float(len(target_event.true_expert_ids) - rank),
            self.config.gate_up_tiles_per_candidate,
            self.config.down_tiles_per_candidate,
        ) for rank, expert_id in enumerate(target_event.true_expert_ids))
        ready = feature_ready_cycle + self.config.predictor_latency_cycles
        prediction_id = (
            f"oracle_p{source_event.request_id}_t{source_event.token_position}_"
            f"l{source_event.layer_id}_to_l{target_event.layer_id}"
        )
        return PredictionEvent(
            source_event.request_id, source_event.token_position, source_event.layer_id,
            target_event.token_position, target_event.layer_id, "cross_layer_oracle_set",
            self.source_point, feature_ready_cycle, self.config.predictor_latency_cycles,
            ready, candidates, prediction_id, len(candidates), "oracle_set", 1.0,
            len(candidates),
        )
