"""Per-token predictor traces and deterministic unique-expert aggregation."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from random import Random
from typing import Literal, Mapping, Protocol

from .batch_routing import BatchRoutingEvent
from .schema import CandidatePrefetch, ModelConfig, ValidationError


@dataclass(frozen=True)
class TokenPrediction:
    """Ranked expert candidates for one token."""

    token_id: int
    candidates: tuple[CandidatePrefetch, ...]


@dataclass(frozen=True)
class BatchPredictorOutput:
    """Per-token predictions and their unique expert aggregation."""

    request_id: int
    token_position: int
    layer_id: int
    prediction_ready_cycle: int
    token_predictions: tuple[TokenPrediction, ...]
    aggregate_mode: Literal["max", "sum"] = "max"

    def aggregate_candidates(self, candidate_count: int | None = None) -> tuple[CandidatePrefetch, ...]:
        """Aggregate scores across tokens and return unique ranked experts."""
        scores: dict[int, float] = {}
        first_rank: dict[int, int] = {}
        for token in self.token_predictions:
            for rank, candidate in enumerate(token.candidates):
                first_rank[candidate.expert_id] = min(first_rank.get(candidate.expert_id, rank), rank)
                if self.aggregate_mode == "max":
                    scores[candidate.expert_id] = max(
                        scores.get(candidate.expert_id, float("-inf")), candidate.score
                    )
                else:
                    scores[candidate.expert_id] = scores.get(candidate.expert_id, 0.0) + candidate.score
        ordered = sorted(scores, key=lambda expert: (-scores[expert], first_rank[expert], expert))
        if candidate_count is not None:
            ordered = ordered[:candidate_count]
        return tuple(CandidatePrefetch(expert, scores[expert]) for expert in ordered)


class BatchPredictor(Protocol):
    """Predictor interface for batch Top-K workloads."""

    def predict(self, event: BatchRoutingEvent) -> BatchPredictorOutput | None:
        """Return per-token candidates without changing exact routes."""
        ...


class PrecomputedBatchPredictor:
    """Lookup precomputed per-token candidate traces."""

    def __init__(self, outputs: Mapping[tuple[int, int, int], BatchPredictorOutput]) -> None:
        self.outputs = dict(outputs)

    def predict(self, event: BatchRoutingEvent) -> BatchPredictorOutput | None:
        """Return the trace row matching request, position, and layer."""
        return self.outputs.get((event.request_id, event.token_position, event.layer_id))

    @classmethod
    def from_jsonl(cls, path: str | Path, model: ModelConfig) -> "PrecomputedBatchPredictor":
        """Load batch predictions from JSONL with one predictions list per row."""
        outputs: dict[tuple[int, int, int], BatchPredictorOutput] = {}
        try:
            lines = Path(path).read_text(encoding="utf-8").splitlines()
        except OSError as exc:
            raise ValidationError(f"cannot read batch predictor trace: {exc}") from exc
        for line_number, line in enumerate(lines, 1):
            if not line.strip():
                continue
            try:
                raw = json.loads(line)
                token_predictions = tuple(TokenPrediction(
                    int(token["token_id"]),
                    tuple(CandidatePrefetch(int(candidate["expert_id"]), float(candidate["score"]))
                          for candidate in token["candidates"]),
                ) for token in raw["predictions"])
                output = BatchPredictorOutput(
                    int(raw["request_id"]), int(raw["token_position"]), int(raw["layer_id"]),
                    int(raw["prediction_ready_cycle"]), token_predictions,
                    raw.get("aggregate_mode", "max"),
                )
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                raise ValidationError(f"invalid batch predictor JSONL line {line_number}: {exc}") from exc
            for token in output.token_predictions:
                ids = [candidate.expert_id for candidate in token.candidates]
                if len(ids) != len(set(ids)) or any(not 0 <= expert < model.num_experts for expert in ids):
                    raise ValidationError(f"invalid candidates at batch predictor line {line_number}")
            key = (output.request_id, output.token_position, output.layer_id)
            if key in outputs:
                raise ValidationError(f"duplicate batch predictor key {key}")
            outputs[key] = output
        return cls(outputs)


class SyntheticBatchPredictor:
    """Oracle-informed deterministic batch predictor for sensitivity analysis."""

    def __init__(
        self, model: ModelConfig, *, candidates_per_token: int,
        true_recall: float, prediction_ready_cycle: int = 0,
        aggregate_mode: Literal["max", "sum"] = "max", seed: int = 0,
    ) -> None:
        if not 1 <= candidates_per_token <= model.num_experts:
            raise ValidationError("candidates_per_token must fit num_experts")
        if not 0.0 <= true_recall <= 1.0 or prediction_ready_cycle < 0:
            raise ValidationError("invalid synthetic batch predictor probability/time")
        self.model = model
        self.candidates_per_token = candidates_per_token
        self.true_recall = true_recall
        self.prediction_ready_cycle = prediction_ready_cycle
        self.aggregate_mode = aggregate_mode
        self.seed = seed

    def predict(self, event: BatchRoutingEvent) -> BatchPredictorOutput:
        """Sample token candidates from truth plus uniform false experts."""
        rng = Random(self.seed + event.request_id * 1_000_003 + event.layer_id)
        predictions = []
        for token_id, route in enumerate(event.routes):
            selected = [expert for expert in route if rng.random() < self.true_recall]
            if len(selected) > self.candidates_per_token:
                selected = selected[:self.candidates_per_token]
            false_pool = [expert for expert in range(self.model.num_experts) if expert not in route]
            needed = self.candidates_per_token - len(selected)
            if needed > len(false_pool):
                raise ValidationError("not enough false experts for synthetic token prediction")
            ids = selected + rng.sample(false_pool, needed)
            candidates = tuple(CandidatePrefetch(
                expert, (1.0 if expert in route else 0.25) + rng.random() * 0.1
            ) for expert in ids)
            candidates = tuple(sorted(candidates, key=lambda item: (-item.score, item.expert_id)))
            predictions.append(TokenPrediction(token_id, candidates))
        return BatchPredictorOutput(
            event.request_id, event.token_position, event.layer_id,
            self.prediction_ready_cycle, tuple(predictions), self.aggregate_mode,
        )
