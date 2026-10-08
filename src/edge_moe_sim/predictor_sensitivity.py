"""Deterministic predictor-rank and calibration traces for M9.3 evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from .batch_predictor import (
    BatchPredictorOutput,
    PrecomputedBatchPredictor,
    TokenPrediction,
)
from .schema import CandidatePrefetch, ValidationError


CalibrationMode = Literal[
    "well_calibrated",
    "overconfident_wrong",
    "underconfident_correct",
    "uniform",
]
CALIBRATION_MODES: tuple[CalibrationMode, ...] = (
    "well_calibrated",
    "overconfident_wrong",
    "underconfident_correct",
    "uniform",
)


@dataclass(frozen=True)
class PredictorSensitivityTrace:
    """One immutable candidate/rank/score trace used by all schedulers."""

    quality: float
    calibration_mode: CalibrationMode
    candidate_ids: tuple[int, ...]
    candidate_scores: tuple[float, ...]
    true_expert_ranks: tuple[int, ...]
    predictor: PrecomputedBatchPredictor


def interpolated_true_ranks(
    candidate_count: int,
    top_k: int,
    quality: float,
    *,
    minimum_quality: float = 0.2,
) -> tuple[int, ...]:
    """Interpolate true ranks monotonically from the head to spread-out ranks."""
    if not 1 <= top_k <= candidate_count:
        raise ValidationError("Top-K must fit the sensitivity candidate count")
    if not minimum_quality <= quality <= 1.0 or minimum_quality >= 1.0:
        raise ValidationError("quality must lie in [minimum_quality, 1]")
    progress = (1.0 - quality) / (1.0 - minimum_quality)
    ranks = []
    previous = 0
    for index in range(top_k):
        best_rank = index + 1
        worst_rank = max(best_rank, round((index + 1) * candidate_count / top_k))
        rank = round(best_rank + progress * (worst_rank - best_rank))
        rank = max(previous + 1, min(candidate_count - top_k + index + 1, rank))
        ranks.append(rank)
        previous = rank
    return tuple(ranks)


def calibration_scores(
    candidate_count: int,
    mode: CalibrationMode,
) -> tuple[float, ...]:
    """Return a rank-preserving score curve for one calibration mode."""
    if candidate_count <= 0 or mode not in CALIBRATION_MODES:
        raise ValidationError("invalid sensitivity calibration request")
    if mode == "uniform":
        return (0.5,) * candidate_count
    endpoints = {
        "well_calibrated": (1.0, 0.20),
        "overconfident_wrong": (0.99, 0.85),
        "underconfident_correct": (0.35, 0.15),
    }
    high, low = endpoints[mode]
    denominator = max(1, candidate_count - 1)
    return tuple(
        high - (high - low) * rank / denominator
        for rank in range(candidate_count)
    )


def build_sensitivity_trace(
    *,
    request_id: int,
    token_position: int,
    layer_id: int,
    num_experts: int,
    top_k: int,
    candidate_count: int,
    quality: float,
    calibration_mode: CalibrationMode,
) -> PredictorSensitivityTrace:
    """Build a deterministic trace while holding candidate membership by quality."""
    if candidate_count > num_experts:
        raise ValidationError("candidate count exceeds number of Experts")
    ranks = interpolated_true_ranks(candidate_count, top_k, quality)
    ordered: list[int | None] = [None] * candidate_count
    true_ids = tuple(range(top_k))
    for expert_id, rank in zip(true_ids, ranks):
        ordered[rank - 1] = expert_id
    false_ids = iter(range(top_k, num_experts))
    for index, expert_id in enumerate(ordered):
        if expert_id is None:
            try:
                ordered[index] = next(false_ids)
            except StopIteration as exc:
                raise ValidationError("insufficient false Experts for trace") from exc
    ids = tuple(int(expert_id) for expert_id in ordered)
    scores = calibration_scores(candidate_count, calibration_mode)
    candidates = tuple(
        CandidatePrefetch(expert_id, score)
        for expert_id, score in zip(ids, scores)
    )
    output = BatchPredictorOutput(
        request_id,
        token_position,
        layer_id,
        0,
        (TokenPrediction(0, candidates),),
    )
    predictor = PrecomputedBatchPredictor(
        {(request_id, token_position, layer_id): output}
    )
    return PredictorSensitivityTrace(
        quality, calibration_mode, ids, scores, ranks, predictor
    )
