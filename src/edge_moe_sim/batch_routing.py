"""Batch Top-K routing representation and weight-reuse statistics."""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil
from statistics import fmean, median
from typing import Mapping

from .schema import ModelConfig, RoutingEvent, ValidationError


@dataclass(frozen=True)
class BatchRoutingEvent:
    """Exact per-token Top-K routes for one MoE layer."""

    request_id: int
    token_position: int
    layer_id: int
    routes: tuple[tuple[int, ...], ...]
    gating_scores: tuple[tuple[float, ...], ...] = ()

    @property
    def batch_size(self) -> int:
        """Return the number of routed tokens."""
        return len(self.routes)

    @property
    def top_k(self) -> int:
        """Return the common number of routes per token."""
        return len(self.routes[0]) if self.routes else 0

    def validate(self, model: ModelConfig, *, expected_top_k: int | None = None) -> None:
        """Validate route shape, IDs, and optional gating scores."""
        if not self.routes:
            raise ValidationError("batch routing event must contain at least one token")
        top_k = expected_top_k if expected_top_k is not None else self.top_k
        if top_k <= 0 or any(len(route) != top_k for route in self.routes):
            raise ValidationError("every token route must have the same positive Top-K")
        for token_id, route in enumerate(self.routes):
            if len(set(route)) != len(route):
                raise ValidationError(f"token {token_id} route contains duplicate experts")
            for expert_id in route:
                if not 0 <= expert_id < model.num_experts:
                    raise ValidationError(f"expert {expert_id} is outside model range")
        if self.gating_scores:
            if len(self.gating_scores) != self.batch_size or any(
                len(scores) != top_k for scores in self.gating_scores
            ):
                raise ValidationError("gating_scores must match routes shape")

    def expert_token_counts(self) -> dict[int, int]:
        """Aggregate token-expert assignments without duplicating weight objects."""
        counts: dict[int, int] = {}
        for route in self.routes:
            for expert_id in route:
                counts[expert_id] = counts.get(expert_id, 0) + 1
        return dict(sorted(counts.items()))

    @classmethod
    def from_legacy(cls, event: RoutingEvent) -> "BatchRoutingEvent":
        """Convert a legacy decode event when token counts describe one common batch.

        Legacy traces do not retain per-token route identity, so this adapter creates
        deterministic synthetic assignment rows matching the aggregate counts.
        """
        top_k = len(event.true_expert_ids)
        remaining = {
            expert_id: event.expert_token_counts[expert_id]
            for expert_id in event.true_expert_ids
        }
        assignment_count = sum(remaining.values())
        if assignment_count % top_k:
            raise ValidationError("legacy expert counts cannot form complete Top-K token routes")
        routes_list = []
        for _ in range(assignment_count // top_k):
            available = sorted(
                (expert for expert, count in remaining.items() if count > 0),
                key=lambda expert: (-remaining[expert], expert),
            )
            if len(available) < top_k:
                raise ValidationError("legacy aggregate counts cannot form duplicate-free Top-K routes")
            route = tuple(available[:top_k])
            routes_list.append(route)
            for expert in route:
                remaining[expert] -= 1
        routes = tuple(routes_list)
        return cls(event.request_id, event.token_position, event.layer_id, routes)


@dataclass(frozen=True)
class BatchWorkloadStats:
    """Reuse and tokens-per-expert distribution for one exact route set."""

    batch_size: int
    router_top_k: int
    token_expert_assignments: int
    unique_active_experts: int
    min_tokens_per_expert: int
    mean_tokens_per_expert: float
    median_tokens_per_expert: float
    p95_tokens_per_expert: float
    max_tokens_per_expert: int
    reuse_factor: float
    tokens_per_expert: Mapping[int, int]


def batch_workload_stats(event: BatchRoutingEvent) -> BatchWorkloadStats:
    """Calculate exact aggregate routing and reuse statistics."""
    counts = event.expert_token_counts()
    values = sorted(counts.values())
    assignments = event.batch_size * event.top_k
    rank = max(0, ceil(0.95 * len(values)) - 1)
    return BatchWorkloadStats(
        event.batch_size, event.top_k, assignments, len(counts), min(values),
        fmean(values), median(values), float(values[rank]), max(values),
        assignments / len(counts), counts,
    )
