"""Helpers for constructing deterministic synthetic decode traces."""

from __future__ import annotations

from collections.abc import Sequence

from .schema import RoutingEvent


def make_decode_event(
    expert_ids: Sequence[int],
    *,
    request_id: int = 0,
    token_position: int = 0,
    layer_id: int = 0,
    token_counts: Sequence[int] | None = None,
    gating_weights: Sequence[float] | None = None,
) -> RoutingEvent:
    """Build one in-memory synthetic decode routing event."""
    ids = tuple(expert_ids)
    counts = tuple(token_counts) if token_counts is not None else (1,) * len(ids)
    weights = tuple(gating_weights) if gating_weights is not None else tuple(
        1.0 / len(ids) for _ in ids
    )
    if len(counts) != len(ids):
        raise ValueError("token_counts length must match expert_ids")
    return RoutingEvent(
        request_id=request_id,
        token_position=token_position,
        layer_id=layer_id,
        phase="decode",
        true_expert_ids=ids,
        true_gating_weights=weights,
        expert_token_counts=dict(zip(ids, counts)),
    )
