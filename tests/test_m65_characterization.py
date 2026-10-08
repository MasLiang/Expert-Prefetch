"""Deterministic characterization inputs and board/configured output checks."""

from __future__ import annotations

from edge_moe_sim.characterization import controlled_routing


def test_controlled_routing_separates_batch_from_reuse() -> None:
    """Equal batch/Top-K can have intentionally different active-expert reuse."""
    reused = controlled_routing(
        batch_size=8, top_k=2, num_experts=32, pattern="max_reuse"
    )
    scattered = controlled_routing(
        batch_size=8, top_k=2, num_experts=32, pattern="low_reuse"
    )
    assert reused.batch_size == scattered.batch_size == 8
    assert reused.top_k == scattered.top_k == 2
    assert len(reused.expert_token_counts()) == 2
    assert reused.expert_token_counts() == {0: 8, 1: 8}
    assert len(scattered.expert_token_counts()) == 16


def test_controlled_random_routing_is_seeded() -> None:
    """Random study rows are reproducible from their stored seed."""
    first = controlled_routing(
        batch_size=12, top_k=3, num_experts=32, pattern="random_uniform", seed=7
    )
    second = controlled_routing(
        batch_size=12, top_k=3, num_experts=32, pattern="random_uniform", seed=7
    )
    assert first.routes == second.routes
