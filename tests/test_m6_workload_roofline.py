"""Milestone 6 batch reuse, F tiling, and Roofline micro-tests."""

from __future__ import annotations

from edge_moe_sim.batch_routing import BatchRoutingEvent, batch_workload_stats
from edge_moe_sim.batch_predictor import BatchPredictorOutput, TokenPrediction
from edge_moe_sim.f_tile import build_f_tile_groups
from edge_moe_sim.hardware_profile import HardwareProfile
from edge_moe_sim.roofline_metrics import calculate_roofline_metrics
from edge_moe_sim.schema import ModelConfig
from edge_moe_sim.schema import CandidatePrefetch


def _model() -> ModelConfig:
    return ModelConfig("m6", 4, 8, 8, 2, 8, 8)


def _profile() -> HardwareProfile:
    return HardwareProfile(
        "micro", 1000, 1, 8_000, 1.0, 1, 1, 32, False,
        8, 4, 1.0, 1, 4096,
    )


def test_batch_reuse_compute_doubles_weight_does_not() -> None:
    one = build_f_tile_groups(_model(), 0, 1, 4)
    two = build_f_tile_groups(_model(), 0, 2, 4)
    assert sum(group.gate_up_ops + group.down_ops for group in two) == 2 * sum(
        group.gate_up_ops + group.down_ops for group in one
    )
    assert sum(group.gate_up_bytes + group.down_bytes for group in two) == sum(
        group.gate_up_bytes + group.down_bytes for group in one
    )


def test_topk_route_aggregation_and_unique_weight() -> None:
    event = BatchRoutingEvent(0, 0, 0, ((0, 1), (1, 2)))
    stats = batch_workload_stats(event)
    assert stats.token_expert_assignments == 4
    assert stats.unique_active_experts == 3
    assert stats.tokens_per_expert == {0: 1, 1: 2, 2: 1}
    bytes_by_expert = {
        expert: sum(group.gate_up_bytes + group.down_bytes for group in build_f_tile_groups(
            _model(), expert, count, 4
        )) for expert, count in stats.tokens_per_expert.items()
    }
    assert bytes_by_expert[1] == bytes_by_expert[0]


def test_f_tile_group_dependency_work_is_exact() -> None:
    groups = build_f_tile_groups(_model(), 2, 3, 3)
    assert [group.f_size for group in groups] == [3, 3, 2]
    assert sum(group.gate_up_ops for group in groups) == 4 * 3 * 4 * 8
    assert sum(group.down_ops for group in groups) == 2 * 3 * 8 * 4
    assert sum(group.nonlinear_elements for group in groups) == 3 * 8


def test_roofline_reuse_moves_toward_compute_bound() -> None:
    profile = _profile()
    weight_bytes = 96
    low = calculate_roofline_metrics(6 * 1 * 4 * 8, weight_bytes, 100, profile)
    high = calculate_roofline_metrics(6 * 64 * 4 * 8, weight_bytes, 1000, profile)
    assert high.arithmetic_intensity_ops_per_byte == 64 * low.arithmetic_intensity_ops_per_byte
    assert low.regime == "memory_bound"
    assert high.regime == "compute_bound"


def test_board_and_configured_rooflines_are_distinct() -> None:
    """A selected AIE subset can saturate while the board stays memory-bound."""
    profile = _profile()
    # AI=4: memory ceiling=32k ops/s, board compute=32k (balanced), while
    # one configured group has only 4k ops/s and is compute-bound.
    metrics = calculate_roofline_metrics(384, 96, 100, profile, configured_aie_units=1)
    assert metrics.board_compute_peak_ops_s == 32_000
    assert metrics.configured_compute_peak_ops_s == 4_000
    assert metrics.board_ridge_ai == 4
    assert metrics.configured_ridge_ai == 0.5
    assert metrics.board_regime == "balanced"
    assert metrics.configured_regime == "compute_bound"


def test_batch_predictor_aggregates_unique_expert_max_or_sum() -> None:
    tokens = (
        TokenPrediction(0, (CandidatePrefetch(1, 0.7), CandidatePrefetch(2, 0.2))),
        TokenPrediction(1, (CandidatePrefetch(1, 0.6), CandidatePrefetch(3, 0.8))),
    )
    maximum = BatchPredictorOutput(0, 0, 0, 0, tokens, "max").aggregate_candidates()
    summed = BatchPredictorOutput(0, 0, 0, 0, tokens, "sum").aggregate_candidates()
    assert tuple(candidate.expert_id for candidate in maximum) == (3, 1, 2)
    assert tuple(candidate.expert_id for candidate in summed) == (1, 3, 2)
    assert len({candidate.expert_id for candidate in summed}) == len(summed)
