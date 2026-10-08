"""Unit tests for deterministic M9.3 quality and calibration traces."""

from __future__ import annotations

from edge_moe_sim.advanced_schema import (
    AdvancedExecutionConfig,
    AdvancedScenario,
    AdvancedSimulationResult,
)
from edge_moe_sim.advanced_simulator import AdvancedSimulator
from edge_moe_sim.batch_routing import BatchRoutingEvent
from edge_moe_sim.hardware_profile import HardwareProfile
from edge_moe_sim.predictor_sensitivity import (
    CALIBRATION_MODES,
    build_sensitivity_trace,
    calibration_scores,
    interpolated_true_ranks,
)
from edge_moe_sim.schema import ModelConfig
from edge_moe_sim.tile_scheduler import BudgetProportionalScheduler


QUALITY_LEVELS = (1.0, .9, .8, .7, .6, .5, .4, .3, .2)


def test_quality_sweep_monotonically_moves_true_experts_down_ranking() -> None:
    """Every true Expert rank is nondecreasing as quality is reduced."""
    for candidate_count in (16, 32, 64):
        ranks = [
            interpolated_true_ranks(candidate_count, 2, quality)
            for quality in QUALITY_LEVELS
        ]
        assert ranks[0] == (1, 2)
        assert ranks[-1] == (candidate_count // 2, candidate_count)
        for left, right in zip(ranks, ranks[1:]):
            assert all(a <= b for a, b in zip(left, right))


def test_calibration_changes_only_scores_not_candidate_set_or_ranking() -> None:
    """All calibration modes preserve the exact candidate-ID sequence."""
    traces = [
        build_sensitivity_trace(
            request_id=93,
            token_position=0,
            layer_id=0,
            num_experts=64,
            top_k=2,
            candidate_count=16,
            quality=.6,
            calibration_mode=mode,
        )
        for mode in CALIBRATION_MODES
    ]
    expected_ids = traces[0].candidate_ids
    assert all(trace.candidate_ids == expected_ids for trace in traces)
    assert len({trace.candidate_scores for trace in traces}) == 4
    event = BatchRoutingEvent(93, 0, 0, ((0, 1),))
    for trace in traces:
        aggregated = trace.predictor.predict(event).aggregate_candidates(16)
        assert tuple(candidate.expert_id for candidate in aggregated) == expected_ids


def test_identical_sensitivity_request_produces_identical_trace() -> None:
    """Repeated construction is byte-for-byte deterministic."""
    arguments = dict(
        request_id=93,
        token_position=0,
        layer_id=0,
        num_experts=128,
        top_k=2,
        candidate_count=32,
        quality=.7,
        calibration_mode="well_calibrated",
    )
    first = build_sensitivity_trace(**arguments)
    second = build_sensitivity_trace(**arguments)
    assert first.candidate_ids == second.candidate_ids
    assert first.candidate_scores == second.candidate_scores
    assert first.true_expert_ranks == second.true_expert_ranks
    event = BatchRoutingEvent(93, 0, 0, ((0, 1),))
    assert first.predictor.predict(event) == second.predictor.predict(event)


def test_calibration_curves_preserve_rank_order() -> None:
    """Score changes cannot silently reorder candidates during aggregation."""
    for mode in CALIBRATION_MODES:
        scores = calibration_scores(64, mode)
        assert all(left >= right for left, right in zip(scores, scores[1:]))


def test_identical_predictor_trace_produces_identical_simulation() -> None:
    """The same trace produces identical summary metrics and physical timeline."""
    model = ModelConfig("m93_test", 4, 8, 8, 1, 8, 8)
    hardware = HardwareProfile(
        "m93_test_hardware", 1000, 2, 8_000, 1.0, 2, 1, 64,
        True, 4, 8, 1.0, 1, 2048, hardware_tag="synthetic_placeholder",
    )
    event = BatchRoutingEvent(93, 0, 0, ((0,),))
    trace = build_sensitivity_trace(
        request_id=93,
        token_position=0,
        layer_id=0,
        num_experts=8,
        top_k=1,
        candidate_count=4,
        quality=.6,
        calibration_mode="well_calibrated",
    )
    config = AdvancedExecutionConfig(
        2, aie_units_per_expert=1, max_parallel_experts=1,
        candidate_count=4, startup_gu_tiles=1,
    )
    simulator = AdvancedSimulator(model, hardware)

    def run() -> AdvancedSimulationResult:
        return simulator.simulate(
            event,
            AdvancedScenario.PREDICTIVE_DSE,
            config,
            predictor=trace.predictor,
            tile_prefetch_scheduler=BudgetProportionalScheduler(),
        )

    first = run()
    second = run()
    assert first.layer_makespan_cycles == second.layer_makespan_cycles
    assert first.tile_metrics == second.tile_metrics
    assert first.memory_metrics == second.memory_metrics
    assert first.timeline == second.timeline
