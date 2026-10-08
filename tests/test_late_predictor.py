"""Predictor output arriving exactly at resolve."""

from edge_moe_sim.simulator import Scenario, Simulator
from predictor_helpers import FixedPredictor


def test_late_predictor_has_no_router_window_benefit(small_model, small_hardware, two_expert_event) -> None:
    """Dispatch wins the same-cycle tie, so prediction cannot prefetch early."""
    simulator = Simulator(small_model, small_hardware)
    exact = simulator.simulate(two_expert_event, Scenario.EXACT_PIPELINE)
    resolve_cycle = exact.result.resolve_cycle
    predictive = simulator.simulate(
        two_expert_event,
        Scenario.PREDICTIVE,
        predictor=FixedPredictor((0, 1), ready=resolve_cycle),
    )
    assert predictive.result.total_cycles == exact.result.total_cycles
    assert predictive.result.ready_recall_at_resolve == 0.0
    assert predictive.result.speculative_bytes == 0

