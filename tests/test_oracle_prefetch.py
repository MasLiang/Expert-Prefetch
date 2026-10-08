"""Oracle and resolve-time readiness tests."""

from edge_moe_sim.predictor import OraclePredictor
from edge_moe_sim.simulator import Scenario, Simulator


def test_oracle_prefetch_is_fully_useful(small_model, small_hardware, two_expert_event) -> None:
    """Oracle candidates have perfect recall and no wasted traffic."""
    result = Simulator(small_model, small_hardware).simulate(
        two_expert_event, Scenario.ORACLE, predictor=OraclePredictor(0)
    ).result
    assert result.static_recall == 1.0
    assert result.full_hit is True
    assert result.useful_speculative_bytes == result.speculative_bytes
    assert result.wasted_speculative_bytes == 0


def test_ready_recall_at_resolve_is_half(small_model, small_hardware, two_expert_event) -> None:
    """One of two sequential speculative Gate-Up transfers finishes by resolve."""
    result = Simulator(small_model, small_hardware).simulate(
        two_expert_event, Scenario.ORACLE, predictor=OraclePredictor(0)
    ).result
    assert result.ready_recall_at_resolve == 0.5

