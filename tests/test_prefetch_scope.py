"""Stage-level prefetch-scope traffic tests."""

from edge_moe_sim.predictor import OraclePredictor
from edge_moe_sim.schema import PrefetchConfig
from edge_moe_sim.simulator import Scenario, Simulator


def test_gate_only_and_full_expert_account_different_traffic(
    small_model, small_hardware, two_expert_event
) -> None:
    """Full-expert oracle speculates Down while gate-only demands it."""
    simulator = Simulator(small_model, small_hardware)
    gate_only = simulator.simulate(
        two_expert_event, Scenario.ORACLE, predictor=OraclePredictor(0),
        prefetch=PrefetchConfig("gate_up_only", 4),
    ).result
    full = simulator.simulate(
        two_expert_event, Scenario.ORACLE, predictor=OraclePredictor(0),
        prefetch=PrefetchConfig("full_expert", 4),
    ).result
    assert gate_only.speculative_bytes == 2 * 64
    assert gate_only.demand_bytes == 2 * 32
    assert full.speculative_bytes == 2 * (64 + 32)
    assert full.demand_bytes == 0
    assert full.useful_speculative_bytes == full.speculative_bytes

