"""No-memory-stall lower-bound tests."""

from edge_moe_sim.simulator import Scenario, Simulator


def test_ideal_is_lower_bound(small_model, small_hardware, two_expert_event) -> None:
    """Constructed scenarios follow ideal <= oracle <= exact <= serial."""
    simulator = Simulator(small_model, small_hardware)
    ideal = simulator.simulate(two_expert_event, Scenario.IDEAL).result
    oracle = simulator.simulate(two_expert_event, Scenario.ORACLE).result
    exact = simulator.simulate(two_expert_event, Scenario.EXACT_PIPELINE).result
    serial = simulator.simulate(two_expert_event, Scenario.DEMAND_SERIAL).result
    assert ideal.total_cycles <= oracle.total_cycles <= exact.total_cycles <= serial.total_cycles
    assert ideal.aie_memory_stall_cycles == 0
    assert ideal.demand_bytes == 0
    assert ideal.speculative_bytes == 0

