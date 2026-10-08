"""Demand-only serial baseline tests."""

from edge_moe_sim.simulator import Scenario, Simulator


def test_demand_serial_is_strict_and_hand_calculable(small_model, small_hardware, two_expert_event) -> None:
    """All operations serialize and the expected total is 64 cycles."""
    output = Simulator(small_model, small_hardware).simulate(two_expert_event, Scenario.DEMAND_SERIAL)
    result = output.result
    assert result.total_cycles == 64
    assert result.speculative_bytes == 0
    assert result.demand_bytes == 2 * (64 + 32)
    assert result.aie_memory_stall_cycles == 2 * (5 + 3)
    ordered = sorted(
        (item for item in output.timeline if item.operation != "memory_stall"),
        key=lambda item: item.start_cycle,
    )
    assert all(left.end_cycle <= right.start_cycle for left, right in zip(ordered, ordered[1:]))
