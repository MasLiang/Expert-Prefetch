"""Exact post-router pipeline tests."""

from edge_moe_sim.simulator import Scenario, Simulator


def test_exact_pipeline_overlaps_dma_and_compute(small_model, small_hardware, two_expert_event) -> None:
    """B Gate-Up loads during A compute and produces no B Gate-Up stall."""
    simulator = Simulator(small_model, small_hardware)
    demand = simulator.simulate(two_expert_event, Scenario.DEMAND_SERIAL)
    exact = simulator.simulate(two_expert_event, Scenario.EXACT_PIPELINE)
    assert exact.result.total_cycles == 53
    assert exact.result.total_cycles <= demand.result.total_cycles
    first, second = exact.result.expert_records
    assert second.gate_up_weight_ready_cycle <= first.gate_up_done_cycle
    assert second.gate_up_start_cycle is not None
    assert second.memory_stall_cycles == 0
    dma_b_gate = next(
        item for item in exact.timeline
        if item.resource == "dma" and item.expert_id == 1 and item.operation.endswith("gate_up_e1")
    )
    assert first.gate_up_start_cycle < dma_b_gate.end_cycle <= first.gate_up_done_cycle
