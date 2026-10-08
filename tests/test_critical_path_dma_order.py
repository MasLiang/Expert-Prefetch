"""Critical-path Exact Pipeline request-order tests."""

from edge_moe_sim.dma_scheduler import AllGateUpThenDownScheduler, CriticalPathAwareScheduler
from edge_moe_sim.schema import AIEConfig, DMAConfig, HardwareConfig, NonlinearConfig, PLControlConfig
from edge_moe_sim.simulator import Scenario, Simulator


def test_critical_path_dma_order_is_strictly_faster(small_model, two_expert_event) -> None:
    """Loading D(E0) before GU(E1) removes a constructed critical stall."""
    hardware = HardwareConfig(
        1, AIEConfig(16, 128, 8, 0, 0), DMAConfig(8, 0),
        NonlinearConfig(100, 0), PLControlConfig(0, 0, 0, 0, 0),
    )
    simulator = Simulator(small_model, hardware)
    all_gate = simulator.simulate(
        two_expert_event, Scenario.EXACT_PIPELINE,
        dma_scheduler=AllGateUpThenDownScheduler(),
    )
    critical = simulator.simulate(
        two_expert_event, Scenario.EXACT_PIPELINE,
        dma_scheduler=CriticalPathAwareScheduler(),
    )
    all_order = [entry.operation for entry in all_gate.timeline if entry.resource == "dma"]
    critical_order = [entry.operation for entry in critical.timeline if entry.resource == "dma"]
    assert all_order == [
        "demand_gate_up_e0", "demand_gate_up_e1", "demand_down_e0", "demand_down_e1",
    ]
    assert critical_order == [
        "demand_gate_up_e0", "demand_down_e0", "demand_gate_up_e1", "demand_down_e1",
    ]
    assert critical.result.total_cycles < all_gate.result.total_cycles

