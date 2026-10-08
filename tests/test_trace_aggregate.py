"""Independent trace aggregation tests."""

from edge_moe_sim.simulator import Scenario, Simulator
from edge_moe_sim.synthetic import make_decode_event


def test_trace_cycles_are_summed_independent_events(small_model, small_hardware) -> None:
    """Trace aggregation sums isolated runs and labels its semantics."""
    simulator = Simulator(small_model, small_hardware)
    events = [make_decode_event((0, 1), request_id=index) for index in range(2)]
    aggregate = simulator.simulate_trace(events, Scenario.EXACT_PIPELINE)
    assert aggregate.num_events == 2
    assert aggregate.total_cycles == 2 * aggregate.per_event[0].total_cycles
    assert "not continuous-batching" in aggregate.cycle_semantics
