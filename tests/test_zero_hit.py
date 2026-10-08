"""False-positive cancellation and non-preemptive DMA tests."""

from edge_moe_sim.schema import ModelConfig
from edge_moe_sim.simulator import Scenario, Simulator
from edge_moe_sim.synthetic import make_decode_event
from predictor_helpers import FixedPredictor


def _case(small_hardware):
    model = ModelConfig("six", 4, 8, 6, 2, 8, 8)
    event = make_decode_event((0, 1), gating_weights=(0.6, 0.4))
    return model, event, Simulator(model, small_hardware)


def test_zero_hit_cancels_queued_and_wastes_transferred(small_hardware) -> None:
    """Resident/in-flight false data is wasted and queued false data is canceled."""
    model, event, simulator = _case(small_hardware)
    result = simulator.simulate(
        event, Scenario.PREDICTIVE,
        predictor=FixedPredictor((2, 3, 4)),
    ).result
    assert result.static_recall == 0.0
    assert result.full_hit is False
    assert result.demand_bytes == 2 * (64 + 32)
    assert result.wasted_speculative_bytes == 2 * 64
    assert result.canceled_speculative_bytes == 64
    assert result.speculative_requests_canceled == 1


def test_nonpreemptive_penalty_and_demand_priority(small_hardware) -> None:
    """An active false burst finishes, then demand runs before queued speculation."""
    model, event, simulator = _case(small_hardware)
    output = simulator.simulate(
        event, Scenario.PREDICTIVE,
        predictor=FixedPredictor((2, 3, 4)),
    )
    assert output.result.demand_blocked_by_speculative_cycles > 0
    dma = [entry for entry in output.timeline if entry.resource == "dma"]
    assert dma[1].operation == "speculative_gate_up_e3"
    assert dma[2].request_type == "demand"

