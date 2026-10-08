"""In-flight commit, queued cancellation, and DMA deduplication tests."""

from edge_moe_sim.schema import AIEConfig, DMAConfig, HardwareConfig, NonlinearConfig, PLControlConfig
from edge_moe_sim.simulator import Scenario, Simulator
from predictor_helpers import FixedPredictor


def test_commit_inflight_request_has_no_duplicate_gate_up(small_model, two_expert_event) -> None:
    """A true speculative request in flight is committed and transferred once."""
    hardware = HardwareConfig(
        1, AIEConfig(16, 16, 16, 2, 1), DMAConfig(8, 1),
        NonlinearConfig(8, 1), PLControlConfig(1, 1, 0, 0, 2),
    )
    output = Simulator(small_model, hardware).simulate(
        two_expert_event, Scenario.PREDICTIVE,
        predictor=FixedPredictor((0,)),
    )
    result = output.result
    assert result.speculative_requests_committed == 1
    assert result.useful_speculative_bytes == 64
    gate_e0 = [
        entry for entry in output.timeline
        if entry.resource == "dma" and entry.expert_id == 0 and "gate_up" in entry.operation
    ]
    assert len(gate_e0) == 1
    assert gate_e0[0].request_type == "speculative"


def test_cancel_queued_request_not_transferred(small_model, two_expert_event) -> None:
    """A third false candidate is canceled before consuming bandwidth."""
    hardware = HardwareConfig(
        1, AIEConfig(16, 16, 16, 2, 1), DMAConfig(8, 1),
        NonlinearConfig(8, 1), PLControlConfig(1, 1, 0, 0, 2),
    )
    output = Simulator(small_model, hardware).simulate(
        two_expert_event, Scenario.PREDICTIVE,
        predictor=FixedPredictor((2, 3)),
    )
    result = output.result
    assert result.speculative_requests_canceled >= 1
    assert result.canceled_speculative_bytes >= 64
    transferred_e3 = [entry for entry in output.timeline if entry.resource == "dma" and entry.expert_id == 3]
    assert transferred_e3 == []
