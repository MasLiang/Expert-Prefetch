"""Partial prediction-hit and ready-first scheduling tests."""

from edge_moe_sim.schema import ModelConfig
from edge_moe_sim.scheduler import ReadyFirstScheduler, RouterOrderScheduler
from edge_moe_sim.simulator import Scenario, Simulator
from edge_moe_sim.synthetic import make_decode_event
from predictor_helpers import FixedPredictor


def test_partial_hit_executes_ready_expert_first(small_hardware) -> None:
    """A ready hit runs before a router-first miss and improves latency."""
    model = ModelConfig("six", 4, 8, 6, 2, 8, 8)
    event = make_decode_event((0, 1), gating_weights=(0.6, 0.4))
    simulator = Simulator(model, small_hardware)
    predictor = FixedPredictor((1, 3))
    ready = simulator.simulate(
        event, Scenario.PREDICTIVE, predictor=predictor,
        expert_scheduler=ReadyFirstScheduler(),
    )
    router = simulator.simulate(
        event, Scenario.PREDICTIVE, predictor=predictor,
        expert_scheduler=RouterOrderScheduler(),
    )
    assert ready.result.static_recall == 0.5
    assert ready.result.expert_records[0].expert_id == 1
    assert ready.result.total_cycles <= router.result.total_cycles
    assert ready.result.demand_bytes == 64 + 2 * 32

