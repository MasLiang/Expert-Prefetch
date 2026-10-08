"""Tile commit/cancel, non-preemption, and resolve readiness tests."""

from edge_moe_sim.prefetch_policy import FixedDepthPolicy
from edge_moe_sim.schema import (
    AIEConfig, CandidatePrefetch, DMAConfig, HardwareConfig, NonlinearConfig,
    PLControlConfig, PredictorOutput, SimulationConfig, TileConfig,
)
from edge_moe_sim.simulator import Scenario, Simulator
from edge_moe_sim.tiling import FixedTileCountPolicy


class TilePredictor:
    """Return progressive candidates at layer start."""

    def __init__(self, candidates) -> None:
        self.candidates = tuple(candidates)

    def predict(self, event, layer_start_cycle):
        """Return candidates keyed to ``event``."""
        return PredictorOutput(
            event.request_id, event.token_position, event.layer_id,
            ready_offset_cycles=0, candidates=self.candidates,
        )


def _run(model, hardware, event, candidates, depth=2):
    return Simulator(model, hardware).simulate(
        event,
        Scenario.PREDICTIVE,
        predictor=TilePredictor(candidates),
        simulation=SimulationConfig("tile"),
        tile_config=TileConfig(1024),
        tiling_policy=FixedTileCountPolicy(4, 2),
        prefetch_policy=FixedDepthPolicy(depth, 0),
    )


def test_tile_commit_no_duplicate(small_model, small_hardware, two_expert_event) -> None:
    """An in-flight true tile commits and is never demand-transferred again."""
    output = _run(
        small_model, small_hardware, two_expert_event,
        (CandidatePrefetch(0, 0.9),), 2,
    )
    tile_zero = [
        entry for entry in output.timeline
        if entry.resource == "dma" and entry.expert_id == 0 and entry.tile_id == 0
        and "_gu_" in entry.operation
    ]
    assert len(tile_zero) == 1
    assert tile_zero[0].request_type == "speculative"
    assert output.result.useful_speculative_tiles == 2


def test_tile_cancel_queued_counts_bytes(small_model, small_hardware, two_expert_event) -> None:
    """All false tiles that did not start are canceled without transfer."""
    output = _run(
        small_model, small_hardware, two_expert_event,
        (CandidatePrefetch(2, 0.9), CandidatePrefetch(3, 0.8)), 4,
    )
    result = output.result
    assert result.canceled_speculative_tiles > 0
    assert result.canceled_speculative_bytes == result.canceled_speculative_tiles * 16
    assert result.speculative_bytes + result.canceled_speculative_bytes == 8 * 16


def test_tile_nonpreemptive_penalty(small_model, two_expert_event) -> None:
    """A long active false tile blocks demand, which then wins over speculation."""
    hardware = HardwareConfig(
        1, AIEConfig(16, 16, 16, 0, 0), DMAConfig(1, 4),
        NonlinearConfig(8, 0), PLControlConfig(0, 0, 0, 0, 0),
    )
    output = _run(
        small_model, hardware, two_expert_event,
        (CandidatePrefetch(2, 0.9), CandidatePrefetch(3, 0.8)), 2,
    )
    assert output.result.demand_blocked_by_speculative_cycles > 0
    dma = [entry for entry in output.timeline if entry.resource == "dma"]
    assert dma[0].request_type == "speculative"
    assert dma[1].request_type == "demand"


def test_ready_recall_and_resident_fraction_at_resolve(
    small_model, two_expert_event
) -> None:
    """Two resident E0 tiles include one of two true first tiles at resolve."""
    hardware = HardwareConfig(
        1, AIEConfig(16, 16, 16, 2, 1), DMAConfig(8, 1),
        NonlinearConfig(8, 1), PLControlConfig(1, 1, 0, 0, 2),
    )
    output = Simulator(small_model, hardware).simulate(
        two_expert_event,
        Scenario.ORACLE,
        simulation=SimulationConfig("tile"),
        tile_config=TileConfig(1024),
        tiling_policy=FixedTileCountPolicy(4, 2),
        prefetch_policy=FixedDepthPolicy(2, 0),
    )
    assert output.result.ready_recall_at_resolve == 0.5
    assert output.result.true_tiles_resident_at_resolve == 2
    assert output.result.true_tile_resident_fraction_at_resolve == 2 / 12
