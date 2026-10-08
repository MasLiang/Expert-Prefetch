"""Tile streaming equivalence, overlap, starvation, and startup tests."""

from edge_moe_sim.schema import (
    AIEConfig, DMAConfig, HardwareConfig, NonlinearConfig, PLControlConfig,
    SimulationConfig, TileConfig,
)
from edge_moe_sim.simulator import Scenario, Simulator
from edge_moe_sim.tiling import FixedTileCountPolicy


def _tile_run(model, event, hardware, gate_tiles=4, down_tiles=2):
    return Simulator(model, hardware).simulate(
        event,
        Scenario.EXACT_PIPELINE,
        simulation=SimulationConfig("tile"),
        tile_config=TileConfig(1024),
        tiling_policy=FixedTileCountPolicy(gate_tiles, down_tiles),
    )


def test_single_tile_equivalent_to_stage(small_model, small_hardware, two_expert_event) -> None:
    """One tile per matrix preserves legacy stage cycles and bytes exactly."""
    simulator = Simulator(small_model, small_hardware)
    for scenario in (Scenario.DEMAND_SERIAL, Scenario.EXACT_PIPELINE, Scenario.IDEAL):
        stage = simulator.simulate(two_expert_event, scenario).result
        tile = simulator.simulate(
            two_expert_event,
            scenario,
            simulation=SimulationConfig("tile"),
            tile_config=TileConfig(1024),
            tiling_policy=FixedTileCountPolicy(1, 1),
        ).result
        assert tile.total_cycles == stage.total_cycles
        assert tile.demand_bytes == stage.demand_bytes
        assert tile.speculative_bytes == stage.speculative_bytes


def test_tile_streaming_overlap_hides_later_dma(small_model, two_expert_event) -> None:
    """Twenty-cycle tile compute hides ten-cycle transfers after tile zero."""
    hardware = HardwareConfig(
        1,
        AIEConfig(16, 2, 2, 0, 0, matrix_startup_cycles=4, tile_pipeline_cycles=4),
        DMAConfig(2, 2),
        NonlinearConfig(100, 0),
        PLControlConfig(0, 0, 0, 0, 0),
    )
    output = _tile_run(small_model, two_expert_event, hardware)
    first = output.result.expert_records[0]
    assert first.gate_up_tile_records[0].compute_start_cycle == 12
    assert all(record.starvation_cycles == 0 for record in first.gate_up_tile_records[1:])
    assert first.inter_tile_stall_cycles == 0
    gu_dma_end = max(
        entry.end_cycle for entry in output.timeline
        if entry.resource == "dma" and entry.expert_id == first.expert_id and "_gu_" in entry.operation
    )
    assert first.gate_up_start_cycle < gu_dma_end < first.gate_up_done_cycle


def test_tile_starvation_is_exact_when_dma_is_slower(small_model, two_expert_event) -> None:
    """Twenty-cycle transfers feeding ten-cycle compute create ten-cycle gaps."""
    hardware = HardwareConfig(
        1,
        AIEConfig(16, 4, 4, 0, 0, matrix_startup_cycles=2, tile_pipeline_cycles=2),
        DMAConfig(1, 4),
        NonlinearConfig(100, 0),
        PLControlConfig(0, 0, 0, 0, 0),
    )
    output = _tile_run(small_model, two_expert_event, hardware)
    gate_records = output.result.expert_records[0].gate_up_tile_records
    assert [record.starvation_cycles for record in gate_records[1:]] == [10, 10, 10]
    assert output.result.aie_inter_tile_stall_cycles > 0
    assert (
        output.result.aie_first_tile_stall_cycles
        + output.result.aie_inter_tile_stall_cycles
        == output.result.aie_memory_stall_cycles
    )


def test_tile_dma_startup_tradeoff(small_model, small_hardware, two_expert_event) -> None:
    """More tiles pay more DMA startups for the same transferred bytes."""
    large = _tile_run(small_model, two_expert_event, small_hardware, 1, 1).result
    small = _tile_run(small_model, two_expert_event, small_hardware, 8, 4).result
    assert small.demand_bytes == large.demand_bytes
    assert small.dma_busy_cycles > large.dma_busy_cycles
    assert small.total_cycles >= large.total_cycles
