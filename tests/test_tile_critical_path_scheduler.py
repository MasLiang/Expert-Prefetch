"""Tile-level critical-path DMA scheduling tests."""

from edge_moe_sim.dma_scheduler import AllGateUpThenDownScheduler, TileCriticalPathAwareScheduler
from edge_moe_sim.schema import (
    AIEConfig, DMAConfig, HardwareConfig, NonlinearConfig, PLControlConfig,
    SimulationConfig, TileConfig,
)
from edge_moe_sim.simulator import Scenario, Simulator
from edge_moe_sim.tiling import FixedTileCountPolicy


def test_tile_critical_path_scheduler_reduces_stall(small_model, two_expert_event) -> None:
    """Current Down tiles beat the next Expert's distant Gate-Up tiles."""
    hardware = HardwareConfig(
        1,
        AIEConfig(16, 8, 8, 0, 0, matrix_startup_cycles=0, tile_pipeline_cycles=0),
        DMAConfig(8, 1),
        NonlinearConfig(100, 0),
        PLControlConfig(0, 0, 0, 0, 0),
    )
    simulator = Simulator(small_model, hardware)
    common = {
        "simulation": SimulationConfig("tile"),
        "tile_config": TileConfig(1024),
        "tiling_policy": FixedTileCountPolicy(4, 2),
    }
    naive = simulator.simulate(
        two_expert_event, Scenario.EXACT_PIPELINE,
        dma_scheduler=AllGateUpThenDownScheduler(), **common,
    )
    critical = simulator.simulate(
        two_expert_event, Scenario.EXACT_PIPELINE,
        dma_scheduler=TileCriticalPathAwareScheduler(), **common,
    )
    naive_order = [entry.operation for entry in naive.timeline if entry.resource == "dma"]
    critical_order = [entry.operation for entry in critical.timeline if entry.resource == "dma"]
    assert naive_order[4] == "demand_gu_e1_t0"
    assert critical_order[4] == "demand_down_e0_t0"
    assert critical.result.aie_memory_stall_cycles < naive.result.aie_memory_stall_cycles
    assert critical.result.total_cycles < naive.result.total_cycles
