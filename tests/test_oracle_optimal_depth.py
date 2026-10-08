"""Small exhaustive Oracle depth-search tests."""

from edge_moe_sim.oracle_optimal import optimize_oracle_depth
from edge_moe_sim.prefetch_policy import FixedDepthPolicy
from edge_moe_sim.schema import (
    AIEConfig, DMAConfig, HardwareConfig, ModelConfig, NonlinearConfig,
    PLControlConfig, SimulationConfig, TileConfig,
)
from edge_moe_sim.simulator import Scenario, Simulator
from edge_moe_sim.synthetic import make_decode_event
from edge_moe_sim.tiling import FixedTileCountPolicy


def test_oracle_optimal_depth_finds_known_minimum() -> None:
    """For the constructed K=1 case, depth two is the first optimal prefix."""
    model = ModelConfig("k1", 4, 8, 4, 1, 8, 8)
    hardware = HardwareConfig(
        1,
        AIEConfig(16, 8, 8, 0, 0, matrix_startup_cycles=0, tile_pipeline_cycles=0),
        DMAConfig(4, 1),
        NonlinearConfig(100, 0),
        PLControlConfig(2, 2, 0, 0, 0),
    )
    event = make_decode_event((0,), gating_weights=(1.0,))
    simulator = Simulator(model, hardware)
    tiling = FixedTileCountPolicy(4, 2)
    optimal = optimize_oracle_depth(
        simulator, event, tile_config=TileConfig(1024), byte_budget=64,
        tiling_policy=tiling,
    )
    oracle_set = simulator.simulate(
        event, Scenario.ORACLE,
        simulation=SimulationConfig("tile"), tile_config=TileConfig(1024),
        tiling_policy=tiling, prefetch_policy=FixedDepthPolicy(1, 0),
    )
    exact = simulator.simulate(
        event, Scenario.EXACT_PIPELINE,
        simulation=SimulationConfig("tile"), tile_config=TileConfig(1024),
        tiling_policy=tiling,
    )
    ideal = simulator.simulate(
        event, Scenario.IDEAL,
        simulation=SimulationConfig("tile"), tile_config=TileConfig(1024),
        tiling_policy=tiling,
    )
    assert optimal.optimal_depths == {0: 2}
    assert optimal.evaluated_combinations == 5
    assert ideal.result.total_cycles <= optimal.optimal_cycles
    assert optimal.optimal_cycles <= oracle_set.result.total_cycles <= exact.result.total_cycles
