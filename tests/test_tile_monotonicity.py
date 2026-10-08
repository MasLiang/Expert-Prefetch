"""Tile-mode monotonicity, convergence, and determinism tests."""

from dataclasses import asdict, replace

from edge_moe_sim.prefetch_policy import FixedDepthPolicy
from edge_moe_sim.predictor import OraclePredictor
from edge_moe_sim.schema import SimulationConfig, TileConfig
from edge_moe_sim.simulator import Scenario, Simulator
from edge_moe_sim.tiling import FixedTileCountPolicy


def _kwargs():
    return {
        "simulation": SimulationConfig("tile"),
        "tile_config": TileConfig(1024),
        "tiling_policy": FixedTileCountPolicy(4, 2),
    }


def test_monotonic_ready_time(small_model, small_hardware, two_expert_event) -> None:
    """An earlier true-only Oracle cannot be slower than a later one."""
    simulator = Simulator(small_model, small_hardware)
    early = simulator.simulate(
        two_expert_event, Scenario.ORACLE, predictor=OraclePredictor(0), **_kwargs()
    )
    late = simulator.simulate(
        two_expert_event, Scenario.ORACLE, predictor=OraclePredictor(5), **_kwargs()
    )
    assert early.result.total_cycles <= late.result.total_cycles


def test_monotonic_dma_bandwidth(small_model, small_hardware, two_expert_event) -> None:
    """Higher DMA bandwidth does not slow Exact or Oracle."""
    faster_hardware = replace(
        small_hardware,
        dma=replace(small_hardware.dma, bytes_per_cycle=64),
    )
    for scenario in (Scenario.EXACT_PIPELINE, Scenario.ORACLE):
        slow = Simulator(small_model, small_hardware).simulate(
            two_expert_event, scenario, **_kwargs()
        )
        fast = Simulator(small_model, faster_hardware).simulate(
            two_expert_event, scenario, **_kwargs()
        )
        assert fast.result.total_cycles <= slow.result.total_cycles


def test_monotonic_aie_throughput_for_ideal(small_model, small_hardware, two_expert_event) -> None:
    """Higher effective AIE rates cannot make the Ideal schedule slower."""
    faster_aie = replace(
        small_hardware.aie,
        gate_up_ops_per_cycle=64,
        down_ops_per_cycle=64,
    )
    fast_hardware = replace(small_hardware, aie=faster_aie)
    slow = Simulator(small_model, small_hardware).simulate(
        two_expert_event, Scenario.IDEAL, **_kwargs()
    )
    fast = Simulator(small_model, fast_hardware).simulate(
        two_expert_event, Scenario.IDEAL, **_kwargs()
    )
    assert fast.result.total_cycles <= slow.result.total_cycles


def test_infinite_dma_bandwidth_convergence(small_model, small_hardware, two_expert_event) -> None:
    """Near-infinite bandwidth leaves at most one-cycle tile-feed bubbles."""
    hardware = replace(
        small_hardware,
        dma=replace(small_hardware.dma, bytes_per_cycle=10**9, startup_cycles=0),
    )
    simulator = Simulator(small_model, hardware)
    exact = simulator.simulate(two_expert_event, Scenario.EXACT_PIPELINE, **_kwargs())
    oracle = simulator.simulate(
        two_expert_event, Scenario.ORACLE,
        prefetch_policy=FixedDepthPolicy(4, 2), **_kwargs(),
    )
    ideal = simulator.simulate(two_expert_event, Scenario.IDEAL, **_kwargs())
    assert exact.result.aie_memory_stall_cycles <= 1
    assert oracle.result.aie_memory_stall_cycles <= 1
    assert ideal.result.total_cycles <= oracle.result.total_cycles <= exact.result.total_cycles
    assert exact.result.total_cycles - ideal.result.total_cycles <= 1


def test_tile_determinism(small_model, small_hardware, two_expert_event) -> None:
    """Tile summary and expanded timeline are exactly repeatable."""
    simulator = Simulator(small_model, small_hardware)
    first = simulator.simulate(two_expert_event, Scenario.EXACT_PIPELINE, **_kwargs())
    second = simulator.simulate(two_expert_event, Scenario.EXACT_PIPELINE, **_kwargs())
    assert asdict(first.result) == asdict(second.result)
    assert first.timeline == second.timeline
