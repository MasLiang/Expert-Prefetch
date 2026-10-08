"""Conservation, exclusion, validation, and determinism tests."""

from dataclasses import asdict

import pytest

from edge_moe_sim.schema import ModelConfig, ValidationError
from edge_moe_sim.simulator import Scenario, Simulator


def test_dma_bandwidth_and_startup(small_model, small_hardware, two_expert_event) -> None:
    """Each interval reserves startup before sufficient payload cycles."""
    output = Simulator(small_model, small_hardware).simulate(two_expert_event, Scenario.EXACT_PIPELINE)
    for interval in (entry for entry in output.timeline if entry.resource == "dma"):
        size = 64 if interval.operation == "fetch_gate_up" else 32
        payload_cycles = interval.end_cycle - interval.start_cycle - small_hardware.dma.startup_cycles
        assert size <= small_hardware.dma.bytes_per_cycle * payload_cycles


def test_aie_intervals_do_not_overlap(small_model, small_hardware, two_expert_event) -> None:
    """The logical AIE remains a single mutual-exclusion resource."""
    output = Simulator(small_model, small_hardware).simulate(two_expert_event, Scenario.EXACT_PIPELINE)
    intervals = sorted((e for e in output.timeline if e.resource == "aie"), key=lambda e: e.start_cycle)
    assert all(left.end_cycle <= right.start_cycle for left, right in zip(intervals, intervals[1:]))


def test_repeated_runs_are_identical(small_model, small_hardware, two_expert_event) -> None:
    """Identical input produces identical summaries and timelines."""
    simulator = Simulator(small_model, small_hardware)
    first = simulator.simulate(two_expert_event, Scenario.EXACT_PIPELINE)
    second = simulator.simulate(two_expert_event, Scenario.EXACT_PIPELINE)
    assert asdict(first.result) == asdict(second.result)
    assert first.timeline == second.timeline


def test_invalid_model_is_rejected() -> None:
    """Configuration failures have explicit validation errors."""
    with pytest.raises(ValidationError, match="top_k"):
        ModelConfig("bad", 4, 8, 2, 3, 8, 8)
