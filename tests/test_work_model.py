"""Tests for formulas and analytical duration rounding."""

from edge_moe_sim.performance_model import aie_runtime_cycles, dma_runtime_cycles
from edge_moe_sim.work_model import (
    down_ops, down_weight_bytes, expert_weight_bytes, gate_up_ops,
    gate_up_weight_bytes, nonlinear_elements, router_ops,
)


def test_small_workload_formulas(small_model, small_hardware) -> None:
    """Match the hand-computed H=4, F=8 formula values."""
    assert router_ops(small_model, 1) == 32
    assert gate_up_ops(small_model, 1) == 128
    assert down_ops(small_model, 1) == 64
    assert nonlinear_elements(small_model, 1) == 8
    assert gate_up_weight_bytes(small_model) == 64
    assert down_weight_bytes(small_model) == 32
    assert expert_weight_bytes(small_model) == 96
    assert aie_runtime_cycles(128, 16, small_hardware.aie, True) == 11
    assert dma_runtime_cycles(64, small_hardware.dma) == 5

