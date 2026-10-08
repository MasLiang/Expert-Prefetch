"""Analytical duration functions with integer-cycle rounding."""

from __future__ import annotations

from .schema import AIEConfig, DMAConfig, NonlinearConfig
from .work_model import ceil_div


def aie_payload_cycles(operation_count: int, ops_per_cycle: int) -> int:
    """Return effective GEMM payload duration in cycles."""
    return ceil_div(operation_count, ops_per_cycle)


def aie_runtime_cycles(
    operation_count: int,
    ops_per_cycle: int,
    config: AIEConfig,
    pays_switch: bool,
) -> int:
    """Return total AIE reservation duration in cycles."""
    return (
        config.startup_cycles
        + (config.operation_switch_cycles if pays_switch else 0)
        + aie_payload_cycles(operation_count, ops_per_cycle)
    )


def dma_runtime_cycles(transfer_bytes: int, config: DMAConfig) -> int:
    """Return DMA startup plus payload duration in cycles."""
    if transfer_bytes <= 0:
        raise ValueError("transfer_bytes must be positive")
    return config.startup_cycles + ceil_div(transfer_bytes, config.bytes_per_cycle)


def nonlinear_runtime_cycles(element_count: int, config: NonlinearConfig) -> int:
    """Return nonlinear startup plus payload duration in cycles."""
    if element_count <= 0:
        raise ValueError("element_count must be positive")
    return config.startup_cycles + ceil_div(element_count, config.elements_per_cycle)
