"""Exact-accounting Roofline metrics for simulated workloads."""

from __future__ import annotations

from dataclasses import dataclass

from .hardware_profile import HardwareProfile
from .schema import ValidationError


@dataclass(frozen=True)
class RooflineMetrics:
    """Board and selected-capacity Roofline ceilings and efficiencies."""

    useful_ops: int
    unique_required_weight_bytes: int
    arithmetic_intensity_ops_per_byte: float
    ridge_point_ops_per_byte: float
    memory_ceiling_ops_per_sec: float
    compute_ceiling_ops_per_sec: float
    board_roofline_ops_per_sec: float
    configured_compute_ceiling_ops_per_sec: float
    configured_roofline_ops_per_sec: float
    achieved_ops_per_sec: float
    board_roofline_efficiency: float
    configured_roofline_efficiency: float
    regime: str
    # Explicit aliases avoid the common error of describing a selected subset as
    # the board.  The legacy fields above remain for API compatibility.
    board_compute_peak_ops_s: float
    configured_compute_peak_ops_s: float
    board_ridge_ai: float
    configured_ridge_ai: float
    board_memory_ceiling_ops_s: float
    configured_memory_ceiling_ops_s: float
    board_roofline_ops_s: float
    configured_roofline_ops_s: float
    board_regime: str
    configured_regime: str


def calculate_roofline_metrics(
    useful_ops: int,
    unique_required_weight_bytes: int,
    makespan_cycles: int,
    profile: HardwareProfile,
    *,
    configured_aie_units: int | None = None,
    balance_tolerance: float = 0.10,
) -> RooflineMetrics:
    """Calculate exact aggregate arithmetic intensity and Roofline position."""
    if useful_ops <= 0 or unique_required_weight_bytes <= 0 or makespan_cycles <= 0:
        raise ValidationError("roofline work, bytes, and makespan must be positive")
    if not 0.0 <= balance_tolerance < 1.0:
        raise ValidationError("balance_tolerance must be in [0, 1)")
    units = profile.total_aie_units if configured_aie_units is None else configured_aie_units
    if not 1 <= units <= profile.total_aie_units:
        raise ValidationError("configured_aie_units must fit the board")
    intensity = useful_ops / unique_required_weight_bytes
    compute = profile.effective_compute_peak_ops_per_sec
    bandwidth = profile.effective_memory_bandwidth_bytes_per_sec
    memory = bandwidth * intensity
    board = min(compute, memory)
    configured_compute = (
        units * profile.ops_per_cycle_per_aie * profile.clock_hz * profile.aie_efficiency
    )
    configured_roof = min(configured_compute, memory)
    achieved = useful_ops * profile.clock_hz / makespan_cycles
    def classify(memory_ceiling: float, compute_ceiling: float) -> str:
        ratio = memory_ceiling / compute_ceiling
        if ratio < 1.0 - balance_tolerance:
            return "memory_bound"
        if ratio > 1.0 + balance_tolerance:
            return "compute_bound"
        return "balanced"

    board_regime = classify(memory, compute)
    configured_regime = classify(memory, configured_compute)
    return RooflineMetrics(
        useful_ops, unique_required_weight_bytes, intensity, compute / bandwidth,
        memory, compute, board, configured_compute, configured_roof, achieved,
        achieved / board, achieved / configured_roof, board_regime,
        compute, configured_compute, compute / bandwidth, configured_compute / bandwidth,
        memory, memory, board, configured_roof, board_regime, configured_regime,
    )
