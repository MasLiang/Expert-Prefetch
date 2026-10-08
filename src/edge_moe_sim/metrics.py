"""Cross-scenario speedup calculations."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

from .schema import SimulationResult


def add_speedups(
    result: SimulationResult,
    demand_serial: SimulationResult,
    exact_pipeline: SimulationResult,
    ideal: SimulationResult | None = None,
) -> SimulationResult:
    """Populate speedup ratios on ``result`` using total layer cycles."""
    result.speedup_vs_demand_serial = demand_serial.total_cycles / result.total_cycles
    result.speedup_vs_exact_pipeline = exact_pipeline.total_cycles / result.total_cycles
    result.speedup_vs_ideal = ideal.total_cycles / result.total_cycles if ideal else None
    return result


def result_to_dict(result: SimulationResult) -> dict[str, Any]:
    """Convert a result, including nested expert records, to JSON data."""
    return asdict(result)

