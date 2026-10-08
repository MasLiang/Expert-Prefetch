"""Lightweight tile-size, ready-time, and byte-budget design sweeps."""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

from .prefetch_policy import FixedByteBudgetPolicy
from .predictor import OraclePredictor
from .schema import PrefetchConfig, RoutingEvent, SimulationConfig, TileConfig
from .simulator import Scenario, Simulator


@dataclass(frozen=True)
class SweepRow:
    """One point in the tile design-space grid."""

    tile_size_bytes: int
    byte_budget: int
    ready_offset_cycles: int
    total_cycles: int
    speedup_vs_tile_exact: float
    aie_memory_stall_cycles: int
    ready_recall: float
    resident_fraction: float
    speculative_bytes: int
    useful_bytes: int
    wasted_bytes: int


def run_tile_sweep(
    simulator: Simulator,
    events: Sequence[RoutingEvent],
    *,
    tile_sizes: Sequence[int],
    byte_budgets: Sequence[int],
    ready_offsets: Sequence[int],
) -> list[SweepRow]:
    """Evaluate the Cartesian product of three tile-prefetch controls."""
    rows: list[SweepRow] = []
    for tile_size in tile_sizes:
        exact_cycles = sum(
            simulator.simulate(
                event, Scenario.EXACT_PIPELINE,
                simulation=SimulationConfig("tile"),
                tile_config=TileConfig(tile_size),
            ).result.total_cycles
            for event in events
        )
        for budget in byte_budgets:
            for ready_offset in ready_offsets:
                results = [
                    simulator.simulate(
                        event,
                        Scenario.ORACLE,
                        predictor=OraclePredictor(ready_offset),
                        simulation=SimulationConfig("tile"),
                        tile_config=TileConfig(tile_size),
                        prefetch=PrefetchConfig(
                            policy="fixed_byte_budget", byte_budget=budget
                        ),
                        prefetch_policy=FixedByteBudgetPolicy(False),
                    ).result
                    for event in events
                ]
                total_cycles = sum(result.total_cycles for result in results)
                rows.append(SweepRow(
                    tile_size,
                    budget,
                    ready_offset,
                    total_cycles,
                    exact_cycles / total_cycles,
                    sum(result.aie_memory_stall_cycles for result in results),
                    sum((result.ready_recall_at_resolve or 0.0) for result in results) / len(results),
                    sum(result.true_tile_resident_fraction_at_resolve for result in results) / len(results),
                    sum(result.speculative_bytes for result in results),
                    sum(result.useful_speculative_bytes for result in results),
                    sum(result.wasted_speculative_bytes for result in results),
                ))
    return rows


def write_sweep_csv(rows: Sequence[SweepRow], path: str | Path) -> None:
    """Write deterministic sweep rows to CSV."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(SweepRow.__dataclass_fields__))
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))
