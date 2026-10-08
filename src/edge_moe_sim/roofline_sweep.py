"""Representative batch/Top-K/bandwidth scenario sweep helpers."""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

from .advanced_schema import AdvancedExecutionConfig, AdvancedScenario
from .advanced_simulator import AdvancedSimulator
from .batch_predictor import BatchPredictor
from .batch_routing import BatchRoutingEvent
from .hardware_profile import HardwareProfile
from .roofline_prefetch_dse import profile_with_effective_bandwidth
from .schema import ModelConfig


@dataclass(frozen=True)
class RooflineSweepRow:
    """One workload/hardware/scenario Roofline result row."""

    scenario: str
    batch_size: int
    top_k: int
    token_expert_assignments: int
    unique_active_experts: int
    reuse_factor: float
    bandwidth_gbps: float
    aie_units_per_expert: int
    max_parallel_experts: int
    candidate_count: int
    prefetch_depth: int
    f_tile_size: int
    arithmetic_intensity: float
    achieved_useful_gbps: float
    physical_memory_utilization: float
    useful_byte_fraction: float
    board_aie_utilization: float
    aie_starvation_cycles: int
    startable_recall: float
    layer_cycles: int
    roofline_efficiency: float
    configured_roofline_efficiency: float
    regime: str
    speculative_bytes: int
    wasted_speculative_bytes: int


def run_roofline_sweep(
    model: ModelConfig, base_profile: HardwareProfile,
    events: Sequence[BatchRoutingEvent], predictor: BatchPredictor,
    scenarios: Sequence[AdvancedScenario | str], *,
    bandwidths_gbps: Sequence[float], config: AdvancedExecutionConfig,
) -> list[RooflineSweepRow]:
    """Run every event/scenario/bandwidth point through the event simulator."""
    rows = []
    for bandwidth in bandwidths_gbps:
        profile = profile_with_effective_bandwidth(base_profile, bandwidth)
        simulator = AdvancedSimulator(model, profile)
        for event in events:
            for raw_scenario in scenarios:
                scenario = AdvancedScenario(raw_scenario)
                result = simulator.simulate(
                    event, scenario, config,
                    predictor=predictor if "predictive" in scenario.value else None,
                )
                memory = result.memory_metrics
                rows.append(RooflineSweepRow(
                    scenario.value, event.batch_size, event.top_k,
                    result.workload.token_expert_assignments,
                    result.workload.unique_active_experts, result.workload.reuse_factor,
                    bandwidth, config.aie_units_per_expert, config.max_parallel_experts,
                    config.candidate_count, config.uniform_gu_prefetch_depth,
                    config.f_tile_size, result.roofline.arithmetic_intensity_ops_per_byte,
                    float(memory["achieved_useful_bandwidth_bytes_per_sec"]) / 1e9,
                    float(memory["physical_bandwidth_utilization"]),
                    float(memory["useful_byte_fraction"]),
                    float(result.aie_metrics["board_aie_utilization"]),
                    result.aie_starvation_cycles, result.startable_recall,
                    result.layer_makespan_cycles,
                    result.roofline.board_roofline_efficiency,
                    result.roofline.configured_roofline_efficiency,
                    result.roofline.regime, int(memory["speculative_bytes"]),
                    int(memory["wasted_speculative_bytes"]),
                ))
    return rows


def write_roofline_sweep_csv(rows: Sequence[RooflineSweepRow], path: str | Path) -> None:
    """Write Roofline sweep rows in stable field order."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fields = list(RooflineSweepRow.__dataclass_fields__)
    with output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))
