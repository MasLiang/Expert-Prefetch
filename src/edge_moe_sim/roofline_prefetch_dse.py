"""User-grid Roofline/prefetch DSE and latency-byte Pareto extraction."""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass, replace
from itertools import product
from pathlib import Path
from typing import Sequence

from .advanced_schema import AdvancedExecutionConfig, AdvancedScenario
from .advanced_simulator import AdvancedSimulator
from .batch_predictor import BatchPredictor
from .batch_routing import BatchRoutingEvent
from .hardware_profile import HardwareProfile
from .schema import ModelConfig, ValidationError


@dataclass(frozen=True)
class DSEPoint:
    """One design point with objective, constraints, and system metrics."""

    batch_size: int
    router_top_k: int
    candidate_count: int
    prefetch_policy: str
    prefetch_depth: int
    aie_units_per_expert: int
    max_parallel_experts: int
    f_tile_size: int
    placement_policy: str
    effective_bandwidth_gbps: float
    makespan_cycles: int
    speculative_bytes: int
    issued_speculative_bytes: int
    useful_speculative_bytes: int
    wasted_speculative_bytes: int
    rejected_prefetch_bytes: int
    peak_ocm_bytes: int
    demand_blocked_cycles: int
    memory_utilization: float
    aie_utilization: float
    aie_starvation_cycles: int
    arithmetic_intensity: float
    board_roofline_efficiency: float
    configured_roofline_efficiency: float
    regime: str
    feasible: bool
    pareto_latency_bytes: bool = False


def allocate_uniform_prefix_under_budget(
    candidate_count: int, num_tiles: int, tile_bytes: int, byte_budget: int,
) -> tuple[int, int]:
    """Return uniform depth and exact used bytes under a shared budget."""
    if min(candidate_count, num_tiles, tile_bytes) <= 0 or byte_budget < 0:
        raise ValidationError("invalid breadth/depth budget inputs")
    depth = min(num_tiles, byte_budget // (candidate_count * tile_bytes))
    return depth, candidate_count * depth * tile_bytes


def profile_with_effective_bandwidth(
    profile: HardwareProfile, effective_bandwidth_gbps: float,
) -> HardwareProfile:
    """Override effective aggregate bandwidth while retaining efficiency metadata."""
    if effective_bandwidth_gbps <= 0:
        raise ValidationError("effective bandwidth must be positive")
    per_channel_peak = (
        effective_bandwidth_gbps * 1e9
        / (profile.num_memory_channels * profile.bandwidth_efficiency)
    )
    return replace(profile, bandwidth_bytes_per_sec_per_channel=per_channel_peak)


def run_roofline_prefetch_dse(
    model: ModelConfig,
    base_profile: HardwareProfile,
    event: BatchRoutingEvent,
    predictor: BatchPredictor,
    *,
    candidate_counts: Sequence[int],
    prefetch_policies: Sequence[str],
    depths: Sequence[int],
    aie_units_per_expert: Sequence[int],
    max_parallel_experts: Sequence[int],
    f_tile_sizes: Sequence[int],
    placement_policies: Sequence[str],
    effective_bandwidth_gbps: Sequence[float],
    speculative_byte_budget: int | None = None,
    ocm_capacity_bytes: int | None = None,
) -> list[DSEPoint]:
    """Search a user-defined grid without replacing event timing by formulas."""
    rows: list[DSEPoint] = []
    for candidates, policy, depth, units, parallel, f_tile, placement, bandwidth in product(
        candidate_counts, prefetch_policies, depths, aie_units_per_expert,
        max_parallel_experts, f_tile_sizes, placement_policies,
        effective_bandwidth_gbps,
    ):
        if candidates > model.num_experts or units > base_profile.total_aie_units:
            continue
        profile = profile_with_effective_bandwidth(base_profile, bandwidth)
        if ocm_capacity_bytes is not None:
            profile = replace(profile, ocm_bytes=ocm_capacity_bytes)
        predictor_output = predictor.predict(event)
        aggregate = predictor_output.aggregate_candidates(candidates) if predictor_output else ()
        max_tiles = (model.intermediate_size + f_tile - 1) // f_tile
        if policy == "uniform":
            explicit = ()
            uniform = depth
            scenario = AdvancedScenario.PREDICTIVE_UNIFORM_PREFIX
            rank_depths = ()
        elif policy == "rank_based":
            explicit = ()
            uniform = depth
            rank_depths = tuple(max(0, depth - rank) for rank in range(candidates))
            scenario = AdvancedScenario.PREDICTIVE_RANKED_PREFIX
        elif policy == "explicit_vector":
            explicit = tuple(min(max_tiles, max(0, depth - rank % 2)) for rank in range(candidates))
            uniform = depth
            rank_depths = ()
            scenario = AdvancedScenario.PREDICTIVE_DSE
        elif policy == "confidence_based":
            maximum = max((candidate.score for candidate in aggregate), default=1.0)
            explicit = tuple(
                min(max_tiles, round(depth * candidate.score / maximum))
                for candidate in aggregate
            )
            uniform = depth
            rank_depths = ()
            scenario = AdvancedScenario.PREDICTIVE_DSE
        else:
            raise ValidationError(f"unsupported DSE prefetch policy {policy!r}")
        config = AdvancedExecutionConfig(
            f_tile, placement, units, parallel,
            candidate_count=candidates,
            uniform_gu_prefetch_depth=uniform,
            rank_based_depths=rank_depths,
            explicit_depths=explicit,
            speculative_byte_budget=speculative_byte_budget,
        )
        result = AdvancedSimulator(model, profile).simulate(
            event, scenario, config, predictor=predictor
        )
        memory = result.memory_metrics
        feasible = (
            (speculative_byte_budget is None or memory["issued_speculative_bytes"] <= speculative_byte_budget)
            and result.peak_ocm_bytes <= profile.ocm_bytes
        )
        rows.append(DSEPoint(
            event.batch_size, event.top_k, candidates, policy, depth, units,
            parallel, f_tile, placement, bandwidth, result.layer_makespan_cycles,
            int(memory["speculative_bytes"]), int(memory["issued_speculative_bytes"]),
            int(memory["useful_speculative_bytes"]),
            int(memory["wasted_speculative_bytes"]), result.rejected_prefetch_bytes,
            result.peak_ocm_bytes, result.demand_blocked_cycles,
            float(memory["physical_bandwidth_utilization"]),
            float(result.aie_metrics["board_aie_utilization"]),
            result.aie_starvation_cycles,
            result.roofline.arithmetic_intensity_ops_per_byte,
            result.roofline.board_roofline_efficiency,
            result.roofline.configured_roofline_efficiency,
            result.roofline.regime, feasible,
        ))
    pareto_indices = _pareto_latency_bytes(rows)
    return [replace(row, pareto_latency_bytes=index in pareto_indices) for index, row in enumerate(rows)]


def _pareto_latency_bytes(rows: Sequence[DSEPoint]) -> set[int]:
    feasible = [(index, row) for index, row in enumerate(rows) if row.feasible]
    return {
        index for index, row in feasible
        if not any(
            other.makespan_cycles <= row.makespan_cycles
            and other.speculative_bytes <= row.speculative_bytes
            and (
                other.makespan_cycles < row.makespan_cycles
                or other.speculative_bytes < row.speculative_bytes
            )
            for other_index, other in feasible if other_index != index
        )
    }


def write_dse_csv(rows: Sequence[DSEPoint], path: str | Path) -> None:
    """Write DSE and Pareto columns without a plotting dependency."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fields = list(DSEPoint.__dataclass_fields__)
    with output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))
