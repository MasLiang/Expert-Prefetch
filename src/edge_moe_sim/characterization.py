"""Reproducible Milestone 6.5 characterization helpers.

This module deliberately drives :class:`AdvancedSimulator`; it does not replace
the event DAG with an analytical latency model.  All numbers it writes are
therefore explicitly synthetic/theoretical unless a measured hardware profile
is supplied by a future calibration workflow.
"""

from __future__ import annotations

import csv
import json
import random
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from statistics import fmean, median
from typing import Any, Iterable, Sequence

from .advanced_schema import AdvancedExecutionConfig, AdvancedScenario
from .advanced_simulator import AdvancedSimulator
from .batch_predictor import BatchPredictorOutput, PrecomputedBatchPredictor, TokenPrediction
from .batch_routing import BatchRoutingEvent
from .hardware_profile import HardwareProfile
from .roofline_prefetch_dse import profile_with_effective_bandwidth
from .schema import CandidatePrefetch, ModelConfig, ValidationError


def controlled_routing(
    *, batch_size: int, top_k: int, num_experts: int, pattern: str,
    active_experts: int | None = None, seed: int = 0,
) -> BatchRoutingEvent:
    """Create deterministic routes where reuse is controlled independently of batch.

    ``max_reuse`` uses the same Top-K experts for every token; ``low_reuse``
    walks expert IDs until exhausted; ``balanced_reuse`` distributes assignments
    across ``active_experts``; ``random_uniform`` and ``zipf_hotspot`` are seeded.
    """
    if batch_size <= 0 or not 0 < top_k <= num_experts:
        raise ValidationError("invalid controlled routing dimensions")
    active = min(num_experts, max(top_k, active_experts or num_experts))
    rng = random.Random(seed)
    routes: list[tuple[int, ...]] = []
    if pattern == "max_reuse":
        routes = [tuple(range(top_k)) for _ in range(batch_size)]
    elif pattern in {"low_reuse", "balanced_reuse"}:
        # Cyclic assignment is exactly balanced (within one) and low_reuse is
        # simply the maximum legal active-expert count.
        pool = num_experts if pattern == "low_reuse" else active
        for token in range(batch_size):
            start = (token * top_k) % pool
            routes.append(tuple((start + offset) % pool for offset in range(top_k)))
    elif pattern == "random_uniform":
        routes = [tuple(sorted(rng.sample(range(active), top_k))) for _ in range(batch_size)]
    elif pattern == "zipf_hotspot":
        weights = [1.0 / (index + 1) for index in range(active)]
        for _ in range(batch_size):
            chosen: set[int] = set()
            while len(chosen) < top_k:
                chosen.add(rng.choices(range(active), weights=weights, k=1)[0])
            routes.append(tuple(sorted(chosen)))
    else:
        raise ValidationError(f"unsupported routing pattern {pattern!r}")
    return BatchRoutingEvent(batch_size * 1000 + top_k * 10 + seed, 0, 0, tuple(routes))


def synthetic_predictor(
    event: BatchRoutingEvent, num_experts: int, *, quality: str = "medium",
    ready_cycle: int = 0, seed: int = 0,
) -> PrecomputedBatchPredictor:
    """Make one deterministic shared predictor trace for policy comparisons."""
    rng = random.Random(seed)
    probability = {"low": 0.25, "medium": 0.60, "high": 0.90}[quality]
    output: list[TokenPrediction] = []
    for token, route in enumerate(event.routes):
        hits = [expert for expert in route if rng.random() < probability]
        false = [expert for expert in range(num_experts) if expert not in route]
        rng.shuffle(false)
        candidates = hits + false[:max(0, 8 * event.top_k - len(hits))]
        # Retain missed true experts at lower rank so broad M sweeps are meaningful.
        candidates.extend(expert for expert in route if expert not in hits)
        unique = list(dict.fromkeys(candidates))
        output.append(TokenPrediction(token, tuple(
            CandidatePrefetch(expert, 1.0 - rank / max(1, len(unique)))
            for rank, expert in enumerate(unique)
        )))
    key = (event.request_id, event.token_position, event.layer_id)
    return PrecomputedBatchPredictor({key: BatchPredictorOutput(
        event.request_id, event.token_position, event.layer_id, ready_cycle,
        tuple(output), "max",
    )})


def controlled_predictor(
    event: BatchRoutingEvent, num_experts: int, *, profile: str,
    candidate_count: int, ready_cycle: int, seed: int = 0,
) -> tuple[PrecomputedBatchPredictor, str, float, bool, bool]:
    """Create a ranked, seed-stable predictor trace with controlled hit coverage.

    The returned expected metrics are computed from the actual aggregate trace,
    rather than inferred from a nominal probability.  This makes policy studies
    compare the same trace identity across full/prefix/DSE scenarios.
    """
    targets = {"perfect": 1.0, "high": 0.85, "medium": 0.60, "low": 0.30, "wrong": 0.0}
    if profile not in targets or not 1 <= candidate_count <= num_experts:
        raise ValidationError("invalid controlled predictor profile or candidate count")
    rng = random.Random(seed)
    true_ids = sorted(event.expert_token_counts())
    hit_count = min(len(true_ids), round(targets[profile] * len(true_ids)))
    # Rotate deterministic true selections so low/medium are not always E0.
    true_order = sorted(true_ids, key=lambda expert: ((expert * 17 + seed) % max(1, len(true_ids)), expert))
    hits = true_order[:hit_count]
    false = [expert for expert in range(num_experts) if expert not in true_ids]
    rng.shuffle(false)
    ranked = hits + false[:max(0, candidate_count - len(hits))]
    if len(ranked) < candidate_count:
        ranked.extend(expert for expert in true_order if expert not in ranked)
    ranked = ranked[:candidate_count]
    # Repeating the aggregate ranking per token ensures that max aggregation has
    # exactly the requested ranked expert identity independent of batch shape.
    predictions = tuple(TokenPrediction(token_id, tuple(
        CandidatePrefetch(expert, float(candidate_count - rank))
        for rank, expert in enumerate(ranked)
    )) for token_id in range(event.batch_size))
    output = BatchPredictorOutput(event.request_id, event.token_position, event.layer_id,
                                  ready_cycle, predictions, "max")
    aggregate = output.aggregate_candidates(candidate_count)
    observed = {candidate.expert_id for candidate in aggregate}
    recall = len(observed & set(true_ids)) / len(true_ids)
    trace_id = f"controlled:{profile}:seed={seed}:event={event.request_id}:M={candidate_count}:ready={ready_cycle}"
    return PrecomputedBatchPredictor({(event.request_id, event.token_position, event.layer_id): output}), trace_id, recall, bool(observed & set(true_ids)), set(true_ids) <= observed


def rank_controlled_predictor(event: BatchRoutingEvent, num_experts: int, *,
                              true_ranks_one_based: Sequence[int], ready_cycle: int = 0,
                              seed: int = 0) -> tuple[PrecomputedBatchPredictor, tuple[int, ...], str]:
    """Build one full ranking with true experts at specified one-based ranks."""
    truths = sorted(event.expert_token_counts())
    if len(truths) != len(true_ranks_one_based) or min(true_ranks_one_based, default=1) < 1:
        raise ValidationError("true rank vector must match active experts and be positive")
    if max(true_ranks_one_based, default=0) > num_experts:
        raise ValidationError("true rank exceeds predictor ranking length")
    ranking: list[int | None] = [None] * num_experts
    for expert, rank in zip(truths, true_ranks_one_based):
        if ranking[rank - 1] is not None:
            raise ValidationError("true ranks must be unique")
        ranking[rank - 1] = expert
    false = [expert for expert in range(num_experts) if expert not in truths]
    random.Random(seed).shuffle(false)
    iterator = iter(false)
    resolved = tuple(item if item is not None else next(iterator) for item in ranking)
    output = BatchPredictorOutput(event.request_id, event.token_position, event.layer_id, ready_cycle,
        tuple(TokenPrediction(token, tuple(CandidatePrefetch(expert, float(len(resolved)-rank)) for rank, expert in enumerate(resolved))) for token in range(event.batch_size)), "max")
    return PrecomputedBatchPredictor({(event.request_id,event.token_position,event.layer_id):output}), resolved, f"rank-controlled:seed={seed}:ranks={tuple(true_ranks_one_based)}"


def _row(
    result: Any, *, experiment_name: str, run_id: str, seed: int,
    model: ModelConfig, profile: HardwareProfile, config: AdvancedExecutionConfig,
    routing_pattern: str, predictor_profile: str = "none",
    allocated_speculative_bytes: int | None = None,
) -> dict[str, Any]:
    """Flatten a simulator result into the stable characterization data schema."""
    memory = result.memory_metrics
    aie = result.aie_metrics
    workload = result.workload
    allocated = allocated_speculative_bytes
    if allocated is None:
        allocated = (
            config.speculative_byte_budget
            if config.speculative_byte_budget is not None and "predictive" in result.scenario
            else 0
        )
    starts = [record.compute_start_time for record in result.expert_records if record.compute_start_time is not None]
    channels = tuple(memory["bytes_per_channel"])
    mean_channel = fmean(channels) if channels else 0.0
    demand_entries = [entry for entry in result.timeline if getattr(entry, "resource", "") == "MEMORY" and getattr(entry, "request_type", "") == "demand"]
    first_by_expert: dict[int, int] = {}
    last_by_expert: dict[int, int] = {}
    channels_by_expert: dict[int, set[int]] = {}
    for entry in demand_entries:
        expert = getattr(entry, "expert_id", None)
        if expert is None:
            continue
        first_by_expert[expert] = min(first_by_expert.get(expert, entry.start_cycle), entry.start_cycle)
        last_by_expert[expert] = max(last_by_expert.get(expert, entry.end_cycle), entry.end_cycle)
        channels_by_expert.setdefault(expert, set()).add(expert % profile.num_memory_channels)
    wave_starts = sorted(set(first_by_expert.values()))
    memory_waves = [
        {"wave_id": index, "start_cycle": start,
         "participating_experts": sorted(expert for expert, first in first_by_expert.items() if first == start),
         "participating_channels": sorted({channel for expert, first in first_by_expert.items() if first == start for channel in channels_by_expert[expert]}),
         "end_cycle": max(last_by_expert[expert] for expert, first in first_by_expert.items() if first == start)}
        for index, start in enumerate(wave_starts)
    ]
    return {
        "experiment_name": experiment_name, "run_id": run_id, "seed": seed,
        "hardware_input_class": "SYNTHETIC / THEORETICAL INPUT",
        "E": model.num_experts, "H": model.hidden_size, "F": model.intermediate_size,
        "weight_bits": model.weight_bits, "batch_size": workload.batch_size,
        "top_k": workload.router_top_k, "routing_pattern": routing_pattern,
        "token_expert_assignments": workload.token_expert_assignments,
        "unique_active_experts": workload.unique_active_experts,
        "reuse_factor": workload.reuse_factor,
        "mean_tokens_per_expert": workload.mean_tokens_per_expert,
        "median_tokens_per_expert": workload.median_tokens_per_expert,
        "max_tokens_per_expert": workload.max_tokens_per_expert,
        "memory_channels": profile.num_memory_channels,
        "peak_bandwidth_gbps": profile.theoretical_memory_bandwidth_bytes_per_sec / 1e9,
        "effective_bandwidth_gbps": profile.effective_memory_bandwidth_bytes_per_sec / 1e9,
        "memory_placement_policy": config.placement_policy,
        "total_aie_compute_groups": profile.total_aie_units,
        "aie_groups_per_expert": config.aie_units_per_expert,
        "max_parallel_experts": config.max_parallel_experts,
        "OCM_bytes": profile.ocm_bytes, "equivalent_full_expert_capacity": profile.ocm_bytes / max(1, result.expert_records[0].weight_bytes),
        "F_tile_size": config.f_tile_size, "number_of_tiles": (model.intermediate_size + config.f_tile_size - 1) // config.f_tile_size,
        "predictor_profile": predictor_profile, "candidate_count": config.candidate_count,
        "depth_policy": result.scenario, "allocated_speculative_bytes": allocated,
        "scenario": result.scenario, "makespan_cycles": result.layer_makespan_cycles,
        "first_true_expert_start": result.first_true_expert_start,
        "last_true_expert_start": result.last_true_expert_start,
        "router_resolve_time": result.router_resolve_cycle,
        "prediction_ready_time": result.prediction_ready_time,
        "prefetch_window_cycles": result.prefetch_window_cycles,
        "router_to_first_true_expert_start": result.router_resolve_to_first_expert_start,
        "transferred_bytes": memory["transferred_bytes"], "demand_bytes": memory["demand_bytes"],
        "speculative_bytes": memory["speculative_bytes"], "issued_speculative_bytes": memory["issued_speculative_bytes"],
        "useful_speculative_bytes": memory["useful_speculative_bytes"], "wasted_speculative_bytes": memory["wasted_speculative_bytes"],
        "canceled_speculative_bytes": memory["canceled_speculative_bytes"],
        "memory_utilization": memory["physical_bandwidth_utilization"],
        "achieved_useful_bandwidth_gbps": memory["achieved_useful_bandwidth_bytes_per_sec"] / 1e9,
        "aie_utilization": aie["board_aie_utilization"], "aie_starvation_cycles": result.aie_starvation_cycles,
        "allocated_aie_cycles": aie["allocated_aie_cycles"], "useful_aie_cycles": aie["useful_aie_cycles"],
        "average_active_aie_groups": aie["average_active_aie_units"],
        "arithmetic_intensity": result.roofline.arithmetic_intensity_ops_per_byte,
        "board_ridge_ai": result.roofline.board_ridge_ai, "configured_ridge_ai": result.roofline.configured_ridge_ai,
        "board_regime": result.roofline.board_regime, "configured_regime": result.roofline.configured_regime,
        "board_roofline_efficiency": result.roofline.board_roofline_efficiency,
        "configured_roofline_efficiency": result.roofline.configured_roofline_efficiency,
        "StartableRecall": result.startable_recall, "WeightedStartableRecall": result.weighted_startable_recall,
        "AnyStartableHit": result.any_startable_hit, "FullStartableHit": result.full_startable_hit,
        "demand_blocking_cycles": result.demand_blocked_cycles,
        "transferred_bytes_during_prefetch_window": result.bytes_transferred_during_prefetch_window,
        "useful_bytes_during_prefetch_window": result.useful_bytes_transferred_during_prefetch_window,
        "false_bytes_during_prefetch_window": max(0, result.bytes_transferred_during_prefetch_window - result.useful_bytes_transferred_during_prefetch_window),
        "memory_idle_cycles_during_prefetch_window": result.memory_idle_cycles_during_prefetch_window,
        "physical_prefetch_window_utilization": result.prefetch_window_utilization,
        "useful_prefetch_window_utilization": result.useful_prefetch_window_utilization,
        "peak_ocm_bytes": result.peak_ocm_bytes, "average_ocm_bytes": result.average_ocm_bytes,
        "channel_imbalance": max(channels) / mean_channel if mean_channel else 0.0,
        "per_channel_bytes": json.dumps(channels), "per_channel_utilization": json.dumps(memory["per_channel_utilization"]),
        "dma_request_count": len([entry for entry in result.timeline if getattr(entry, "resource", "") == "MEMORY"]),
        "setup_overhead_cycles": len([entry for entry in result.timeline if getattr(entry, "resource", "") == "MEMORY"]) * profile.dma_setup_cycles,
        "overlap_cycles": result.dma_compute_overlap_cycles,
        "expert_execution_wave_count": max(1, (len(result.expert_records) + config.max_parallel_experts - 1) // config.max_parallel_experts),
        "peak_parallel_experts": min(config.max_parallel_experts, len(result.expert_records)),
        "average_parallel_experts": len(result.expert_records) / max(1, (len(result.expert_records) + config.max_parallel_experts - 1) // config.max_parallel_experts),
        "cycles_blocked_by_aie_capacity": 0, "cycles_blocked_by_memory_channel": result.aie_starvation_cycles,
        "cycles_blocked_by_ocm_capacity": result.rejected_prefetch_bytes and result.aie_starvation_cycles or 0,
        "number_of_demand_DMA_requests": len([entry for entry in result.timeline if getattr(entry, "resource", "") == "MEMORY" and getattr(entry, "request_type", "") == "demand"]),
        "number_of_speculative_DMA_requests": len([entry for entry in result.timeline if getattr(entry, "resource", "") == "MEMORY" and getattr(entry, "request_type", "") == "speculative"]),
        "first_start_latency": min(starts) - result.router_resolve_cycle if starts else 0,
        "memory_transfer_wave_count": len(memory_waves),
        "memory_wave_records": json.dumps(memory_waves),
        "mean_startup_readiness": result.mean_startup_readiness,
        "weighted_startup_readiness": result.weighted_startup_readiness,
        "total_startup_gap_bytes": result.total_startup_gap_bytes,
    }


def simulate_row(*, experiment_name: str, run_id: str, model: ModelConfig,
                 profile: HardwareProfile, event: BatchRoutingEvent,
                 scenario: AdvancedScenario, config: AdvancedExecutionConfig,
                 routing_pattern: str, seed: int = 0, predictor_quality: str = "medium",
                 ready_cycle: int = 0) -> dict[str, Any]:
    """Run one event DAG point and return its reproducibility-complete row."""
    predictive = "predictive" in scenario.value or scenario is AdvancedScenario.ORACLE_OPTIMAL_DEPTH
    predictor = synthetic_predictor(event, model.num_experts, quality=predictor_quality,
                                    ready_cycle=ready_cycle, seed=seed) if predictive else None
    result = AdvancedSimulator(model, profile).simulate(event, scenario, config, predictor=predictor)
    return _row(result, experiment_name=experiment_name, run_id=run_id, seed=seed,
                model=model, profile=profile, config=config, routing_pattern=routing_pattern,
                predictor_profile=predictor_quality if predictive else "none")


def write_rows(rows: Iterable[dict[str, Any]], path: Path) -> list[dict[str, Any]]:
    """Write homogeneous CSV rows in deterministic first-seen column order."""
    materialized = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in materialized for key in row))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(materialized)
    return materialized


def sanity_rows(model: ModelConfig, profile: HardwareProfile) -> list[dict[str, Any]]:
    """Run tiny invariant cases; raise clearly if foundational accounting fails."""
    rows = []
    for batch in (1, 2):
        event = controlled_routing(batch_size=batch, top_k=1, num_experts=model.num_experts, pattern="max_reuse")
        config = AdvancedExecutionConfig(min(64, model.intermediate_size), aie_units_per_expert=1)
        rows.append(simulate_row(experiment_name="sanity", run_id=f"reuse-{batch}", model=model, profile=profile,
                                 event=event, scenario=AdvancedScenario.EXACT_TILE_STREAMING, config=config,
                                 routing_pattern="max_reuse"))
    if rows[1]["demand_bytes"] != rows[0]["demand_bytes"] or rows[1]["arithmetic_intensity"] <= rows[0]["arithmetic_intensity"]:
        raise RuntimeError("sanity failure: reuse must increase AI without duplicating expert weights")
    slow = replace(profile, bandwidth_bytes_per_sec_per_channel=profile.bandwidth_bytes_per_sec_per_channel / 2)
    fast = replace(profile, bandwidth_bytes_per_sec_per_channel=profile.bandwidth_bytes_per_sec_per_channel * 2)
    event = controlled_routing(batch_size=1, top_k=1, num_experts=model.num_experts, pattern="max_reuse")
    cfg = AdvancedExecutionConfig(min(64, model.intermediate_size), aie_units_per_expert=1)
    slow_row = simulate_row(experiment_name="sanity", run_id="bandwidth-slow", model=model, profile=slow, event=event, scenario=AdvancedScenario.EXACT_TILE_STREAMING, config=cfg, routing_pattern="max_reuse")
    fast_row = simulate_row(experiment_name="sanity", run_id="bandwidth-fast", model=model, profile=fast, event=event, scenario=AdvancedScenario.EXACT_TILE_STREAMING, config=cfg, routing_pattern="max_reuse")
    if fast_row["makespan_cycles"] > slow_row["makespan_cycles"]:
        raise RuntimeError("sanity failure: increasing memory bandwidth increased makespan")
    roomy = replace(profile, ocm_bytes=max(profile.ocm_bytes, 64 * 1024 * 1024))
    full = simulate_row(experiment_name="sanity", run_id="full-load-bytes", model=model, profile=roomy,
                        event=event, scenario=AdvancedScenario.EXACT_FULL_EXPERT_LOAD, config=cfg,
                        routing_pattern="max_reuse")
    stream = simulate_row(experiment_name="sanity", run_id="stream-bytes", model=model, profile=roomy,
                          event=event, scenario=AdvancedScenario.EXACT_TILE_STREAMING, config=cfg,
                          routing_pattern="max_reuse")
    if full["demand_bytes"] != stream["demand_bytes"]:
        raise RuntimeError("sanity failure: full load and streaming demand bytes differ")
    more_groups = simulate_row(experiment_name="sanity", run_id="more-aie-groups", model=model, profile=roomy,
                               event=event, scenario=AdvancedScenario.EXACT_TILE_STREAMING,
                               config=replace(cfg, aie_units_per_expert=min(2, profile.total_aie_units)),
                               routing_pattern="max_reuse")
    if more_groups["allocated_aie_cycles"] / 2 > stream["allocated_aie_cycles"]:
        raise RuntimeError("sanity failure: more AIE groups increased pure compute-task duration")
    return rows + [slow_row, fast_row, full, stream, more_groups]
