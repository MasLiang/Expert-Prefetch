"""Resident-Router V80 study with two-layer predictions and a weight barrier."""
from __future__ import annotations

import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "examples")]

from edge_moe_sim.advanced_schema import AdvancedExecutionConfig
from edge_moe_sim.batch_routing import BatchRoutingEvent
from edge_moe_sim.hardware_profile import HardwareProfile
from edge_moe_sim.predictor_sensitivity import build_sensitivity_trace
from edge_moe_sim.unified_schema import UnifiedLayerSequence, UnifiedPredictionSpec
from edge_moe_sim.unified_sequence_simulator import UnifiedSequenceSimulator
from m91_realistic_evaluation import WORKLOAD_FILES, load_workload_profile, write_csv


def run_case(workload: Any, gap_us: int, latency_us: int, mode: str,
             policy: str, quality: float) -> tuple[Any, Any, int]:
    """Run four layers; predictions cover only layers two and three in every mode."""
    cfg = yaml.safe_load((ROOT / "configs/hardware/v80_80percent.yaml").read_text())["resource_study"]
    clock = 300_000_000
    dsps = int(cfg["dsp_engines"] * cfg["resource_fraction"])
    # Reserve Router weights for all MoE layers in the declared workload.
    router_bytes = workload.num_moe_layers * ((
        workload.model.hidden_size * workload.model.num_experts * workload.model.weight_bits + 7
    ) // 8)
    ocm = int((cfg["block_ram_megabits"] + cfg["ultraram_megabits"])
              * cfg["memory_megabit_multiplier"] / 8 * cfg["resource_fraction"])
    profile = HardwareProfile(
        "v80_80percent_resident_router", clock, 8,
        cfg["hbm_bandwidth_bytes_per_sec"] / 8, .8, 80, 1, 4096,
        True, 2, dsps, 1.0, 30, ocm - router_bytes,
        hardware_tag="THEORETICAL_80_PERCENT_RESOURCE_BUDGET",
    )
    config = AdvancedExecutionConfig(
        workload.f_tile_size, aie_units_per_expert=1, max_parallel_experts=2,
        router_aie_units=2, candidate_count=workload.candidate_count,
        # At most two predictions can coexist. Keep half the weight OCM free
        # for current-layer demand and streaming buffers.
        speculative_byte_budget=profile.ocm_bytes // 4,
    )
    events = tuple(BatchRoutingEvent(8200, 0, layer, ((0, 1),)) for layer in range(4))
    sequence = UnifiedLayerSequence(
        events, {(layer, layer + 1): gap_us * clock // 1_000_000 for layer in range(3)}
    )
    specs = []
    predictors = {}
    if mode != "no_prefetch":
        for target in (2, 3):
            source = target - (1 if mode == "one_layer_gated" else 2)
            spec = UnifiedPredictionSpec(
                source, target, "source_layer_start",
                predictor_latency_cycles=latency_us * clock // 1_000_000,
                candidate_count=workload.candidate_count,
                prefetch_policy=policy,
                prefix_depth=1 if policy == "uniform_prefix" else 4,
                wait_for_previous_layer_weights=mode != "two_layer_immediate",
            )
            trace = build_sensitivity_trace(
                request_id=8200, token_position=0, layer_id=target,
                num_experts=workload.model.num_experts, top_k=2,
                candidate_count=workload.candidate_count, quality=quality,
                calibration_mode="well_calibrated",
            )
            specs.append(spec)
            predictors[(source, target)] = trace.predictor
    result = UnifiedSequenceSimulator(workload.model, profile).simulate(
        sequence, config, predictions=tuple(specs), predictors=predictors,
        trace_enabled=True,
    )
    assert result.peak_ocm_bytes <= profile.ocm_bytes
    for record in result.speculative_request_records:
        if mode.endswith("gated") and record["issue_cycle"] is not None:
            previous = int(record["target_layer"]) - 1
            weight_ends = [int(x["cycle"]) for x in result.trace
                           if x.get("event_type") == "dma_done" and x.get("layer_id") == previous]
            assert int(record["issue_cycle"]) >= max(weight_ends)
    return profile, result, router_bytes


def main() -> None:
    """Save sequence/layer timings and representative causal traces."""
    out = ROOT / "results/v80_two_layer_prefetch"
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    timing_rows = []
    for path in WORKLOAD_FILES:
        workload = load_workload_profile(path)
        for gap in (0, 10):
            _, baseline, _ = run_case(workload, gap, 0, "no_prefetch", "uniform_prefix", 1.0)
            baseline_stall = sum(x.aie_starvation_cycles for x in baseline.layer_results.values())
            for latency in (0, 1):
                for quality in (1.0, .8, .6, .4, .2):
                    for policy in ("uniform_prefix", "ranked_prefix", "full_expert_prefetch"):
                        for mode in ("one_layer_gated", "two_layer_gated", "two_layer_immediate"):
                            profile, result, router_bytes = run_case(workload, gap, latency, mode, policy, quality)
                            row = {
                                "workload": workload.name, "mode": mode, "policy": policy,
                                "quality": quality, "predictor_latency_us": latency,
                                "inter_layer_gap_us": gap, "clock_hz": profile.clock_hz,
                                "router_resident_bytes": router_bytes,
                                "weight_ocm_bytes": profile.ocm_bytes,
                                "sequence_us": result.makespan_cycles / profile.clock_hz * 1e6,
                                "baseline_sequence_us": baseline.makespan_cycles / profile.clock_hz * 1e6,
                                "speedup": baseline.makespan_cycles / result.makespan_cycles,
                                "latency_reduction_percent": (1-result.makespan_cycles/baseline.makespan_cycles)*100,
                                "weight_wait_reduction_cycles": baseline_stall - sum(x.aie_starvation_cycles for x in result.layer_results.values()),
                                "useful_bytes": sum(x.useful_bytes for x in result.predictions),
                                "wasted_bytes": sum(x.wasted_bytes for x in result.predictions),
                                "rejected_bytes": result.rejected_future_prefetch_bytes,
                                "blocked_demand_cycles": result.demand_blocked_by_future_speculation_cycles,
                                "peak_weight_ocm_bytes": result.peak_ocm_bytes,
                            }
                            rows.append(row)
                            for prediction in result.predictions:
                                previous = prediction.target_layer_id - 1
                                weight_done = max(int(x["cycle"]) for x in result.trace
                                                  if x.get("event_type") == "dma_done" and x.get("layer_id") == previous)
                                timing_rows.append({
                                    **row, "source_layer": prediction.source_layer_id,
                                    "target_layer": prediction.target_layer_id,
                                    "prediction_ready_cycle": prediction.predictor_ready_cycle,
                                    "previous_weights_done_cycle": weight_done,
                                    "prefetch_release_cycle": prediction.prefetch_release_cycle,
                                    "target_start_cycle": prediction.target_layer_start_cycle,
                                    "target_resolve_cycle": prediction.target_router_resolve_cycle,
                                    "ready_recall": result.layer_results[prediction.target_layer_id].startable_recall,
                                })
                            if quality == 1 and latency == 0 and policy == "uniform_prefix" and mode == "two_layer_gated":
                                stem = f"{workload.name}_gap{gap}"
                                (out / f"{stem}_trace.json").write_text(json.dumps(result.trace, indent=2) + "\n")
                                (out / f"{stem}_predictions.json").write_text(json.dumps([asdict(x) for x in result.predictions], indent=2) + "\n")
        print(f"completed {workload.name}", flush=True)
    write_csv(out / "comparison.csv", rows)
    write_csv(out / "prediction_timing.csv", timing_rows)
    grouped = {}
    for row in rows:
        key = (row["workload"], row["policy"], row["quality"],
               row["inter_layer_gap_us"], row["predictor_latency_us"])
        grouped.setdefault(key, {})[row["mode"]] = row
    summary = {
        "sequence_rows": len(rows), "prediction_timing_rows": len(timing_rows),
        "simulated_layers": 4, "predicted_target_layers": [2, 3],
        "router_weights_resident": True,
        "gate": "all actual predecessor true GU and Down DMA completed",
        "per_prediction_budget_fraction": 0.25,
        "maximum_one_vs_two_layer_gated_latency_difference_us": max(
            abs(group["one_layer_gated"]["sequence_us"]
                - group["two_layer_gated"]["sequence_us"])
            for group in grouped.values()),
        "best_gated_cases": {
            name: max((row for row in rows if row["mode"] == "two_layer_gated"
                       and row["workload"] == name), key=lambda row: row["speedup"])
            for name in ("small_edge_moe", "medium_moe", "large_moe")
        },
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"Wrote {len(rows)} sequence runs, {len(timing_rows)} prediction timings")


if __name__ == "__main__":
    main()
