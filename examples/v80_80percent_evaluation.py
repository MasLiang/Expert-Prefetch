"""Evaluate V80 80-percent resource envelopes with the existing event engine."""
from __future__ import annotations

import json
import sys
from dataclasses import replace
from pathlib import Path
from statistics import fmean
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "examples")]

from edge_moe_sim.advanced_schema import AdvancedScenario
from edge_moe_sim.advanced_simulator import AdvancedSimulator
from edge_moe_sim.batch_routing import BatchRoutingEvent
from edge_moe_sim.hardware_profile import HardwareProfile
from edge_moe_sim.predictor_sensitivity import build_sensitivity_trace
from edge_moe_sim.tile_scheduler import scheduler_by_name
from m91_realistic_evaluation import WORKLOAD_FILES, execution_config, load_workload_profile, write_csv


def main() -> None:
    """Run independent resource and Router-mapping sensitivities, then save results."""
    config_path = ROOT / "configs/hardware/v80_80percent.yaml"
    cfg = yaml.safe_load(config_path.read_text())["resource_study"]
    fraction = cfg["resource_fraction"]
    dsp_budget = int(cfg["dsp_engines"] * fraction)
    # Two equal Expert engines use the entire integer DSP budget.
    assert dsp_budget % 2 == 0
    capacities = {
        "bram_uram": int((cfg["block_ram_megabits"] + cfg["ultraram_megabits"])
                         * cfg["memory_megabit_multiplier"] / 8 * fraction),
        "total_pl_upper_envelope": int(cfg["total_pl_memory_megabits"]
                                      * cfg["memory_megabit_multiplier"] / 8 * fraction),
    }
    rows: list[dict[str, Any]] = []
    for clock in cfg["clocks_hz"]:
        for duty in cfg["compute_duty_factors"]:
            for memory_mode, capacity in capacities.items():
                profile = HardwareProfile(
                    name="v80_80percent_dsp_streaming",
                    clock_hz=clock,
                    num_memory_channels=cfg["abstract_dma_channels"],
                    bandwidth_bytes_per_sec_per_channel=(
                        cfg["hbm_bandwidth_bytes_per_sec"] / cfg["abstract_dma_channels"]
                    ),
                    bandwidth_efficiency=fraction,
                    dma_setup_cycles=cfg["dma_setup_cycles"],
                    max_inflight_per_channel=1,
                    burst_bytes=4096,
                    supports_channel_striping=True,
                    total_aie_units=2,
                    ops_per_cycle_per_aie=(dsp_budget // 2
                                          * cfg["macs_per_dsp_per_cycle"]
                                          * cfg["operations_per_mac"]),
                    aie_efficiency=duty,
                    aie_task_startup_cycles=cfg["compute_startup_cycles"],
                    ocm_bytes=capacity,
                    hardware_tag="THEORETICAL_80_PERCENT_RESOURCE_BUDGET",
                )
                for workload_file in WORKLOAD_FILES:
                    workload = load_workload_profile(workload_file)
                    event = BatchRoutingEvent(8000, 0, 0, ((0, 1),))
                    for router_groups in cfg["router_compute_groups"]:
                        execution = replace(
                            execution_config(workload, {
                                "dma_mapping": "round_robin_expert",
                                "aie_units_per_expert": 1,
                            }),
                            router_aie_units=router_groups,
                        )
                        simulator = AdvancedSimulator(workload.model, profile)
                        ideal = simulator.simulate(event, AdvancedScenario.IDEAL, execution)
                        for quality in (1.0, .8, .6, .4, .2):
                            trace = build_sensitivity_trace(
                                request_id=8000, token_position=0, layer_id=0,
                                num_experts=workload.model.num_experts, top_k=2,
                                candidate_count=workload.candidate_count,
                                quality=quality, calibration_mode="well_calibrated",
                            )
                            results = {}
                            for policy in ("no_prefetch", "full_expert", "budget_proportional",
                                           "hardware_greedy", "ranked_tile"):
                                results[policy] = simulator.simulate(
                                    event, AdvancedScenario.PREDICTIVE_DSE, execution,
                                    predictor=trace.predictor,
                                    tile_prefetch_scheduler=scheduler_by_name(policy),
                                )
                            baseline = results["no_prefetch"]
                            for policy, result in results.items():
                                metrics = result.tile_metrics
                                assert metrics["allocated_transfer_bytes"] == (
                                    metrics["useful_prefetched_bytes"]
                                    + metrics["wasted_prefetched_bytes"]
                                    + metrics["cancelled_transfer_bytes"]
                                )
                                assert result.peak_ocm_bytes <= capacity
                                rows.append({
                                    "clock_hz": clock, "compute_duty_factor": duty,
                                    "dsp_budget": dsp_budget, "memory_mode": memory_mode,
                                    "ocm_bytes": capacity, "router_compute_groups": router_groups,
                                    "effective_compute_TOPS": profile.effective_compute_peak_ops_per_sec / 1e12,
                                    "effective_hbm_GB_per_s": profile.effective_memory_bandwidth_bytes_per_sec / 1e9,
                                    "workload": workload.name, "expert_bytes": workload.expert_bytes,
                                    "quality": quality, "scheduler": policy,
                                    "latency_cycles": result.layer_makespan_cycles,
                                    "latency_us": result.layer_makespan_cycles / clock * 1e6,
                                    "baseline_us": baseline.layer_makespan_cycles / clock * 1e6,
                                    "ideal_us": ideal.layer_makespan_cycles / clock * 1e6,
                                    "speedup": baseline.layer_makespan_cycles / result.layer_makespan_cycles,
                                    "latency_reduction_percent": (1 - result.layer_makespan_cycles / baseline.layer_makespan_cycles) * 100,
                                    "memory_stall_reduction_cycles": baseline.aie_starvation_cycles - result.aie_starvation_cycles,
                                    "allocated_bytes": metrics["allocated_transfer_bytes"],
                                    "useful_bytes": metrics["useful_prefetched_bytes"],
                                    "wasted_bytes": metrics["wasted_prefetched_bytes"],
                                    "peak_ocm_bytes": result.peak_ocm_bytes,
                                })
    output = ROOT / "results/v80_80percent"
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "evaluation.csv", rows)
    summary = {
        "specifications_and_assumptions": cfg,
        "dsp_budget": dsp_budget, "ocm_capacities_bytes": capacities,
        "effective_bandwidth_GB_per_s": cfg["hbm_bandwidth_bytes_per_sec"] * fraction / 1e9,
        "rows": len(rows),
        "by_workload": {
            workload: {
                "mean_tile_speedup": fmean(row["speedup"] for row in rows
                    if row["workload"] == workload and row["scheduler"] in
                    ("budget_proportional", "hardware_greedy", "ranked_tile")),
                "best_tile_speedup": max(row["speedup"] for row in rows
                    if row["workload"] == workload and row["scheduler"] in
                    ("budget_proportional", "hardware_greedy", "ranked_tile")),
            } for workload in ("small_edge_moe", "medium_moe", "large_moe")
        },
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
