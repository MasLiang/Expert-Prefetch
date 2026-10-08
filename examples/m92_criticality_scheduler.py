"""M9.2 critical-path-aware scheduling and fixed-prediction Oracle study."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from statistics import fmean
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "examples")]

from edge_moe_sim.tile_scheduler import (
    CriticalityAwareTileScheduler,
    FixedPredictionOracleScheduler,
    FixedPrefetchPlanScheduler,
    PrefetchPlan,
    scheduler_by_name,
)
from m91_realistic_evaluation import (
    HARDWARE_FILES,
    OCM_SCALES,
    QUALITIES,
    WORKLOAD_FILES,
    flatten,
    load_hardware,
    load_workload_profile,
    planning_inputs,
    scaled_hardware,
    search_oracle,
    simulate,
    write_csv,
)

OUT = ROOT / "results" / "m92_criticality_scheduler"
ONLINE = (
    "no_prefetch", "full_expert", "uniform_tile", "ranked_tile",
    "budget_proportional", "hardware_greedy", "criticality_aware",
)
ALL_SCHEDULERS = ONLINE + ("fixed_prediction_oracle", "full_oracle")
BANDWIDTH_SCALE = 1.0


def _fixed_scheduler(plan: PrefetchPlan, name: str) -> FixedPrefetchPlanScheduler:
    """Convert an analyzed plan into normal event-engine replay form."""
    order = tuple((item.tile.expert_id, item.tile.tile_id) for item in plan.transfers)
    return FixedPrefetchPlanScheduler(order, name)


def search_fixed_prediction_oracle(
    workload: Any,
    base_hardware: Any,
    execution: dict[str, Any],
    *,
    quality: str,
    ocm_scale: float,
) -> tuple[PrefetchPlan, int, int]:
    """Optimize rank-preserving allocations for one unchanged predictor output."""
    prediction, geometry, hardware, timeline = planning_inputs(
        workload,
        base_hardware,
        quality=quality,
        ocm_scale=ocm_scale,
        bandwidth_scale=BANDWIDTH_SCALE,
    )
    online_plans = [
        (
            CriticalityAwareTileScheduler()
            if name == "criticality_aware" else scheduler_by_name(name)
        ).schedule(prediction, geometry, hardware, timeline)
        for name in ONLINE
    ]

    def evaluate(plan: PrefetchPlan) -> int:
        _, result = simulate(
            workload,
            base_hardware,
            execution,
            scheduler=_fixed_scheduler(plan, "fixed_prediction_candidate"),
            quality=quality,
            ocm_scale=ocm_scale,
            bandwidth_scale=BANDWIDTH_SCALE,
        )
        return result.layer_makespan_cycles

    result = FixedPredictionOracleScheduler().optimize(
        prediction,
        geometry,
        hardware,
        timeline,
        evaluate,
        extra_plans=online_plans,
    )
    return result.plan, result.latency_cycles, result.evaluated_plans


def _group_key(row: dict[str, Any]) -> tuple[Any, ...]:
    """Return one workload/hardware/quality/OCM configuration key."""
    return tuple(
        row[name]
        for name in ("workload", "hardware", "prediction_quality", "ocm_scale")
    )


def annotate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Add latency, traffic-efficiency, and signed/positive Oracle-gap metrics."""
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[_group_key(row)].append(row)
    decomposition: list[dict[str, Any]] = []
    for key, group in groups.items():
        by_name = {str(row["scheduler"]): row for row in group}
        baseline = int(by_name["no_prefetch"]["latency_cycles"])
        baseline_stall = int(by_name["no_prefetch"]["memory_stall_cycles"])
        fixed = int(by_name["fixed_prediction_oracle"]["latency_cycles"])
        full = int(by_name["full_oracle"]["latency_cycles"])
        if full > fixed:
            raise AssertionError("full-information Oracle is worse than fixed prediction")
        for row in group:
            latency = int(row["latency_cycles"])
            transferred = int(row["transferred_speculative_bytes"])
            tiles = int(row["allocated_tile_transfers"])
            latency_reduction = baseline - latency
            critical_path_reduction = baseline_stall - int(row["memory_stall_cycles"])
            row["speedup_vs_no_prefetch"] = baseline / latency
            row["latency_reduction_cycles"] = latency_reduction
            row["scheduler_efficiency_cycles_per_byte"] = (
                latency_reduction / transferred if transferred else 0.0
            )
            row["critical_path_reduction_cycles"] = critical_path_reduction
            row["critical_path_efficiency_cycles_per_tile"] = (
                critical_path_reduction / tiles if tiles else 0.0
            )
            # The first pair follows the exact signed formulas in the request.
            row["prediction_gap_cycles_full_minus_fixed"] = full - fixed
            row["allocation_gap_cycles_fixed_minus_scheduler"] = fixed - latency
            # Positive loss forms are easier to plot and sum to total Oracle loss.
            row["prediction_loss_cycles"] = fixed - full
            row["allocation_loss_cycles"] = latency - fixed
            row["total_loss_to_full_oracle_cycles"] = latency - full

        item: dict[str, Any] = {
            "workload": key[0],
            "hardware": key[1],
            "prediction_quality": key[2],
            "ocm_scale": key[3],
            "bandwidth_scale": BANDWIDTH_SCALE,
            "no_prefetch_latency_cycles": baseline,
            "full_oracle_latency_cycles": full,
            "fixed_prediction_oracle_latency_cycles": fixed,
            "prediction_gap_cycles_full_minus_fixed": full - fixed,
            "prediction_loss_cycles": fixed - full,
        }
        for name in ("budget_proportional", "hardware_greedy", "criticality_aware"):
            latency = int(by_name[name]["latency_cycles"])
            total = latency - full
            item[f"{name}_latency_cycles"] = latency
            item[f"{name}_allocation_gap_cycles_fixed_minus_scheduler"] = fixed - latency
            item[f"{name}_allocation_loss_cycles"] = latency - fixed
            item[f"{name}_prediction_loss_fraction"] = (
                (fixed - full) / total if total > 0 else 0.0
            )
            item[f"{name}_allocation_loss_fraction"] = (
                (latency - fixed) / total if total > 0 else 0.0
            )
        decomposition.append(item)
    return decomposition


def criticality_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Build one focused Criticality-Aware comparison row per configuration."""
    groups: dict[tuple[Any, ...], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        groups[_group_key(row)][str(row["scheduler"])] = row
    result = []
    for key, group in groups.items():
        critical = group["criticality_aware"]
        full_expert = group["full_expert"]
        result.append({
            "workload": key[0],
            "hardware": key[1],
            "prediction_quality": key[2],
            "ocm_scale": key[3],
            "expert_bytes": critical["expert_bytes"],
            "criticality_latency_cycles": critical["latency_cycles"],
            "criticality_speedup_vs_no_prefetch": critical["speedup_vs_no_prefetch"],
            "criticality_speedup_vs_full_expert": (
                int(full_expert["latency_cycles"]) / int(critical["latency_cycles"])
            ),
            "criticality_scheduler_efficiency_cycles_per_byte": (
                critical["scheduler_efficiency_cycles_per_byte"]
            ),
            "criticality_critical_path_efficiency_cycles_per_tile": (
                critical["critical_path_efficiency_cycles_per_tile"]
            ),
            "criticality_transferred_bytes": critical["transferred_speculative_bytes"],
            "criticality_allocated_tiles": critical["allocated_tile_transfers"],
            "improvement_vs_hardware_greedy_percent": (
                int(group["hardware_greedy"]["latency_cycles"])
                / int(critical["latency_cycles"]) - 1
            ) * 100,
            "improvement_vs_budget_proportional_percent": (
                int(group["budget_proportional"]["latency_cycles"])
                / int(critical["latency_cycles"]) - 1
            ) * 100,
            "prediction_loss_cycles": critical["prediction_loss_cycles"],
            "allocation_loss_cycles": critical["allocation_loss_cycles"],
        })
    return result


def make_figures(
    output: Path,
    rows: list[dict[str, Any]],
    decomposition: list[dict[str, Any]],
    focused: list[dict[str, Any]],
) -> list[str]:
    """Generate the five required M9.2 figures as PNG and PDF."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    written: list[str] = []

    def save(figure: Any, stem: str) -> None:
        for suffix in ("png", "pdf"):
            path = output / f"{stem}.{suffix}"
            figure.savefig(path, dpi=220)
            written.append(path.name)
        plt.close(figure)

    # Figure 1: positive loss decomposition, normalized by full Oracle latency.
    prediction = []
    allocation = []
    for quality in QUALITIES:
        subset = [item for item in decomposition if item["prediction_quality"] == quality]
        prediction.append(100 * fmean(
            int(item["prediction_loss_cycles"]) / int(item["full_oracle_latency_cycles"])
            for item in subset
        ))
        allocation.append(100 * fmean(
            int(item["criticality_aware_allocation_loss_cycles"])
            / int(item["full_oracle_latency_cycles"])
            for item in subset
        ))
    figure, axis = plt.subplots(figsize=(8, 4.8))
    axis.bar(QUALITIES, prediction, label="Prediction uncertainty", color="#377eb8")
    axis.bar(
        QUALITIES, allocation, bottom=prediction,
        label="Criticality allocation loss", color="#ff9800",
    )
    axis.set(
        ylabel="Mean latency loss / full Oracle latency (%)",
        title="Figure 1: Oracle gap decomposition",
    )
    axis.grid(axis="y", alpha=.25)
    axis.legend()
    figure.tight_layout()
    save(figure, "fig01_oracle_gap_decomposition")

    # Figure 2: normalized scheduler latency.
    labels = {
        "no_prefetch": "No Prefetch", "full_expert": "Full Expert",
        "budget_proportional": "Budget Proportional",
        "hardware_greedy": "Hardware Greedy",
        "criticality_aware": "Criticality Aware",
        "fixed_prediction_oracle": "Fixed-Prediction Oracle",
        "full_oracle": "Full Oracle",
    }
    shown = tuple(labels)
    normalized = [
        fmean(
            1 / float(row["speedup_vs_no_prefetch"])
            for row in rows if row["scheduler"] == scheduler
        )
        for scheduler in shown
    ]
    figure, axis = plt.subplots(figsize=(10, 4.8))
    axis.bar(range(len(shown)), normalized, color="#4c78a8")
    axis.set_xticks(range(len(shown)), tuple(labels[name] for name in shown), rotation=25, ha="right")
    axis.set(
        ylabel="Mean latency / no-prefetch latency",
        title="Figure 2: Scheduler latency comparison",
    )
    axis.grid(axis="y", alpha=.25)
    figure.tight_layout()
    save(figure, "fig02_scheduler_latency_comparison")

    # Figure 3: cycles saved per MiB transferred by online prefetch policies.
    policies = (
        "full_expert", "uniform_tile", "ranked_tile", "budget_proportional",
        "hardware_greedy", "criticality_aware",
    )
    efficiency = [
        fmean(
            float(row["scheduler_efficiency_cycles_per_byte"]) * (1 << 20)
            for row in rows if row["scheduler"] == scheduler
        )
        for scheduler in policies
    ]
    figure, axis = plt.subplots(figsize=(9, 4.8))
    axis.bar(range(len(policies)), efficiency, color="#59a14f")
    axis.set_xticks(
        range(len(policies)),
        tuple(name.replace("_", " ").title() for name in policies),
        rotation=25, ha="right",
    )
    axis.set(
        ylabel="Mean latency reduction (cycles) per MiB transferred",
        title="Figure 3: Prefetch traffic efficiency",
    )
    axis.grid(axis="y", alpha=.25)
    figure.tight_layout()
    save(figure, "fig03_latency_reduction_per_byte")

    # Figure 4: Criticality improvement relative to two current heuristics.
    versus_hardware = []
    versus_budget = []
    for quality in QUALITIES:
        subset = [item for item in focused if item["prediction_quality"] == quality]
        versus_hardware.append(fmean(
            float(item["improvement_vs_hardware_greedy_percent"])
            for item in subset
        ))
        versus_budget.append(fmean(
            float(item["improvement_vs_budget_proportional_percent"])
            for item in subset
        ))
    x = range(len(QUALITIES))
    figure, axis = plt.subplots(figsize=(8, 4.8))
    axis.plot(x, versus_hardware, marker="o", label="vs Hardware Greedy")
    axis.plot(x, versus_budget, marker="s", label="vs Budget Proportional")
    axis.axhline(0, color="gray", linewidth=.8)
    axis.set_xticks(x, tuple(item.title() for item in QUALITIES))
    axis.set(
        ylabel="Mean latency improvement (%)",
        title="Figure 4: Criticality-aware scheduler improvement",
    )
    axis.grid(alpha=.25)
    axis.legend()
    figure.tight_layout()
    save(figure, "fig04_criticality_improvement")

    # Figure 5: Expert size and benefit over Full Expert.
    workloads = ("small_edge_moe", "medium_moe", "large_moe")
    figure, axis = plt.subplots(figsize=(8, 4.8))
    for policy, label, marker in (
        ("uniform_tile", "Uniform Tile", "o"),
        ("budget_proportional", "Budget Proportional", "s"),
        ("criticality_aware", "Criticality Aware", "^"),
    ):
        x_values = []
        y_values = []
        for workload in workloads:
            subset = [row for row in rows if row["workload"] == workload]
            expert_bytes = int(subset[0]["expert_bytes"])
            groups: dict[tuple[Any, ...], dict[str, dict[str, Any]]] = defaultdict(dict)
            for row in subset:
                groups[_group_key(row)][str(row["scheduler"])] = row
            benefit = fmean(
                int(group["full_expert"]["latency_cycles"])
                / int(group[policy]["latency_cycles"])
                for group in groups.values()
            )
            x_values.append(expert_bytes / (1 << 20))
            y_values.append(benefit)
        axis.plot(x_values, y_values, marker=marker, label=label)
    axis.axhline(1, color="gray", linewidth=.8)
    axis.set_xscale("log", base=2)
    axis.set(
        xlabel="Expert weight size (MiB, log2 scale)",
        ylabel="Mean speedup vs Full Expert",
        title="Figure 5: Workload scaling of tile scheduling benefit",
    )
    axis.grid(alpha=.25)
    axis.legend()
    figure.tight_layout()
    save(figure, "fig05_expert_size_tile_benefit")
    return written


def summarize(
    output: Path,
    rows: list[dict[str, Any]],
    decomposition: list[dict[str, Any]],
    focused: list[dict[str, Any]],
    plots: list[str],
    oracle_simulations: int,
) -> None:
    """Write machine-readable and Markdown M9.2 summaries."""
    mean_speedup = {
        scheduler: fmean(
            float(row["speedup_vs_no_prefetch"])
            for row in rows if row["scheduler"] == scheduler
        )
        for scheduler in ALL_SCHEDULERS
    }
    critical_vs_hardware = [
        float(item["improvement_vs_hardware_greedy_percent"]) for item in focused
    ]
    critical_vs_budget = [
        float(item["improvement_vs_budget_proportional_percent"]) for item in focused
    ]
    gap_by_quality = {}
    for quality in QUALITIES:
        subset = [item for item in decomposition if item["prediction_quality"] == quality]
        gap_by_quality[quality] = {
            "prediction_loss_percent_of_full_oracle": 100 * fmean(
                int(item["prediction_loss_cycles"])
                / int(item["full_oracle_latency_cycles"])
                for item in subset
            ),
            "criticality_allocation_loss_percent_of_full_oracle": 100 * fmean(
                int(item["criticality_aware_allocation_loss_cycles"])
                / int(item["full_oracle_latency_cycles"])
                for item in subset
            ),
        }
    groups: dict[tuple[Any, ...], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        groups[_group_key(row)][str(row["scheduler"])] = row
    full_expert_wins = 0
    for group in groups.values():
        best = min(int(group[name]["latency_cycles"]) for name in ONLINE)
        if int(group["full_expert"]["latency_cycles"]) == best:
            full_expert_wins += 1
    largest = max(
        focused,
        key=lambda item: float(item["criticality_speedup_vs_full_expert"]),
    )
    summary = {
        "supported": [
            "simulator-based relative scheduling comparison",
            "allocation/prediction gap decomposition",
            "hardware-friendly scheduler trends",
        ],
        "not_claimed": [
            "real FPGA performance", "real predictor accuracy", "RTL implementation",
        ],
        "configurations": len(groups),
        "scheduler_rows": len(rows),
        "oracle_candidate_simulations": oracle_simulations,
        "bandwidth_scale": BANDWIDTH_SCALE,
        "mean_speedup_vs_no_prefetch": mean_speedup,
        "gap_decomposition_by_quality": gap_by_quality,
        "criticality_vs_hardware_greedy": {
            "mean_improvement_percent": fmean(critical_vs_hardware),
            "better_points": sum(value > 0 for value in critical_vs_hardware),
            "tie_points": sum(value == 0 for value in critical_vs_hardware),
            "worse_points": sum(value < 0 for value in critical_vs_hardware),
        },
        "criticality_vs_budget_proportional": {
            "mean_improvement_percent": fmean(critical_vs_budget),
            "better_points": sum(value > 0 for value in critical_vs_budget),
            "tie_points": sum(value == 0 for value in critical_vs_budget),
            "worse_points": sum(value < 0 for value in critical_vs_budget),
        },
        "full_expert_online_winner_membership": full_expert_wins,
        "largest_criticality_benefit_vs_full_expert": largest,
        "plots": plots,
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    report = f"""# M9.2 Criticality-Aware Scheduling Summary

- Configurations: {len(groups)}
- Scheduler rows: {len(rows)}
- Offline candidate simulations: {oracle_simulations}
- Bandwidth scale: {BANDWIDTH_SCALE}x
- Mean speedups vs No Prefetch: `{mean_speedup}`
- Gap decomposition by quality: `{gap_by_quality}`
- Criticality vs Hardware Greedy: `{summary['criticality_vs_hardware_greedy']}`
- Criticality vs Budget Proportional: `{summary['criticality_vs_budget_proportional']}`
- Full Expert online winner memberships including ties: {full_expert_wins}/{len(groups)}

The exact signed formulas requested are retained in the CSV. Positive `loss`
columns are used for plots: prediction loss = fixed - full, allocation loss =
scheduler - fixed.

SUPPORTED: simulator-based relative scheduling comparison; allocation/prediction
gap decomposition; hardware-friendly scheduler trends.

NOT CLAIMED: real FPGA performance; real predictor accuracy; RTL implementation.
"""
    (output / "SUMMARY.md").write_text(report, encoding="utf-8")


def validate(
    output: Path,
    rows: list[dict[str, Any]],
    decomposition: list[dict[str, Any]],
) -> None:
    """Validate coverage, conservation, capacity, and both Oracle bounds."""
    if len(rows) != 3 * 3 * 4 * 4 * 9:
        raise AssertionError("incomplete M9.2 scheduler matrix")
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[_group_key(row)].append(row)
        if int(row["allocated_bytes"]) != (
            int(row["useful_bytes"])
            + int(row["wasted_bytes"])
            + int(row["cancelled_bytes"])
        ):
            raise AssertionError("M9.2 byte conservation failed")
        if int(row["peak_ocm_bytes"]) > int(row["ocm_capacity_bytes"]):
            raise AssertionError("M9.2 OCM limit exceeded")
        if int(row["allocated_bytes"]) > int(row["plan_budget_bytes"]):
            raise AssertionError("M9.2 bandwidth/OCM planning budget exceeded")
    for group in groups.values():
        by_name = {row["scheduler"]: row for row in group}
        full = int(by_name["full_oracle"]["latency_cycles"])
        fixed = int(by_name["fixed_prediction_oracle"]["latency_cycles"])
        if full > fixed or any(full > int(row["latency_cycles"]) for row in group):
            raise AssertionError("M9.2 full Oracle dominance failed")
        for name in ONLINE:
            if fixed > int(by_name[name]["latency_cycles"]):
                raise AssertionError("M9.2 fixed-prediction Oracle dominance failed")
    if len(decomposition) != 144:
        raise AssertionError("incomplete M9.2 decomposition")
    (output / "artifact_validation.json").write_text(
        json.dumps({
            "status": "PASS",
            "rows": len(rows),
            "configurations": len(groups),
            "fixed_prediction_trace_identity": True,
            "tile_and_byte_conservation": True,
            "ocm_bandwidth_dma_constraints": True,
            "full_oracle_dominance": True,
            "fixed_prediction_oracle_dominance": True,
            "hardware_validation": False,
        }, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    """Run the independent M9.2 analysis without rewriting older artifacts."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=OUT)
    args = parser.parse_args()
    workloads = tuple(load_workload_profile(path) for path in WORKLOAD_FILES)
    hardware_profiles = tuple(load_hardware(path) for path in HARDWARE_FILES)
    full_cache: dict[tuple[str, str, float], tuple[PrefetchPlan, int, int]] = {}
    fixed_cache: dict[tuple[str, str, str, float], tuple[PrefetchPlan, int, int]] = {}
    oracle_simulations = 0

    for workload in workloads:
        for base, execution, _ in hardware_profiles:
            for ocm_scale in OCM_SCALES:
                full = search_oracle(
                    workload, base, execution, ocm_scale=ocm_scale,
                    bandwidth_scale=BANDWIDTH_SCALE,
                )
                full_cache[(workload.name, base.name, ocm_scale)] = full
                oracle_simulations += full[2]
                for quality in QUALITIES:
                    fixed = search_fixed_prediction_oracle(
                        workload, base, execution, quality=quality,
                        ocm_scale=ocm_scale,
                    )
                    fixed_cache[(workload.name, base.name, quality, ocm_scale)] = fixed
                    oracle_simulations += fixed[2]
                print(
                    f"oracles {workload.name}/{base.name} ocm={ocm_scale}: "
                    f"full={full[1]} cycles"
                )

    rows: list[dict[str, Any]] = []
    for workload in workloads:
        for base, execution, _ in hardware_profiles:
            for quality in QUALITIES:
                for ocm_scale in OCM_SCALES:
                    schedulers: list[tuple[str, Any, int]] = []
                    for name in ONLINE:
                        scheduler = (
                            CriticalityAwareTileScheduler()
                            if name == "criticality_aware" else scheduler_by_name(name)
                        )
                        schedulers.append((name, scheduler, 0))
                    fixed_plan, _, fixed_count = fixed_cache[
                        (workload.name, base.name, quality, ocm_scale)
                    ]
                    full_plan, _, full_count = full_cache[
                        (workload.name, base.name, ocm_scale)
                    ]
                    schedulers.extend((
                        (
                            "fixed_prediction_oracle",
                            _fixed_scheduler(fixed_plan, "fixed_prediction_oracle"),
                            fixed_count,
                        ),
                        (
                            "full_oracle",
                            _fixed_scheduler(full_plan, "full_oracle"),
                            full_count,
                        ),
                    ))
                    for name, scheduler, count in schedulers:
                        scaled, result = simulate(
                            workload,
                            base,
                            execution,
                            scheduler=scheduler,
                            quality=quality,
                            ocm_scale=ocm_scale,
                            bandwidth_scale=BANDWIDTH_SCALE,
                        )
                        row = flatten(
                            workload,
                            base,
                            scaled,
                            result,
                            scheduler=name,
                            quality=quality,
                            ocm_scale=ocm_scale,
                            bandwidth_scale=BANDWIDTH_SCALE,
                            oracle_plans=count,
                        )
                        rows.append(row)

    decomposition = annotate(rows)
    focused = criticality_rows(rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "scheduler_comparison.csv", rows)
    write_csv(args.output_dir / "oracle_decomposition.csv", decomposition)
    write_csv(args.output_dir / "criticality_analysis.csv", focused)
    plots = make_figures(args.output_dir, rows, decomposition, focused)
    summarize(
        args.output_dir, rows, decomposition, focused, plots, oracle_simulations
    )
    validate(args.output_dir, rows, decomposition)
    print(f"Wrote {len(rows)} M9.2 scheduler rows to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
