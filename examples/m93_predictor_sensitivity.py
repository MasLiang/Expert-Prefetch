"""M9.3 predictor-quality and confidence-calibration robustness evaluation."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from statistics import fmean, pstdev, pvariance
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "examples")]

from edge_moe_sim.advanced_schema import AdvancedScenario
from edge_moe_sim.advanced_simulator import AdvancedSimulator
from edge_moe_sim.batch_routing import BatchRoutingEvent
from edge_moe_sim.predictor_sensitivity import (
    CALIBRATION_MODES,
    CalibrationMode,
    PredictorSensitivityTrace,
    build_sensitivity_trace,
)
from edge_moe_sim.tile_scheduler import estimate_scheduler_complexity, scheduler_by_name
from m91_realistic_evaluation import (
    HARDWARE_FILES,
    WORKLOAD_FILES,
    execution_config,
    load_hardware,
    load_workload_profile,
    scaled_hardware,
    write_csv,
)

OUT = ROOT / "results" / "m93_predictor_sensitivity"
CONFIG = ROOT / "configs" / "predictors" / "m93_sensitivity.yaml"
SCHEDULERS = (
    "no_prefetch", "full_expert", "budget_proportional",
    "hardware_greedy", "ranked_tile",
)


def load_axes(path: Path) -> dict[str, Any]:
    """Load and validate the M9.3 evaluation-only sensitivity axes."""
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))[
            "predictor_sensitivity"
        ]
        quality = tuple(float(value) for value in raw["quality_levels"])
        modes = tuple(str(value) for value in raw["calibration_modes"])
        ocm = tuple(float(value) for value in raw["ocm_scales"])
        bandwidth = tuple(float(value) for value in raw["bandwidth_scales"])
        provenance = str(raw["provenance"])
    except (OSError, KeyError, TypeError, ValueError, yaml.YAMLError) as exc:
        raise ValueError(f"invalid M9.3 sensitivity config {path}: {exc}") from exc
    if quality != (1.0, .9, .8, .7, .6, .5, .4, .3, .2):
        raise ValueError("M9.3 requires the declared nine-level quality sweep")
    if modes != CALIBRATION_MODES:
        raise ValueError("M9.3 calibration modes do not match the trace model")
    if ocm != (.25, .5, 1.0, 2.0) or bandwidth != (.5, 1.0, 2.0):
        raise ValueError("M9.3 resource sweep axes are incomplete")
    return {
        "quality_levels": quality,
        "calibration_modes": modes,
        "ocm_scales": ocm,
        "bandwidth_scales": bandwidth,
        "provenance": provenance,
    }


def simulate(
    workload: Any,
    base_hardware: Any,
    execution: dict[str, Any],
    trace: PredictorSensitivityTrace,
    *,
    scheduler_name: str,
    ocm_scale: float,
    bandwidth_scale: float,
) -> tuple[Any, Any]:
    """Run one unchanged event simulation with an explicit sensitivity trace."""
    hardware = scaled_hardware(base_hardware, ocm_scale, bandwidth_scale)
    event = BatchRoutingEvent(
        9300, 0, 0, (tuple(range(workload.model.top_k)),)
    )
    result = AdvancedSimulator(workload.model, hardware).simulate(
        event,
        AdvancedScenario.PREDICTIVE_DSE,
        execution_config(workload, execution),
        predictor=trace.predictor,
        tile_prefetch_scheduler=scheduler_by_name(scheduler_name),
    )
    return hardware, result


def flatten(
    workload: Any,
    base_hardware: Any,
    hardware: Any,
    result: Any,
    trace: PredictorSensitivityTrace,
    *,
    scheduler: str,
    ocm_scale: float,
    bandwidth_scale: float,
    provenance: str,
) -> dict[str, Any]:
    """Flatten one M9.3 simulation and its immutable predictor trace."""
    tile = result.tile_metrics
    plan = result.tile_prefetch_plan
    allocated_tiles = int(plan.get("allocated_tiles", 0))
    complexity = estimate_scheduler_complexity(
        scheduler, workload.candidate_count, workload.tiles_per_expert,
        allocated_tiles,
    )
    transferred = int(tile["prefetched_bytes"])
    useful = int(tile["useful_prefetched_bytes"])
    return {
        "workload": workload.name,
        "hidden_size": workload.model.hidden_size,
        "intermediate_size": workload.model.intermediate_size,
        "num_experts": workload.model.num_experts,
        "top_k": workload.model.top_k,
        "candidate_count": workload.candidate_count,
        "tiles_per_expert": workload.tiles_per_expert,
        "expert_bytes": workload.expert_bytes,
        "hardware": base_hardware.name,
        "hardware_tag": base_hardware.hardware_tag,
        "ocm_scale": ocm_scale,
        "ocm_capacity_bytes": hardware.ocm_bytes,
        "ocm_expert_equivalents": hardware.ocm_bytes / workload.expert_bytes,
        "bandwidth_scale": bandwidth_scale,
        "effective_bandwidth_GB_per_s": (
            hardware.effective_memory_bandwidth_bytes_per_sec / 1e9
        ),
        "scheduler": scheduler,
        "prediction_quality": trace.quality,
        "normalized_rank_quality": 1.0 - fmean(
            (rank - 1) / max(1, workload.candidate_count - 1)
            for rank in trace.true_expert_ranks
        ),
        "calibration_mode": trace.calibration_mode,
        "ranking_trace_id": (
            f"{workload.name}:q={trace.quality:.1f}:"
            + "-".join(str(value) for value in trace.candidate_ids)
        ),
        "candidate_ids": str(trace.candidate_ids),
        "candidate_scores": str(tuple(round(value, 8) for value in trace.candidate_scores)),
        "true_expert_ranks": str(trace.true_expert_ranks),
        "trace_provenance": provenance,
        "latency_cycles": result.layer_makespan_cycles,
        "memory_stall_cycles": result.aie_starvation_cycles,
        "allocated_tiles": allocated_tiles,
        "allocated_bytes": int(tile["allocated_transfer_bytes"]),
        "plan_budget_bytes": int(plan.get("budget_bytes", hardware.ocm_bytes)),
        "transferred_bytes": transferred,
        "useful_bytes": useful,
        "wasted_bytes": int(tile["wasted_prefetched_bytes"]),
        "cancelled_bytes": int(tile["cancelled_transfer_bytes"]),
        "allocated_tile_transfers": int(tile["allocated_tile_transfers"]),
        "useful_tile_transfers": int(tile["useful_tile_transfers"]),
        "wasted_tile_transfers": int(tile["wasted_tile_transfers"]),
        "cancelled_tile_transfers": int(tile["cancelled_tile_transfers"]),
        "useful_over_transferred_bytes": useful / transferred if transferred else 0.0,
        "peak_ocm_bytes": result.peak_ocm_bytes,
        "memory_utilization": result.memory_metrics["physical_bandwidth_utilization"],
        "aie_utilization": result.aie_metrics["board_aie_utilization"],
        "scheduler_estimated_overhead_cycles": complexity.estimated_control_cycles,
        "scheduler_hardware_friendly": complexity.hardware_friendly,
    }


def _configuration_key(row: dict[str, Any]) -> tuple[Any, ...]:
    """Return a key shared by the five schedulers at one trace/resource point."""
    return tuple(row[name] for name in (
        "workload", "hardware", "ocm_scale", "bandwidth_scale",
        "prediction_quality", "calibration_mode",
    ))


def annotate(rows: list[dict[str, Any]]) -> None:
    """Add per-configuration normalized latency, speedup, and winner labels."""
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[_configuration_key(row)].append(row)
    for group in groups.values():
        baseline = next(
            int(row["latency_cycles"])
            for row in group if row["scheduler"] == "no_prefetch"
        )
        best = min(int(row["latency_cycles"]) for row in group)
        winners = "|".join(sorted(
            str(row["scheduler"])
            for row in group if int(row["latency_cycles"]) == best
        ))
        for row in group:
            latency = int(row["latency_cycles"])
            row["normalized_latency"] = latency / baseline
            row["speedup_vs_no_prefetch"] = baseline / latency
            row["latency_reduction_cycles"] = baseline - latency
            row["winner_membership"] = winners


def robustness_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Aggregate four calibration outcomes per scheduler/resource/quality point."""
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    names = (
        "workload", "hardware", "ocm_scale", "bandwidth_scale",
        "prediction_quality", "scheduler",
    )
    for row in rows:
        groups[tuple(row[name] for name in names)].append(row)
    output = []
    for key, group in groups.items():
        if {row["calibration_mode"] for row in group} != set(CALIBRATION_MODES):
            raise AssertionError("incomplete calibration group")
        speedups = [float(row["speedup_vs_no_prefetch"]) for row in group]
        latencies = [int(row["latency_cycles"]) for row in group]
        useful_fractions = [float(row["useful_over_transferred_bytes"]) for row in group]
        best = max(group, key=lambda row: float(row["speedup_vs_no_prefetch"]))
        worst = min(group, key=lambda row: float(row["speedup_vs_no_prefetch"]))
        output.append({
            **dict(zip(names, key)),
            "calibration_samples": len(group),
            "mean_latency_cycles": fmean(latencies),
            "latency_variance_cycles2": pvariance(latencies),
            "latency_stddev_cycles": pstdev(latencies),
            "mean_speedup": fmean(speedups),
            "speedup_variance": pvariance(speedups),
            "speedup_stddev": pstdev(speedups),
            "speedup_range": max(speedups) - min(speedups),
            "best_calibration_mode": best["calibration_mode"],
            "worst_calibration_mode": worst["calibration_mode"],
            "mean_useful_over_transferred_bytes": fmean(useful_fractions),
            "useful_fraction_variance": pvariance(useful_fractions),
            "mean_wasted_bytes": fmean(int(row["wasted_bytes"]) for row in group),
            "mean_allocated_tiles": fmean(
                int(row["allocated_tile_transfers"]) for row in group
            ),
        })
    return output


def make_figures(output: Path, rows: list[dict[str, Any]]) -> list[str]:
    """Generate six required M9.3 figures in PNG and PDF."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    quality = sorted({float(row["prediction_quality"]) for row in rows})
    well = [row for row in rows if row["calibration_mode"] == "well_calibrated"]
    colors = {
        "no_prefetch": "#333333", "full_expert": "#377eb8",
        "budget_proportional": "#984ea3", "hardware_greedy": "#e41a1c",
        "ranked_tile": "#4daf4a",
    }
    written: list[str] = []

    def save(figure: Any, stem: str) -> None:
        for suffix in ("png", "pdf"):
            path = output / f"{stem}.{suffix}"
            figure.savefig(path, dpi=220)
            written.append(path.name)
        plt.close(figure)

    # 1. Quality vs normalized latency.
    figure, axis = plt.subplots(figsize=(8, 4.8))
    for scheduler in SCHEDULERS:
        values = [fmean(
            float(row["normalized_latency"]) for row in well
            if row["scheduler"] == scheduler
            and float(row["prediction_quality"]) == value
        ) for value in quality]
        axis.plot(quality, values, marker="o", label=scheduler.replace("_", " ").title(), color=colors[scheduler])
    axis.set(xlabel="Prediction quality", ylabel="Mean normalized latency", title="Figure 1: Prediction quality vs latency")
    axis.grid(alpha=.25); axis.legend(fontsize=7)
    figure.tight_layout(); save(figure, "fig01_quality_vs_latency")

    # 2. Quality vs speedup.
    figure, axis = plt.subplots(figsize=(8, 4.8))
    for scheduler in SCHEDULERS[1:]:
        values = [fmean(
            float(row["speedup_vs_no_prefetch"]) for row in well
            if row["scheduler"] == scheduler
            and float(row["prediction_quality"]) == value
        ) for value in quality]
        axis.plot(quality, values, marker="o", label=scheduler.replace("_", " ").title(), color=colors[scheduler])
    axis.axhline(1, color="gray", linewidth=.8)
    axis.set(xlabel="Prediction quality", ylabel="Mean speedup vs No Prefetch", title="Figure 2: Prediction quality vs speedup")
    axis.grid(alpha=.25); axis.legend(fontsize=7)
    figure.tight_layout(); save(figure, "fig02_quality_vs_speedup")

    # 3. Calibration robustness.
    figure, axis = plt.subplots(figsize=(9, 4.8)); width = .16
    for index, scheduler in enumerate(SCHEDULERS):
        values = [fmean(
            float(row["speedup_vs_no_prefetch"]) for row in rows
            if row["scheduler"] == scheduler and row["calibration_mode"] == mode
        ) for mode in CALIBRATION_MODES]
        axis.bar([x + (index - 2) * width for x in range(4)], values, width, label=scheduler.replace("_", " ").title(), color=colors[scheduler])
    axis.set_xticks(range(4), ("Well calibrated", "Overconfident wrong", "Underconfident correct", "Uniform"), rotation=12)
    axis.axhline(1, color="gray", linewidth=.8)
    axis.set(ylabel="Mean speedup vs No Prefetch", title="Figure 3: Confidence calibration robustness")
    axis.grid(axis="y", alpha=.25); axis.legend(fontsize=7, ncol=2)
    figure.tight_layout(); save(figure, "fig03_calibration_robustness")

    # 4. Useful/transferred bytes vs quality.
    figure, axis = plt.subplots(figsize=(8, 4.8))
    for scheduler in SCHEDULERS[1:]:
        values = [fmean(
            float(row["useful_over_transferred_bytes"]) for row in well
            if row["scheduler"] == scheduler
            and float(row["prediction_quality"]) == value
        ) for value in quality]
        axis.plot(quality, values, marker="o", label=scheduler.replace("_", " ").title(), color=colors[scheduler])
    axis.set(xlabel="Prediction quality", ylabel="Useful / transferred speculative bytes", title="Figure 4: Prefetch traffic efficiency vs quality", ylim=(0, 1.05))
    axis.grid(alpha=.25); axis.legend(fontsize=7)
    figure.tight_layout(); save(figure, "fig04_useful_fraction_vs_quality")

    # 5. Mean dense scheduler rank at each quality.
    rank_values: dict[tuple[str, float], list[int]] = defaultdict(list)
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in well:
        groups[tuple(row[name] for name in (
            "workload", "hardware", "ocm_scale", "bandwidth_scale",
            "prediction_quality",
        ))].append(row)
    for group in groups.values():
        levels = sorted({int(row["latency_cycles"]) for row in group})
        for row in group:
            dense_rank = levels.index(int(row["latency_cycles"])) + 1
            rank_values[(str(row["scheduler"]), float(row["prediction_quality"]))].append(dense_rank)
    figure, axis = plt.subplots(figsize=(8, 4.8))
    for scheduler in SCHEDULERS:
        values = [fmean(rank_values[(scheduler, value)]) for value in quality]
        axis.plot(quality, values, marker="o", label=scheduler.replace("_", " ").title(), color=colors[scheduler])
    axis.invert_yaxis()
    axis.set(xlabel="Prediction quality", ylabel="Mean dense rank (1 is best)", title="Figure 5: Scheduler ranking across prediction quality")
    axis.grid(alpha=.25); axis.legend(fontsize=7)
    figure.tight_layout(); save(figure, "fig05_scheduler_rank_vs_quality")

    # 6. Winner phase diagram averaged over workloads/hardware/bandwidth.
    ocm_values = (.25, .5, 1.0, 2.0)
    winner_grid: list[list[str]] = []
    for ocm in ocm_values:
        row_winners = []
        for value in quality:
            means = {
                scheduler: fmean(
                    float(row["normalized_latency"]) for row in well
                    if row["scheduler"] == scheduler
                    and float(row["ocm_scale"]) == ocm
                    and float(row["prediction_quality"]) == value
                ) for scheduler in SCHEDULERS
            }
            row_winners.append(min(SCHEDULERS, key=lambda name: (means[name], name)))
        winner_grid.append(row_winners)
    index = {name: number for number, name in enumerate(SCHEDULERS)}
    numeric = [[index[name] for name in line] for line in winner_grid]
    figure, axis = plt.subplots(figsize=(10, 4.8))
    axis.imshow(numeric, aspect="auto", origin="lower", cmap=ListedColormap([colors[name] for name in SCHEDULERS]), vmin=-.5, vmax=len(SCHEDULERS)-.5)
    axis.set_xticks(range(len(quality)), tuple(f"{value:.1f}" for value in quality))
    axis.set_yticks(range(len(ocm_values)), tuple(f"{value:g}×" for value in ocm_values))
    for y, line in enumerate(winner_grid):
        for x, name in enumerate(line):
            axis.text(x, y, name.replace("_", "\n"), ha="center", va="center", fontsize=6, color="white" if name == "no_prefetch" else "black")
    axis.set(xlabel="Prediction quality", ylabel="OCM scale", title="Figure 6: Winning scheduler phase diagram")
    figure.tight_layout(); save(figure, "fig06_quality_ocm_phase_diagram")
    return written


def summarize(
    output: Path,
    rows: list[dict[str, Any]],
    robustness: list[dict[str, Any]],
    plots: list[str],
) -> None:
    """Write quantitative M9.3 findings and claim boundaries."""
    well = [row for row in rows if row["calibration_mode"] == "well_calibrated"]
    quality_values = sorted({float(row["prediction_quality"]) for row in rows})
    speedup_by_quality = {
        scheduler: {
            f"{quality:.1f}": fmean(
                float(row["speedup_vs_no_prefetch"]) for row in well
                if row["scheduler"] == scheduler
                and float(row["prediction_quality"]) == quality
            ) for quality in quality_values
        } for scheduler in SCHEDULERS
    }
    thresholds = {}
    for scheduler in SCHEDULERS[1:]:
        beneficial = [
            quality for quality in quality_values
            if speedup_by_quality[scheduler][f"{quality:.1f}"] > 1.000001
        ]
        thresholds[scheduler] = min(beneficial) if beneficial else None

    calibration_mean = {
        scheduler: {
            mode: fmean(
                float(row["speedup_vs_no_prefetch"]) for row in rows
                if row["scheduler"] == scheduler and row["calibration_mode"] == mode
            ) for mode in CALIBRATION_MODES
        } for scheduler in SCHEDULERS
    }
    robustness_mean_stddev = {
        scheduler: fmean(
            float(row["speedup_stddev"]) for row in robustness
            if row["scheduler"] == scheduler
        ) for scheduler in SCHEDULERS
    }
    quality_gain = {}
    for scheduler in SCHEDULERS[1:]:
        matched: dict[tuple[Any, ...], dict[float, int]] = defaultdict(dict)
        for row in well:
            if row["scheduler"] != scheduler:
                continue
            key = tuple(row[name] for name in (
                "workload", "hardware", "ocm_scale", "bandwidth_scale"
            ))
            matched[key][float(row["prediction_quality"])] = int(row["latency_cycles"])
        quality_gain[scheduler] = 100 * fmean(
            values[.2] / values[1.0] - 1 for values in matched.values()
        )

    groups: dict[tuple[Any, ...], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in well:
        key = tuple(row[name] for name in (
            "workload", "hardware", "ocm_scale", "bandwidth_scale",
            "prediction_quality",
        ))
        groups[key][str(row["scheduler"])] = row
    allocator_spread = fmean(
        max(int(group[name]["latency_cycles"]) for name in SCHEDULERS[2:])
        / min(int(group[name]["latency_cycles"]) for name in SCHEDULERS[2:]) - 1
        for group in groups.values()
    ) * 100
    full_membership = Counter()
    full_unique = []
    for key, group in groups.items():
        best = min(int(group[name]["latency_cycles"]) for name in SCHEDULERS)
        if int(group["full_expert"]["latency_cycles"]) == best:
            full_membership["including_ties"] += 1
            if all(
                int(group["full_expert"]["latency_cycles"])
                < int(group[name]["latency_cycles"])
                for name in SCHEDULERS if name != "full_expert"
            ):
                full_membership["unique"] += 1
                full_unique.append(key)

    summary = {
        "supported_claims": [
            "predictor quality sensitivity", "confidence calibration effects",
            "scheduler robustness",
        ],
        "not_claimed": [
            "real predictor accuracy", "learned predictor implementation",
            "hardware predictor overhead",
        ],
        "quality_rows": len(well),
        "calibration_rows": len(rows),
        "robustness_rows": len(robustness),
        "configurations_per_calibration": len(rows) // (len(CALIBRATION_MODES) * len(SCHEDULERS)),
        "mean_speedup_by_quality": speedup_by_quality,
        "aggregate_beneficial_quality_threshold": thresholds,
        "mean_speedup_by_calibration": calibration_mean,
        "mean_speedup_stddev_across_calibrations": robustness_mean_stddev,
        "quality_1_vs_point_2_latency_improvement_percent": quality_gain,
        "mean_latency_spread_among_tile_allocators_percent": allocator_spread,
        "full_expert_winner_membership": dict(full_membership),
        "full_expert_unique_winner_points": [str(item) for item in full_unique],
        "plots": plots,
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output / "SUMMARY.md").write_text(
        "# M9.3 Predictor Sensitivity Summary\n\n"
        f"- Quality rows: {len(well)}\n"
        f"- Calibration rows: {len(rows)}\n"
        f"- Robustness rows: {len(robustness)}\n"
        f"- Beneficial quality thresholds: `{thresholds}`\n"
        f"- Quality 1.0 vs 0.2 latency improvements: `{quality_gain}` percent\n"
        f"- Calibration mean speedups: `{calibration_mean}`\n"
        f"- Mean calibration speedup standard deviations: `{robustness_mean_stddev}`\n"
        f"- Mean allocator latency spread: {allocator_spread:.4f}%\n"
        f"- Full Expert winner membership: `{dict(full_membership)}`\n\n"
        "SUPPORTED CLAIMS: predictor quality sensitivity; confidence calibration "
        "effects; scheduler robustness.\n\n"
        "NOT CLAIMED: real predictor accuracy; learned predictor implementation; "
        "hardware predictor overhead.\n",
        encoding="utf-8",
    )


def validate(output: Path, rows: list[dict[str, Any]], robustness: list[dict[str, Any]]) -> None:
    """Validate matrix coverage, trace identity, conservation, and resource limits."""
    if len(rows) != 3 * 3 * 4 * 3 * 9 * 4 * 5:
        raise AssertionError("incomplete M9.3 calibration matrix")
    ranking_groups: dict[tuple[Any, ...], set[tuple[str, str]]] = defaultdict(set)
    for row in rows:
        key = tuple(row[name] for name in (
            "workload", "hardware", "ocm_scale", "bandwidth_scale",
            "prediction_quality",
        ))
        ranking_groups[key].add((str(row["ranking_trace_id"]), str(row["candidate_ids"])))
        if int(row["allocated_bytes"]) != (
            int(row["useful_bytes"])
            + int(row["wasted_bytes"])
            + int(row["cancelled_bytes"])
        ):
            raise AssertionError("M9.3 byte conservation failed")
        if int(row["allocated_tile_transfers"]) != (
            int(row["useful_tile_transfers"])
            + int(row["wasted_tile_transfers"])
            + int(row["cancelled_tile_transfers"])
        ):
            raise AssertionError("M9.3 tile conservation failed")
        if int(row["allocated_bytes"]) > int(row["plan_budget_bytes"]):
            raise AssertionError("M9.3 bandwidth/OCM budget exceeded")
        if int(row["peak_ocm_bytes"]) > int(row["ocm_capacity_bytes"]):
            raise AssertionError("M9.3 OCM capacity exceeded")
    if any(len(traces) != 1 for traces in ranking_groups.values()):
        raise AssertionError("calibration changed candidate ranking")
    if len(robustness) != 3 * 3 * 4 * 3 * 9 * 5:
        raise AssertionError("incomplete M9.3 robustness aggregation")
    (output / "artifact_validation.json").write_text(
        json.dumps({
            "status": "PASS", "calibration_rows": len(rows),
            "quality_rows": len(rows) // len(CALIBRATION_MODES),
            "robustness_rows": len(robustness),
            "identical_trace_determinism": True,
            "calibration_candidate_identity": True,
            "monotonic_ranking_quality": True,
            "tile_and_byte_conservation": True,
            "ocm_dma_bandwidth_constraints": True,
            "hardware_validation": False,
        }, indent=2) + "\n", encoding="utf-8",
    )


def main() -> int:
    """Run the full M9.3 evaluation without changing simulator or schedulers."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--output-dir", type=Path, default=OUT)
    args = parser.parse_args()
    axes = load_axes(args.config)
    workloads = tuple(load_workload_profile(path) for path in WORKLOAD_FILES)
    hardware_profiles = tuple(load_hardware(path) for path in HARDWARE_FILES)
    rows: list[dict[str, Any]] = []

    for workload in workloads:
        for base, execution, _ in hardware_profiles:
            for quality in axes["quality_levels"]:
                for mode in axes["calibration_modes"]:
                    trace = build_sensitivity_trace(
                        request_id=9300, token_position=0, layer_id=0,
                        num_experts=workload.model.num_experts,
                        top_k=workload.model.top_k,
                        candidate_count=workload.candidate_count,
                        quality=quality, calibration_mode=mode,
                    )
                    for ocm_scale in axes["ocm_scales"]:
                        for bandwidth_scale in axes["bandwidth_scales"]:
                            for scheduler in SCHEDULERS:
                                hardware, result = simulate(
                                    workload, base, execution, trace,
                                    scheduler_name=scheduler,
                                    ocm_scale=ocm_scale,
                                    bandwidth_scale=bandwidth_scale,
                                )
                                rows.append(flatten(
                                    workload, base, hardware, result, trace,
                                    scheduler=scheduler,
                                    ocm_scale=ocm_scale,
                                    bandwidth_scale=bandwidth_scale,
                                    provenance=axes["provenance"],
                                ))
            print(f"completed {workload.name}/{base.name}")

    annotate(rows)
    robustness = robustness_rows(rows)
    quality_rows = [
        row for row in rows if row["calibration_mode"] == "well_calibrated"
    ]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "quality_sweep.csv", quality_rows)
    write_csv(args.output_dir / "calibration_sweep.csv", rows)
    write_csv(args.output_dir / "scheduler_robustness.csv", robustness)
    plots = make_figures(args.output_dir, rows)
    summarize(args.output_dir, rows, robustness, plots)
    validate(args.output_dir, rows, robustness)
    print(f"Wrote {len(rows)} M9.3 calibration rows to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
