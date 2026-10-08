"""Run the generic single-token M8 resource-aware tile-scheduling sweep."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from statistics import fmean
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from edge_moe_sim.advanced_schema import AdvancedExecutionConfig, AdvancedScenario
from edge_moe_sim.advanced_simulator import AdvancedSimulator
from edge_moe_sim.batch_predictor import (
    BatchPredictorOutput, PrecomputedBatchPredictor, TokenPrediction,
)
from edge_moe_sim.batch_routing import BatchRoutingEvent
from edge_moe_sim.hardware_profile import HardwareProfile
from edge_moe_sim.schema import CandidatePrefetch, ModelConfig, ValidationError
from edge_moe_sim.tile_scheduler import scheduler_by_name
from edge_moe_sim.work_model import expert_weight_bytes

DEFAULT_CONFIG = ROOT / "configs/m8_tile_scheduler.yaml"
DEFAULT_OUTPUT = ROOT / "results/m8_tile_scheduler"
QUALITY_TRUE_RANKS = {
    "perfect": (1, 2),
    "high": (1, 4),
    "medium": (3, 8),
    "low": (8, 16),
}


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write deterministic non-empty CSV rows."""
    if not rows:
        raise ValueError(f"empty M8 artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def load_config(path: Path) -> dict[str, Any]:
    """Load and minimally validate the standalone M8 experiment configuration."""
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ValidationError(f"cannot load M8 configuration: {exc}") from exc
    if not isinstance(data, dict) or not {"model", "hardware", "experiment"} <= set(data):
        raise ValidationError("M8 configuration needs model, hardware, and experiment sections")
    return data


def model_from(raw: dict[str, Any]) -> ModelConfig:
    """Construct the generic model configuration."""
    return ModelConfig(
        str(raw["name"]), int(raw["hidden_size"]), int(raw["intermediate_size"]),
        int(raw["num_experts"]), int(raw["top_k"]), int(raw["weight_bits"]),
        int(raw["activation_bits"]),
    )


def profile_from(raw: dict[str, Any], *, channels: int, ocm_scale: float,
                 bandwidth_scale: float) -> HardwareProfile:
    """Construct one generic synthetic hardware sensitivity point."""
    return HardwareProfile(
        name="m8_generic_synthetic",
        clock_hz=int(raw["clock_hz"]),
        num_memory_channels=channels,
        bandwidth_bytes_per_sec_per_channel=(
            float(raw["bandwidth_bytes_per_sec_per_channel"]) * bandwidth_scale
        ),
        bandwidth_efficiency=float(raw["bandwidth_efficiency"]),
        dma_setup_cycles=int(raw["dma_setup_cycles"]),
        max_inflight_per_channel=1,
        burst_bytes=int(raw["burst_bytes"]),
        supports_channel_striping=True,
        total_aie_units=int(raw["total_aie_units"]),
        ops_per_cycle_per_aie=float(raw["ops_per_cycle_per_aie"]),
        aie_efficiency=float(raw["aie_efficiency"]),
        aie_task_startup_cycles=int(raw["aie_task_startup_cycles"]),
        ocm_bytes=round(int(raw["base_ocm_bytes"]) * ocm_scale),
        router_ops_per_cycle_per_aie=float(raw["router_ops_per_cycle_per_aie"]),
        hardware_tag="synthetic_placeholder_not_hardware_validated",
    )


def predictor(model: ModelConfig, quality: str, candidate_count: int,
              ready_cycle: int) -> PrecomputedBatchPredictor:
    """Build a deterministic ranked trace with true Experts at controlled ranks."""
    ranks = QUALITY_TRUE_RANKS[quality]
    if max(ranks) > candidate_count or model.top_k != len(ranks):
        raise ValidationError("quality ranks must fit candidates and Top-K")
    ordered: list[int | None] = [None] * candidate_count
    true_ids = tuple(range(model.top_k))
    for expert_id, rank_one_based in zip(true_ids, ranks):
        ordered[rank_one_based - 1] = expert_id
    false_ids = iter(range(model.top_k, model.num_experts))
    for index, expert_id in enumerate(ordered):
        if expert_id is None:
            ordered[index] = next(false_ids)
    candidates = tuple(
        CandidatePrefetch(int(expert_id), 1.0 - rank * 0.75 / max(1, candidate_count - 1))
        for rank, expert_id in enumerate(ordered)
    )
    output = BatchPredictorOutput(
        8000, 0, 0, ready_cycle, (TokenPrediction(0, candidates),)
    )
    return PrecomputedBatchPredictor({(8000, 0, 0): output})


def run_one(config: dict[str, Any], *, scheduler_name: str, quality: str,
            tile_size_name: str, f_tile_size: int, ocm_scale: float,
            bandwidth_scale: float, channels: int) -> dict[str, Any]:
    """Run one point and flatten tile/resource metrics into a CSV row."""
    model = model_from(config["model"])
    experiment = config["experiment"]
    profile = profile_from(
        config["hardware"], channels=channels, ocm_scale=ocm_scale,
        bandwidth_scale=bandwidth_scale,
    )
    event = BatchRoutingEvent(8000, 0, 0, (tuple(range(model.top_k)),))
    execution = AdvancedExecutionConfig(
        f_tile_size=f_tile_size,
        placement_policy="round_robin_expert",
        aie_units_per_expert=int(experiment["aie_units_per_expert"]),
        max_parallel_experts=model.top_k,
        startup_gu_tiles=int(experiment["startup_gu_tiles"]),
        candidate_count=int(experiment["candidate_count"]),
    )
    result = AdvancedSimulator(model, profile).simulate(
        event, AdvancedScenario.PREDICTIVE_DSE, execution,
        predictor=predictor(
            model, quality, int(experiment["candidate_count"]),
            int(experiment["predictor_ready_cycle"]),
        ),
        tile_prefetch_scheduler=scheduler_by_name(scheduler_name),
    )
    tile = result.tile_metrics
    plan = result.tile_prefetch_plan
    num_tiles = (model.intermediate_size + f_tile_size - 1) // f_tile_size
    return {
        "scheduler": scheduler_name,
        "prediction_quality": quality,
        "true_expert_ranks": str(QUALITY_TRUE_RANKS[quality]),
        "tile_size": tile_size_name,
        "f_tile_elements": f_tile_size,
        "tiles_per_expert": num_tiles,
        "ocm_scale": ocm_scale,
        "ocm_capacity_bytes": profile.ocm_bytes,
        "ocm_capacity_expert_equivalents": profile.ocm_bytes / expert_weight_bytes(model),
        "bandwidth_scale": bandwidth_scale,
        "effective_bandwidth_GB_per_s_per_channel": (
            profile.effective_bytes_per_cycle_per_channel * profile.clock_hz / 1e9
        ),
        "dma_channels": channels,
        "latency_cycles": result.layer_makespan_cycles,
        "router_resolve_cycle": result.router_resolve_cycle,
        "tile_startup_latency_cycles": tile["tile_startup_latency_cycles"],
        "memory_stall_cycles": tile["memory_stall_cycles"],
        "compute_idle_aie_unit_cycles": tile["compute_idle_aie_unit_cycles"],
        "memory_bandwidth_utilization": result.memory_metrics["physical_bandwidth_utilization"],
        "aie_utilization": result.aie_metrics["board_aie_utilization"],
        "peak_ocm_bytes": result.peak_ocm_bytes,
        "ocm_utilization": result.peak_ocm_bytes / profile.ocm_bytes,
        "planned_tiles": int(plan.get("allocated_tiles", 0)),
        "planned_bytes": int(plan.get("allocated_bytes", 0)),
        "plan_budget_bytes": int(plan.get("budget_bytes", profile.ocm_bytes)),
        "depths_by_expert": json.dumps(plan.get("depths_by_expert", {}), sort_keys=True),
        "prefetched_tiles": tile["prefetched_tiles"],
        "useful_prefetched_tiles": tile["useful_prefetched_tiles"],
        "wasted_prefetched_tiles": tile["wasted_prefetched_tiles"],
        "prefetched_bytes": tile["prefetched_bytes"],
        "useful_prefetched_bytes": tile["useful_prefetched_bytes"],
        "wasted_prefetched_bytes": tile["wasted_prefetched_bytes"],
        "prefetch_efficiency": tile["prefetch_efficiency"],
        "candidate_tile_coverage": tile["candidate_tile_coverage"],
        "rejected_prefetch_bytes": result.rejected_prefetch_bytes,
        "evicted_speculative_bytes": result.evicted_speculative_bytes,
        "demand_blocked_cycles": result.demand_blocked_cycles,
        "hardware_tag": profile.hardware_tag,
        "batch_size": 1,
        "simulator_scope": "single_token_single_layer_tile_scheduling",
    }


def add_comparisons(rows: list[dict[str, Any]]) -> None:
    """Annotate latency speedup and exact winner within every hardware/workload point."""
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = tuple(row[name] for name in (
            "prediction_quality", "tile_size", "ocm_scale",
            "bandwidth_scale", "dma_channels",
        ))
        groups[key].append(row)
    for group in groups.values():
        baseline = next(int(row["latency_cycles"]) for row in group if row["scheduler"] == "no_prefetch")
        best = min(int(row["latency_cycles"]) for row in group)
        winners = "|".join(sorted(row["scheduler"] for row in group if int(row["latency_cycles"]) == best))
        full = next(int(row["latency_cycles"]) for row in group if row["scheduler"] == "full_expert")
        for row in group:
            row["speedup_vs_no_prefetch"] = baseline / int(row["latency_cycles"])
            row["speedup_vs_full_expert"] = full / int(row["latency_cycles"])
            row["latency_winners"] = winners
            row["is_latency_winner"] = int(row["latency_cycles"]) == best


def make_plots(output: Path, rows: list[dict[str, Any]]) -> list[str]:
    """Create three static methodology plots from the experiment CSV."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = {
        "no_prefetch": "No Prefetch", "full_expert": "Full Expert",
        "uniform_tile": "Uniform Tile", "ranked_tile": "Ranked Tile",
    }
    colors = {
        "no_prefetch": "#333333", "full_expert": "#377eb8",
        "uniform_tile": "#ff9800", "ranked_tile": "#4daf4a",
    }
    written: list[str] = []

    # Controlled slice: medium tile, 2 channels, 1x bandwidth; mean over qualities.
    subset = [row for row in rows if row["tile_size"] == "medium" and row["dma_channels"] == 2 and row["bandwidth_scale"] == 1.0]
    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    for policy in labels:
        points = []
        for scale in (0.5, 1.0, 2.0):
            values = [float(row["speedup_vs_no_prefetch"]) for row in subset if row["scheduler"] == policy and row["ocm_scale"] == scale]
            points.append(fmean(values))
        ax.plot((0.5, 1.0, 2.0), points, marker="o", label=labels[policy], color=colors[policy])
    ax.axhline(1.0, color="gray", linewidth=0.8)
    ax.set(xlabel="OCM capacity scale", ylabel="Mean speedup vs no prefetch", title="Tile allocation benefit under OCM pressure")
    ax.grid(alpha=0.25); ax.legend(); fig.tight_layout()
    for suffix in ("png", "pdf"):
        path = output / f"fig_m8_latency_vs_ocm.{suffix}"; fig.savefig(path, dpi=220); written.append(path.name)
    plt.close(fig)

    # Efficiency against controlled predictor quality.
    order = ("perfect", "high", "medium", "low")
    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    for policy in ("full_expert", "uniform_tile", "ranked_tile"):
        values = [fmean(float(row["prefetch_efficiency"]) for row in rows if row["scheduler"] == policy and row["prediction_quality"] == quality) for quality in order]
        ax.plot(order, values, marker="o", label=labels[policy], color=colors[policy])
    ax.set(xlabel="Prediction quality", ylabel="Useful / transferred speculative bytes", title="Prefetch efficiency vs prediction quality", ylim=(0, 1.05))
    ax.grid(alpha=0.25); ax.legend(); fig.tight_layout()
    for suffix in ("png", "pdf"):
        path = output / f"fig_m8_efficiency_vs_quality.{suffix}"; fig.savefig(path, dpi=220); written.append(path.name)
    plt.close(fig)

    # Ranked versus full latency advantage averaged across qualities/tile sizes/channels.
    matrix = []
    for ocm in (0.5, 1.0, 2.0):
        line = []
        for bandwidth in (0.5, 1.0, 2.0):
            keys = defaultdict(dict)
            for row in rows:
                if row["ocm_scale"] == ocm and row["bandwidth_scale"] == bandwidth:
                    key = (row["prediction_quality"], row["tile_size"], row["dma_channels"])
                    keys[key][row["scheduler"]] = int(row["latency_cycles"])
            line.append(fmean((values["full_expert"] / values["ranked_tile"] - 1) * 100 for values in keys.values()))
        matrix.append(line)
    fig, ax = plt.subplots(figsize=(6.4, 4.5))
    image = ax.imshow(matrix, cmap="RdYlGn", aspect="auto")
    for y, line in enumerate(matrix):
        for x, value in enumerate(line): ax.text(x, y, f"{value:.2f}%", ha="center", va="center")
    ax.set_xticks(range(3), ("0.5×", "1×", "2×")); ax.set_yticks(range(3), ("0.5×", "1×", "2×"))
    ax.set(xlabel="Bandwidth scale", ylabel="OCM scale", title="Ranked-tile latency advantage over full-Expert")
    fig.colorbar(image, ax=ax, label="Mean latency reduction (%)"); fig.tight_layout()
    for suffix in ("png", "pdf"):
        path = output / f"fig_m8_ranked_vs_full_heatmap.{suffix}"; fig.savefig(path, dpi=220); written.append(path.name)
    plt.close(fig)
    return written


def summarize(rows: list[dict[str, Any]], output: Path, plots: list[str]) -> None:
    """Write machine-readable and human-readable M8 summaries."""
    winners = Counter()
    groups = defaultdict(list)
    for row in rows:
        key = (row["prediction_quality"], row["tile_size"], row["ocm_scale"], row["bandwidth_scale"], row["dma_channels"])
        groups[key].append(row)
    for group in groups.values():
        best = min(int(row["latency_cycles"]) for row in group)
        for row in group:
            if int(row["latency_cycles"]) == best:
                winners[row["scheduler"]] += 1
    ranked = [row for row in rows if row["scheduler"] == "ranked_tile"]
    uniform = [row for row in rows if row["scheduler"] == "uniform_tile"]
    full = [row for row in rows if row["scheduler"] == "full_expert"]
    ranked_vs_full_delta = [
        int(next(item for item in group if item["scheduler"] == "full_expert")["latency_cycles"])
        - int(next(item for item in group if item["scheduler"] == "ranked_tile")["latency_cycles"])
        for group in groups.values()
    ]
    ranked_vs_full_by_ocm = {
        str(scale): fmean(
            float(row["speedup_vs_full_expert"]) for row in ranked
            if float(row["ocm_scale"]) == scale
        ) for scale in (0.5, 1.0, 2.0)
    }
    quality_speedups = {
        quality: {
            policy: fmean(
                float(row["speedup_vs_no_prefetch"]) for row in rows
                if row["prediction_quality"] == quality and row["scheduler"] == policy
            ) for policy in ("full_expert", "uniform_tile", "ranked_tile")
        } for quality in ("perfect", "high", "medium", "low")
    }
    summary = {
        "simulator_scope": "single_token_single_layer_tile_scheduling",
        "hardware_validation": False,
        "configuration_count": len(groups),
        "simulation_count": len(rows),
        "winner_memberships_including_ties": dict(sorted(winners.items())),
        "mean_ranked_speedup_vs_no_prefetch": fmean(float(row["speedup_vs_no_prefetch"]) for row in ranked),
        "mean_ranked_speedup_vs_full_expert": fmean(float(row["speedup_vs_full_expert"]) for row in ranked),
        "mean_prefetch_efficiency": {
            "full_expert": fmean(float(row["prefetch_efficiency"]) for row in full),
            "uniform_tile": fmean(float(row["prefetch_efficiency"]) for row in uniform),
            "ranked_tile": fmean(float(row["prefetch_efficiency"]) for row in ranked),
        },
        "ranked_vs_full_configuration_counts": {
            "ranked_faster": sum(delta > 0 for delta in ranked_vs_full_delta),
            "exact_tie": sum(delta == 0 for delta in ranked_vs_full_delta),
            "ranked_slower": sum(delta < 0 for delta in ranked_vs_full_delta),
        },
        "mean_ranked_speedup_vs_full_by_ocm_scale": ranked_vs_full_by_ocm,
        "mean_speedup_vs_no_prefetch_by_quality": quality_speedups,
        "plots": plots,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report = f"""# M8 Tile Scheduler Experiment Summary

This is simulator-based methodology validation on a generic synthetic profile; it is not hardware validation.

- Controlled configurations: {len(groups)}
- Simulator runs: {len(rows)}
- Mean ranked-tile speedup vs no prefetch: {summary['mean_ranked_speedup_vs_no_prefetch']:.4f}×
- Mean ranked-tile speedup vs full-Expert prefetch: {summary['mean_ranked_speedup_vs_full_expert']:.4f}×
- Winner memberships including ties: `{summary['winner_memberships_including_ties']}`
- Mean useful-byte efficiency: `{summary['mean_prefetch_efficiency']}`
- Ranked vs full-Expert configurations: `{summary['ranked_vs_full_configuration_counts']}`

## Answers to the four experiment questions

1. **Does tile scheduling outperform full-Expert prefetch?** On average, ranked tile is {summary['mean_ranked_speedup_vs_full_expert']:.4f}× faster, but not universally: it is faster in {summary['ranked_vs_full_configuration_counts']['ranked_faster']} configurations, tied in {summary['ranked_vs_full_configuration_counts']['exact_tie']}, and slower in {summary['ranked_vs_full_configuration_counts']['ranked_slower']}.
2. **Under what resource conditions?** Mean ranked/full speedup by OCM scale is `{summary['mean_ranked_speedup_vs_full_by_ocm_scale']}`. The strongest benefit occurs at 0.5× OCM, where only one full Expert fits but many startup prefixes fit. With more OCM, full Expert can tie or win. Multiple DMA channels make breadth easier to exploit; one channel can serialize speculative work enough to erase the advantage.
3. **How does quality affect allocation?** Speedups by quality are `{summary['mean_speedup_vs_no_prefetch_by_quality']}`. Ranked is strongest for useful high-ranked confidence, whereas uniform breadth is safer at low quality because it invests less depth in false early candidates.
4. **Does fine granularity reduce waste?** Mean useful-byte efficiency rises from {summary['mean_prefetch_efficiency']['full_expert']:.3f} for full Expert to {summary['mean_prefetch_efficiency']['uniform_tile']:.3f} for uniform and {summary['mean_prefetch_efficiency']['ranked_tile']:.3f} for ranked allocation. This is an aggregate result; individual configurations can reverse it.

The correct conclusion is conditional: policy benefits depend on OCM, bandwidth, channel count, tile size, and true-Expert ranks. This is methodology validation on a synthetic profile, not hardware validation.

Artifacts:

- `experiment.csv`
- `summary.json`
- `{'`, `'.join(plots)}`
"""
    (output / "SUMMARY.md").write_text(report, encoding="utf-8")


def validate(rows: list[dict[str, Any]], output: Path) -> None:
    """Validate sweep completeness and conservation before declaring success."""
    if len(rows) != 3 * 3 * 3 * 4 * 3 * 4:
        raise AssertionError(f"unexpected M8 sweep size: {len(rows)}")
    for row in rows:
        if int(row["planned_bytes"]) > int(row["plan_budget_bytes"]):
            raise AssertionError("plan exceeded budget")
        if int(row["peak_ocm_bytes"]) > int(row["ocm_capacity_bytes"]):
            raise AssertionError("OCM capacity exceeded")
        if int(row["prefetched_tiles"]) != int(row["useful_prefetched_tiles"]) + int(row["wasted_prefetched_tiles"]):
            raise AssertionError("tile conservation failed")
        if int(row["prefetched_bytes"]) != int(row["useful_prefetched_bytes"]) + int(row["wasted_prefetched_bytes"]):
            raise AssertionError("byte conservation failed")
    required = ["experiment.csv", "summary.json", "SUMMARY.md"]
    required += [f"fig_m8_{name}.{suffix}" for name in ("latency_vs_ocm", "efficiency_vs_quality", "ranked_vs_full_heatmap") for suffix in ("png", "pdf")]
    if not all((output / name).is_file() and (output / name).stat().st_size > 0 for name in required):
        raise AssertionError("missing M8 artifact")
    (output / "artifact_validation.json").write_text(json.dumps({
        "status": "PASS", "rows": len(rows), "configurations": len(rows) // 4,
        "tile_and_byte_conservation": True, "ocm_capacity_invariant": True,
        "hardware_validation": False,
    }, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    """Execute the complete bounded M8 sweep and generate tables, report, and plots."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    config = load_config(args.config)
    experiment = config["experiment"]
    rows = [
        run_one(
            config, scheduler_name=scheduler, quality=quality,
            tile_size_name=tile_name, f_tile_size=int(f_tile_size),
            ocm_scale=float(ocm), bandwidth_scale=float(bandwidth), channels=int(channels),
        )
        for ocm in experiment["ocm_scales"]
        for bandwidth in experiment["bandwidth_scales"]
        for channels in experiment["dma_channels"]
        for quality in experiment["prediction_qualities"]
        for tile_name, f_tile_size in experiment["tile_sizes"].items()
        for scheduler in experiment["schedulers"]
    ]
    add_comparisons(rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "experiment.csv", rows)
    plots = make_plots(args.output_dir, rows)
    summarize(rows, args.output_dir, plots)
    validate(rows, args.output_dir)
    print(f"Wrote {len(rows)} M8 simulations to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
