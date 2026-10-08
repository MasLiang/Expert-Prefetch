"""合成 PE/带宽/容量扫描；使用主模拟器，寻找预测预取的经验收益区域。"""
from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
import hashlib
from itertools import product
import json
import multiprocessing as mp
from pathlib import Path
import random
import sys
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from edge_moe_sim.advanced_schema import AdvancedExecutionConfig
from edge_moe_sim.batch_predictor import BatchPredictorOutput, PrecomputedBatchPredictor, TokenPrediction
from edge_moe_sim.batch_routing import BatchRoutingEvent
from edge_moe_sim.hardware_profile import HardwareProfile
from edge_moe_sim.schema import CandidatePrefetch, ModelConfig
from edge_moe_sim.unified_schema import UnifiedLayerSequence, UnifiedPredictionSpec
from edge_moe_sim.unified_sequence_simulator import UnifiedSequenceSimulator


def make_inputs(settings: dict[str, Any]) -> list[dict[str, Any]]:
    """生成共享truth和嵌套质量等级；质量同时控制rank-1命中和候选recall。"""
    if settings["top_k"] != 1 or settings["candidate_count"] != 2:
        raise ValueError("phase experiment currently requires Top-1, two candidates")
    if settings["num_experts"] < 3 or settings["layers"] < 2 or settings["samples"] < 1:
        raise ValueError("invalid synthetic trace geometry")
    if any(not 0 <= q <= 1 for q in settings["qualities"]):
        raise ValueError("quality must be in [0,1]")
    rng = random.Random(settings["seed"])
    count = settings["samples"] * (settings["layers"] - 1)
    positions = list(range(count))
    rng.shuffle(positions)
    quality_order = {position: rank for rank, position in enumerate(positions)}
    records = []
    for sample in range(settings["samples"]):
        truth = [rng.randrange(settings["num_experts"]) for _ in range(settings["layers"])]
        targets = []
        for target in range(1, settings["layers"]):
            targets.append({"target": target, "wrong_ids": rng.sample(
                [e for e in range(settings["num_experts"]) if e != truth[target]], 2),
                "quality_order": quality_order[sample * (settings["layers"] - 1) + target - 1]})
        records.append({"sample": sample, "truth": truth, "targets": targets})
    return records


def candidates(record: dict[str, Any], target: dict[str, Any], quality: float,
               decisions: int, oracle: bool = False) -> tuple[int, ...]:
    """Oracle仅输出truth；其他质量等级命中时truth在rank-1，否则完全miss。"""
    true = record["truth"][target["target"]]
    if oracle:
        return (true,)
    if target["quality_order"] < int(quality * decisions + 0.5):
        return (true, target["wrong_ids"][1])
    return tuple(target["wrong_ids"])


def simulate_job(job: tuple[dict[str, Any], dict[str, Any], dict[str, Any]]) -> dict[str, Any]:
    """在单顺序Expert上用全部PE；不将增加PE误当成增加Expert并行度。"""
    settings, point, record = job
    h, f = settings["hidden_size"], settings["f_tile_size"] * point["tile_groups"]
    model = ModelConfig(name="synthetic_phase", hidden_size=h, intermediate_size=f,
        num_experts=settings["num_experts"], top_k=1, weight_bits=8, activation_bits=8,
        moe_layer_indices=tuple(range(settings["layers"])))
    expert_bytes = 3 * h * f
    capacity = int(expert_bytes * point["ocm_expert_equivalents"])
    hardware = HardwareProfile(name="synthetic_PE_not_board", clock_hz=settings["clock_hz"],
        num_memory_channels=settings["channels"],
        bandwidth_bytes_per_sec_per_channel=point["bytes_per_cycle_per_channel"] * settings["clock_hz"],
        bandwidth_efficiency=1, dma_setup_cycles=settings["dma_setup_cycles"],
        max_inflight_per_channel=1, burst_bytes=16384, supports_channel_striping=True,
        total_aie_units=point["pe_count"], ops_per_cycle_per_aie=settings["ops_per_cycle_per_pe"],
        aie_efficiency=1, aie_task_startup_cycles=settings["compute_startup_cycles"], ocm_bytes=capacity)
    sequence = UnifiedLayerSequence(tuple(BatchRoutingEvent(record["sample"], 0, layer, ((e,),))
        for layer, e in enumerate(record["truth"])))
    specs, predictors = [], {}
    oracle = point["policy"] == "oracle_all_gu"
    count = 1 if oracle else settings["candidate_count"]
    depth = min(point["prefix_depth"], point["tile_groups"])
    if point["policy"] != "no_prediction":
        decisions = settings["samples"] * (settings["layers"] - 1)
        for target in record["targets"]:
            layer = target["target"]
            ids = candidates(record, target, point["quality"], decisions, oracle)
            output = BatchPredictorOutput(record["sample"], 0, layer, 0,
                (TokenPrediction(0, tuple(CandidatePrefetch(e, 1 / (i + 1)) for i, e in enumerate(ids))),))
            predictors[(layer - 1, layer)] = PrecomputedBatchPredictor({(record["sample"], 0, layer): output})
            specs.append(UnifiedPredictionSpec(layer - 1, layer, "source_router_done",
                settings["predictor_latency_cycles"], count, "uniform_prefix", depth,
                wait_for_previous_layer_weights=False))
    execution = AdvancedExecutionConfig(settings["f_tile_size"], placement_policy="striped_tile",
        aie_units_per_expert=point["pe_count"], router_aie_units=point["pe_count"],
        max_parallel_experts=1, candidate_count=count, speculative_byte_budget=capacity,
        nonlinear_elements_per_cycle=settings["nonlinear_elements_per_cycle"])
    result = UnifiedSequenceSimulator(model, hardware).simulate(sequence, execution,
        predictions=tuple(specs), predictors=predictors)
    if result.peak_ocm_bytes > capacity or len(result.layer_results) != settings["layers"]:
        raise AssertionError("capacity violation or incomplete simulation")
    metrics = result.memory_metrics
    if metrics["speculative_bytes"] != metrics["useful_speculative_bytes"] + metrics["wasted_speculative_bytes"]:
        raise AssertionError("speculative byte accounting violation")
    return {**point, "sample": record["sample"], "expert_bytes": expert_bytes,
        "ocm_bytes": capacity, "effective_depth": depth,
        "compute_to_memory_payload_ratio": 2 * settings["channels"] * point["bytes_per_cycle_per_channel"] /
            (point["pe_count"] * settings["ops_per_cycle_per_pe"]),
        "prefetched_gu_fraction_of_expert": 2 * depth / (3 * point["tile_groups"]),
        "total_cycles": result.inference_makespan_cycles, "peak_ocm_bytes": result.peak_ocm_bytes,
        "first_start_sum_cycles": sum(r.first_true_expert_start - r.router_resolve_cycle
            for r in result.layer_results.values()),
        "future_speculation_blocking_cycles": result.critical_path_future_speculation_blocking_cycles,
        "evicted_future_bytes": result.evicted_future_speculative_bytes,
        "demand_bytes": sum(r.demand_bytes for r in result.layer_results.values()),
        "speculative_bytes": metrics["speculative_bytes"],
        "useful_speculative_bytes": metrics["useful_speculative_bytes"],
        "wasted_speculative_bytes": metrics["wasted_speculative_bytes"],
        "useful_completed_before_router_bytes": sum(p.useful_completed_speculative_bytes_before_router
            for p in result.predictions),
        "realized_lead_sum_cycles": sum(p.realized_lead_cycles for p in result.predictions)}


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """写出确定字段顺序的非空CSV。"""
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def make_plots(directory: Path, rows: list[dict[str, Any]], settings: dict[str, Any]) -> None:
    """生成固定深度曲线与明确标记为事后最优深度的相图。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    groups = settings["tile_groups"]
    capacities = settings["ocm_expert_equivalents"]
    fig, axes = plt.subplots(len(groups), len(capacities), figsize=(13, 10), squeeze=False)
    for i, tiles in enumerate(groups):
        for j, capacity in enumerate(capacities):
            ax = axes[i][j]
            for quality in settings["qualities"]:
                selected = [r for r in rows if r["policy"] == "uniform_prefix" and
                    r["tile_groups"] == tiles and r["ocm_expert_equivalents"] == capacity and
                    r["quality"] == quality and r["prefix_depth"] == 4]
                for pe in settings["pe_counts"]:
                    points = sorted([r for r in selected if r["pe_count"] == pe],
                        key=lambda r: r["compute_to_memory_payload_ratio"])
                    ax.plot([r["compute_to_memory_payload_ratio"] for r in points],
                        [r["speedup"] for r in points], marker=".",
                        label=f"q={quality}; PE={pe}")
            ax.axhline(1, color="black", linestyle="--")
            ax.axhline(1.05, color="grey", linestyle=":")
            ax.set_xscale("log")
            ax.set_title(f"{tiles} F groups; OCM={capacity} experts")
            ax.set_xlabel("Compute / memory payload time (analytical)")
            ax.set_ylabel("Speedup vs no-prediction streaming")
    axes[0][-1].legend(fontsize=6)
    fig.suptitle("Synthetic single-expert execution; fixed requested GU depth=4")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(directory / f"fixed_depth_curves.{ext}")
    plt.close(fig)
    fig, axes = plt.subplots(len(groups), len(capacities), figsize=(12, 9), squeeze=False)
    for i, tiles in enumerate(groups):
        for j, capacity in enumerate(capacities):
            matrix = []
            for pe in settings["pe_counts"]:
                matrix.append([max(r["speedup"] for r in rows if r["policy"] == "uniform_prefix"
                    and r["tile_groups"] == tiles and r["ocm_expert_equivalents"] == capacity
                    and r["quality"] == 0.9 and r["pe_count"] == pe
                    and r["bytes_per_cycle_per_channel"] == bw)
                    for bw in settings["bytes_per_cycle_per_channel"]])
            ax = axes[i][j]
            im = ax.imshow(matrix, origin="lower", vmin=0.85, vmax=1.15, cmap="RdYlGn", aspect="auto")
            for y, line in enumerate(matrix):
                for x, value in enumerate(line):
                    ax.text(x, y, f"{value:.3f}", ha="center", va="center", fontsize=8)
            ax.set_xticks(range(len(settings["bytes_per_cycle_per_channel"])), settings["bytes_per_cycle_per_channel"])
            ax.set_yticks(range(len(settings["pe_counts"])), settings["pe_counts"])
            ax.set_xlabel("Bytes/cycle/channel (4 channels)")
            ax.set_ylabel("PE count")
            ax.set_title(f"{tiles} groups; OCM={capacity} experts")
    fig.colorbar(im, ax=axes.ravel().tolist(), label="Speedup", shrink=0.6)
    fig.suptitle("q=0.9; hindsight best tested depth (NOT fixed deployable policy)")
    for ext in ("png", "pdf"):
        fig.savefig(directory / f"benefit_phase.{ext}")
    plt.close(fig)


def main() -> None:
    """运行完整合成矩阵、保存原始输入/结果、经验阈值、图和来源校验码。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/experiments/prefetch_phase.yaml")
    parser.add_argument("--output", type=Path, default=ROOT / "results/prefetch_phase")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if args.workers < 1:
        raise ValueError("workers must be positive")
    args.config = args.config.resolve()
    args.output = args.output.resolve()
    settings = yaml.safe_load(args.config.read_text())
    records = make_inputs(settings)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "inputs.json").write_text(json.dumps(records, indent=2))
    (args.output / "summary.json").write_text(json.dumps({"status": "running"}))
    raw, rows = [], []
    with mp.get_context("spawn").Pool(args.workers) as pool:
        for tiles, pe, bw, capacity in product(settings["tile_groups"], settings["pe_counts"],
                settings["bytes_per_cycle_per_channel"], settings["ocm_expert_equivalents"]):
            base = {"tile_groups": tiles, "pe_count": pe, "bytes_per_cycle_per_channel": bw,
                    "ocm_expert_equivalents": capacity}
            points = [{**base, "policy": "no_prediction", "quality": 0.0, "prefix_depth": 0},
                      {**base, "policy": "oracle_all_gu", "quality": 1.0, "prefix_depth": tiles}]
            points += [{**base, "policy": "uniform_prefix", "quality": q, "prefix_depth": depth}
                       for q, depth in product(settings["qualities"], settings["prefix_depths"])]
            values = pool.map(simulate_job, [(settings, point, record) for point in points for record in records])
            raw.extend(values)
            baseline = sum(r["total_cycles"] for r in values[:len(records)])
            for index, point in enumerate(points):
                samples = values[index * len(records):(index + 1) * len(records)]
                row = dict(samples[0])
                del row["sample"]
                for key in ["total_cycles", "first_start_sum_cycles", "future_speculation_blocking_cycles",
                        "evicted_future_bytes", "demand_bytes", "speculative_bytes", "useful_speculative_bytes",
                        "wasted_speculative_bytes", "useful_completed_before_router_bytes", "realized_lead_sum_cycles"]:
                    row[key] = sum(r[key] for r in samples)
                row["peak_ocm_bytes"] = max(r["peak_ocm_bytes"] for r in samples)
                row["samples"] = len(records)
                row["observed_recall"] = (int(point["quality"] * settings["samples"] *
                    (settings["layers"] - 1) + 0.5) / (settings["samples"] * (settings["layers"] - 1)))
                row["speedup"] = baseline / row["total_cycles"]
                row["net_saved_cycles"] = baseline - row["total_cycles"]
                rows.append(row)
            write_csv(args.output / "comparison.csv", rows)
            print(f"completed groups={tiles} PE={pe} B/cycle/ch={bw} OCM/Expert={capacity}", flush=True)
    write_csv(args.output / "per_sample.csv", raw)
    thresholds = []
    for policy, q, depth in [("oracle_all_gu", 1.0, -1)] + [
            ("uniform_prefix", q, depth) for q, depth in product(settings["qualities"], settings["prefix_depths"])]:
        for ratio in sorted({r["compute_to_memory_payload_ratio"] for r in rows}):
            selected = [r for r in rows if r["policy"] == policy and r["quality"] == q and
                (depth == -1 or r["prefix_depth"] == depth) and r["compute_to_memory_payload_ratio"] == ratio]
            for threshold in settings["benefit_thresholds"]:
                thresholds.append({"policy": policy, "quality": q, "requested_depth": depth,
                    "compute_to_memory_payload_ratio": ratio, "speedup_threshold": threshold,
                    "points": len(selected), "passing_points": sum(r["speedup"] >= threshold for r in selected),
                    "min_speedup": min(r["speedup"] for r in selected),
                    "max_speedup": max(r["speedup"] for r in selected)})
    write_csv(args.output / "thresholds.csv", thresholds)
    make_plots(args.output, rows, settings)
    sources = [args.config, Path(__file__), *sorted((ROOT / "src/edge_moe_sim").rglob("*.py"))]
    summary = {"status": "complete", "simulator": "UnifiedSequenceSimulator", "settings": settings,
        "sample_results": len(raw), "configuration_points": len(rows),
        "source_sha256": {str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p):
            hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
        "limits": ["synthetic models/hardware; not measured", "Top-1, sequential experts, PE allocated to each GEMM",
            "quality jointly varies recall and rank-1 precision; candidate confidence is not calibrated",
            "Oracle is true-only all-GU heuristic, NOT optimal schedule or guaranteed latency lower bound",
            "no burst arbitration; no new scheduler semantics", "ratio excludes setup, nonlinear and placement constraints",
            "5 samples and 5 layers per point; boundaries are sampled empirical observations, not universal proof"]}
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
