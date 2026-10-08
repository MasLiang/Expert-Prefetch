"""多seed收益转折加密扫描；复用既有主模拟器执行函数，不改变调度语义。"""
from __future__ import annotations

import argparse
import csv
import hashlib
from itertools import product
import json
import multiprocessing as mp
from pathlib import Path
from statistics import mean, stdev
from typing import Any

import yaml

from prefetch_phase_sweep import ROOT, make_inputs, simulate_job, write_csv

HardwareKey = tuple[int, int, float, float]
PolicyKey = tuple[str, float, int]


def build_points(config: dict[str, Any]) -> dict[HardwareKey, list[PolicyKey]]:
    """生成主矩阵和局部质量/深度消融；按有效深度去重等价配置。"""
    if len(set(config["seeds"])) != len(config["seeds"]) or len(config["seeds"]) < 2:
        raise ValueError("dense scan requires at least two distinct seeds")
    if config["samples_per_seed"] <= 0 or any(r <= 0 for r in config["compute_to_memory_ratios"]):
        raise ValueError("sample count and ratios must be positive")
    if not 0 <= config["fixed_quality"] <= 1 or config["fixed_depth"] < 1:
        raise ValueError("invalid fixed policy")
    points: dict[HardwareKey, set[PolicyKey]] = {}
    for groups, pe, capacity, ratio in product(config["tile_groups"], config["pe_counts"],
            config["ocm_expert_equivalents"], config["compute_to_memory_ratios"]):
        key = (groups, pe, float(ratio), float(capacity))
        points[key] = {("no_prediction", 0., 0), ("oracle_all_gu", 1., groups),
            ("uniform_prefix", config["fixed_quality"], min(config["fixed_depth"], groups))}
    sensitivity = config["sensitivity"]
    for groups, ratio, quality, depth in product(sensitivity["tile_groups"], sensitivity["ratios"],
            sensitivity["qualities"], sensitivity["depths"]):
        if not 0 <= quality <= 1 or depth < 1:
            raise ValueError("invalid sensitivity quality/depth")
        key = (groups, sensitivity["pe_count"], float(ratio), float(sensitivity["ocm_expert_equivalents"]))
        points.setdefault(key, {("no_prediction", 0., 0), ("oracle_all_gu", 1., groups)})
        points[key].add(("uniform_prefix", quality, min(depth, groups)))
    return {key: sorted(value) for key, value in sorted(points.items())}


def aggregate_seeds(rows: list[dict[str, Any]], seed_count: int,
                    thresholds: list[float]) -> list[dict[str, Any]]:
    """按配对seed速度比统计均值/标准差/极值；极值带不冒充置信区间。"""
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    identity = ["tile_groups", "pe_count", "ratio", "ocm_expert_equivalents", "policy", "quality", "prefix_depth"]
    for row in rows:
        grouped.setdefault(tuple(row[k] for k in identity), []).append(row)
    result = []
    for key, samples in sorted(grouped.items()):
        if len(samples) != seed_count or len({r["seed"] for r in samples}) != seed_count:
            raise ValueError("incomplete or duplicate seed group")
        values = [r["speedup"] for r in samples]
        row = dict(zip(identity, key))
        row.update(seeds=seed_count, speedup_mean=mean(values), speedup_std=stdev(values),
            speedup_min=min(values), speedup_max=max(values),
            pooled_speedup=sum(r["baseline_cycles"] for r in samples) / sum(r["total_cycles"] for r in samples),
            net_saved_cycles_mean=mean(r["net_saved_cycles"] for r in samples),
            first_start_saved_cycles_mean=mean(r["first_start_saved_cycles"] for r in samples),
            future_blocking_cycles_mean=mean(r["future_speculation_blocking_cycles"] for r in samples),
            useful_bytes_mean=mean(r["useful_speculative_bytes"] for r in samples),
            wasted_bytes_mean=mean(r["wasted_speculative_bytes"] for r in samples),
            extra_transfer_bytes_mean=mean(r["extra_transfer_bytes"] for r in samples))
        for threshold in thresholds:
            row[f"seeds_ge_{threshold:g}"] = sum(v >= threshold for v in values)
        result.append(row)
    return result


def sampled_regions(rows: list[dict[str, Any]], config: dict[str, Any]) -> list[dict[str, Any]]:
    """输出固定策略的连续通过采样段；不推断点间插值的充分条件。"""
    result = []
    for groups, pe, capacity, threshold in product(config["tile_groups"], config["pe_counts"],
            config["ocm_expert_equivalents"], config["thresholds"]):
        selected = sorted([r for r in rows if r["tile_groups"] == groups and r["pe_count"] == pe
            and r["ocm_expert_equivalents"] == capacity and r["policy"] == "uniform_prefix"
            and r["quality"] == config["fixed_quality"]
            and r["prefix_depth"] == min(config["fixed_depth"], groups)
            and r["ratio"] in config["compute_to_memory_ratios"]], key=lambda r: r["ratio"])
        segments, current = [], []
        for row in selected:
            if row["speedup_min"] >= threshold:
                current.append(row["ratio"])
            elif current:
                segments.append(current)
                current = []
        if current:
            segments.append(current)
        result.append({"tile_groups": groups, "pe_count": pe, "ocm_expert_equivalents": capacity,
            "quality": config["fixed_quality"], "depth": config["fixed_depth"], "threshold": threshold,
            "tested_points": len(selected), "all_seeds_passing_points": sum(len(s) for s in segments),
            "consecutive_passing_sample_segments": json.dumps(segments)})
    return result


def plot_results(directory: Path, rows: list[dict[str, Any]], config: dict[str, Any]) -> None:
    """画固定策略均值/min-max带和局部质量深度消融；不展示事后最优包络。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(len(config["tile_groups"]), len(config["ocm_expert_equivalents"]),
        figsize=(12, 10), squeeze=False)
    for i, groups in enumerate(config["tile_groups"]):
        for j, capacity in enumerate(config["ocm_expert_equivalents"]):
            ax = axes[i][j]
            for pe in config["pe_counts"]:
                points = sorted([r for r in rows if r["tile_groups"] == groups and r["pe_count"] == pe
                    and r["ocm_expert_equivalents"] == capacity and r["policy"] == "uniform_prefix"
                    and r["quality"] == config["fixed_quality"]
                    and r["prefix_depth"] == min(config["fixed_depth"], groups)], key=lambda r: r["ratio"])
                x = [r["ratio"] for r in points]
                line, = ax.plot(x, [r["speedup_mean"] for r in points], marker=".", label=f"PE={pe}")
                ax.fill_between(x, [r["speedup_min"] for r in points], [r["speedup_max"] for r in points],
                    color=line.get_color(), alpha=.2)
            ax.axhline(1, color="black", linestyle="--")
            ax.axhline(1.05, color="grey", linestyle=":")
            ax.set_xscale("log")
            ax.set_title(f"{groups} F groups; OCM={capacity} Experts")
            ax.set_xlabel("Ideal compute / memory payload time")
            ax.set_ylabel("Speedup vs no-prediction streaming")
            ax.grid(alpha=.2)
            ax.legend()
    fig.suptitle("Fixed q=0.9 / GU depth=1; mean and min-max across 5 seeds (NOT CI)")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(directory / f"dense_curves.{ext}")
    plt.close(fig)
    sensitivity = config["sensitivity"]
    fig, axes = plt.subplots(len(sensitivity["tile_groups"]), 2, figsize=(12, 7), squeeze=False)
    for i, groups in enumerate(sensitivity["tile_groups"]):
        for j, ratio in enumerate([1., 2.]):
            ax = axes[i][j]
            for depth in sorted({min(d, groups) for d in sensitivity["depths"]}):
                points = sorted([r for r in rows if r["policy"] == "uniform_prefix" and
                    r["tile_groups"] == groups and r["pe_count"] == sensitivity["pe_count"] and
                    r["ocm_expert_equivalents"] == sensitivity["ocm_expert_equivalents"] and
                    r["ratio"] == ratio and r["prefix_depth"] == depth], key=lambda r: r["quality"])
                ax.errorbar([r["quality"] for r in points], [r["speedup_mean"] for r in points],
                    yerr=[r["speedup_std"] for r in points], marker="o", label=f"GU depth={depth}")
            ax.axhline(1, color="black", linestyle="--")
            ax.set_title(f"{groups} groups; ratio={ratio}")
            ax.set_xlabel("Quality (recall AND rank-1 precision)")
            ax.set_ylabel("Mean speedup +/- seed sample SD")
            ax.legend(fontsize=8)
    fig.suptitle("Local sensitivity: PE=1; OCM=4 Experts; no policy optimization")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(directory / f"quality_depth_sensitivity.{ext}")
    plt.close(fig)


def main() -> None:
    """完成多seed扫描、约束校验、配对统计、采样段和图表。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/experiments/prefetch_dense.yaml")
    parser.add_argument("--output", type=Path, default=ROOT / "results/prefetch_dense")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if args.workers < 1:
        raise ValueError("workers must be positive")
    args.config, args.output = args.config.resolve(), args.output.resolve()
    config = yaml.safe_load(args.config.read_text())
    base_path = ROOT / config["base_config"]
    base = yaml.safe_load(base_path.read_text())
    base["samples"] = config["samples_per_seed"]
    base["qualities"] = sorted(set([config["fixed_quality"], *config["sensitivity"]["qualities"]]))
    points = build_points(config)
    traces = {seed: make_inputs({**base, "seed": seed}) for seed in config["seeds"]}
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "inputs.json").write_text(json.dumps(traces, indent=2))
    (args.output / "summary.json").write_text(json.dumps({"status": "running", "config": config}))
    raw, seed_rows = [], []
    summed = ["total_cycles", "first_start_sum_cycles", "future_speculation_blocking_cycles", "evicted_future_bytes",
        "demand_bytes", "speculative_bytes", "useful_speculative_bytes", "wasted_speculative_bytes",
        "useful_completed_before_router_bytes", "realized_lead_sum_cycles"]
    with mp.get_context("spawn").Pool(args.workers) as pool:
        for index, ((groups, pe, ratio, capacity), policies) in enumerate(points.items()):
            bw = ratio * pe * base["ops_per_cycle_per_pe"] / (2 * base["channels"])
            batches = []
            for seed in config["seeds"]:
                settings = {**base, "seed": seed}
                for policy, quality, depth in policies:
                    point = dict(tile_groups=groups, pe_count=pe, bytes_per_cycle_per_channel=bw,
                        ocm_expert_equivalents=capacity, policy=policy, quality=quality, prefix_depth=depth)
                    batches.extend((settings, point, record) for record in traces[seed])
            values = pool.map(simulate_job, batches)
            offset = 0
            for seed in config["seeds"]:
                block = values[offset:offset + len(policies) * base["samples"]]
                baseline_samples = [r for r in block if r["policy"] == "no_prediction"]
                baseline = sum(r["total_cycles"] for r in baseline_samples)
                first_start = sum(r["first_start_sum_cycles"] for r in baseline_samples)
                traffic = sum(r["demand_bytes"] for r in baseline_samples)
                for pi, (policy, quality, depth) in enumerate(policies):
                    samples = block[pi * base["samples"]:(pi + 1) * base["samples"]]
                    for sample in samples:
                        raw.append({"seed": seed, "ratio": ratio, **sample})
                    row = {"seed": seed, "ratio": ratio, **samples[0]}
                    del row["sample"]
                    for key in summed:
                        row[key] = sum(r[key] for r in samples)
                    row["peak_ocm_bytes"] = max(r["peak_ocm_bytes"] for r in samples)
                    row.update(samples=base["samples"], baseline_cycles=baseline,
                        speedup=baseline / row["total_cycles"], net_saved_cycles=baseline - row["total_cycles"],
                        first_start_saved_cycles=first_start - row["first_start_sum_cycles"],
                        extra_transfer_bytes=row["demand_bytes"] + row["speculative_bytes"] - traffic)
                    seed_rows.append(row)
                offset += len(policies) * base["samples"]
            write_csv(args.output / "per_seed.csv", seed_rows)
            if (index + 1) % 12 == 0 or index + 1 == len(points):
                print(f"completed hardware points {index + 1}/{len(points)}", flush=True)
    write_csv(args.output / "per_sample.csv", raw)
    aggregate = aggregate_seeds(seed_rows, len(config["seeds"]), config["thresholds"])
    write_csv(args.output / "comparison.csv", aggregate)
    write_csv(args.output / "benefit_regions.csv", sampled_regions(aggregate, config))
    plot_results(args.output, aggregate, config)
    sources = [args.config, base_path, Path(__file__), ROOT / "examples/prefetch_phase_sweep.py",
               *sorted((ROOT / "src/edge_moe_sim").rglob("*.py"))]
    summary = {"status": "complete", "simulator": "UnifiedSequenceSimulator", "config": config, "base_settings": base,
        "hardware_points": len(points), "configuration_points": len(aggregate), "seed_points": len(seed_rows),
        "sample_results": len(raw), "source_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sources}, "limits": ["synthetic Top-1, sequential Experts, same unchanged main simulator",
            "quality changes both recall and rank-1 precision; fixed candidate count=2",
            "mean/std/min/max across seeds, NOT confidence intervals",
            "passing segments are sampled points, NOT universal or interpolated guarantees",
            "whole tile DMA non-preemptive; Oracle true-only all-GU is a heuristic not an optimal bound"]}
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
