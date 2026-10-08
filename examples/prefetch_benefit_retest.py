"""复测历史收益点，再以配对激励扫描尺寸、Top-K、深度和容量；不改模拟器。"""
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
from statistics import mean, stdev
from typing import Any

import yaml

from prefetch_phase_sweep import (ROOT, AdvancedExecutionConfig, BatchPredictorOutput,
    BatchRoutingEvent, CandidatePrefetch, HardwareProfile, ModelConfig,
    PrecomputedBatchPredictor, TokenPrediction, UnifiedLayerSequence,
    UnifiedPredictionSpec, UnifiedSequenceSimulator, candidates, simulate_job, write_csv)


def extend_record(record: dict[str, Any], seed: int, top_k: int,
                  num_experts: int) -> dict[str, Any]:
    """保留原 Top-1 truth/rank；嵌套扩展 truth，命中决策在各 K 中完全相关。"""
    if top_k < 1 or 3 * top_k > num_experts:
        raise ValueError("Top-K requires E >= 3K for disjoint miss candidates")
    rng = random.Random(seed * 1000 + record["sample"])
    truths = []
    for original in record["truth"]:
        # Always draw the maximum tested expansion, so K is a nested prefix.
        extras = rng.sample([e for e in range(num_experts) if e != original], num_experts - 1)
        truths.append([original, *extras[:top_k - 1]])
    targets = []
    for old in record["targets"]:
        truth = truths[old["target"]]
        wrong = [e for e in old["wrong_ids"] if e not in truth]
        tail = [e for e in range(num_experts) if e not in truth and e not in wrong]
        rng.shuffle(tail)
        wrong = (wrong + tail)[:2 * top_k]
        targets.append({**old, "wrong_ids": wrong})
    return {"sample": record["sample"], "truth": truths, "targets": targets}


def prediction_ids(record: dict[str, Any], target: dict[str, Any],
                   quality: float, decisions: int) -> tuple[int, ...]:
    """命中层包含全部 K 个 truth 且排名在前；miss 层 2K 候选均错。"""
    truth = record["truth"][target["target"]]
    k = len(truth)
    if target["quality_order"] < int(quality * decisions + .5):
        # Match historical Top-1's second candidate exactly.
        wrong = target["wrong_ids"][1:] + target["wrong_ids"][:1]
        return tuple(truth + wrong[:k])
    return tuple(target["wrong_ids"])


def build_points(config: dict[str, Any], base: dict[str, Any]) -> list[dict[str, Any]]:
    """构建去重后的硬件/策略点；baseline不按预取深度重复运行。"""
    if len(set(config["seeds"])) != len(config["seeds"]) or len(config["seeds"]) < 2:
        raise ValueError("at least two distinct seeds required")
    if not 0 <= config["quality"] <= 1:
        raise ValueError("quality must be in [0,1]")
    if any(k < 1 or 3 * k > base["num_experts"] for k in config["top_ks"]):
        raise ValueError("invalid top_ks")
    if set(config["capacity_modes"]) - {"scaled_expert", "fixed_bytes"}:
        raise ValueError("unknown capacity mode")
    if set(config["depth_modes"]) - {"shallow", "proportional"}:
        raise ValueError("unknown depth mode")
    points = []
    for groups, ratio in product(config["historical_groups"], config["historical_ratios"]):
        for policy in ("no_prediction", "uniform_prefix"):
            points.append(dict(stage="historical", tile_groups=groups, ratio=ratio, top_k=1,
                capacity_mode="scaled_expert", depth_mode="shallow", prefix_depth=int(policy != "no_prediction"),
                policy=policy))
    for groups, ratio, k, capacity in product(config["growth_groups"], config["growth_ratios"],
                                            config["top_ks"], config["capacity_modes"]):
        common = dict(stage="growth", tile_groups=groups, ratio=ratio, top_k=k, capacity_mode=capacity)
        points.append(dict(**common, depth_mode="none", prefix_depth=0, policy="no_prediction"))
        seen = set()
        for mode in config["depth_modes"]:
            depth = 1 if mode == "shallow" else max(1, (groups + 1) // 2)
            if depth not in seen:
                points.append(dict(**common, depth_mode=mode, prefix_depth=depth, policy="uniform_prefix"))
                seen.add(depth)
    for point in points:
        if point["tile_groups"] < 1 or point["ratio"] <= 0:
            raise ValueError("groups and ratios must be positive")
        size_groups = point["tile_groups"] if point["capacity_mode"] == "scaled_expert" else config["anchor_groups"]
        point["ocm_bytes"] = int(3 * base["hidden_size"] * base["f_tile_size"] * size_groups *
                                 config["ocm_expert_equivalents"])
        point["bytes_per_cycle_per_channel"] = point["ratio"] * base["ops_per_cycle_per_pe"] / (2 * base["channels"])
        point["quality"] = config["quality"] if point["policy"] != "no_prediction" else 0.
    return points


def worker(job: tuple[dict[str, Any], dict[str, Any], int, dict[str, Any]]) -> dict[str, Any]:
    """运行当前主模型；历史阶段原封不动复用历史执行函数和保存 trace。"""
    base, point, seed, record = job
    groups, k = point["tile_groups"], point["top_k"]
    if point["stage"] == "historical":
        legacy_point = dict(tile_groups=groups, pe_count=1,
            bytes_per_cycle_per_channel=point["bytes_per_cycle_per_channel"],
            ocm_expert_equivalents=point["ocm_bytes"] / (3 * base["hidden_size"] * base["f_tile_size"] * groups),
            policy=point["policy"], quality=point["quality"], prefix_depth=point["prefix_depth"])
        values = simulate_job(({**base, "seed": seed}, legacy_point, record))
        return {**values, **point, "seed": seed}
    extended = extend_record(record, seed, k, base["num_experts"])
    model = ModelConfig(name="synthetic_benefit_growth", hidden_size=base["hidden_size"],
        intermediate_size=base["f_tile_size"] * groups, num_experts=base["num_experts"],
        top_k=k, weight_bits=8, activation_bits=8, moe_layer_indices=tuple(range(base["layers"])))
    hardware = HardwareProfile(name="synthetic_not_board", clock_hz=base["clock_hz"],
        num_memory_channels=base["channels"], bandwidth_bytes_per_sec_per_channel=
            point["bytes_per_cycle_per_channel"] * base["clock_hz"], bandwidth_efficiency=1,
        dma_setup_cycles=base["dma_setup_cycles"], max_inflight_per_channel=1, burst_bytes=16384,
        supports_channel_striping=True, total_aie_units=1, ops_per_cycle_per_aie=base["ops_per_cycle_per_pe"],
        aie_efficiency=1, aie_task_startup_cycles=base["compute_startup_cycles"], ocm_bytes=point["ocm_bytes"])
    sequence = UnifiedLayerSequence(tuple(BatchRoutingEvent(record["sample"], 0, layer, (tuple(ids),))
        for layer, ids in enumerate(extended["truth"])))
    specs, predictors = [], {}
    if point["policy"] != "no_prediction":
        for target in extended["targets"]:
            layer = target["target"]
            ids = prediction_ids(extended, target, point["quality"], base["samples"] * (base["layers"] - 1))
            output = BatchPredictorOutput(record["sample"], 0, layer, 0,
                (TokenPrediction(0, tuple(CandidatePrefetch(e, 1 / (i + 1)) for i, e in enumerate(ids))),))
            predictors[(layer - 1, layer)] = PrecomputedBatchPredictor({(record["sample"], 0, layer): output})
            specs.append(UnifiedPredictionSpec(layer - 1, layer, "source_router_done",
                base["predictor_latency_cycles"], 2 * k, "uniform_prefix", point["prefix_depth"],
                wait_for_previous_layer_weights=False))
    execution = AdvancedExecutionConfig(base["f_tile_size"], placement_policy="striped_tile",
        aie_units_per_expert=1, router_aie_units=1, max_parallel_experts=1, candidate_count=2 * k,
        speculative_byte_budget=point["ocm_bytes"], nonlinear_elements_per_cycle=base["nonlinear_elements_per_cycle"])
    result = UnifiedSequenceSimulator(model, hardware).simulate(sequence, execution,
        predictions=tuple(specs), predictors=predictors)
    metrics = result.memory_metrics
    if result.peak_ocm_bytes > point["ocm_bytes"] or len(result.layer_results) != base["layers"]:
        raise AssertionError("OCM overflow or incomplete layers")
    if metrics["speculative_bytes"] != metrics["useful_speculative_bytes"] + metrics["wasted_speculative_bytes"]:
        raise AssertionError("speculative byte conservation failed")
    if point["policy"] == "no_prediction" and metrics["speculative_bytes"]:
        raise AssertionError("baseline issued speculation")
    return {**point, "seed": seed, "sample": record["sample"], "expert_bytes": 3 * model.hidden_size * model.intermediate_size,
        "total_cycles": result.inference_makespan_cycles, "peak_ocm_bytes": result.peak_ocm_bytes,
        "first_start_sum_cycles": sum(r.first_true_expert_start - r.router_resolve_cycle for r in result.layer_results.values()),
        "future_speculation_blocking_cycles": result.critical_path_future_speculation_blocking_cycles,
        "evicted_future_bytes": result.evicted_future_speculative_bytes,
        "demand_bytes": sum(r.demand_bytes for r in result.layer_results.values()),
        **{name: metrics[name] for name in ("speculative_bytes", "useful_speculative_bytes", "wasted_speculative_bytes")},
        "useful_completed_before_router_bytes": sum(p.useful_completed_speculative_bytes_before_router for p in result.predictions),
        "realized_lead_sum_cycles": sum(p.realized_lead_cycles for p in result.predictions)}


def aggregate(raw: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """逐seed配对总周期，再统计均值/范围和pooled比；要求样本集合一致。"""
    hardware = ("stage", "tile_groups", "ratio", "top_k", "capacity_mode", "ocm_bytes")
    baseline = {}
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in raw:
        key = tuple(row[n] for n in hardware)
        if row["policy"] == "no_prediction":
            identity = (*key, row["seed"], row["sample"])
            if identity in baseline:
                raise ValueError("duplicate baseline")
            baseline[identity] = row
        groups.setdefault((*key, row["policy"], row["prefix_depth"], row["depth_mode"]), []).append(row)
    output = []
    for key, values in sorted(groups.items()):
        seed_ratios, base_sum = [], 0
        for seed in sorted({r["seed"] for r in values}):
            subset = [r for r in values if r["seed"] == seed]
            ids = {r["sample"] for r in subset}
            expected = {i[-1] for i in baseline if i[:-2] == key[:len(hardware)] and i[-2] == seed}
            if ids != expected or len(ids) != len(subset):
                raise ValueError("missing or duplicate paired samples")
            count = sum(baseline[(*key[:len(hardware)], seed, r["sample"])]["total_cycles"] for r in subset)
            base_sum += count
            seed_ratios.append(count / sum(r["total_cycles"] for r in subset))
        current = sum(r["total_cycles"] for r in values)
        row = {n: values[0][n] for n in (*hardware, "policy", "prefix_depth", "depth_mode", "quality", "bytes_per_cycle_per_channel")}
        row.update(samples=len(values), seeds=len(seed_ratios), baseline_cycles=base_sum, total_cycles=current,
            mean_cycles=current / len(values), speedup_mean=mean(seed_ratios),
            speedup_std=stdev(seed_ratios) if len(seed_ratios) > 1 else 0,
            speedup_min=min(seed_ratios), speedup_max=max(seed_ratios), pooled_speedup=base_sum / current,
            net_saved_cycles=base_sum - current, prefetched_gu_fraction=2 * row["prefix_depth"] / (3 * row["tile_groups"]))
        for name in ("first_start_sum_cycles", "future_speculation_blocking_cycles", "evicted_future_bytes",
                     "demand_bytes", "speculative_bytes", "useful_speculative_bytes", "wasted_speculative_bytes",
                     "useful_completed_before_router_bytes", "realized_lead_sum_cycles"):
            row[name] = sum(r[name] for r in values)
        output.append(row)
    return output


def historical_deltas(raw: list[dict[str, Any]], directory: Path) -> list[dict[str, Any]]:
    """每个历史样本核对cycles、容量与流量，不只比较四舍五入speedup。"""
    with (directory / "per_sample.csv").open() as stream:
        old = list(csv.DictReader(stream))
    fields = ("seed", "sample", "tile_groups", "policy", "prefix_depth", "quality")
    lookup = {}
    for row in old:
        if int(row["pe_count"]) == 1 and float(row["ocm_expert_equivalents"]) == 4.:
            key = (*tuple(row[n] for n in fields), float(row["ratio"]))
            if key in lookup:
                raise ValueError("duplicate historical sample")
            lookup[key] = row
    result = []
    for r in raw:
        if r["stage"] != "historical":
            continue
        key = (*tuple(str(r[n]) for n in fields), float(r["ratio"]))
        if key not in lookup:
            raise ValueError("historical sample missing")
        previous = lookup[key]
        delta = {n: r[n] for n in ("seed", "sample", "tile_groups", "ratio", "policy")}
        for metric in ("total_cycles", "peak_ocm_bytes", "demand_bytes", "speculative_bytes",
                       "useful_speculative_bytes", "wasted_speculative_bytes"):
            delta[f"delta_{metric}"] = r[metric] - int(previous[metric])
        result.append(delta)
    return result


def plots(directory: Path, rows: list[dict[str, Any]]) -> None:
    """固定策略曲线；所有图均注明streaming分母，阴影不是置信区间。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    for ratio in (.9, 2.):
        fig, axes = plt.subplots(2, 3, figsize=(14, 8), squeeze=False)
        for i, capacity in enumerate(("scaled_expert", "fixed_bytes")):
            for j, k in enumerate((1, 2, 4)):
                ax = axes[i][j]
                for depth in ("shallow", "proportional"):
                    points = sorted([r for r in rows if r["stage"] == "growth" and r["policy"] == "uniform_prefix"
                        and r["ratio"] == ratio and r["capacity_mode"] == capacity and r["top_k"] == k
                        and (r["depth_mode"] == depth or r["tile_groups"] == 2)], key=lambda r: r["tile_groups"])
                    line, = ax.plot([r["tile_groups"] for r in points], [r["pooled_speedup"] for r in points], marker="o", label=depth)
                    ax.fill_between([r["tile_groups"] for r in points], [r["speedup_min"] for r in points],
                        [r["speedup_max"] for r in points], alpha=.15, color=line.get_color())
                ax.axhline(1, color="black", linestyle="--")
                ax.set_xscale("log", base=2)
                ax.set_title(f"K={k}; OCM={capacity}")
                ax.set_xlabel("F groups / Expert")
                ax.set_ylabel("Speedup vs paired streaming demand")
                ax.legend()
        fig.suptitle(f"Synthetic; rho={ratio}; pooled ratio + seed range (not CI)")
        fig.tight_layout()
        for extension in ("png", "pdf"):
            fig.savefig(directory / f"growth_rho_{ratio:g}.{extension}")
        plt.close(fig)


def main() -> None:
    """保存配置/trace/hash、完成两阶段扫描、核对历史、输出CSV和中文报告。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/experiments/prefetch_benefit_retest.yaml")
    parser.add_argument("--output", type=Path, default=ROOT / "results/prefetch_benefit_retest_20261007")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    if args.workers < 1:
        raise ValueError("workers must be positive")
    if args.output.exists():
        raise ValueError("output exists; choose a new directory")
    config = yaml.safe_load(args.config.read_text())
    base = yaml.safe_load((ROOT / config["base_config"]).read_text())
    traces = json.loads((ROOT / config["historical_directory"] / "inputs.json").read_text())
    base["samples"] = len(traces[str(config["seeds"][0])])
    if any(len(traces[str(s)]) != base["samples"] for s in config["seeds"]):
        raise ValueError("inconsistent saved sample counts")
    points = build_points(config, base)
    args.output.mkdir(parents=True)
    sources = [args.config.resolve(), Path(__file__), ROOT / "examples/prefetch_phase_sweep.py",
               ROOT / config["base_config"], *sorted((ROOT / "src/edge_moe_sim").rglob("*.py"))]
    hashes = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    summary = dict(status="RUNNING", config=config, base_settings=base, source_sha256=hashes,
                   expected_simulations=len(points) * len(config["seeds"]) * base["samples"])
    (args.output / "inputs.json").write_text(json.dumps({"historical": traces, "expanded": {
        str(s): {str(k): [extend_record(r, s, k, base["num_experts"]) for r in traces[str(s)]]
                 for k in config["top_ks"]} for s in config["seeds"]}}, indent=2))
    raw = []
    try:
        with mp.get_context("spawn").Pool(args.workers) as pool:
            for index, point in enumerate(points):
                values = pool.map(worker, [(base, point, seed, record) for seed in config["seeds"] for record in traces[str(seed)]])
                raw.extend(values)
                summary.update(completed_simulations=len(raw), completed_configurations=index + 1)
                (args.output / "summary.json").write_text(json.dumps(summary, indent=2))
                if (index + 1) % 12 == 0 or index + 1 == len(points):
                    print(f"completed {index + 1}/{len(points)} configurations; {len(raw)} simulations", flush=True)
        # Historical and growth workers emit different auxiliary keys; normalize CSV.
        common = list(raw[-1])
        write_csv(args.output / "per_sample.csv", [{n: r[n] for n in common} for r in raw])
        rows = aggregate(raw)
        deltas = historical_deltas(raw, ROOT / config["historical_directory"])
        write_csv(args.output / "historical_deltas.csv", deltas)
        write_csv(args.output / "comparison.csv", rows)
        plots(args.output, rows)
        changed = sum(any(v for k, v in r.items() if k.startswith("delta_")) for r in deltas)
        if any(hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != digest for name, digest in hashes.items()):
            raise RuntimeError("source changed during experiment")
        summary.update(status="COMPLETE", configurations=len(rows), simulations=len(raw),
                       historical_samples=len(deltas), historical_changed_samples=changed,
                       execution_rules=asdict(AdvancedExecutionConfig(base["f_tile_size"]).execution_rules),
                       limits=["synthetic geometry, not real model traces or board throughput",
                           "Top-K hits correlated per layer; candidate count scales as 2K",
                           "scaled OCM changes bytes with F; fixed OCM stays at anchor capacity",
                           "whole matrix fragment nonpreemptive DMA; no burst arbitration",
                           "seed ranges not confidence intervals; no interpolation guarantees"])
        report = ["# 历史收益复测与逐项扩展", "", f"当前主模型完成 {len(raw)} 次模拟；历史 {len(deltas)} 样本中 {changed} 个样本指标变化。",
            "", "所有速度比相对同硬件、同激励 streaming 无预测。固定32 ops/cycle，4通道，200MHz，提前一层预测。",
            "Top-K扩展同时增加候选数到2K；每层全部truth同时hit/miss，质量90%，不代表真实Top-K独立预测。",
            "scaled_expert容量=4个当前Expert；fixed_bytes始终为196608 bytes。比例深度=ceil(groups/2)。",
            "阴影为5seed范围，不是置信区间。startup/Router/非线性不计入rho。未修改模拟器、调度器或RTL。", "",
            "|阶段|groups|rho|K|容量|深度|pooled speedup|seed min|max|", "|---|---:|---:|---:|---|---:|---:|---:|---:|"]
        report += [f"|{r['stage']}|{r['tile_groups']}|{r['ratio']}|{r['top_k']}|{r['capacity_mode']}|{r['prefix_depth']}|{r['pooled_speedup']:.6f}|{r['speedup_min']:.6f}|{r['speedup_max']:.6f}|"
                   for r in rows if r["policy"] != "no_prediction"]
        (args.output / "REPORT.md").write_text("\n".join(report) + "\n")
    except Exception as error:
        summary.update(status="FAILED", error=str(error))
        raise
    finally:
        (args.output / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: summary[k] for k in ("status", "simulations", "historical_changed_samples")}), flush=True)


if __name__ == "__main__":
    main()
