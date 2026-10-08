"""OLMoE 全部执行 case 与 Mixtral streaming-only 的共享激励矩阵。"""
from __future__ import annotations

import argparse
import csv
from dataclasses import replace
import hashlib
import json
import multiprocessing
from pathlib import Path
import random
from typing import Any

import yaml

from switch_prefetch_comparison import ROOT, run_sample, write_csv
from edge_moe_sim.real_model_profile import load_real_moe_profiles
from edge_moe_sim.schema import ModelConfig
from edge_moe_sim.work_model import expert_weight_bytes

STREAMING = ("no_prediction", "full_expert_prefetch", "uniform_prefix", "ranked_prefix")
WHOLE = ("demand_full_expert", "demand_tile_serial", "nonpipeline_full_expert_prefetch")


def run_worker(task: tuple[Any, ...]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """独立样本 worker；重跑 checkpoint 样本必须逐字段一致，顺序由父进程确定。"""
    model, settings, record, capacity, rate, policy, cached = task
    timing: list[dict[str, Any]] = []
    row = run_sample(model, settings, record, capacity, rate, policy, timing)
    if cached is not None and row != cached:
        raise AssertionError("checkpoint sample differs from rerun")
    return row, timing


def make_batch_inputs(model: ModelConfig, settings: dict[str, Any]) -> list[dict[str, Any]]:
    """Top-K 覆盖率按所有真实 Expert 计；命中排名随机，不强行将命中排在前列。"""
    samples, coverage, candidates = settings["samples"], settings["coverage"], settings["candidate_count"]
    layers = model.moe_layer_indices
    if samples <= 0 or len(layers) < 2 or not 0 <= coverage <= 1 or not model.top_k <= candidates <= model.num_experts:
        raise ValueError("invalid sample, layer, coverage or candidate settings")
    rng = random.Random(settings["seed"])
    total = samples * (len(layers) - 1) * model.top_k
    hits = int(total * coverage + 0.5)
    flags = [True] * hits + [False] * (total - hits)
    rng.shuffle(flags)
    records = []
    cursor = 0
    for sample in range(samples):
        truth = [rng.sample(range(model.num_experts), model.top_k) for _ in layers]
        predictions = []
        for index, (source, target) in enumerate(zip(layers, layers[1:])):
            selected = [expert for expert, hit in zip(truth[index + 1], flags[cursor:cursor + model.top_k]) if hit]
            cursor += model.top_k
            wrong = [expert for expert in range(model.num_experts) if expert not in truth[index + 1]]
            if candidates - len(selected) > len(wrong):
                raise ValueError("candidate size cannot preserve the requested coverage")
            ids = selected + rng.sample(wrong, candidates - len(selected))
            rng.shuffle(ids)
            predictions.append({"source": source, "target": target, "candidate_ids": ids,
                "scores": [(candidates - rank) / candidates for rank in range(candidates)],
                "true_ranks": [ids.index(expert) if expert in ids else -1 for expert in truth[index + 1]]})
        records.append({"sample": sample, "truth": truth, "predictions": predictions})
    return records


def make_plot(directory: Path, rows: list[dict[str, Any]]) -> None:
    """所有模型画相对 streaming 的收益，OLMoE 另画相对完整按需加载的收益。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    for metric in ("speedup_vs_streaming", "speedup_vs_whole_demand"):
        models = list(dict.fromkeys(r["model"] for r in rows if r[metric] is not None))
        fig, axes = plt.subplots(len(models), 2, figsize=(11, 4 * len(models)), squeeze=False)
        for i, model in enumerate(models):
            for j, rate in enumerate(sorted({r["ops_per_cycle"] for r in rows})):
                ax = axes[i][j]
                for policy in STREAMING + WHOLE:
                    points = [r for r in rows if r["model"] == model and r["ops_per_cycle"] == rate and r["policy"] == policy]
                    if points:
                        ax.plot([r["ocm_bytes"] / 1048576 for r in points], [r[metric] for r in points], marker="o", label=policy)
                ax.axhline(1, color="gray", linestyle="--")
                ax.set_title(f"{model}; {rate} configured ops/cycle")
                ax.set_xlabel("Expert OCM budget (MiB)")
                ax.set_ylabel(metric)
                ax.legend(fontsize=6)
        fig.suptitle("Real model geometry / synthetic Top-K recall 84.4% / configured hardware")
        fig.tight_layout()
        for extension in ("png", "pdf"):
            fig.savefig(directory / f"{metric}.{extension}")
        plt.close(fig)


def main() -> None:
    """运行所有声明点；保存激励、逐样本/聚合 CSV、预测时序、图和来源 hashes。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/experiments/olmoe_mixtral_cases.yaml")
    parser.add_argument("--output", type=Path, default=ROOT / "results/olmoe_mixtral_cases")
    parser.add_argument("--workers", type=int, default=4, help="主机模拟进程数，不是硬件 PE 数")
    parser.add_argument("--resume", action="store_true", help="复用本次中断运行的无预测样本；预测样本重跑校验并补齐 timing")
    args = parser.parse_args()
    if args.workers <= 0:
        raise ValueError("workers must be positive")
    experiment = yaml.safe_load(args.config.read_text())
    base = yaml.safe_load((ROOT / experiment["base_config"]).read_text())
    profiles = load_real_moe_profiles(ROOT / experiment["catalogue"], weight_bits=experiment["weight_bits"], activation_bits=experiment["activation_bits"])
    if set(experiment["models"]) - {p.model.name for p in profiles}:
        raise ValueError("unknown model selection")
    args.output.mkdir(parents=True, exist_ok=True)
    cached: dict[tuple[Any, ...], dict[str, Any]] = {}
    if args.resume:
        with (args.output / "per_sample.csv").open(newline="") as stream:
            for entry in csv.DictReader(stream):
                row = {k: v if k in {"model", "policy"} else int(v) for k, v in entry.items()}
                cached[(row["model"], row["ocm_bytes"], row["ops_per_cycle"], row["policy"], row["sample"])] = row
    (args.output / "summary.json").write_text(json.dumps({"status": "running"}))
    raw, aggregate, timings, geometry, coverage = [], [], [], [], {}
    pool = multiprocessing.get_context("spawn").Pool(args.workers) if args.workers > 1 else None
    reused = 0
    for profile in profiles:
        if profile.model.name not in experiment["models"]:
            continue
        count = experiment["simulated_layers"]
        if not 2 <= count <= profile.num_moe_layers:
            raise ValueError("invalid simulated MoE depth")
        model = replace(profile.model, moe_layer_indices=tuple(range(count)))
        settings = {**base, "samples": experiment["samples"], "seed": experiment["seed"],
                    "coverage": experiment["coverage"], "candidate_count": min(model.num_experts, model.top_k * experiment["candidate_multiplier"])}
        records = make_batch_inputs(model, settings)
        if args.resume:
            saved = json.loads((args.output / f"{model.name}_inputs.json").read_text()) if (args.output / f"{model.name}_inputs.json").exists() else None
            if saved is not None and saved != records:
                raise ValueError("checkpoint input trace changed")
        hit_count = sum(rank >= 0 for r in records for p in r["predictions"] for rank in p["true_ranks"])
        total = len(records) * (count - 1) * model.top_k
        coverage[model.name] = {"hits": hit_count, "true_expert_decisions": total, "static_recall": hit_count / total}
        (args.output / f"{model.name}_inputs.json").write_text(json.dumps(records, indent=2))
        geometry.append({**profile.geometry(settings["f_tile_size"]), "simulated_layers": count,
                         "candidate_count": settings["candidate_count"]})
        policies = STREAMING + WHOLE if model.name.startswith("olmoe") else STREAMING
        for capacity in settings["ocm_capacity_bytes"]:
            for rate in settings["ops_per_cycle"]:
                group = []
                for policy in policies:
                    samples, tasks = [], []
                    for record in records:
                        previous = cached.get((model.name, capacity, rate, policy, record["sample"]))
                        if previous is not None and policy in {"no_prediction", "demand_full_expert", "demand_tile_serial"}:
                            samples.append(previous)
                            reused += 1
                        else:
                            tasks.append((model, settings, record, capacity, rate, policy, previous))
                    results = pool.map(run_worker, tasks) if pool is not None else map(run_worker, tasks)
                    for sample, prediction_timing in results:
                        samples.append(sample)
                        timings.extend(prediction_timing)
                    samples.sort(key=lambda row: row["sample"])
                    raw.extend(samples)
                    row = {key: samples[0][key] for key in ("model", "ocm_bytes", "ops_per_cycle", "policy")}
                    for key in sorted(samples[0].keys() - row.keys() - {"sample", "peak_ocm_bytes"}):
                        row[key] = sum(sample[key] for sample in samples)
                    row["peak_ocm_bytes"] = max(sample["peak_ocm_bytes"] for sample in samples)
                    row.update(samples=len(samples), top_k=model.top_k, candidate_count=settings["candidate_count"], simulated_layers=count)
                    group.append(row)
                by_policy = {row["policy"]: row for row in group}
                for row in group:
                    row["speedup_vs_streaming"] = by_policy["no_prediction"]["total_cycles"] / row["total_cycles"]
                    row["speedup_vs_whole_demand"] = (by_policy["demand_full_expert"]["total_cycles"] / row["total_cycles"]
                                                     if "demand_full_expert" in by_policy else None)
                    row["speedup_vs_full_target_streaming"] = by_policy["full_expert_prefetch"]["total_cycles"] / row["total_cycles"]
                aggregate.extend(group)
                write_csv(args.output / "comparison.csv", aggregate)
                write_csv(args.output / "per_sample.csv", raw)
                if timings:
                    write_csv(args.output / "per_prediction.csv", timings)
                print(f"completed {model.name} OCM={capacity} ops/cycle={rate}", flush=True)
    if pool is not None:
        pool.close()
        pool.join()
    write_csv(args.output / "model_geometry.csv", geometry)
    write_csv(args.output / "per_prediction.csv", timings)
    make_plot(args.output, aggregate)
    sources = [args.config, ROOT / experiment["base_config"], ROOT / experiment["catalogue"], Path(__file__),
               ROOT / "examples/switch_prefetch_comparison.py", *sorted((ROOT / "src/edge_moe_sim").rglob("*.py"))]
    summary = {"status": "complete", "simulator": "UnifiedSequenceSimulator", "experiment": experiment,
        "base_settings": base, "simulation_count": len(raw), "configuration_count": len(aggregate),
        "host_workers": args.workers, "reused_nonpredictive_samples": reused,
        "coverage": coverage, "model_geometry": geometry,
        "source_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
        "infeasible_whole_expert_cases": [p.model.name for p in profiles if p.model.name.startswith("mixtral")],
        "limits": ["INT8 assumed; no pruning or real predictor; synthetic ranking randomized",
            "six consecutive MoE layers, 50 independent cold-start single-token sequences, not full model inference",
            "Top-K static recall 84.4% is a quality assumption, not measured CEPR on these models",
            "Mixtral full target streaming may truncate; not whole-Expert residency",
            "actual-byte OCM, no burst arbitration, configured hardware not board measurements",
            "uniform/ranked use same configured prefix_depth, not equal issued speculative byte budgets",
            "Mixtral 8x22B total routed INT8 weights exceed U200 external DDR; no single-card deployment claim"]}
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
