"""未剪枝 Switch 三类预取对照；主模拟器、共享保存的 CEPR-inspired 合成激励。"""
from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
import hashlib
import json
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
from edge_moe_sim.config import load_model_config
from edge_moe_sim.hardware_profile import HardwareProfile
from edge_moe_sim.schema import CandidatePrefetch, ModelConfig
from edge_moe_sim.unified_schema import UnifiedLayerSequence, UnifiedPredictionSpec
from edge_moe_sim.unified_sequence_simulator import UnifiedSequenceSimulator
from edge_moe_sim.work_model import expert_weight_bytes

POLICIES = ("no_prediction", "full_expert_prefetch", "uniform_prefix", "ranked_prefix",
            "demand_full_expert", "demand_tile_serial")


def make_inputs(model: ModelConfig, settings: dict[str, Any]) -> list[dict[str, Any]]:
    """生成精确总体覆盖率（四舍五入）和均衡命中排名；truth 与策略无关。"""
    if model.top_k != 1 or settings["candidate_count"] != 2:
        raise ValueError("this CEPR-inspired experiment requires Top-1 and two candidates")
    if settings["samples"] <= 0 or not 0 <= settings["coverage"] <= 1:
        raise ValueError("invalid sample count or coverage")
    layers = model.moe_layer_indices
    if len(layers) < 2:
        raise ValueError("at least two MoE layers required")
    rng = random.Random(settings["seed"])
    count = settings["samples"] * (len(layers) - 1)
    hits = int(count * settings["coverage"] + 0.5)
    ranks = [index % 2 for index in range(hits)] + [-1] * (count - hits)
    rng.shuffle(ranks)
    records = []
    for sample in range(settings["samples"]):
        truth = [rng.randrange(model.num_experts) for _ in layers]
        predictions = []
        for index, (source, target) in enumerate(zip(layers, layers[1:])):
            wrong = rng.sample([e for e in range(model.num_experts) if e != truth[index + 1]], 2)
            rank = ranks[sample * (len(layers) - 1) + index]
            if rank >= 0:
                wrong[rank] = truth[index + 1]
            predictions.append({"source": source, "target": target, "candidate_ids": wrong,
                                "scores": [0.6, 0.4], "true_rank": rank})
        records.append({"sample": sample, "truth": truth, "predictions": predictions})
    return records


def run_sample(model: ModelConfig, settings: dict[str, Any], record: dict[str, Any],
               capacity: int, rate: int, policy: str,
               timing_rows: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    """独立单 token 六层模拟；实际 byte OCM，无 RTL 固定 slot 或延迟拟合。"""
    hardware = HardwareProfile(**settings["hardware"], ocm_bytes=capacity, ops_per_cycle_per_aie=rate)
    sequence = UnifiedLayerSequence(tuple(
        BatchRoutingEvent(record["sample"], 0, layer,
                          (tuple(expert) if isinstance(expert, (list, tuple)) else (expert,),))
        for layer, expert in zip(model.moe_layer_indices, record["truth"])
    ))
    predictors = {}
    specs = []
    for prediction in record["predictions"]:
        source, target = prediction["source"], prediction["target"]
        output = BatchPredictorOutput(record["sample"], 0, target, 0, (TokenPrediction(0, tuple(
            CandidatePrefetch(e, score) for e, score in zip(prediction["candidate_ids"], prediction["scores"])
        )),))
        predictors[(source, target)] = PrecomputedBatchPredictor({(record["sample"], 0, target): output})
        if policy not in {"no_prediction", "demand_full_expert", "demand_tile_serial"}:
            specs.append(UnifiedPredictionSpec(source, target, "source_router_done",
                settings["predictor_latency_cycles"], settings["candidate_count"],
                "full_expert_prefetch" if policy == "nonpipeline_full_expert_prefetch" else policy,
                settings["prefix_depth"],
                wait_for_previous_layer_weights=settings["wait_for_previous_layer_weights"]))
    config = AdvancedExecutionConfig(settings["f_tile_size"], placement_policy="striped_tile",
        candidate_count=settings["candidate_count"], nonlinear_elements_per_cycle=settings["nonlinear_elements_per_cycle"],
        speculative_byte_budget=capacity,
        demand_loading={"demand_full_expert": "full_expert_load", "demand_tile_serial": "tile_serial",
                        "nonpipeline_full_expert_prefetch": "full_expert_load"}.get(policy, "tile_streaming"))
    result = UnifiedSequenceSimulator(model, hardware).simulate(sequence, config,
        predictions=tuple(specs), predictors=predictors)
    if len(result.layer_results) != len(sequence.layers) or result.peak_ocm_bytes > capacity:
        raise AssertionError("incomplete sequence or capacity violation")
    if timing_rows is not None:
        for prediction in result.predictions:
            row = {"model": model.name, "ocm_bytes": capacity, "ops_per_cycle": rate,
                   "policy": policy, "sample": record["sample"], **asdict(prediction)}
            row["release_to_router_cycles"] = (prediction.target_router_resolve_cycle -
                prediction.prefetch_release_cycle if prediction.prefetch_release_cycle is not None else None)
            timing_rows.append(row)
    tile_count = (model.intermediate_size + settings["f_tile_size"] - 1) // settings["f_tile_size"]
    required = {(matrix, tile) for matrix in ("gate_up", "down") for tile in range(tile_count)}
    completed: dict[tuple[int, int], set[tuple[str, int]]] = {}
    for request in result.speculative_request_records:
        target = int(request["target_layer"])
        end = request["completion_cycle"]
        if request["issue_cycle"] is not None and end is not None and end <= result.layer_results[target].router_resolve_cycle:
            completed.setdefault((target, int(request["expert_id"])), set()).add((request["matrix_type"], request["tile_id"]))
    return {
        "model": model.name, "ocm_bytes": capacity, "ops_per_cycle": rate, "policy": policy,
        "sample": record["sample"], "total_cycles": result.inference_makespan_cycles,
        "memory_wait_cycles": sum(r.aie_starvation_cycles for r in result.layer_results.values()),
        "peak_ocm_bytes": result.peak_ocm_bytes,
        "demand_bytes": sum(r.demand_bytes for r in result.layer_results.values()),
        "speculative_bytes": result.memory_metrics["speculative_bytes"],
        "useful_speculative_bytes": result.memory_metrics["useful_speculative_bytes"],
        "wasted_speculative_bytes": result.memory_metrics["wasted_speculative_bytes"],
        "completed_speculative_bytes_before_router": sum(p.completed_speculative_bytes_before_router for p in result.predictions),
        "full_experts_transferred_before_router": sum(required <= keys for keys in completed.values()),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """保存非空 CSV；字段顺序确定。"""
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot(directory: Path, rows: list[dict[str, Any]]) -> None:
    """输出速度比和等待周期对照 PNG/PDF。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    models = list(dict.fromkeys(r["model"] for r in rows))
    rates = sorted({r["ops_per_cycle"] for r in rows})
    for metric, ylabel in (("speedup_vs_no_prediction", "Speedup vs demand tile streaming"),
                           ("memory_wait_cycles", "Aggregate AIE starvation cycles")):
        fig, axes = plt.subplots(len(models), len(rates), figsize=(11, 7), squeeze=False)
        for i, model in enumerate(models):
            for j, rate in enumerate(rates):
                ax = axes[i][j]
                for policy in POLICIES:
                    points = [r for r in rows if r["model"] == model and r["ops_per_cycle"] == rate and r["policy"] == policy]
                    ax.plot([r["ocm_bytes"] / 1048576 for r in points], [r[metric] for r in points], marker="o", label=policy)
                ax.set_title(f"{model}; {rate} configured ops/cycle")
                ax.set_xlabel("Expert OCM budget (MiB)")
                ax.set_ylabel(ylabel)
                ax.legend(fontsize=7)
        fig.suptitle("Unpruned Switch / synthetic two-candidate coverage 84.4% / configured hardware")
        fig.tight_layout()
        for extension in ("png", "pdf"):
            fig.savefig(directory / f"{metric}.{extension}")
        plt.close(fig)


def main() -> None:
    """保存激励、完整配置、逐样本结果、聚合结果及来源校验码。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/experiments/switch_prefetch.yaml")
    parser.add_argument("--output", type=Path, default=ROOT / "results/switch_prefetch_baselines")
    parser.add_argument("--reference", type=Path, help="历史 barrier 输出目录，用于比较；不会覆盖")
    args = parser.parse_args()
    settings = yaml.safe_load(args.config.read_text())
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "summary.json").write_text(json.dumps({"status": "running"}))
    raw, aggregate, inputs, timings = [], [], {}, []
    for model_path in settings["models"]:
        model = load_model_config(ROOT / model_path)
        records = make_inputs(model, settings)
        inputs[model.name] = records
        (args.output / f"{model.name}_inputs.json").write_text(json.dumps(records, indent=2))
        for capacity in settings["ocm_capacity_bytes"]:
            for rate in settings["ops_per_cycle"]:
                group = []
                for policy in POLICIES:
                    samples = [run_sample(model, settings, record, capacity, rate, policy, timings) for record in records]
                    raw.extend(samples)
                    row = {key: samples[0][key] for key in ("model", "ocm_bytes", "ops_per_cycle", "policy")}
                    for key in sorted(samples[0].keys() - row.keys() - {"sample", "peak_ocm_bytes"}):
                        row[key] = sum(sample[key] for sample in samples)
                    row["peak_ocm_bytes"] = max(sample["peak_ocm_bytes"] for sample in samples)
                    row["expert_bytes"] = expert_weight_bytes(model)
                    row["samples"] = len(samples)
                    group.append(row)
                baseline, full = group[0]["total_cycles"], group[1]["total_cycles"]
                weak_full, weak_tile = group[4]["total_cycles"], group[5]["total_cycles"]
                for row in group:
                    row["speedup_vs_no_prediction"] = baseline / row["total_cycles"]
                    row["speedup_vs_full_expert"] = full / row["total_cycles"]
                    row["speedup_vs_demand_full_expert"] = weak_full / row["total_cycles"]
                    row["speedup_vs_demand_tile_serial"] = weak_tile / row["total_cycles"]
                aggregate.extend(group)
                print(f"completed {model.name} OCM={capacity} ops/cycle={rate}", flush=True)
    write_csv(args.output / "per_sample.csv", raw)
    write_csv(args.output / "comparison.csv", aggregate)
    if timings:
        write_csv(args.output / "per_prediction.csv", timings)
    if args.reference:
        old_summary = json.loads((args.reference / "summary.json").read_text())
        old_settings = dict(old_summary["settings"])
        old_settings["wait_for_previous_layer_weights"] = settings["wait_for_previous_layer_weights"]
        if old_summary["status"] != "complete" or old_settings != settings:
            raise ValueError("reference differs in settings beyond weight barrier")
        with (args.reference / "comparison.csv").open(newline="") as stream:
            previous = {(r["model"], int(r["ocm_bytes"]), int(r["ops_per_cycle"]), r["policy"]): r
                        for r in csv.DictReader(stream)}
        deltas = []
        for row in aggregate:
            if row["policy"] in {"demand_full_expert", "demand_tile_serial"}:
                continue
            old = previous[(row["model"], row["ocm_bytes"], row["ops_per_cycle"], row["policy"])]
            if row["policy"] == "no_prediction" and int(old["total_cycles"]) != row["total_cycles"]:
                raise AssertionError("no-prediction baseline changed")
            if (old_summary["settings"]["wait_for_previous_layer_weights"] == settings["wait_for_previous_layer_weights"]
                    and int(old["total_cycles"]) != row["total_cycles"]):
                raise AssertionError("existing policy cycles changed while adding weak baselines")
            deltas.append({**row, "barrier_total_cycles": int(old["total_cycles"]),
                           "speedup_vs_barrier": int(old["total_cycles"]) / row["total_cycles"]})
        write_csv(args.output / "barrier_comparison.csv", deltas)
    plot(args.output, aggregate)
    sources = [args.config, Path(__file__), *(ROOT / p for p in settings["models"]),
               *sorted((ROOT / "src/edge_moe_sim").rglob("*.py"))]
    summary = {"status": "complete", "simulator": "UnifiedSequenceSimulator", "settings": settings,
        "simulation_count": len(raw), "configuration_count": len(aggregate),
        "observed_candidate_coverage": {name: sum(p["true_rank"] >= 0 for r in records for p in r["predictions"]) /
            sum(len(r["predictions"]) for r in records) for name, records in inputs.items()},
        "source_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
        "limits": ["no pruning; no real CEPR; synthetic ranking evenly splits hits between rank 1/2",
            "50 independent cold-start single-token sequences, not persistent autoregressive cache",
            "one MoE-layer-ahead; previous-weight barrier selectable in settings; no attention/encoder work",
            "actual-byte OCM, not RTL slots; one configured logical PL engine (legacy field name AIE)",
            "full-expert policy targets all tiles but admission/window may prevent full completion",
            "tile prefixes prefetch input projection only; full targets input and down",
            "weak baselines: full demand residency before compute, or serial per-matrix tile DMA/compute",
            "hardware and ReLU throughput are configured assumptions, not measurements"]}
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
