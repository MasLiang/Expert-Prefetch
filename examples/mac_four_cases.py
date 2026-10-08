"""同一MAC宽度/带宽/容量/输入下比较整Expert与tile streaming四case。"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
from itertools import product
import json
import multiprocessing as mp
from pathlib import Path
from statistics import mean, stdev
from typing import Any

import yaml

from rtl_rules_pe_sweep import (ROOT, AdvancedExecutionConfig, BatchPredictorOutput,
    BatchRoutingEvent, CandidatePrefetch, ExecutionRules, HardwareProfile, ModelConfig,
    PrecomputedBatchPredictor, TokenPrediction, UnifiedLayerSequence, UnifiedPredictionSpec,
    UnifiedSequenceSimulator, build_models, write_csv)
from edge_moe_sim.work_model import expert_weight_bytes

CASES = ("A_whole_demand", "B_stream_demand", "C_whole_prefetch", "D_stream_tile_prefetch")


def common_rules(model: ModelConfig, geometry: dict[str, Any], config: dict[str, Any]) -> ExecutionRules:
    """四case共享byte容量与DMA/控制成本；不启用整Expert路径不支持的pair规则。"""
    count = min(model.num_experts, model.top_k * geometry["candidate_multiplier"])
    return ExecutionRules(**config["execution_rules"],
        router_control_cycles=model.num_experts * model.top_k + 2 * model.top_k + 34 * (model.top_k - 1) + 3,
        prefetch_allocator_cycles=count + 5 + (count - 1).bit_length())


def run_case(job: tuple[ModelConfig, dict[str, Any], dict[str, Any], dict[str, Any],
                        dict[str, Any], dict[str, Any], dict[str, Any]]) -> dict[str, Any]:
    """运行单样本；A/C必须等整Expert全部GU/DOWN权重驻留才计算。"""
    model, base, geometry, config, entry, point, record = job
    whole = point["case"] in {"A_whole_demand", "C_whole_prefetch"}
    predict = point["case"] in {"C_whole_prefetch", "D_stream_tile_prefetch"}
    rules = common_rules(model, geometry, config)
    fields = dict(base["hardware"])
    fields.update(name="shared_four_case_configured_MAC_width", total_aie_units=1,
        burst_bytes=geometry["burst_bytes"],
        hardware_tag="COMMON_BYTE_OCM_NOT_FINITE_SLOT_RTL_NOT_MEASURED",
        bandwidth_bytes_per_sec_per_channel=point["bytes_per_cycle_per_channel"] * fields["clock_hz"])
    hardware = HardwareProfile(**fields,
        ops_per_cycle_per_aie=geometry["ops_per_cycle_per_group"] * point["mac_groups"],
        ocm_bytes=point["ocm_bytes"])
    sequence = UnifiedLayerSequence(tuple(BatchRoutingEvent(record["sample"], 0, layer, (tuple(ids),))
        for layer, ids in zip(model.moe_layer_indices, record["truth"])))
    specs, predictors = [], {}
    candidates = min(model.num_experts, model.top_k * geometry["candidate_multiplier"])
    if predict:
        for prediction in record["predictions"]:
            source, target = prediction["source"], prediction["target"]
            output = BatchPredictorOutput(record["sample"], 0, target, 0, (TokenPrediction(0, tuple(
                CandidatePrefetch(e, score) for e, score in zip(prediction["candidate_ids"], prediction["scores"]))),))
            predictors[(source, target)] = PrecomputedBatchPredictor({(record["sample"], 0, target): output})
            specs.append(UnifiedPredictionSpec(source, target, "source_router_done", base["predictor_latency_cycles"],
                candidates, "full_expert_prefetch" if whole else "uniform_prefix", geometry["prefix_depth"],
                wait_for_previous_layer_weights=config["wait_for_previous_layer_weights"]))
    execution = AdvancedExecutionConfig(entry["f_tile_size"], placement_policy="striped_tile",
        aie_units_per_expert=1, router_aie_units=1, max_parallel_experts=1,
        candidate_count=candidates, speculative_byte_budget=point["ocm_bytes"],
        nonlinear_elements_per_cycle=base["nonlinear_elements_per_cycle"], execution_rules=rules,
        demand_loading="full_expert_load" if whole else "tile_streaming")
    result = UnifiedSequenceSimulator(model, hardware).simulate(sequence, execution,
        predictions=tuple(specs), predictors=predictors)
    metrics = result.memory_metrics
    if len(result.layer_results) != len(sequence.layers) or result.peak_ocm_bytes > point["ocm_bytes"]:
        raise AssertionError("incomplete sequence or OCM overflow")
    if metrics["speculative_bytes"] != metrics["useful_speculative_bytes"] + metrics["wasted_speculative_bytes"]:
        raise AssertionError("speculative byte conservation")
    if not predict and metrics["speculative_bytes"]:
        raise AssertionError("unrequested speculation")
    row = {**point, "sample":record["sample"], "total_cycles":result.inference_makespan_cycles,
        "peak_ocm_bytes":result.peak_ocm_bytes,
        "memory_wait_sum_cycles":sum(r.aie_starvation_cycles for r in result.layer_results.values()),
        "speculative_bytes":metrics["speculative_bytes"], "useful_speculative_bytes":metrics["useful_speculative_bytes"],
        "wasted_speculative_bytes":metrics["wasted_speculative_bytes"],
        "expert_bytes":expert_weight_bytes(model), "payload_compute_cycles":result.aie_metrics["useful_aie_cycles"]}
    if config.get("include_prediction_diagnostics", False):
        details = []
        for prediction in result.predictions:
            target = result.layer_results[prediction.target_layer_id]
            required = {(matrix,tile) for matrix in ("gate_up","down")
                for tile in range((model.intermediate_size+entry["f_tile_size"]-1)//entry["f_tile_size"])}
            transferred: dict[int,set[tuple[str,int]]] = {}
            for request in result.speculative_request_records:
                end = request["completion_cycle"]
                if request["target_layer"] == prediction.target_layer_id and end is not None and end <= target.router_resolve_cycle:
                    transferred.setdefault(request["expert_id"],set()).add((request["matrix_type"],request["tile_id"]))
            full = {expert for expert,tiles in transferred.items() if required <= tiles}
            truth = set(target.true_expert_ids)
            details.append({**asdict(prediction), "candidate_hit":bool(truth.intersection(prediction.candidate_ids)),
                "full_true_resident_at_router":target.startable_recall if whole else None,
                "full_candidates_transferred_before_router":len(full),
                "full_true_transferred_before_router":len(full.intersection(truth)),
                "startup_resident_fraction_at_router":target.weighted_startup_readiness,
                "prediction_to_release_cycles":prediction.prefetch_release_cycle-prediction.predictor_ready_cycle
                    if prediction.prefetch_release_cycle is not None else None,
                "release_to_router_cycles":target.router_resolve_cycle-prediction.prefetch_release_cycle
                    if prediction.prefetch_release_cycle is not None else None})
        row["prediction_diagnostics"] = details
    return row


def aggregate(seed_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """分别计算相对A、B、C和各case自身16-lane基线的配对速度比。"""
    keys = ("model", "bytes_per_cycle_per_channel", "mac_groups", "seed", "case")
    lookup = {tuple(r[k] for k in keys):r for r in seed_rows}
    if len(lookup) != len(seed_rows):
        raise ValueError("duplicate seed result")
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in seed_rows:
        identity = tuple(row[k] for k in keys[:-1])
        for case, label in zip(CASES[:3], ("A", "B", "C")):
            row[f"speedup_vs_{label}"] = lookup[(*identity, case)]["total_cycles"] / row["total_cycles"]
        row["speedup_vs_16_lanes"] = lookup[(row["model"], row["bytes_per_cycle_per_channel"], 1,
            row["seed"], row["case"])]["total_cycles"] / row["total_cycles"]
        groups.setdefault(tuple(row[k] for k in keys if k != "seed"), []).append(row)
    output = []
    for _, values in sorted(groups.items()):
        row = {k:values[0][k] for k in ("model", "bytes_per_cycle_per_channel", "mac_groups", "case", "ocm_bytes", "expert_bytes")}
        row.update(lanes=16*row["mac_groups"], seeds=len(values), samples=sum(v["samples"] for v in values),
            cycles_per_sequence=sum(v["total_cycles"] for v in values)/sum(v["samples"] for v in values),
            peak_ocm_bytes=max(v["peak_ocm_bytes"] for v in values))
        for field in ("speedup_vs_A", "speedup_vs_B", "speedup_vs_C", "speedup_vs_16_lanes"):
            row[field] = mean(v[field] for v in values)
            row[field+"_std"] = stdev(v[field] for v in values)
        for field in ("speculative_bytes", "useful_speculative_bytes", "wasted_speculative_bytes"):
            row[field] = sum(v[field] for v in values)
        output.append(row)
    return output


def plot(directory: Path, rows: list[dict[str, Any]]) -> None:
    """每个供数条件下绘制四case相对A的收益以及绝对cycles。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    panels = sorted({(r["model"], r["bytes_per_cycle_per_channel"]) for r in rows})
    fig, axes = plt.subplots(len(panels), 2, figsize=(11, 3*len(panels)), squeeze=False)
    for i, (model, bw) in enumerate(panels):
        for case in CASES:
            selected = sorted([r for r in rows if r["model"] == model and
                r["bytes_per_cycle_per_channel"] == bw and r["case"] == case], key=lambda r:r["lanes"])
            for j, metric in enumerate(("speedup_vs_A", "cycles_per_sequence")):
                axes[i][j].plot([r["lanes"] for r in selected], [r[metric] for r in selected], marker="o", label=case)
        for j in range(2):
            axes[i][j].set_xscale("log", base=2)
            axes[i][j].set_title(f"{model}; nominal {bw} B/cycle/channel")
            axes[i][j].set_xlabel("Hypothetical MAC lanes in one Engine")
            axes[i][j].set_ylabel("Speedup vs A at same lane count" if j==0 else "Cycles / sequence")
            axes[i][j].legend(fontsize=7)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(directory/f"four_cases.{ext}")
    plt.close(fig)


def main() -> None:
    """复用上一轮保存的输入，完整重跑四case，保存原始及配对结果。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT/"configs/experiments/mac_four_cases.yaml")
    parser.add_argument("--output", type=Path, default=ROOT/"results/mac_four_cases")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if args.workers < 1:
        raise ValueError("workers must be positive")
    config = yaml.safe_load(args.config.read_text())
    geometry = yaml.safe_load((ROOT/config["geometry_config"]).read_text())
    base = yaml.safe_load((ROOT/geometry["base_config"]).read_text())
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output/"summary.json").write_text(json.dumps({"status":"running"}))
    raw, seeds, declarations, input_paths = [], [], [], []
    with mp.get_context("spawn").Pool(args.workers) as pool:
        for model, entry in build_models(geometry):
            path = ROOT/config["inputs_directory"]/f"{model.name}_inputs.json"
            input_paths.append(path)
            traces = json.loads(path.read_text())
            (args.output/path.name).write_text(json.dumps(traces, indent=2))
            slot = (model.input_projection_count*model.hidden_size*entry["f_tile_size"]*model.weight_bits+7)//8
            capacity = slot*geometry["ocm_banks"]*geometry["slots_per_bank"]
            if expert_weight_bytes(model)>capacity:
                raise ValueError(f"{model.name}: OCM cannot fit complete Expert")
            declarations.append({"model":model.name, "ocm_bytes":capacity, "expert_bytes":expert_weight_bytes(model),
                "rules":asdict(common_rules(model,geometry,config))})
            for groups, bw in product(entry.get("mac_groups",geometry["mac_groups"]),
                    entry.get("bytes_per_cycle_per_channel",geometry["bytes_per_cycle_per_channel"])):
                points = [dict(model=model.name, mac_groups=groups, bytes_per_cycle_per_channel=bw,
                    case=case, seed=seed, ocm_bytes=capacity) for seed,case in product(geometry["seeds"],CASES)]
                values = pool.map(run_case, [(model,base,geometry,config,entry,p,rec)
                    for p in points for rec in traces[str(p["seed"])]])
                raw.extend(values)
                for index, point in enumerate(points):
                    samples = values[index*entry["samples"]:(index+1)*entry["samples"]]
                    row = {**point, "samples":len(samples), "expert_bytes":expert_weight_bytes(model),
                        "peak_ocm_bytes":max(r["peak_ocm_bytes"] for r in samples)}
                    for field in ("total_cycles", "memory_wait_sum_cycles", "payload_compute_cycles",
                            "speculative_bytes", "useful_speculative_bytes", "wasted_speculative_bytes"):
                        row[field] = sum(r[field] for r in samples)
                    seeds.append(row)
                write_csv(args.output/"checkpoint.csv",seeds)
                print(f"completed {model.name}; groups={groups}; BW/ch={bw}", flush=True)
    comparison = aggregate(seeds)
    write_csv(args.output/"per_sample.csv",raw)
    write_csv(args.output/"per_seed.csv",seeds)
    write_csv(args.output/"comparison.csv",comparison)
    plot(args.output,comparison)
    sources = [args.config.resolve(),Path(__file__),ROOT/config["geometry_config"],ROOT/geometry["base_config"],
        ROOT/geometry["catalogue"],ROOT/"examples/rtl_rules_pe_sweep.py",*input_paths,
        *sorted((ROOT/"src/edge_moe_sim").rglob("*.py"))]
    summary = {"status":"complete", "simulator":"UnifiedSequenceSimulator", "config":config,
        "geometry":geometry, "base":base, "declarations":declarations,
        "configuration_points":len(comparison), "seed_points":len(seeds), "sample_results":len(raw),
        "source_sha256":{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
        "limits":["no RTL changes; hypothetical wider single Engine, not board measurements",
            "all cases byte OCM/unlimited credits/expert_locked; not previous finite-slot RTL-rules dataset",
            "all cases use identical BW/burst/payload timing/control parameters and saved routing inputs",
            "C requests full candidates but admission/budget can truncate; compute always waits for complete Expert",
            "D uses uniform GU prefix, not full-target streaming or RTL HardwareGreedy",
            "previous-layer weights barrier true for both predictive cases",
            "synthetic routing; OLMoE one sample per seed with rounded 91.67% recall",
            "whole demand baseline has no within-layer DMA/Expert-compute overlap; cross-layer speculation remains allowed"]}
    (args.output/"summary.json").write_text(json.dumps(summary,indent=2))


if __name__ == "__main__":
    main()
