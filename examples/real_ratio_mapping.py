"""将合成收益比例映射到真实模型结构，保留Top-K并明确大OCM理论边界。"""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import hashlib
from itertools import product
import json
import multiprocessing as mp
from pathlib import Path
from statistics import mean, stdev
from typing import Any

import yaml

from olmoe_mixtral_cases import make_batch_inputs
from prefetch_phase_sweep import (ROOT, AdvancedExecutionConfig, BatchPredictorOutput, BatchRoutingEvent,
    CandidatePrefetch, HardwareProfile, ModelConfig, PrecomputedBatchPredictor, TokenPrediction,
    UnifiedLayerSequence, UnifiedPredictionSpec, UnifiedSequenceSimulator, write_csv)
from edge_moe_sim.config import load_model_config
from edge_moe_sim.real_model_profile import load_real_moe_profiles
from edge_moe_sim.work_model import expert_weight_bytes

POLICIES = ("no_prediction", "uniform_random", "uniform_hits_first", "oracle_all_gu")


def layouts(model: ModelConfig, config: dict[str, Any]) -> list[tuple[str, int, int]]:
    """产生粗tile和细tile对照；细tile等GU bytes对照用于排除单纯预算差异。"""
    result = []
    for groups in config["mapped_groups"]:
        if model.intermediate_size % groups:
            raise ValueError("mapped groups must divide intermediate size")
        result.append((f"mapped_{groups}_groups", model.intermediate_size // groups, 1))
    if not model.name.startswith("mixtral") or config["mixtral_include_fine"]:
        fine = config["fine_tile_size"]
        if model.intermediate_size % (2 * fine):
            raise ValueError("equal-byte fine prefix requires integer depth")
        result.extend([("fine_original", fine, config["fine_original_depth"]),
            ("fine_equal_half_gu", fine, model.intermediate_size // (2 * fine))])
    return result


def ordered_candidates(record: dict[str, Any], prediction: dict[str, Any], policy: str) -> tuple[int, ...]:
    """随机和hits-first共享候选集合；后者是truth辅助排名消融，不是真实预测器。"""
    true = record["truth"][prediction["target"]]
    ids = prediction["candidate_ids"]
    if policy == "oracle_all_gu":
        return tuple(true)
    if policy == "uniform_hits_first":
        true_set = set(true)
        return tuple([e for e in ids if e in true_set] + [e for e in ids if e not in true_set])
    return tuple(ids)


def simulate(job: tuple[ModelConfig, dict[str, Any], dict[str, Any], dict[str, Any]]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """用全部PE执行当前Expert；完成原始metrics与逐预测诊断，检查资源约束。"""
    model, settings, point, record = job
    pe = point["pe_count"]
    capacity = point["ocm_bytes"]
    fields = dict(settings["hardware"])
    fields.update(name="real_geometry_ratio_mapping_unmeasured", total_aie_units=pe,
        hardware_tag="SYNTHETIC_EFFECTIVE_RATES_NOT_BOARD_IMPLEMENTATION",
        bandwidth_bytes_per_sec_per_channel=point["bytes_per_cycle_per_channel"] * fields["clock_hz"])
    hardware = HardwareProfile(**fields, ocm_bytes=capacity,
        ops_per_cycle_per_aie=point["ops_per_cycle_per_pe"])
    sequence = UnifiedLayerSequence(tuple(BatchRoutingEvent(record["sample"],0,layer,(tuple(ids),))
        for layer,ids in zip(model.moe_layer_indices,record["truth"])))
    policy = point["policy"]
    count = model.top_k if policy == "oracle_all_gu" else settings["candidate_count"]
    groups = model.intermediate_size // point["f_tile_size"]
    depth = groups if policy == "oracle_all_gu" else point["prefix_depth"]
    specs, predictors = [], {}
    if policy != "no_prediction":
        for prediction in record["predictions"]:
            source,target = prediction["source"],prediction["target"]
            ids = ordered_candidates(record,prediction,policy)
            output = BatchPredictorOutput(record["sample"],0,target,0,(TokenPrediction(0,tuple(
                CandidatePrefetch(e,(len(ids)-i)/len(ids)) for i,e in enumerate(ids))),))
            predictors[(source,target)] = PrecomputedBatchPredictor({(record["sample"],0,target):output})
            specs.append(UnifiedPredictionSpec(source,target,"source_router_done",
                settings["predictor_latency_cycles"],count,"uniform_prefix",depth,
                wait_for_previous_layer_weights=False))
    execution = AdvancedExecutionConfig(point["f_tile_size"],placement_policy="striped_tile",
        aie_units_per_expert=pe,router_aie_units=pe,max_parallel_experts=1,candidate_count=count,
        nonlinear_elements_per_cycle=settings["nonlinear_elements_per_cycle"],speculative_byte_budget=capacity)
    result = UnifiedSequenceSimulator(model,hardware).simulate(sequence,execution,
        predictions=tuple(specs),predictors=predictors)
    metrics = result.memory_metrics
    if result.peak_ocm_bytes > capacity or len(result.layer_results) != len(sequence.layers):
        raise AssertionError("capacity violation or incomplete sequence")
    if metrics["speculative_bytes"] != metrics["useful_speculative_bytes"] + metrics["wasted_speculative_bytes"]:
        raise AssertionError("speculative byte accounting violation")
    row = {**point,"sample":record["sample"],"top_k":model.top_k,"candidate_count":count,
        "tile_groups":groups,"effective_gu_depth":depth if policy != "no_prediction" else 0,
        "expert_bytes":expert_weight_bytes(model),"total_cycles":result.inference_makespan_cycles,
        "peak_ocm_bytes":result.peak_ocm_bytes,
        "first_start_sum_cycles":sum(r.first_true_expert_start-r.router_resolve_cycle for r in result.layer_results.values()),
        "future_blocking_cycles":result.critical_path_future_speculation_blocking_cycles,
        "demand_bytes":sum(r.demand_bytes for r in result.layer_results.values()),
        "speculative_bytes":metrics["speculative_bytes"],"useful_speculative_bytes":metrics["useful_speculative_bytes"],
        "wasted_speculative_bytes":metrics["wasted_speculative_bytes"]}
    timings = [{"model":model.name,"seed":point["seed"],"sample":record["sample"],
        "layout":point["layout"],"pe_count":pe,"ratio":point["ratio"],"policy":policy,**asdict(p)}
        for p in result.predictions]
    return row,timings


def aggregate_seeds(rows: list[dict[str, Any]], seeds: list[int]) -> list[dict[str, Any]]:
    """对配对seed速度比汇总，速度比分母始终为相同layout的无预测streaming。"""
    grouped: dict[tuple[Any,...],list[dict[str,Any]]] = {}
    for row in rows:
        grouped.setdefault(tuple(row[k] for k in ["model","layout","pe_count","ratio","policy"]),[]).append(row)
    result = []
    for _, group in sorted(grouped.items()):
        if sorted(r["seed"] for r in group) != sorted(seeds):
            raise ValueError("missing or duplicate seed")
        row = {k:v for k,v in group[0].items() if k not in {"seed","speedup","baseline_cycles","total_cycles",
            "peak_ocm_bytes","first_start_sum_cycles","future_blocking_cycles","demand_bytes","speculative_bytes",
            "useful_speculative_bytes","wasted_speculative_bytes","samples"}}
        values = [r["speedup"] for r in group]
        row.update(seeds=len(group),samples=sum(r["samples"] for r in group),speedup_mean=mean(values),
            speedup_std=stdev(values),speedup_min=min(values),speedup_max=max(values),
            pooled_speedup=sum(r["baseline_cycles"] for r in group)/sum(r["total_cycles"] for r in group),
            total_cycles=sum(r["total_cycles"] for r in group),peak_ocm_bytes=max(r["peak_ocm_bytes"] for r in group))
        for key in ["first_start_sum_cycles","future_blocking_cycles","demand_bytes","speculative_bytes",
                    "useful_speculative_bytes","wasted_speculative_bytes"]:
            row[key] = sum(r[key] for r in group)
        result.append(row)
    return result


def plot(directory: Path, rows: list[dict[str, Any]]) -> None:
    """对各真实模型结构画固定策略曲线；不把大OCM结果标成可部署。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    models = list(dict.fromkeys(r["model"] for r in rows))
    pes = sorted({r["pe_count"] for r in rows})
    for policy in ["uniform_random","uniform_hits_first"]:
        fig,axes = plt.subplots(len(models),len(pes),figsize=(12,3*len(models)),squeeze=False)
        for i,model in enumerate(models):
            for j,pe in enumerate(pes):
                ax = axes[i][j]
                for layout in dict.fromkeys(r["layout"] for r in rows if r["model"]==model):
                    points = sorted([r for r in rows if r["model"]==model and r["pe_count"]==pe
                        and r["layout"]==layout and r["policy"]==policy],key=lambda r:r["ratio"])
                    line, = ax.plot([r["ratio"] for r in points],[r["speedup_mean"] for r in points],marker="o",label=layout)
                    ax.fill_between([r["ratio"] for r in points],[r["speedup_min"] for r in points],
                        [r["speedup_max"] for r in points],alpha=.15,color=line.get_color())
                budget = "HYPOTHETICAL LARGE OCM" if model.startswith("mixtral") else "OCM within 24 MiB budget"
                ax.set_title(f"{model}; PE={pe}\n{budget}",fontsize=9)
                ax.axhline(1,color="black",linestyle="--")
                ax.axhline(1.05,color="grey",linestyle=":")
                ax.set_xlabel("Ideal compute / memory payload ratio")
                ax.set_ylabel("Speedup vs same-layout streaming")
                ax.legend(fontsize=6)
        fig.suptitle(f"Real model geometry; synthetic 90% recall; {policy}; seed min-max")
        fig.tight_layout()
        for ext in ["png","pdf"]:
            fig.savefig(directory/f"{policy}.{ext}")
        plt.close(fig)


def main() -> None:
    """运行比例映射与细tile对照，保存输入、配对结果、图与出处。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",type=Path,default=ROOT/"configs/experiments/real_ratio_mapping.yaml")
    parser.add_argument("--output",type=Path,default=ROOT/"results/real_ratio_mapping")
    parser.add_argument("--workers",type=int,default=4)
    args = parser.parse_args()
    if args.workers < 1:
        raise ValueError("workers must be positive")
    args.config,args.output = args.config.resolve(),args.output.resolve()
    config = yaml.safe_load(args.config.read_text())
    if len(config["seeds"]) < 2 or len(set(config["seeds"])) != len(config["seeds"]):
        raise ValueError("at least two distinct seeds required")
    base = yaml.safe_load((ROOT/config["base_config"]).read_text())
    profiles = load_real_moe_profiles(ROOT/config["catalogue"],weight_bits=config["weight_bits"],activation_bits=config["activation_bits"])
    models = [load_model_config(ROOT/p) for p in config["switch_models"]] + [
        p.model for p in profiles if p.model.name in config["catalogue_models"]]
    models = [replace(m,moe_layer_indices=tuple(range(config["layers"]))) for m in models]
    args.output.mkdir(parents=True,exist_ok=True)
    (args.output/"summary.json").write_text(json.dumps({"status":"running"}))
    raw,seed_rows,timings,inputs,geometry = [],[],[],{},[]
    with mp.get_context("spawn").Pool(args.workers) as pool:
        for model in models:
            settings = {**base,"samples":config["samples_per_seed"],"coverage":config["coverage"],
                "candidate_count":min(model.num_experts,model.top_k*config["candidate_multiplier"])}
            traces = {seed:make_batch_inputs(model,{**settings,"seed":seed}) for seed in config["seeds"]}
            inputs[model.name] = traces
            (args.output/f"{model.name}_inputs.json").write_text(json.dumps(traces,indent=2))
            capacity = int(expert_weight_bytes(model)*config["ocm_expert_equivalents"])
            feasible = capacity <= config["current_ocm_budget_bytes"]
            geometry.append({**asdict(model),"expert_bytes":expert_weight_bytes(model),"ocm_bytes":capacity,
                "within_current_ocm_budget":feasible,"candidate_count":settings["candidate_count"]})
            for pe,ratio,(layout,tile,depth) in product(config["pe_counts"],config["ratios"],layouts(model,config)):
                points = [{"model":model.name,"layout":layout,"f_tile_size":tile,"prefix_depth":depth,
                    "pe_count":pe,"ops_per_cycle_per_pe":config["ops_per_cycle_per_pe"],"ratio":ratio,
                    "bytes_per_cycle_per_channel":ratio*pe*config["ops_per_cycle_per_pe"]/(2*base["hardware"]["num_memory_channels"]),
                    "ocm_bytes":capacity,"within_current_ocm_budget":feasible,"seed":seed,"policy":policy}
                    for seed,policy in product(config["seeds"],POLICIES)]
                values = pool.map(simulate,[(model,settings,point,record) for point in points for record in traces[point["seed"]]])
                for index,point in enumerate(points):
                    samples = values[index*settings["samples"]:(index+1)*settings["samples"]]
                    raw.extend(r for r,_ in samples)
                    timings.extend(t for _,ts in samples for t in ts)
                    row = dict(samples[0][0]);del row["sample"]
                    for key in ["total_cycles","first_start_sum_cycles","future_blocking_cycles","demand_bytes",
                            "speculative_bytes","useful_speculative_bytes","wasted_speculative_bytes"]:
                        row[key] = sum(r[key] for r,_ in samples)
                    row["peak_ocm_bytes"] = max(r["peak_ocm_bytes"] for r,_ in samples)
                    row["samples"] = len(samples)
                    seed_rows.append(row)
                current = seed_rows[-len(points):]
                for row in current:
                    baseline = next(r["total_cycles"] for r in current if r["seed"]==row["seed"] and r["policy"]=="no_prediction")
                    row.update(baseline_cycles=baseline,speedup=baseline/row["total_cycles"])
                write_csv(args.output/"per_seed.csv",seed_rows)
                print(f"completed {model.name} PE={pe} ratio={ratio} {layout}",flush=True)
    write_csv(args.output/"per_sample.csv",raw)
    write_csv(args.output/"per_prediction.csv",timings)
    aggregate = aggregate_seeds(seed_rows,config["seeds"])
    write_csv(args.output/"comparison.csv",aggregate)
    plot(args.output,aggregate)
    sources = [args.config,Path(__file__),ROOT/config["base_config"],ROOT/config["catalogue"],
        ROOT/"examples/olmoe_mixtral_cases.py",ROOT/"examples/switch_prefetch_comparison.py",
        ROOT/"examples/prefetch_phase_sweep.py",*(ROOT/p for p in config["switch_models"]),
        *sorted((ROOT/"src/edge_moe_sim").rglob("*.py"))]
    summary = {"status":"complete","simulator":"UnifiedSequenceSimulator","config":config,"base_settings":base,
        "models":geometry,"configuration_points":len(aggregate),"seed_points":len(seed_rows),"sample_results":len(raw),
        "source_sha256":{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
        "limits":["real geometry; synthetic routing and prediction, no checkpoint inference or measured board performance",
            "Mixtral mapping requires hypothetical 672/1152 MiB OCM; not deployable under current 24 MiB budget",
            "Top-K and candidate count preserved; hits-first is truth-assisted ranking ablation",
            "speedup denominator is same-layout no-prediction streaming, not original fine-layout baseline",
            "Oracle all-GU is diagnostic, not optimal schedule; all PEs assigned current GEMM; sequential Experts",
            "three seeds, ten samples each, five MoE layers; no full-model inference or RTL tile feasibility claim"]}
    (args.output/"summary.json").write_text(json.dumps(summary,indent=2))


if __name__ == "__main__":
    main()
