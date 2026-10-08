"""主模拟器RTL架构规则逐项对照与单Engine MAC规模扫描，不修改RTL。"""
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
from edge_moe_sim.execution_rules import ExecutionRules
from edge_moe_sim.real_model_profile import load_real_moe_profiles

MODES = ("legacy", "tile_scheduler", "burst_fifo", "rtl_resources")


def rules_for(mode: str, model: ModelConfig, tile: int, config: dict[str, Any]) -> ExecutionRules:
    """slot/pair与tile调度共同启用，再加入burst/FIFO、控制及层间退休。

    不混用expert锁定与Router序pair admission：Top-K局部预取在容量饱和时
    可能锁定非游标Expert，无法腾出游标所需slot，不能作为有效消融配置。
    """
    if mode not in MODES:
        raise ValueError("unknown rule mode")
    if mode == "legacy":
        return ExecutionRules()
    slot = (model.input_projection_count * model.hidden_size * tile * model.weight_bits + 7)//8
    values: dict[str,Any] = dict(ocm_slot_bytes=slot,ocm_banks=config["ocm_banks"],
        reserved_demand_slots=2,atomic_tile_pair=True,retry_future_prefetch=True,request_order="pair_fifo")
    if mode in {"tile_scheduler","burst_fifo","rtl_resources"}:
        values["compute_policy"] = "tile_ready_first"
    if mode in {"burst_fifo","rtl_resources"}:
        values.update(burst_arbitration=True,descriptor_entries=config["descriptor_entries"],
            reserved_demand_descriptors=2,dma_fifo_depth=config["dma_fifo_depth"])
    if mode == "rtl_resources":
        values.update(config["control"])
        # RTL top-K scan + EXP + selected-logit divide/emit结构估计，不拟合makespan。
        values["router_control_cycles"] = model.num_experts*model.top_k+2*model.top_k+34*(model.top_k-1)+3
        candidates = min(model.num_experts,model.top_k*config["candidate_multiplier"])
        values["prefetch_allocator_cycles"] = candidates+5+(candidates-1).bit_length()
    return ExecutionRules(**values)


def build_models(config: dict[str,Any]) -> list[tuple[ModelConfig,dict[str,Any]]]:
    """保持真实OLMoE Top-8；合成模型用于低成本密集供数研究。"""
    catalogue = {p.model.name:p.model for p in load_real_moe_profiles(ROOT/config["catalogue"])}
    result = []
    for row in config["models"]:
        if row["name"] in catalogue:
            model = catalogue[row["name"]]
        else:
            model = ModelConfig(row["name"],row["hidden_size"],row["intermediate_size"],
                row["num_experts"],row["top_k"],8,8)
        result.append((replace(model,moe_layer_indices=tuple(range(config["layers"]))),row))
    return result


def simulate(job: tuple[ModelConfig,dict[str,Any],dict[str,Any],dict[str,Any],dict[str,Any]]) -> dict[str,Any]:
    """一个Engine内扩大MAC速率；严格保留单共享Engine和tile ownership。"""
    model,settings,config,point,record = job
    rules = rules_for(point["mode"],model,point["f_tile_size"],config)
    fields = dict(settings["hardware"])
    fields.update(name="rtl_rules_hypothetical_MAC_width",total_aie_units=1,burst_bytes=config["burst_bytes"],
        hardware_tag="ARCHITECTURE_RULES_NOT_RTL_CYCLE_OR_BOARD_MEASUREMENT",
        bandwidth_bytes_per_sec_per_channel=point["bytes_per_cycle_per_channel"]*fields["clock_hz"])
    hardware = HardwareProfile(**fields,ops_per_cycle_per_aie=config["ops_per_cycle_per_group"]*point["mac_groups"],
        ocm_bytes=point["ocm_bytes"])
    sequence = UnifiedLayerSequence(tuple(BatchRoutingEvent(record["sample"],0,layer,(tuple(ids),))
        for layer,ids in zip(model.moe_layer_indices,record["truth"])))
    specs,predictors = [],{}
    # 当前 RTL 和主模型均按资源可用性放行，不施加前层完整权重屏障。
    wait = False
    if point["policy"] == "uniform_prefix":
        for prediction in record["predictions"]:
            source,target = prediction["source"],prediction["target"]
            ids = prediction["candidate_ids"]
            output = BatchPredictorOutput(record["sample"],0,target,0,(TokenPrediction(0,tuple(
                CandidatePrefetch(e,score) for e,score in zip(ids,prediction["scores"]))),))
            predictors[(source,target)] = PrecomputedBatchPredictor({(record["sample"],0,target):output})
            specs.append(UnifiedPredictionSpec(source,target,"source_router_done",settings["predictor_latency_cycles"],
                settings["candidate_count"],"uniform_prefix",config["prefix_depth"],wait_for_previous_layer_weights=wait))
    execution = AdvancedExecutionConfig(point["f_tile_size"],placement_policy="striped_tile",
        aie_units_per_expert=1,router_aie_units=1,max_parallel_experts=1,
        candidate_count=settings["candidate_count"],speculative_byte_budget=point["ocm_bytes"],
        nonlinear_elements_per_cycle=settings["nonlinear_elements_per_cycle"],execution_rules=rules)
    result = UnifiedSequenceSimulator(model,hardware).simulate(sequence,execution,
        predictions=tuple(specs),predictors=predictors)
    metrics = result.memory_metrics
    if len(result.layer_results)!=len(sequence.layers) or result.peak_ocm_bytes>point["ocm_bytes"]:
        raise AssertionError("incomplete sequence or capacity violation")
    if metrics["speculative_bytes"]!=metrics["useful_speculative_bytes"]+metrics["wasted_speculative_bytes"]:
        raise AssertionError("speculative byte classification mismatch")
    reserved_peak = max(metrics["fifo_reserved_credit_peaks"].values(),default=0)
    occupancy_peak = max(metrics["fifo_occupancy_peaks"].values(),default=0)
    if rules.descriptor_entries and metrics["descriptor_peak"]>rules.descriptor_entries:
        raise AssertionError("descriptor credit overflow")
    if rules.dma_fifo_depth and max(reserved_peak,occupancy_peak)>rules.dma_fifo_depth:
        raise AssertionError("FIFO reservation/occupancy overflow")
    useful_cycles = result.aie_metrics["useful_aie_cycles"]
    allocated_cycles = result.aie_metrics["allocated_aie_cycles"]
    return {**point,"sample":record["sample"],"top_k":model.top_k,"candidate_count":settings["candidate_count"],
        "total_cycles":result.inference_makespan_cycles,"peak_ocm_bytes":result.peak_ocm_bytes,
        "payload_compute_cycles":useful_cycles,"allocated_compute_cycles":allocated_cycles,
        "memory_wait_sum_cycles":sum(r.aie_starvation_cycles for r in result.layer_results.values()),
        "future_blocking_cycles":result.critical_path_future_speculation_blocking_cycles,
        "demand_bytes":sum(r.demand_bytes for r in result.layer_results.values()),
        "speculative_bytes":metrics["speculative_bytes"],"useful_speculative_bytes":metrics["useful_speculative_bytes"],
        "wasted_speculative_bytes":metrics["wasted_speculative_bytes"],
        "descriptor_peak":metrics.get("descriptor_peak",metrics.get("live_descriptor_peak",0)),
        "fifo_reserved_credit_peak":reserved_peak,"fifo_occupancy_peak":occupancy_peak,
        "wait_for_previous_weights":wait,"logical_engines":1,
        "hypothetical_mac_lanes":16*point["mac_groups"]}


def aggregate_seeds(rows: list[dict[str,Any]], seeds: list[int]) -> list[dict[str,Any]]:
    """汇总单Engine利用率及配对速度比；总延迟和prefetch净收益分开。"""
    grouped: dict[tuple[Any,...],list[dict[str,Any]]] = {}
    for row in rows:
        grouped.setdefault(tuple(row[k] for k in ["model","mode","mac_groups","bytes_per_cycle_per_channel","policy"]),[]).append(row)
    result = []
    for key,group in sorted(grouped.items()):
        if sorted(r["seed"] for r in group)!=sorted(seeds):
            raise ValueError("missing or duplicate seed")
        row = {k:v for k,v in group[0].items() if k not in {"seed","samples","total_cycles","peak_ocm_bytes",
            "payload_compute_cycles","allocated_compute_cycles","memory_wait_sum_cycles","future_blocking_cycles",
            "demand_bytes","speculative_bytes","useful_speculative_bytes","wasted_speculative_bytes",
            "speedup_vs_no_prediction","speedup_vs_group1","baseline_cycles","group1_cycles","descriptor_peak",
            "fifo_reserved_credit_peak","fifo_occupancy_peak"}}
        samples = sum(r["samples"] for r in group)
        values = [r["speedup_vs_no_prediction"] for r in group]
        row.update(seeds=len(group),samples=samples,cycles_per_sequence=sum(r["total_cycles"] for r in group)/samples,
            prefetch_speedup_mean=mean(values),prefetch_speedup_min=min(values),prefetch_speedup_max=max(values),
            prefetch_speedup_std=stdev(values),mac_scaling_speedup=mean(r["speedup_vs_group1"] for r in group),
            payload_utilization=sum(r["payload_compute_cycles"] for r in group)/sum(r["total_cycles"] for r in group),
            allocated_utilization=sum(r["allocated_compute_cycles"] for r in group)/sum(r["total_cycles"] for r in group),
            peak_ocm_bytes=max(r["peak_ocm_bytes"] for r in group),descriptor_peak=max(r["descriptor_peak"] for r in group))
        row["fifo_reserved_credit_peak"] = max(r["fifo_reserved_credit_peak"] for r in group)
        row["fifo_occupancy_peak"] = max(r["fifo_occupancy_peak"] for r in group)
        for field in ["memory_wait_sum_cycles","future_blocking_cycles","demand_bytes","speculative_bytes",
                      "useful_speculative_bytes","wasted_speculative_bytes"]:
            row[field] = sum(r[field] for r in group)
        result.append(row)
    return result


def plot(directory: Path, rows: list[dict[str,Any]]) -> None:
    """同时显示总cycles、payload利用率和预取速度比，不用百分比替代总性能。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    panels = sorted({(r["model"],r["bytes_per_cycle_per_channel"]) for r in rows})
    fig,axes = plt.subplots(len(panels),3,figsize=(14,3*len(panels)),squeeze=False)
    for i,(model,bw) in enumerate(panels):
        for j,(metric,label) in enumerate([("cycles_per_sequence","Cycles / sequence"),
                ("payload_utilization","Compute payload utilization"),("prefetch_speedup_mean","Prefetch speedup")]):
            ax = axes[i][j]
            for mode,policy in product(["legacy","rtl_resources"],["no_prediction","uniform_prefix"]):
                if j==2 and policy=="no_prediction":
                    continue
                selected = sorted([r for r in rows if r["model"]==model and r["bytes_per_cycle_per_channel"]==bw
                    and r["mode"]==mode and r["policy"]==policy],key=lambda r:r["mac_groups"])
                ax.plot([r["mac_groups"] for r in selected],[r[metric] for r in selected],marker="o",label=f"{mode}/{policy}")
            if j==2:
                ax.axhline(1,color="black",linestyle="--")
            ax.set_xscale("log",base=2)
            ax.set_title(f"{model}; {bw} B/cycle/ch",fontsize=9)
            ax.set_xlabel("16-lane MAC groups in ONE hypothetical Engine")
            ax.set_ylabel(label)
            ax.legend(fontsize=6)
    fig.tight_layout()
    for ext in ["png","pdf"]:
        fig.savefig(directory/f"mac_scale.{ext}")
    plt.close(fig)


def main() -> None:
    """扫描既有规则组合，保存配置/输入、逐样本、规则消融、图和来源。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",type=Path,default=ROOT/"configs/experiments/rtl_rules_pe_sweep.yaml")
    parser.add_argument("--output",type=Path,default=ROOT/"results/rtl_rules_pe_sweep")
    parser.add_argument("--workers",type=int,default=4)
    args = parser.parse_args()
    if args.workers<1:
        raise ValueError("workers must be positive")
    args.config,args.output = args.config.resolve(),args.output.resolve()
    config = yaml.safe_load(args.config.read_text())
    base = yaml.safe_load((ROOT/config["base_config"]).read_text())
    args.output.mkdir(parents=True,exist_ok=True)
    (args.output/"summary.json").write_text(json.dumps({"status":"running"}))
    raw,seed_rows,declarations = [],[],[]
    with mp.get_context("spawn").Pool(args.workers) as pool:
        for model,entry in build_models(config):
            tile = entry["f_tile_size"]
            slot = rules_for("rtl_resources",model,tile,config).ocm_slot_bytes
            capacity = slot*config["ocm_banks"]*config["slots_per_bank"]
            settings = {**base,"samples":entry["samples"],"coverage":config["coverage"],
                "candidate_count":min(model.num_experts,model.top_k*config["candidate_multiplier"])}
            traces = {seed:make_batch_inputs(model,{**settings,"seed":seed}) for seed in config["seeds"]}
            (args.output/f"{model.name}_inputs.json").write_text(json.dumps(traces,indent=2))
            for mode in MODES:
                declarations.append({"model":model.name,"mode":mode,"rules":asdict(rules_for(mode,model,tile,config)),
                    "wait_for_previous_weights":mode!="legacy","slot_payload_bytes":slot,"ocm_bytes":capacity})
            for groups,bw in product(entry.get("mac_groups",config["mac_groups"]),
                    entry.get("bytes_per_cycle_per_channel",config["bytes_per_cycle_per_channel"])):
                modes = MODES if groups==1 and bw==16 else ("legacy","rtl_resources")
                points = [dict(model=model.name,mode=mode,mac_groups=groups,bytes_per_cycle_per_channel=bw,
                    policy=policy,seed=seed,f_tile_size=tile,ocm_bytes=capacity)
                    for seed,mode,policy in product(config["seeds"],modes,["no_prediction","uniform_prefix"])]
                values = pool.map(simulate,[(model,settings,config,p,rec) for p in points for rec in traces[p["seed"]]])
                raw.extend(values)
                for index,point in enumerate(points):
                    samples = values[index*entry["samples"]:(index+1)*entry["samples"]]
                    row = dict(samples[0]);del row["sample"]
                    for field in ["total_cycles","payload_compute_cycles","allocated_compute_cycles","memory_wait_sum_cycles",
                            "future_blocking_cycles","demand_bytes","speculative_bytes","useful_speculative_bytes","wasted_speculative_bytes"]:
                        row[field] = sum(r[field] for r in samples)
                    row["peak_ocm_bytes"] = max(r["peak_ocm_bytes"] for r in samples)
                    row["descriptor_peak"] = max(r["descriptor_peak"] for r in samples)
                    row["fifo_reserved_credit_peak"] = max(r["fifo_reserved_credit_peak"] for r in samples)
                    row["fifo_occupancy_peak"] = max(r["fifo_occupancy_peak"] for r in samples)
                    row["samples"] = len(samples)
                    seed_rows.append(row)
                write_csv(args.output/"checkpoint.csv",seed_rows)
                print(f"completed {model.name} MAC groups={groups} BW/ch={bw}",flush=True)
    lookup = {(r["model"],r["mode"],r["mac_groups"],r["bytes_per_cycle_per_channel"],r["policy"],r["seed"]):r for r in seed_rows}
    for row in seed_rows:
        identity = (row["model"],row["mode"],row["mac_groups"],row["bytes_per_cycle_per_channel"])
        baseline = lookup[(*identity,"no_prediction",row["seed"])]["total_cycles"]
        group1 = lookup[(row["model"],row["mode"],1,row["bytes_per_cycle_per_channel"],row["policy"],row["seed"])]["total_cycles"]
        row.update(baseline_cycles=baseline,group1_cycles=group1,speedup_vs_no_prediction=baseline/row["total_cycles"],
            speedup_vs_group1=group1/row["total_cycles"])
    aggregate = aggregate_seeds(seed_rows,config["seeds"])
    write_csv(args.output/"per_sample.csv",raw)
    write_csv(args.output/"per_seed.csv",seed_rows)
    write_csv(args.output/"comparison.csv",aggregate)
    write_csv(args.output/"rule_ablation.csv",[r for r in aggregate if r["mac_groups"]==1 and r["bytes_per_cycle_per_channel"]==16])
    (args.output/"rule_declarations.json").write_text(json.dumps(declarations,indent=2))
    plot(args.output,aggregate)
    sources = [args.config,Path(__file__),ROOT/config["base_config"],ROOT/config["catalogue"],
        ROOT/"examples/olmoe_mixtral_cases.py",ROOT/"examples/switch_prefetch_comparison.py",
        ROOT/"examples/prefetch_phase_sweep.py",*sorted((ROOT/"src/edge_moe_sim").rglob("*.py"))]
    rtl_sources = sorted((ROOT/"hardware/m10_u200/rtl").glob("*.sv"))
    summary = {"status":"complete","simulator":"UnifiedSequenceSimulator","config":config,
        "configuration_points":len(aggregate),"seed_points":len(seed_rows),"sample_results":len(raw),
        "source_sha256":{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
        "rtl_sha256":{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in rtl_sources},
        "limits":["no RTL modified or executed; MAC groups>1 are hypothetical wider single Engine",
            "finite slot/pair/FIFO/burst/scheduler rules correspond to RTL architecture, not complete cycle equivalence",
            "Uniform allocator differs from current RTL HardwareGreedy; not an end-to-end RTL validation claim",
            "control estimates generalized from declared RTL structure, not calibrated model-sized cycle costs",
            "all modes release future prefetch without a previous-layer complete-weight barrier",
            "OLMoE one sample per seed for cost; realized recall rounded, not exactly 90%",
            "memory_wait_sum is aggregate Expert wait, not a disjoint portion of makespan",
            "payload utilization includes Router+Experts, excludes startup; no wide-OCM feasibility or DSP mapping proof"]}
    (args.output/"summary.json").write_text(json.dumps(summary,indent=2))


if __name__=="__main__":
    main()
