"""Switch Top-1双候选FLAME条件对照：四case、放行屏障、完整就绪与净延迟收益。"""
from __future__ import annotations

import argparse
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

from mac_four_cases import ROOT, common_rules, run_case, write_csv
from switch_prefetch_comparison import make_inputs
from edge_moe_sim.config import load_model_config
from edge_moe_sim.schema import ModelConfig

CASES = (("A_whole_demand",False),("B_stream_demand",False),
    ("C_whole_prefetch",False),("D_stream_tile_prefetch",False),
    ("C_whole_prefetch",True),("D_stream_tile_prefetch",True))


def make_traces(model: ModelConfig, settings: dict[str,Any]) -> list[dict[str,Any]]:
    """固定Top-2质量，可选独立控制Top-1质量；论文最大值不冒充联合实测分布。"""
    records = make_inputs(model,settings)
    top1 = settings.get("top1_coverage")
    if top1 is not None:
        if not 0 <= top1 <= settings["coverage"]:
            raise ValueError("top1_coverage must be between zero and Top-2 coverage")
        hits = [(record,prediction) for record in records for prediction in record["predictions"]
            if prediction["true_rank"]>=0]
        random.Random(settings["seed"]).shuffle(hits)
        head_count = int(settings["samples"]*(len(model.moe_layer_indices)-1)*top1+.5)
        for index,(_,prediction) in enumerate(hits):
            rank = 0 if index<head_count else 1
            if rank != prediction["true_rank"]:
                prediction["candidate_ids"].reverse()
                prediction["true_rank"] = rank
    for record in records:
        record["truth"] = [[expert] for expert in record["truth"]]
    return records


def simulate(job: tuple[Any,...]) -> tuple[dict[str,Any],list[dict[str,Any]]]:
    """重用四case实现，屏障设置按点传入；诊断不改变真实路由或执行。"""
    model,base,geometry,common,entry,point,record = job
    settings = {**common,"include_prediction_diagnostics":True,
        "wait_for_previous_layer_weights":point["barrier"]}
    result = run_case((model,base,geometry,settings,entry,point,record))
    details = result.pop("prediction_diagnostics")
    for item in details:
        item.update({k:point[k] for k in ("model","mac_groups","bytes_per_cycle_per_channel","ocm_bytes","case","barrier","seed")})
        item["sample"] = record["sample"]
    result.update(prediction_targets=len(details),candidate_hits=sum(p["candidate_hit"] for p in details),
        full_true_resident_hits=sum(p["full_true_resident_at_router"] or 0 for p in details),
        full_true_transferred_hits=sum(p["full_true_transferred_before_router"] for p in details),
        useful_physical_bytes_before_router=sum(p["useful_physical_speculative_bytes_by_router"] for p in details),
        release_delay_sum_cycles=sum(p["prediction_to_release_cycles"] or 0 for p in details),
        released_targets=sum(p["prefetch_release_cycle"] is not None for p in details))
    return result,details


def aggregate(rows: list[dict[str,Any]]) -> list[dict[str,Any]]:
    """同模型/MAC/BW/容量/seed配对A、B与屏障开启参考；完整ready指标仅用于C。"""
    dimensions = ("model","mac_groups","bytes_per_cycle_per_channel","ocm_bytes","seed")
    lookup = {tuple(r[k] for k in dimensions)+(r["case"],r["barrier"]):r for r in rows}
    if len(lookup) != len(rows):
        raise ValueError("duplicate seed points")
    grouped: dict[tuple[Any,...],list[dict[str,Any]]] = {}
    for row in rows:
        key = tuple(row[k] for k in dimensions)
        a,b = lookup[key+("A_whole_demand",False)],lookup[key+("B_stream_demand",False)]
        row.update(speedup_vs_A=a["total_cycles"]/row["total_cycles"],
            speedup_vs_B=b["total_cycles"]/row["total_cycles"],
            net_saved_cycles_vs_A=a["total_cycles"]-row["total_cycles"],
            aggregate_wait_reduction_vs_A=a["memory_wait_sum_cycles"]-row["memory_wait_sum_cycles"],
            speedup_vs_barrier_on=lookup[key+(row["case"],True)]["total_cycles"]/row["total_cycles"]
                if row["case"].startswith(("C_","D_")) else 1.0)
        grouped.setdefault(key[:-1]+(row["case"],row["barrier"]),[]).append(row)
    output = []
    for _,values in sorted(grouped.items()):
        first = values[0]
        row = {k:first[k] for k in (*dimensions[:-1],"case","barrier","expert_bytes")}
        samples = sum(r["samples"] for r in values)
        targets = sum(r["prediction_targets"] for r in values)
        row.update(lanes=16*row["mac_groups"],seeds=len(values),samples=samples,
            cycles_per_sequence=sum(r["total_cycles"] for r in values)/samples,
            candidate_recall=sum(r["candidate_hits"] for r in values)/targets if targets else None,
            full_true_ready_recall=sum(r["full_true_resident_hits"] for r in values)/targets
                if targets and row["case"]=="C_whole_prefetch" else None,
            full_true_transfer_recall=sum(r["full_true_transferred_hits"] for r in values)/targets if targets else None,
            release_delay_per_target=sum(r["release_delay_sum_cycles"] for r in values)/targets if targets else None,
            useful_physical_bytes_before_router=sum(r["useful_physical_bytes_before_router"] for r in values),
            peak_ocm_bytes=max(r["peak_ocm_bytes"] for r in values),
            net_saved_cycles_per_sequence=sum(r["net_saved_cycles_vs_A"] for r in values)/samples,
            aggregate_wait_reduction_per_sequence=sum(r["aggregate_wait_reduction_vs_A"] for r in values)/samples)
        for metric in ("speedup_vs_A","speedup_vs_B","speedup_vs_barrier_on"):
            row[metric] = mean(r[metric] for r in values)
            row[metric+"_std"] = stdev(r[metric] for r in values) if len(values)>1 else None
        for metric in ("speculative_bytes","useful_speculative_bytes","wasted_speculative_bytes"):
            row[metric] = sum(r[metric] for r in values)
        output.append(row)
    return output


def plot(directory: Path, rows: list[dict[str,Any]]) -> None:
    """固定容量画PE/供数收益和C的static recall、完整驻留recall，避免混淆指标。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    capacity = max(r["ocm_bytes"] for r in rows)
    models = sorted({r["model"] for r in rows})
    groups = sorted({r["mac_groups"] for r in rows})
    for metric in ("speedup_vs_A","full_true_ready_recall"):
        fig,axes = plt.subplots(len(models),len(groups),figsize=(14,7),squeeze=False)
        for i,model in enumerate(models):
            for j,group in enumerate(groups):
                for case,barrier in CASES:
                    if metric=="full_true_ready_recall" and case!="C_whole_prefetch":
                        continue
                    points = sorted([r for r in rows if r["model"]==model and r["mac_groups"]==group
                        and r["case"]==case and r["barrier"]==barrier and r["ocm_bytes"]==capacity],
                        key=lambda r:r["bytes_per_cycle_per_channel"])
                    axes[i][j].plot([r["bytes_per_cycle_per_channel"] for r in points],
                        [r[metric] for r in points],marker="o",label=f"{case}; barrier={barrier}")
                axes[i][j].axhline(.844 if metric=="full_true_ready_recall" else 1,color="gray",linestyle="--")
                axes[i][j].set_title(f"{model}; {group*16} hypothetical lanes",fontsize=9)
                axes[i][j].set_xlabel("Nominal bytes/cycle/channel; payload multiplier=2")
                axes[i][j].set_ylabel(metric)
                axes[i][j].legend(fontsize=6)
        fig.tight_layout()
        for ext in ("png","pdf"):
            fig.savefig(directory/f"{metric}.{ext}")
        plt.close(fig)


def main() -> None:
    """完整受控扫描；主比较解除屏障，旧屏障仅保留为独立消融。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",type=Path,default=ROOT/"configs/experiments/flame_condition_comparison.yaml")
    parser.add_argument("--output",type=Path,default=ROOT/"results/flame_condition_comparison")
    parser.add_argument("--workers",type=int,default=4)
    args = parser.parse_args()
    if args.workers<1:
        raise ValueError("workers must be positive")
    geometry = yaml.safe_load(args.config.read_text())
    if geometry["primary_wait_for_previous_layer_weights"] is not False:
        raise ValueError("primary FLAME condition experiment must disable previous-weights barrier")
    base = yaml.safe_load((ROOT/geometry["base_config"]).read_text())
    common = yaml.safe_load((ROOT/geometry["common_config"]).read_text())
    args.output.mkdir(parents=True,exist_ok=True)
    (args.output/"summary.json").write_text(json.dumps({"status":"running"}))
    raw,seed_rows,details,declarations = [],[],[],[]
    with mp.get_context("spawn").Pool(args.workers) as pool:
        for model_path in geometry["models"]:
            model = load_model_config(ROOT/model_path)
            settings = {**base,"samples":geometry["samples"],"coverage":geometry["coverage"],"candidate_count":2}
            if "top1_coverage" in geometry:
                settings["top1_coverage"] = geometry["top1_coverage"]
            traces = {seed:make_traces(model,{**settings,"seed":seed}) for seed in geometry["seeds"]}
            (args.output/f"{model.name}_inputs.json").write_text(json.dumps(traces,indent=2))
            declarations.append({"model":asdict(model),"rules":asdict(common_rules(model,geometry,common))})
            points = []
            for groups,bw in product(geometry["mac_groups"],geometry["bytes_per_cycle_per_channel"]):
                seeds = geometry["seeds"] if (groups,bw)==(geometry["reference_mac_groups"],
                    geometry["reference_bytes_per_cycle_per_channel"]) else geometry["seeds"][:1]
                capacities = [geometry["ocm_bytes"]]
                if model.name==geometry["capacity_probe_model"] and len(seeds)>1:
                    capacities.append(geometry["capacity_probe_bytes"])
                for capacity,seed,(case,barrier) in product(capacities,seeds,CASES):
                    points.append(dict(model=model.name,mac_groups=groups,bytes_per_cycle_per_channel=bw,
                        ocm_bytes=capacity,case=case,barrier=barrier,seed=seed))
            for start in range(0,len(points),6):
                batch = points[start:start+6]
                values = pool.map(simulate,[(model,base,geometry,common,{"f_tile_size":geometry["f_tile_size"]},p,record)
                    for p in batch for record in traces[p["seed"]]])
                for index,point in enumerate(batch):
                    results = values[index*geometry["samples"]:(index+1)*geometry["samples"]]
                    samples = [r for r,_ in results]
                    raw.extend(samples)
                    details.extend(d for _,items in results for d in items)
                    row = {**point,"samples":len(samples),"expert_bytes":samples[0]["expert_bytes"],
                        "peak_ocm_bytes":max(r["peak_ocm_bytes"] for r in samples)}
                    for field in ("total_cycles","memory_wait_sum_cycles","payload_compute_cycles","speculative_bytes",
                            "useful_speculative_bytes","wasted_speculative_bytes","prediction_targets","candidate_hits",
                            "full_true_resident_hits","full_true_transferred_hits","useful_physical_bytes_before_router",
                            "release_delay_sum_cycles","released_targets"):
                        row[field] = sum(r[field] for r in samples)
                    seed_rows.append(row)
                write_csv(args.output/"checkpoint.csv",seed_rows)
                print(f"completed {model.name}; points {start+len(batch)}/{len(points)}",flush=True)
    comparison = aggregate(seed_rows)
    write_csv(args.output/"per_sample.csv",raw)
    write_csv(args.output/"per_seed.csv",seed_rows)
    write_csv(args.output/"per_prediction.csv",details)
    write_csv(args.output/"comparison.csv",comparison)
    plot(args.output,comparison)
    sources = [args.config.resolve(),Path(__file__),ROOT/geometry["common_config"],ROOT/geometry["base_config"],
        *[ROOT/p for p in geometry["models"]],ROOT/"examples/mac_four_cases.py",ROOT/"examples/switch_prefetch_comparison.py",
        *sorted((ROOT/"src/edge_moe_sim").rglob("*.py"))]
    summary = {"status":"complete","simulator":"UnifiedSequenceSimulator","geometry":geometry,"base":base,
        "declarations":declarations,"configuration_points":len(comparison),"seed_points":len(seed_rows),
        "sample_results":len(raw),"prediction_records":len(details),
        "source_sha256":{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
        "limits":["Switch structure / Top-1 / two candidates / exact 84.4% synthetic recall; not measured CEPR",
            "no pruning; INT8 assumed; six MoE layers only; first layer cold, no previous-token prediction",
            "four cases share byte OCM/unlimited credits/expert_locked; not finite-slot RTL validation",
            "MAC and BW configured, not board measurements; this driver does not simulate RTL",
            "main prediction cases barrier=false; barrier=true is only a separate ablation",
            "16 MiB shared weight capacity, not two isolated PA-BUF cache implementation",
            "full ready recall is actual resident true Expert at Router; completed transfer count is a separate metric",
            "net saved cycles is measured paired makespan change, not a causal decomposition into DMA/control",
            "optional Top-1 quality uses independent reported maxima as controlled sensitivity, not measured joint CEPR distribution",
            "main scan one seed; reference point and capacity probe three seeds"]}
    (args.output/"summary.json").write_text(json.dumps(summary,indent=2))


if __name__=="__main__":
    main()
