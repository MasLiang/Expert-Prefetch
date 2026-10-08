"""固定headroom模型的tile粒度扫描：streaming最佳粒度与预取最佳粒度分开。"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from math import ceil
import multiprocessing as mp
from pathlib import Path
from statistics import mean, stdev
from typing import Any

import yaml

from prefetch_phase_sweep import (ROOT, AdvancedExecutionConfig, BatchPredictorOutput,
    BatchRoutingEvent, CandidatePrefetch, HardwareProfile, ModelConfig,
    PrecomputedBatchPredictor, TokenPrediction, UnifiedLayerSequence,
    UnifiedPredictionSpec, UnifiedSequenceSimulator, write_csv)


class ObservedSimulator(UnifiedSequenceSimulator):
    """仅保存已有事件状态引用用于只读测量；不修改事件、队列或资源行为。"""

    def _handle(self, state: Any, event: Any) -> None:
        """按原处理器执行，再保留可读的资源/请求记录。"""
        super()._handle(state, event)
        self.observed_state = state


def idle_windows(blocked: list[tuple[int, int]], start_cycle: int,
                 end_cycle: int) -> list[tuple[int, int]]:
    """求半开窗口中没有baseline已issue未完成demand的空档；合并重叠区间。"""
    if end_cycle <= start_cycle:
        return []
    cursor, result = start_cycle, []
    for start, end in sorted(blocked):
        if end <= cursor or start >= end_cycle:
            continue
        if start > cursor:
            result.append((cursor, min(start, end_cycle)))
        cursor = max(cursor, min(end, end_cycle))
    if cursor < end_cycle:
        result.append((cursor, end_cycle))
    return result


def slack_metrics(windows: list[tuple[int, int]], request_cycles: int) -> dict[str, int | float]:
    """事后baseline窗口能装下一个GU DMA的比例；不是在线调度安全保证。"""
    if request_cycles <= 0:
        raise ValueError("request_cycles must be positive")
    fitting = [end-start for start,end in windows if end-start >= request_cycles]
    total = sum(end-start for start,end in windows)
    return dict(safe_slack_windows=len(windows), fitting_slack_windows=len(fitting),
        safe_slack_cycles=total, fitting_slack_cycles=sum(fitting),
        safe_slack_fit_ratio=len(fitting)/len(windows) if windows else 0.,
        safe_slack_cycle_fit_ratio=sum(fitting)/total if total else 0.)


def validate(config: dict[str, Any], base: dict[str, Any]) -> None:
    """保证固定F，不用groups固定而误改变Expert大小；各tile精确整除GU预算。"""
    if len(set(config["seeds"]))!=len(config["seeds"]) or len(config["seeds"])<2:
        raise ValueError("distinct multiple seeds required")
    if config["hidden_size"]!=base["hidden_size"] or config["intermediate_size"]<=0:
        raise ValueError("invalid fixed workload")
    if not config["f_tile_sizes"] or len(set(config["f_tile_sizes"]))!=len(config["f_tile_sizes"]):
        raise ValueError("unique tile sizes required")
    for tile in config["f_tile_sizes"]:
        gu_bytes=2*config["hidden_size"]*tile
        if tile<=0 or config["intermediate_size"]%tile or config["fixed_gu_budget_bytes"]%gu_bytes:
            raise ValueError("tile must divide F and fixed GU budget exactly")
        if not 1<=config["fixed_gu_budget_bytes"]//gu_bytes<=config["intermediate_size"]//tile:
            raise ValueError("fixed GU budget outside Expert geometry")
    if config["ocm_bytes"] < 3*config["hidden_size"]*max(config["f_tile_sizes"]):
        raise ValueError("OCM cannot contain one tile")
    if any(r<=0 for r in config["ratios"]) or any(not 0<=q<=1 for q in config["accuracies"]):
        raise ValueError("invalid ratio/accuracy")


def worker(job: tuple[dict[str, Any], dict[str, Any], int, float, float, str, int, dict[str, Any]]) -> dict[str, Any]:
    """运行当前主模型，读真实请求/interval，不改变调度；单位cycle、byte。"""
    base, config, tile, ratio, accuracy, policy, seed, record = job
    if policy not in {"OFF","LIGHT","GU32","FULL"}:
        raise ValueError("unknown policy")
    h,f=config["hidden_size"],config["intermediate_size"]
    groups=f//tile; gu_bytes=2*h*tile
    bandwidth=ratio*base["ops_per_cycle_per_pe"]/(2*base["channels"])
    model=ModelConfig(name="synthetic_fixed_headroom_tile_sweep",hidden_size=h,intermediate_size=f,
        num_experts=base["num_experts"],top_k=1,weight_bits=8,activation_bits=8,
        moe_layer_indices=tuple(range(base["layers"])))
    hardware=HardwareProfile(name="synthetic_not_board",clock_hz=base["clock_hz"],
        num_memory_channels=base["channels"],bandwidth_bytes_per_sec_per_channel=bandwidth*base["clock_hz"],
        bandwidth_efficiency=1,dma_setup_cycles=base["dma_setup_cycles"],max_inflight_per_channel=1,
        burst_bytes=16384,supports_channel_striping=True,total_aie_units=1,
        ops_per_cycle_per_aie=base["ops_per_cycle_per_pe"],aie_efficiency=1,
        aie_task_startup_cycles=base["compute_startup_cycles"],ocm_bytes=config["ocm_bytes"])
    sequence=UnifiedLayerSequence(tuple(BatchRoutingEvent(record["sample"],0,layer,((expert,),))
        for layer,expert in enumerate(record["truth"])))
    specs,predictors=[],{}
    depth=0 if policy=="OFF" else (1 if policy=="LIGHT" else
        config["fixed_gu_budget_bytes"]//gu_bytes if policy=="GU32" else groups)
    for target in record["targets"] if policy!="OFF" else []:
        layer=target["target"]
        hit=target["quality_order"]<int(accuracy*base["samples"]*(base["layers"]-1)+.5)
        expert=record["truth"][layer] if hit else target["wrong_ids"][0]
        output=BatchPredictorOutput(record["sample"],0,layer,0,
            (TokenPrediction(0,(CandidatePrefetch(expert,1.),)),))
        predictors[layer-1,layer]=PrecomputedBatchPredictor({(record["sample"],0,layer):output})
        specs.append(UnifiedPredictionSpec(layer-1,layer,"source_router_done",base["predictor_latency_cycles"],
            1,"full_expert_prefetch" if policy=="FULL" else "uniform_prefix",depth,
            wait_for_previous_layer_weights=False))
    execution=AdvancedExecutionConfig(tile,placement_policy="striped_tile",max_parallel_experts=1,
        candidate_count=1,speculative_byte_budget=config["ocm_bytes"],
        nonlinear_elements_per_cycle=base["nonlinear_elements_per_cycle"])
    simulator=ObservedSimulator(model,hardware)
    result=simulator.simulate(sequence,execution,predictions=tuple(specs),predictors=predictors)
    state=simulator.observed_state
    starts=[r for r in state.requests if r.started]
    intervals=[interval for channel in state.memory.channels for interval in channel.intervals]
    occupied=sum(i.end_cycle-i.start_cycle for i in intervals)
    setup=len(intervals)*base["dma_setup_cycles"]
    drain=result.final_resource_idle_cycle; cycles=result.inference_makespan_cycles
    metrics=result.memory_metrics
    if policy=="OFF" and drain!=cycles:
        raise AssertionError("baseline has DMA tail")
    if occupied!=metrics["memory_busy_cycles"] or len(intervals)!=len(starts)*base["channels"]:
        raise AssertionError("striped request/interval counting mismatch")
    if result.peak_ocm_bytes>config["ocm_bytes"] or setup>occupied:
        raise AssertionError("capacity or setup accounting violation")
    if metrics["speculative_bytes"]!=metrics["useful_speculative_bytes"]+metrics["wasted_speculative_bytes"]:
        raise AssertionError("byte conservation failed")
    if accuracy==1 and metrics["wasted_speculative_bytes"]:
        raise AssertionError("perfect predictor wasted bytes")
    windows=[]
    if policy=="OFF":
        blocked=[(r.issue_cycle,r.end_cycle) for r in starts]
        for layer in range(1,base["layers"]):
            ready=result.layer_results[layer-1].router_resolve_cycle+base["predictor_latency_cycles"]
            deadline=result.layer_results[layer].router_resolve_cycle
            windows.extend(idle_windows(blocked,ready,deadline))
    required=base["dma_setup_cycles"]+ceil(gu_bytes/(base["channels"]*bandwidth))
    return dict(f_tile_size=tile,tile_groups=groups,gu_tile_bytes=gu_bytes,down_tile_bytes=h*tile,
        logical_tile_bytes=3*h*tile,ratio=ratio,accuracy=accuracy,policy=policy,prefix_depth=depth,
        seed=seed,sample=record["sample"],total_cycles=cycles,drain_cycles=drain,
        request_count=len(starts),issued_request_count=len(state.requests),
        canceled_before_start_requests=sum(r.canceled and not r.started for r in state.requests),
        channel_interval_count=len(intervals),dma_setup_channel_cycles=setup,dma_busy_channel_cycles=occupied,
        dma_setup_fraction=setup/occupied if occupied else 0.,
        transferred_bytes=metrics["transferred_bytes"],speculative_bytes=metrics["speculative_bytes"],
        useful_speculative_bytes=metrics["useful_speculative_bytes"],wasted_speculative_bytes=metrics["wasted_speculative_bytes"],
        payload_utilization=metrics["transferred_bytes"]/(base["channels"]*bandwidth*drain),
        memory_occupancy=occupied/(base["channels"]*drain),
        demand_blocking_cycles=result.demand_blocked_by_future_speculation_cycles,
        critical_future_blocking_cycles=result.critical_path_future_speculation_blocking_cycles,
        aie_busy_cycles=result.aie_metrics["allocated_aie_cycles"],
        aie_payload_cycles=result.aie_metrics["useful_aie_cycles"],
        aie_utilization=result.aie_metrics["allocated_aie_cycles"]/cycles,
        aie_payload_utilization=result.aie_metrics["useful_aie_cycles"]/cycles,
        peak_ocm_bytes=result.peak_ocm_bytes,gu_dma_runtime_cycles=required,**slack_metrics(windows,required))


def aggregate(raw: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """按同tile的OFF配对；利用率/分数取计数比，不平均sample比值。"""
    baseline={(r["f_tile_size"],r["ratio"],r["seed"],r["sample"]):r for r in raw if r["policy"]=="OFF"}
    if len(baseline)!=sum(r["policy"]=="OFF" for r in raw):
        raise ValueError("duplicate baseline")
    grouped={}
    for r in raw:grouped.setdefault((r["f_tile_size"],r["ratio"],r["accuracy"],r["policy"]),[]).append(r)
    rows=[]
    for (tile,ratio,accuracy,policy),values in sorted(grouped.items()):
        expected={(k[2],k[3]) for k in baseline if k[:2]==(tile,ratio)}
        if len(values)!=len(expected) or {(r["seed"],r["sample"]) for r in values}!=expected:
            raise ValueError("missing or duplicate paired samples")
        bases=[baseline[tile,ratio,r["seed"],r["sample"]] for r in values]
        total=sum(r["total_cycles"] for r in values);base_total=sum(r["total_cycles"] for r in bases)
        ratios=[sum(r["total_cycles"] for r in bases if r["seed"]==seed)/
                sum(r["total_cycles"] for r in values if r["seed"]==seed) for seed in sorted({r["seed"] for r in values})]
        row={n:values[0][n] for n in ("f_tile_size","tile_groups","gu_tile_bytes","down_tile_bytes",
            "logical_tile_bytes","ratio","accuracy","policy","prefix_depth","gu_dma_runtime_cycles")}
        for name in ("total_cycles","drain_cycles","request_count","issued_request_count","channel_interval_count",
            "canceled_before_start_requests","dma_setup_channel_cycles","dma_busy_channel_cycles","transferred_bytes",
            "speculative_bytes","useful_speculative_bytes","wasted_speculative_bytes","demand_blocking_cycles",
            "critical_future_blocking_cycles","aie_busy_cycles","aie_payload_cycles"):
            row[name]=sum(r[name] for r in values)
        row.update(samples=len(values),mean_cycles=total/len(values),speedup=base_total/total,
            speedup_mean=mean(ratios),speedup_std=stdev(ratios) if len(ratios)>1 else 0.,
            speedup_min=min(ratios),speedup_max=max(ratios),baseline_total_cycles=base_total,
            dma_setup_fraction=row["dma_setup_channel_cycles"]/row["dma_busy_channel_cycles"],
            payload_utilization=sum(r["payload_utilization"]*r["drain_cycles"] for r in values)/row["drain_cycles"],
            memory_occupancy=sum(r["memory_occupancy"]*r["drain_cycles"] for r in values)/row["drain_cycles"],
            aie_utilization=row["aie_busy_cycles"]/total,aie_payload_utilization=row["aie_payload_cycles"]/total,
            peak_ocm_bytes=max(r["peak_ocm_bytes"] for r in values))
        # Every policy gets the same counterfactual baseline slack metric.
        for name in ("safe_slack_windows","fitting_slack_windows","safe_slack_cycles","fitting_slack_cycles"):
            row[name]=sum(r[name] for r in bases)
        row["safe_slack_fit_ratio"]=row["fitting_slack_windows"]/row["safe_slack_windows"] if row["safe_slack_windows"] else 0.
        row["safe_slack_cycle_fit_ratio"]=row["fitting_slack_cycles"]/row["safe_slack_cycles"] if row["safe_slack_cycles"] else 0.
        rows.append(row)
    return rows


def optima(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """分别报告baseline最佳tile、增量speedup最佳tile、overall latency最佳tile及并列。"""
    result=[]
    for ratio in sorted({r["ratio"] for r in rows}):
        baseline=[r for r in rows if r["ratio"]==ratio and r["policy"]=="OFF"]
        best_base=min(r["mean_cycles"] for r in baseline)
        base_tiles=[r["f_tile_size"] for r in baseline if r["mean_cycles"]==best_base]
        for accuracy in (.9,1.):
            for policy in ("LIGHT","GU32","FULL"):
                selected=[r for r in rows if r["ratio"]==ratio and r["accuracy"]==accuracy and r["policy"]==policy]
                if not selected:continue
                speed=max(r["speedup"] for r in selected);latency=min(r["mean_cycles"] for r in selected)
                incremental=[r["f_tile_size"] for r in selected if r["speedup"]==speed]
                overall=[r["f_tile_size"] for r in selected if r["mean_cycles"]==latency]
                result.append(dict(ratio=ratio,accuracy=accuracy,policy=policy,
                    baseline_best_tiles=json.dumps(base_tiles),incremental_best_tiles=json.dumps(incremental),
                    overall_best_tiles=json.dumps(overall),best_incremental_speedup=speed,
                    best_overall_cycles=latency,baseline_best_cycles=best_base,
                    best_overall_vs_best_stream=best_base/latency,
                    incremental_optimum_differs=not bool(set(base_tiles)&set(incremental)),
                    overall_optimum_differs=not bool(set(base_tiles)&set(overall))))
    return result


def plots(directory: Path,rows: list[dict[str,Any]]) -> None:
    """三组主图保留固定policy，整体延迟图包含OFF；不以最优包络冒充部署。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ratios=sorted({r["ratio"] for r in rows})
    for kind in ("A_streaming_baseline","B_prefetch_incremental","C_overall_latency"):
        fig,axes=plt.subplots(2,3,figsize=(14,8),squeeze=False)
        for ax,ratio in zip(axes.ravel(),ratios):
            policies=[(0.,"OFF")] if kind.startswith("A") else [(q,p) for q in (.9,1.) for p in ("LIGHT","GU32","FULL")]
            if kind.startswith("C"):policies=[(0.,"OFF"),*policies]
            for q,p in policies:
                points=sorted([r for r in rows if r["ratio"]==ratio and r["accuracy"]==q and r["policy"]==p],key=lambda r:r["f_tile_size"])
                label="Stream-only" if p=="OFF" else f"{p} / q={q:g}"
                ax.plot([r["f_tile_size"] for r in points],[r["speedup"] if kind.startswith("B") else r["mean_cycles"] for r in points],
                    marker="o",label=label,linestyle="-" if q!=.9 else "--")
            if kind.startswith("B"):ax.axhline(1,color="black",linestyle=":")
            ax.set_xscale("log",base=2);ax.set_xticks([8,16,32,64,128],[8,16,32,64,128])
            ax.set_xlabel("F tile width (H=128, F=256 fixed)")
            ax.set_ylabel("Paired B/D speedup" if kind.startswith("B") else "Mean five-layer latency (cycles)")
            ax.set_title(f"rho={ratio:g}");ax.grid(alpha=.2)
        axes[0][0].legend(fontsize=7);fig.suptitle(f"{kind}: synthetic, causal one-layer lookahead")
        fig.tight_layout()
        for ext in ("png","pdf"):fig.savefig(directory/f"{kind}.{ext}")
        plt.close(fig)


def main() -> None:
    """运行小矩阵，保存输入/metrics/最佳粒度/三图；拒绝覆盖历史输出。"""
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",type=Path,default=ROOT/"configs/experiments/prefetch_tile_size.yaml")
    parser.add_argument("--output",type=Path,default=ROOT/"results/prefetch_tile_size_20261008")
    parser.add_argument("--workers",type=int,default=8)
    args=parser.parse_args();config=yaml.safe_load(args.config.read_text())
    base=yaml.safe_load((ROOT/config["base_config"]).read_text());validate(config,base)
    if args.workers<1 or args.output.exists():raise ValueError("positive workers and new output directory required")
    trace_path=ROOT/config["historical_directory"]/"inputs.json"
    original=json.loads(trace_path.read_text());traces={str(s):original[str(s)] for s in config["seeds"]}
    base["samples"]=len(traces[str(config["seeds"][0])])
    args.output.mkdir(parents=True);(args.output/"inputs.json").write_text(json.dumps(traces,indent=2))
    sources=[args.config.resolve(),Path(__file__),ROOT/config["base_config"],trace_path,
        ROOT/"examples/prefetch_phase_sweep.py",*sorted((ROOT/"src/edge_moe_sim").rglob("*.py"))]
    hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    summary=dict(status="RUNNING",config=config,base_settings=base,source_sha256=hashes)
    raw=[]
    try:
        with mp.get_context("spawn").Pool(args.workers) as pool:
            for ratio in config["ratios"]:
                for tile in config["f_tile_sizes"]:
                    policies=[(0.,"OFF")]+[(q,p) for q in config["accuracies"] for p in ("LIGHT","GU32","FULL")]
                    raw.extend(pool.map(worker,[(base,config,tile,ratio,q,p,seed,r)
                        for q,p in policies for seed in config["seeds"] for r in traces[str(seed)]]))
                    summary["completed_simulations"]=len(raw)
                    (args.output/"summary.json").write_text(json.dumps(summary,indent=2))
                print(f"rho={ratio:g} completed; {len(raw)} simulations",flush=True)
        rows=aggregate(raw);best=optima(rows)
        write_csv(args.output/"per_sample.csv",raw);write_csv(args.output/"comparison.csv",rows)
        write_csv(args.output/"optima.csv",best);plots(args.output,rows)
        if any(hashlib.sha256((ROOT/n).read_bytes()).hexdigest()!=d for n,d in hashes.items()):raise RuntimeError("source changed")
        summary.update(status="COMPLETE",simulations=len(raw),configurations=len(rows),
            execution_rules=asdict(AdvancedExecutionConfig(64).execution_rules),
            differing_incremental_optima=sum(r["incremental_optimum_differs"] for r in best),
            differing_overall_optima=sum(r["overall_optimum_differs"] for r in best),
            definitions=dict(safe_slack="posthoc OFF demand-free windows after source router+20 and before target router; fraction fitting one GU DMA",
                setup_fraction="setup channel-cycles / occupied channel-cycles through resource drain",
                aie_utilization="allocated GEMM cycles including startup / inference cycles; one logical engine",
                requests="started matrix DMA descriptors; each spans four channels, not AXI beats",
                realistic="synthetic single-candidate 90% Top-1 accuracy, not measured predictor"))
    except Exception as error:summary.update(status="FAILED",error=str(error));raise
    finally:(args.output/"summary.json").write_text(json.dumps(summary,indent=2))
    print(json.dumps({k:summary[k] for k in ("status","simulations","differing_incremental_optima","differing_overall_optima")}),flush=True)


if __name__=="__main__":main()
