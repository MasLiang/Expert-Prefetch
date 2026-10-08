"""测量stream-only内存占用与完美预测收益，并生成固定策略的准确率相图。"""
from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
from pathlib import Path
from statistics import mean, stdev
from typing import Any

import yaml

from prefetch_phase_sweep import (ROOT, AdvancedExecutionConfig, BatchPredictorOutput,
    BatchRoutingEvent, CandidatePrefetch, HardwareProfile, ModelConfig,
    PrecomputedBatchPredictor, TokenPrediction, UnifiedLayerSequence,
    UnifiedPredictionSpec, UnifiedSequenceSimulator, write_csv)


def measure_utilization(busy_channel_cycles: int, transferred_bytes: int,
                        channels: int, bandwidth_bytes_per_cycle: float,
                        makespan_cycles: int) -> tuple[float, float]:
    """返回含setup的通道占用率、有效payload带宽利用率；分母均用推理makespan。"""
    if min(channels, bandwidth_bytes_per_cycle, makespan_cycles) <= 0:
        raise ValueError("utilization denominators must be positive")
    occupied = busy_channel_cycles / (channels * makespan_cycles)
    payload = transferred_bytes / (channels * bandwidth_bytes_per_cycle * makespan_cycles)
    if not 0 <= payload <= occupied + 1e-9 or not 0 <= occupied <= 1 + 1e-9:
        raise AssertionError("memory utilization outside capacity")
    return occupied, payload


def run_sample(job: tuple[dict[str, Any], dict[str, Any], float, float, str, int, dict[str, Any]]) -> dict[str, Any]:
    """单候选Top-1；accuracy=1是因果perfect prediction，不是零内存等待ideal。"""
    base, config, ratio, accuracy, policy, seed, record = job
    if policy not in {"OFF", "LIGHT", "PREFETCH"}:
        raise ValueError("unknown policy")
    h, f = base["hidden_size"], base["f_tile_size"] * config["tile_groups"]
    bandwidth = ratio * base["ops_per_cycle_per_pe"] / (2 * base["channels"])
    capacity = int(3 * h * f * config["ocm_expert_equivalents"])
    model = ModelConfig(name="synthetic_memory_headroom", hidden_size=h, intermediate_size=f,
        num_experts=base["num_experts"], top_k=1, weight_bits=8, activation_bits=8,
        moe_layer_indices=tuple(range(base["layers"])))
    hardware = HardwareProfile(name="synthetic_not_board", clock_hz=base["clock_hz"],
        num_memory_channels=base["channels"], bandwidth_bytes_per_sec_per_channel=bandwidth * base["clock_hz"],
        bandwidth_efficiency=1, dma_setup_cycles=base["dma_setup_cycles"], max_inflight_per_channel=1,
        burst_bytes=16384, supports_channel_striping=True, total_aie_units=1,
        ops_per_cycle_per_aie=base["ops_per_cycle_per_pe"], aie_efficiency=1,
        aie_task_startup_cycles=base["compute_startup_cycles"], ocm_bytes=capacity)
    sequence = UnifiedLayerSequence(tuple(BatchRoutingEvent(record["sample"], 0, layer, ((e,),))
        for layer, e in enumerate(record["truth"])))
    specs, predictors = [], {}
    hits = 0
    decisions = base["samples"] * (base["layers"] - 1)
    if policy != "OFF":
        for target in record["targets"]:
            layer = target["target"]
            hit = target["quality_order"] < int(accuracy * decisions + .5)
            hits += hit
            expert = record["truth"][layer] if hit else target["wrong_ids"][0]
            output = BatchPredictorOutput(record["sample"], 0, layer, 0,
                (TokenPrediction(0, (CandidatePrefetch(expert, 1.),)),))
            predictors[(layer - 1, layer)] = PrecomputedBatchPredictor({(record["sample"], 0, layer): output})
            specs.append(UnifiedPredictionSpec(layer - 1, layer, "source_router_done",
                base["predictor_latency_cycles"], 1,
                "uniform_prefix" if policy == "LIGHT" else "full_expert_prefetch",
                1 if policy == "LIGHT" else config["tile_groups"], wait_for_previous_layer_weights=False))
    execution = AdvancedExecutionConfig(base["f_tile_size"], placement_policy="striped_tile",
        max_parallel_experts=1, candidate_count=1, speculative_byte_budget=capacity,
        nonlinear_elements_per_cycle=base["nonlinear_elements_per_cycle"])
    result = UnifiedSequenceSimulator(model, hardware).simulate(sequence, execution,
        predictions=tuple(specs), predictors=predictors)
    metrics = result.memory_metrics
    # Baseline has no tail DMA; otherwise busy cycles could exceed the inference window.
    if policy == "OFF" and result.final_resource_idle_cycle != result.inference_makespan_cycles:
        raise AssertionError("baseline has resource tail; utilization requires clipped intervals")
    if result.peak_ocm_bytes > capacity:
        raise AssertionError("OCM overflow")
    if metrics["speculative_bytes"] != metrics["useful_speculative_bytes"] + metrics["wasted_speculative_bytes"]:
        raise AssertionError("speculative conservation failed")
    if accuracy == 1 and metrics["wasted_speculative_bytes"]:
        raise AssertionError("perfect predictor wasted bytes")
    occupied, payload = measure_utilization(int(metrics["memory_busy_cycles"]), int(metrics["transferred_bytes"]),
        base["channels"], bandwidth, result.final_resource_idle_cycle)
    return dict(ratio=ratio, accuracy=accuracy, policy=policy, seed=seed, sample=record["sample"],
        total_cycles=result.inference_makespan_cycles, drain_cycles=result.final_resource_idle_cycle,
        occupied_utilization=occupied, payload_utilization=payload,
        memory_busy_channel_cycles=metrics["memory_busy_cycles"], bandwidth_bytes_per_cycle_per_channel=bandwidth,
        transferred_bytes=metrics["transferred_bytes"], speculative_bytes=metrics["speculative_bytes"],
        useful_speculative_bytes=metrics["useful_speculative_bytes"], wasted_speculative_bytes=metrics["wasted_speculative_bytes"],
        first_start_sum_cycles=sum(r.first_true_expert_start-r.router_resolve_cycle for r in result.layer_results.values()),
        future_blocking_cycles=result.critical_path_future_speculation_blocking_cycles,
        evicted_future_bytes=result.evicted_future_speculative_bytes, peak_ocm_bytes=result.peak_ocm_bytes,
        hit_decisions=hits, prediction_decisions=0 if policy == "OFF" else base["layers"]-1)


def aggregate(raw: list[dict[str, Any]], channels: int) -> list[dict[str, Any]]:
    """用匹配baseline生成pooled利用率和速度比，另存seed范围。"""
    baseline = {}
    groups: dict[tuple[float, float, str], list[dict[str, Any]]] = {}
    for row in raw:
        if row["policy"] == "OFF":
            key = row["ratio"], row["seed"], row["sample"]
            if key in baseline:
                raise ValueError("duplicate baseline")
            baseline[key] = row
        groups.setdefault((row["ratio"], row["accuracy"], row["policy"]), []).append(row)
    result = []
    for (ratio, accuracy, policy), values in sorted(groups.items()):
        expected = {(r["seed"], r["sample"]) for r in baseline.values() if r["ratio"] == ratio}
        if {(r["seed"],r["sample"]) for r in values} != expected or len(values) != len(expected):
            raise ValueError("missing or duplicate paired sample")
        bases = [baseline[ratio, r["seed"], r["sample"]] for r in values]
        baseline_cycles, cycles = sum(r["total_cycles"] for r in bases), sum(r["total_cycles"] for r in values)
        occupied, payload = measure_utilization(sum(r["memory_busy_channel_cycles"] for r in bases),
            sum(r["transferred_bytes"] for r in bases), channels,
            values[0]["bandwidth_bytes_per_cycle_per_channel"], baseline_cycles)
        seed_ratios = [sum(r["total_cycles"] for r in bases if r["seed"] == seed) /
                       sum(r["total_cycles"] for r in values if r["seed"] == seed)
                       for seed in sorted({r["seed"] for r in values})]
        result.append(dict(ratio=ratio, accuracy=accuracy, policy=policy, samples=len(values),
            baseline_memory_utilization=occupied, baseline_payload_utilization=payload,
            baseline_total_cycles=baseline_cycles, total_cycles=cycles, speedup=baseline_cycles/cycles,
            speedup_mean=mean(seed_ratios), speedup_std=stdev(seed_ratios) if len(seed_ratios)>1 else 0,
            speedup_min=min(seed_ratios), speedup_max=max(seed_ratios),
            actual_accuracy=sum(r["hit_decisions"] for r in values)/max(1,sum(r["prediction_decisions"] for r in values)),
            net_saved_cycles=baseline_cycles-cycles,
            extra_transfer_bytes=sum(r["transferred_bytes"] for r in values)-sum(r["transferred_bytes"] for r in bases),
            speculative_bytes=sum(r["speculative_bytes"] for r in values),
            wasted_speculative_bytes=sum(r["wasted_speculative_bytes"] for r in values),
            first_start_saved_cycles=sum(r["first_start_sum_cycles"] for r in bases)-sum(r["first_start_sum_cycles"] for r in values),
            future_blocking_cycles=sum(r["future_blocking_cycles"] for r in values)))
    return result


def recommend(points: list[dict[str, Any]], minimum_speedup: float, tie_fraction: float) -> str:
    """仅当所有seed超过阈值才开预取；近似等延迟优先轻量策略。事后分类，非部署算法。"""
    eligible = [r for r in points if r["policy"] != "OFF" and r["speedup_min"] >= minimum_speedup]
    if not eligible:
        return "OFF"
    fastest = min(r["total_cycles"] for r in eligible)
    near = [r for r in eligible if r["total_cycles"] <= fastest * (1 + tie_fraction)]
    return min(near, key=lambda r: (r["speculative_bytes"], r["policy"] != "LIGHT"))["policy"]


def plot(directory: Path, rows: list[dict[str, Any]], phase: list[dict[str, Any]]) -> None:
    """真实利用率坐标绘图；相图用散点保留不均匀采样，不插值成连续边界。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter
    fig, ax = plt.subplots(figsize=(8, 5))
    for policy in ("LIGHT", "PREFETCH"):
        selected = sorted([r for r in rows if r["accuracy"] == 1 and r["policy"] == policy],
                          key=lambda r: r["baseline_memory_utilization"])
        x = [r["baseline_memory_utilization"] for r in selected]
        ax.plot(x, [r["speedup"] for r in selected], marker="o", label=f"Perfect predictor / {policy}")
        ax.fill_between(x, [r["speedup_min"] for r in selected], [r["speedup_max"] for r in selected], alpha=.15)
        for r in selected:
            if policy == "PREFETCH":
                ax.annotate(f"rho={r['ratio']:g}", (r["baseline_memory_utilization"],r["speedup"]),fontsize=6,xytext=(3,3),textcoords="offset points")
    ax.axhline(1, color="black", linestyle="--")
    ax.set_xlabel("Stream-only mean channel occupancy (includes setup)")
    ax.xaxis.set_major_formatter(PercentFormatter(1))
    ax.set_ylabel("Speedup vs paired stream-only")
    ax.set_title("100% correct causal prediction, one-layer lookahead\nFixed policies; seed range is not CI")
    ax.legend(); ax.grid(alpha=.2); fig.tight_layout()
    for ext in ("png","pdf"):
        fig.savefig(directory/f"perfect_prefetch_vs_utilization.{ext}")
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = {"OFF":"#777777", "LIGHT":"#ffa500", "PREFETCH":"#2ca02c"}
    for label,color in colors.items():
        selected = [r for r in phase if r["recommendation"] == label]
        ax.scatter([r["baseline_memory_utilization"] for r in selected], [r["accuracy"] for r in selected],
                   c=color, marker="s", s=45, label=label, edgecolors="white",linewidths=.4)
    ax.set_xlabel("Stream-only mean channel occupancy (includes setup)")
    ax.xaxis.set_major_formatter(PercentFormatter(1)); ax.set_ylabel("Top-1 prediction accuracy")
    ax.set_yticks(sorted({r["accuracy"] for r in phase})); ax.set_xlim(0,1.02)
    ax.set_title("Sampled policy phase diagram, not interpolated regions\nAll seeds >=1% gain; within 0.5% latency choose lower speculation")
    ax.legend(); ax.grid(alpha=.2); fig.tight_layout()
    for ext in ("png","pdf"):
        fig.savefig(directory/f"accuracy_utilization_phase.{ext}")
    plt.close(fig)


def main() -> None:
    """运行小矩阵、保存配对原始数据/图和可审计输入；拒绝覆盖结果。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",type=Path,default=ROOT/"configs/experiments/prefetch_memory_headroom.yaml")
    parser.add_argument("--output",type=Path,default=ROOT/"results/prefetch_memory_headroom_20261007")
    parser.add_argument("--workers",type=int,default=8)
    args=parser.parse_args()
    config=yaml.safe_load(args.config.read_text()); base=yaml.safe_load((ROOT/config["base_config"]).read_text())
    if args.workers < 1 or args.output.exists():
        raise ValueError("workers must be positive and output must not exist")
    if len(set(config["seeds"])) < 2 or len(set(config["seeds"])) != len(config["seeds"]):
        raise ValueError("distinct multiple seeds required")
    if 1 not in config["accuracies"] or any(not 0<=q<=1 for q in config["accuracies"]):
        raise ValueError("accuracies must be in [0,1] and include 1")
    if any(r<=0 for r in config["ratios"]) or config["minimum_robust_speedup"]<=1 or config["cost_tie_latency_fraction"]<0:
        raise ValueError("invalid ratios or phase thresholds")
    trace_path=ROOT/config["historical_directory"]/"inputs.json"
    original=json.loads(trace_path.read_text()); traces={str(s):original[str(s)] for s in config["seeds"]}
    base["samples"]=len(traces[str(config["seeds"][0])])
    if any(len(t)!=base["samples"] for t in traces.values()):
        raise ValueError("inconsistent saved trace size")
    args.output.mkdir(parents=True)
    (args.output/"inputs.json").write_text(json.dumps(traces,indent=2))
    sources=[args.config.resolve(),Path(__file__),ROOT/config["base_config"],trace_path,
             ROOT/"examples/prefetch_phase_sweep.py",*sorted((ROOT/"src/edge_moe_sim").rglob("*.py"))]
    hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    summary=dict(status="RUNNING",config=config,base_settings=base,source_sha256=hashes)
    (args.output/"summary.json").write_text(json.dumps(summary,indent=2))
    raw=[]
    try:
        with mp.get_context("spawn").Pool(args.workers) as pool:
            for ratio in config["ratios"]:
                policies=[(0.,"OFF")]+[(q,p) for q in config["accuracies"] for p in ("LIGHT","PREFETCH")]
                raw.extend(pool.map(run_sample,[(base,config,ratio,q,p,seed,record)
                    for q,p in policies for seed in config["seeds"] for record in traces[str(seed)]]))
                print(f"rho={ratio:g} complete; {len(raw)} simulations",flush=True)
                summary["completed_simulations"]=len(raw)
                (args.output/"summary.json").write_text(json.dumps(summary,indent=2))
        rows=aggregate(raw,base["channels"])
        phase=[]
        for ratio in config["ratios"]:
            for accuracy in config["accuracies"]:
                selected=[r for r in rows if r["ratio"]==ratio and r["accuracy"]==accuracy and r["policy"]!="OFF"]
                phase.append(dict(ratio=ratio,accuracy=accuracy,baseline_memory_utilization=selected[0]["baseline_memory_utilization"],
                    recommendation=recommend(selected,config["minimum_robust_speedup"],config["cost_tie_latency_fraction"]),
                    light_speedup=next(r["speedup"] for r in selected if r["policy"]=="LIGHT"),
                    full_speedup=next(r["speedup"] for r in selected if r["policy"]=="PREFETCH")))
        write_csv(args.output/"per_sample.csv",raw); write_csv(args.output/"comparison.csv",rows)
        write_csv(args.output/"phase.csv",phase)
        write_csv(args.output/"perfect_prefetch.csv",[r for r in rows if r["accuracy"]==1])
        plot(args.output,rows,phase)
        if any(hashlib.sha256((ROOT/n).read_bytes()).hexdigest()!=d for n,d in hashes.items()):
            raise RuntimeError("source changed during run")
        summary.update(status="COMPLETE",simulations=len(raw),configurations=len(rows),phase_points=len(phase),
            utilization_definition="sum baseline occupied channel cycles / (channels * inference makespan); setup included",
            perfect_definition="single correct candidate at previous router done +20 cycles; resource constrained, not ideal",
            phase_definition="hindsight choice among OFF/GU1/fullExpert; all seed speedup >=1.01; 0.5% latency cost tie",
            limits=["synthetic Top-1, one layer lookahead, fixed F=256 and OCM=384KiB",
                    "utilization is whole-sequence average, not location of causally usable idle windows",
                    "nonpreemptive matrix fragment DMA; no burst arbitration or board calibration",
                    "one candidate removes breadth/confidence effects; not the previous two-candidate experiment",
                    "sampled cells, not universal or interpolated continuous boundaries"])
    except Exception as error:
        summary.update(status="FAILED",error=str(error)); raise
    finally:
        (args.output/"summary.json").write_text(json.dumps(summary,indent=2))
    print(json.dumps({n:summary[n] for n in ("status","simulations","phase_points")}),flush=True)


if __name__=="__main__":
    main()
