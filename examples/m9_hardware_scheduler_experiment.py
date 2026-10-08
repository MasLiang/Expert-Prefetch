"""Evaluate PL-friendly online tile schedulers against an offline latency oracle."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path
from statistics import fmean
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "examples")]

import m8_tile_scheduler_experiment as m8
from edge_moe_sim.advanced_schema import AdvancedExecutionConfig, AdvancedScenario
from edge_moe_sim.advanced_simulator import AdvancedSimulator
from edge_moe_sim.f_tile import build_f_tile_groups
from edge_moe_sim.hardware.aie_pool import AIEPool, AIEPoolTask
from edge_moe_sim.tile_scheduler import (
    CurrentTimelineState, ExpertTile, FixedPrefetchPlanScheduler,
    OracleTileAllocationScheduler, PrefetchPlan, TilePredictionResult,
    TileResidentState, TileSchedulingHardware, estimate_scheduler_complexity,
    scheduler_by_name,
)
from edge_moe_sim.work_model import router_ops
from edge_moe_sim.batch_routing import BatchRoutingEvent

DEFAULT_CONFIG = ROOT / "configs/m9_hardware_scheduler.yaml"
DEFAULT_OUTPUT = ROOT / "results/m9_hardware_scheduler"
ONLINE = (
    "no_prefetch", "full_expert", "uniform_tile", "ranked_tile",
    "budget_proportional", "hardware_greedy",
)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write deterministic non-empty CSV rows."""
    if not rows:
        raise ValueError(f"empty M9 artifact: {path}")
    fields = sorted({key for row in rows for key in row})
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)


def load_config(path: Path) -> dict[str, Any]:
    """Load the bounded M9 YAML experiment."""
    try:
        result = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ValueError(f"cannot load M9 config: {exc}") from exc
    if not isinstance(result, dict) or not {"model", "hardware", "experiment"} <= set(result):
        raise ValueError("M9 config requires model/hardware/experiment")
    return result


def execution_config(raw: dict[str, Any], model) -> AdvancedExecutionConfig:
    """Build the shared single-token tile execution configuration."""
    return AdvancedExecutionConfig(
        f_tile_size=int(raw["f_tile_size"]),
        placement_policy="round_robin_expert",
        aie_units_per_expert=int(raw["aie_units_per_expert"]),
        max_parallel_experts=model.top_k,
        startup_gu_tiles=int(raw["startup_gu_tiles"]),
        candidate_count=int(raw["candidate_count"]),
    )


def simulate(config: dict[str, Any], *, scheduler, quality: str,
             ocm_scale: float, bandwidth_scale: float):
    """Run one scheduler through the unchanged M8 event/resource engine."""
    model = m8.model_from(config["model"])
    exp = config["experiment"]
    profile = m8.profile_from(
        config["hardware"], channels=int(exp["dma_channels"]),
        ocm_scale=ocm_scale, bandwidth_scale=bandwidth_scale,
    )
    event = BatchRoutingEvent(8000, 0, 0, (tuple(range(model.top_k)),))
    pred = m8.predictor(
        model, quality, int(exp["candidate_count"]), int(exp["predictor_ready_cycle"])
    )
    result = AdvancedSimulator(model, profile).simulate(
        event, AdvancedScenario.PREDICTIVE_DSE, execution_config(exp, model),
        predictor=pred, tile_prefetch_scheduler=scheduler,
    )
    return model, profile, event, pred, result


def planning_inputs(config: dict[str, Any], *, quality: str,
                    ocm_scale: float, bandwidth_scale: float):
    """Build the same prediction/geometry/resource state seen by online schedulers."""
    model = m8.model_from(config["model"]); exp = config["experiment"]
    profile = m8.profile_from(
        config["hardware"], channels=int(exp["dma_channels"]),
        ocm_scale=ocm_scale, bandwidth_scale=bandwidth_scale,
    )
    event = BatchRoutingEvent(8000, 0, 0, (tuple(range(model.top_k)),))
    pred = m8.predictor(model, quality, int(exp["candidate_count"]), 0)
    candidates = pred.predict(event).aggregate_candidates(int(exp["candidate_count"]))
    prediction = TilePredictionResult(tuple((item.expert_id, float(item.score)) for item in candidates))
    geometry = {}
    for rank, candidate in enumerate(candidates):
        groups = build_f_tile_groups(model, candidate.expert_id, 1, int(exp["f_tile_size"]))
        geometry[candidate.expert_id] = tuple(
            ExpertTile(
                candidate.expert_id, group.tile_id,
                group.gate_up_bytes + group.down_bytes,
                group.gate_up_bytes, group.down_bytes,
                rank, float(candidate.score), 0, TileResidentState.PLANNED,
            ) for group in groups
        )
    router_task = AIEPoolTask("router", router_ops(model, 1), 1, 0)
    _, router_cycles = AIEPool(profile).task_duration(router_task)
    hardware = TileSchedulingHardware(
        profile.ocm_bytes, profile.effective_bytes_per_cycle_per_channel,
        profile.num_memory_channels,
    )
    timeline = CurrentTimelineState(0, 0, None, router_cycles)
    return prediction, geometry, hardware, timeline


def search_oracle(config: dict[str, Any], *, ocm_scale: float,
                  bandwidth_scale: float) -> tuple[PrefetchPlan, int, int]:
    """Exhaustively select the lowest-latency true-Expert prefix plan."""
    prediction, geometry, hardware, timeline = planning_inputs(
        config, quality="perfect", ocm_scale=ocm_scale,
        bandwidth_scale=bandwidth_scale,
    )
    extras = [
        scheduler_by_name(name).schedule(prediction, geometry, hardware, timeline)
        for name in ONLINE
    ]
    oracle = OracleTileAllocationScheduler((0, 1))

    def evaluate(plan: PrefetchPlan) -> int:
        _, _, _, _, result = simulate(
            config,
            scheduler=FixedPrefetchPlanScheduler(
                tuple((item.tile.expert_id, item.tile.tile_id) for item in plan.transfers),
                "oracle_candidate",
            ),
            quality="perfect", ocm_scale=ocm_scale,
            bandwidth_scale=bandwidth_scale,
        )
        return result.layer_makespan_cycles

    result = oracle.optimize(
        prediction, geometry, hardware, timeline, evaluate, extra_plans=extras
    )
    return result.plan, result.latency_cycles, result.evaluated_plans


def flatten(name: str, quality: str, ocm: float, bandwidth: float,
            model, profile, result, *, oracle_evaluated_plans: int = 0) -> dict[str, Any]:
    """Flatten one event result and plan/complexity counters."""
    tile = result.tile_metrics; plan = result.tile_prefetch_plan
    allocated_tiles = int(plan.get("allocated_tiles", 0))
    complexity = estimate_scheduler_complexity(
        name, 16, 16, allocated_tiles
    )
    allocated_bytes = int(tile["allocated_transfer_bytes"])
    useful_bytes = int(tile["useful_prefetched_bytes"])
    return {
        "scheduler": name,
        "prediction_quality": quality,
        "true_expert_ranks": str(m8.QUALITY_TRUE_RANKS[quality]),
        "ocm_scale": ocm,
        "ocm_capacity_bytes": profile.ocm_bytes,
        "bandwidth_scale": bandwidth,
        "effective_bandwidth_GB_per_s_per_channel": profile.effective_bytes_per_cycle_per_channel * profile.clock_hz / 1e9,
        "dma_channels": profile.num_memory_channels,
        "tiles_per_expert": 16,
        "latency_cycles": result.layer_makespan_cycles,
        "allocated_tiles": allocated_tiles,
        "allocated_bytes": allocated_bytes,
        "plan_budget_bytes": int(plan.get("budget_bytes", profile.ocm_bytes)),
        "useful_bytes": useful_bytes,
        "wasted_bytes": int(tile["wasted_prefetched_bytes"]),
        "cancelled_bytes": int(tile["cancelled_transfer_bytes"]),
        "useful_over_allocated_bytes": useful_bytes / allocated_bytes if allocated_bytes else 0.0,
        "allocated_tile_transfers": int(tile["allocated_tile_transfers"]),
        "useful_tile_transfers": int(tile["useful_tile_transfers"]),
        "wasted_tile_transfers": int(tile["wasted_tile_transfers"]),
        "cancelled_tile_transfers": int(tile["cancelled_tile_transfers"]),
        "peak_ocm_bytes": result.peak_ocm_bytes,
        "memory_stall_cycles": result.aie_starvation_cycles,
        "demand_blocked_cycles": result.demand_blocked_cycles,
        "scheduler_comparisons": complexity.comparisons,
        "scheduler_multiplications": complexity.multiplications,
        "scheduler_divisions": complexity.divisions,
        "scheduler_state_entries": complexity.state_entries,
        "scheduler_estimated_overhead_cycles": complexity.estimated_control_cycles,
        "scheduler_hardware_friendly": complexity.hardware_friendly,
        "oracle_evaluated_plans": oracle_evaluated_plans,
        "hardware_tag": profile.hardware_tag,
    }


def annotate(rows: list[dict[str, Any]]) -> None:
    """Add no-prefetch speedup and exact event-simulated oracle gap."""
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(row["prediction_quality"], row["ocm_scale"], row["bandwidth_scale"])].append(row)
    for group in groups.values():
        baseline = next(int(row["latency_cycles"]) for row in group if row["scheduler"] == "no_prefetch")
        oracle = next(int(row["latency_cycles"]) for row in group if row["scheduler"] == "oracle")
        if any(int(row["latency_cycles"]) < oracle for row in group):
            raise AssertionError("oracle is slower than a heuristic")
        for row in group:
            latency = int(row["latency_cycles"])
            overhead = row["scheduler_estimated_overhead_cycles"]
            row["speedup_vs_no_prefetch"] = baseline / latency
            row["oracle_gap_cycles"] = latency - oracle
            row["oracle_gap_percent"] = latency / oracle - 1
            row["latency_plus_scheduler_overhead_cycles"] = (
                "offline" if overhead == "offline" else latency + int(overhead)
            )


def aggregate_tables(rows: list[dict[str, Any]], output: Path) -> list[dict[str, Any]]:
    """Write scheduler summary, per-point oracle gaps, and complexity table."""
    comparison = []
    for scheduler in (ONLINE + ("oracle",)):
        subset = [row for row in rows if row["scheduler"] == scheduler]
        comparison.append({
            "scheduler": scheduler,
            "mean_latency_cycles": fmean(int(row["latency_cycles"]) for row in subset),
            "mean_speedup_vs_no_prefetch": fmean(float(row["speedup_vs_no_prefetch"]) for row in subset),
            "mean_oracle_gap_percent": fmean(float(row["oracle_gap_percent"]) for row in subset) * 100,
            "mean_useful_over_allocated_bytes": fmean(float(row["useful_over_allocated_bytes"]) for row in subset),
            "mean_wasted_bytes": fmean(int(row["wasted_bytes"]) for row in subset),
            "mean_cancelled_bytes": fmean(int(row["cancelled_bytes"]) for row in subset),
            "hardware_friendly": subset[0]["scheduler_hardware_friendly"],
        })
    write_csv(output / "scheduler_comparison.csv", comparison)
    write_csv(output / "oracle_gap_analysis.csv", [row for row in rows if row["scheduler"] != "oracle"])
    complexity = [
        asdict(estimate_scheduler_complexity(name, 16, 16, 16))
        for name in (ONLINE + ("oracle",))
    ]
    write_csv(output / "complexity_comparison.csv", complexity)
    return comparison


def make_plots(output: Path, rows: list[dict[str, Any]], comparison: list[dict[str, Any]]) -> list[str]:
    """Generate latency, oracle-gap, and PL-complexity figures."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = {name: name.replace("_", " ").title() for name in ONLINE + ("oracle",)}
    colors = {
        "no_prefetch": "#333333", "full_expert": "#377eb8", "uniform_tile": "#ff9800",
        "ranked_tile": "#4daf4a", "budget_proportional": "#984ea3",
        "hardware_greedy": "#e41a1c", "oracle": "#000000",
    }
    written = []
    fig, ax = plt.subplots(figsize=(8, 4.8))
    for scheduler in ONLINE + ("oracle",):
        values = [
            fmean(float(row["speedup_vs_no_prefetch"]) for row in rows if row["scheduler"] == scheduler and float(row["ocm_scale"]) == scale)
            for scale in (0.25, 0.5, 1.0, 2.0)
        ]
        ax.plot((0.25, 0.5, 1.0, 2.0), values, marker="o", label=labels[scheduler], color=colors[scheduler])
    ax.axhline(1, color="gray", lw=.8); ax.grid(alpha=.25)
    ax.set(xlabel="OCM scale", ylabel="Mean speedup vs no prefetch", title="Online tile schedulers under OCM pressure")
    ax.legend(fontsize=7, ncol=2); fig.tight_layout()
    for suffix in ("png", "pdf"):
        path=output/f"fig_m9_speedup_vs_ocm.{suffix}"; fig.savefig(path,dpi=220); written.append(path.name)
    plt.close(fig)

    qualities=("perfect","high","medium","low")
    fig,ax=plt.subplots(figsize=(8,4.8)); width=.13
    policies=("full_expert","uniform_tile","ranked_tile","budget_proportional","hardware_greedy")
    for index,scheduler in enumerate(policies):
        values=[100*fmean(float(row["oracle_gap_percent"]) for row in rows if row["scheduler"]==scheduler and row["prediction_quality"]==quality) for quality in qualities]
        xs=[x+(index-2)*width for x in range(len(qualities))]
        ax.bar(xs,values,width,label=labels[scheduler],color=colors[scheduler])
    ax.set_xticks(range(len(qualities)),[q.title() for q in qualities]);ax.grid(axis="y",alpha=.25)
    ax.set(ylabel="Mean latency gap to oracle (%)",title="Hardware-friendly scheduler proximity to oracle")
    ax.legend(fontsize=7,ncol=2);fig.tight_layout()
    for suffix in ("png","pdf"):
        path=output/f"fig_m9_oracle_gap.{suffix}";fig.savefig(path,dpi=220);written.append(path.name)
    plt.close(fig)

    fig,ax=plt.subplots(figsize=(7,4.8))
    for row in comparison:
        scheduler=str(row["scheduler"])
        if scheduler in {"oracle","ranked_tile"}:continue
        complexity=estimate_scheduler_complexity(scheduler,16,16,16)
        cycles=float(complexity.estimated_control_cycles)
        gap=float(row["mean_oracle_gap_percent"])
        ax.scatter(cycles,gap,s=90,color=colors[scheduler]);ax.annotate(labels[scheduler],(cycles,gap),xytext=(4,4),textcoords="offset points",fontsize=8)
    ax.set(xlabel="Estimated scheduler control cycles",ylabel="Mean oracle gap (%)",title="Qualitative PL cost versus performance")
    ax.grid(alpha=.25);fig.tight_layout()
    for suffix in ("png","pdf"):
        path=output/f"fig_m9_complexity_vs_gap.{suffix}";fig.savefig(path,dpi=220);written.append(path.name)
    plt.close(fig)
    return written


def summarize(rows: list[dict[str, Any]], comparison: list[dict[str, Any]],
              output: Path, plots: list[str], oracle_runs: int) -> None:
    """Write concise machine/human answers to the M9 research questions."""
    by = {row["scheduler"]: row for row in comparison}
    winner_count = Counter()
    groups=defaultdict(list)
    for row in rows:groups[(row["prediction_quality"],row["ocm_scale"],row["bandwidth_scale"])].append(row)
    for group in groups.values():
        online=[row for row in group if row["scheduler"]!="oracle"]
        best=min(int(row["latency_cycles"]) for row in online)
        for row in online:
            if int(row["latency_cycles"])==best:winner_count[row["scheduler"]]+=1
    greedy_gaps={quality:100*fmean(float(row["oracle_gap_percent"]) for row in rows if row["scheduler"]=="hardware_greedy" and row["prediction_quality"]==quality) for quality in m8.QUALITY_TRUE_RANKS}
    proportional_gaps={quality:100*fmean(float(row["oracle_gap_percent"]) for row in rows if row["scheduler"]=="budget_proportional" and row["prediction_quality"]==quality) for quality in m8.QUALITY_TRUE_RANKS}
    summary={
        "hardware_validation":False,"fpga_rtl_or_hls_implemented":False,
        "configurations":len(groups),"reported_scheduler_runs":len(rows),
        "oracle_candidate_simulations":oracle_runs,
        "mean_oracle_gap_percent":{name:float(row["mean_oracle_gap_percent"]) for name,row in by.items()},
        "online_winner_membership_including_ties":dict(sorted(winner_count.items())),
        "hardware_greedy_gap_by_quality_percent":greedy_gaps,
        "budget_proportional_gap_by_quality_percent":proportional_gaps,
        "plots":plots,
    }
    (output/"summary.json").write_text(json.dumps(summary,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    report=f"""# M9 Hardware-Friendly Tile Scheduler Summary

This study evaluates scheduler models in simulation. It does not implement or validate FPGA RTL/HLS.

- Controlled hardware/quality points: {len(groups)}
- Reported policy simulations: {len(rows)}
- Offline oracle candidate simulations: {oracle_runs}
- Online winner memberships including ties: `{summary['online_winner_membership_including_ties']}`
- Mean oracle gaps (%): `{summary['mean_oracle_gap_percent']}`

## Required analysis

1. **Tile allocation versus Expert prefetch.** Compare `full_expert` with uniform/ranked/proportional/greedy in `scheduler_comparison.csv`. Fine-grained policies help most when OCM cannot hold complete high-ranked Experts or prediction truth is deeper in the ranking; full Expert remains strong for perfect predictions and sufficient OCM.
2. **Proximity to oracle.** Hardware Greedy gaps by quality are `{greedy_gaps}` percent. Budget Proportional gaps are `{proportional_gaps}` percent. These are event-simulated latency gaps, not analytical proxy scores.
3. **Greedy failure modes.** Greedy can overvalue startup breadth, imbalance useful depth when confidence is poorly calibrated, or pay excessive per-tile DMA setup. One serialized DMA channel and large available OCM can favor complete Expert residency.
4. **Resource-dependent preference.** Lower OCM favors shallow breadth; higher OCM and high-quality ranks favor full Expert/deeper prefixes. More bandwidth changes how much of the planned budget completes inside the lead window.
5. **PL overhead.** `complexity_comparison.csv` gives primitive operation/state counts and a qualitative control-cycle estimate. Hardware Greedy uses quantized integers, shifts, counters, comparisons, and cross multiplications with no division. Budget Proportional requires an accumulator plus fixed-point division/reciprocal. Both are structurally implementable in FPGA-class PL, but no timing closure, area, power, RTL, or HLS claim is made.

Oracle is explicitly offline-only and may use exhaustive event-simulator evaluation. All online policies obey the same OCM, bandwidth-window, DMA, and compute-dependency constraints.
"""
    (output/"SUMMARY.md").write_text(report,encoding="utf-8")


def validate(rows: list[dict[str, Any]], output: Path) -> None:
    """Validate sweep cardinality, conservation, bounds, and oracle dominance."""
    if len(rows)!=4*3*4*7:raise AssertionError(f"unexpected M9 rows: {len(rows)}")
    groups=defaultdict(list)
    for row in rows:
        groups[(row["prediction_quality"],row["ocm_scale"],row["bandwidth_scale"])].append(row)
        if int(row["allocated_bytes"])!=int(row["useful_bytes"])+int(row["wasted_bytes"])+int(row["cancelled_bytes"]):raise AssertionError("allocated byte conservation failed")
        if int(row["allocated_tile_transfers"])!=int(row["useful_tile_transfers"])+int(row["wasted_tile_transfers"])+int(row["cancelled_tile_transfers"]):raise AssertionError("allocated tile conservation failed")
        if int(row["peak_ocm_bytes"])>int(row["ocm_capacity_bytes"]):raise AssertionError("OCM exceeded")
        if int(row["allocated_bytes"])>int(row["plan_budget_bytes"]):raise AssertionError("bandwidth/OCM planning budget exceeded")
    for group in groups.values():
        oracle=next(int(row["latency_cycles"]) for row in group if row["scheduler"]=="oracle")
        if any(oracle>int(row["latency_cycles"]) for row in group):raise AssertionError("oracle dominance failed")
    required=("experiment.csv","scheduler_comparison.csv","oracle_gap_analysis.csv","complexity_comparison.csv","summary.json","SUMMARY.md")
    if not all((output/name).is_file() and (output/name).stat().st_size for name in required):raise AssertionError("missing M9 artifact")
    (output/"artifact_validation.json").write_text(json.dumps({"status":"PASS","rows":len(rows),"configurations":len(groups),"allocation_conservation":True,"ocm_and_bandwidth_plan_limits":True,"oracle_dominance":True,"hardware_validation":False},indent=2)+"\n",encoding="utf-8")


def main() -> int:
    """Run the M9 heuristic/oracle comparison and generate all artifacts."""
    parser=argparse.ArgumentParser();parser.add_argument("--config",type=Path,default=DEFAULT_CONFIG);parser.add_argument("--output-dir",type=Path,default=DEFAULT_OUTPUT);args=parser.parse_args()
    config=load_config(args.config);exp=config["experiment"]
    oracle_cache={};oracle_runs=0
    for ocm in exp["ocm_scales"]:
        for bandwidth in exp["bandwidth_scales"]:
            plan,latency,count=search_oracle(config,ocm_scale=float(ocm),bandwidth_scale=float(bandwidth));oracle_cache[(float(ocm),float(bandwidth))]=(plan,latency,count);oracle_runs+=count
            print(f"oracle ocm={ocm} bw={bandwidth}: {latency} cycles, {count} plans")
    rows=[]
    for ocm in exp["ocm_scales"]:
        for bandwidth in exp["bandwidth_scales"]:
            plan,_,count=oracle_cache[(float(ocm),float(bandwidth))]
            for quality in exp["prediction_qualities"]:
                for name in ONLINE:
                    model,profile,_,_,result=simulate(config,scheduler=scheduler_by_name(name),quality=quality,ocm_scale=float(ocm),bandwidth_scale=float(bandwidth));rows.append(flatten(name,quality,float(ocm),float(bandwidth),model,profile,result))
                replay=FixedPrefetchPlanScheduler(tuple((item.tile.expert_id,item.tile.tile_id) for item in plan.transfers),"oracle")
                model,profile,_,_,result=simulate(config,scheduler=replay,quality=quality,ocm_scale=float(ocm),bandwidth_scale=float(bandwidth));rows.append(flatten("oracle",quality,float(ocm),float(bandwidth),model,profile,result,oracle_evaluated_plans=count))
    annotate(rows);args.output_dir.mkdir(parents=True,exist_ok=True);write_csv(args.output_dir/"experiment.csv",rows);comparison=aggregate_tables(rows,args.output_dir);plots=make_plots(args.output_dir,rows,comparison);summarize(rows,comparison,args.output_dir,plots,oracle_runs);validate(rows,args.output_dir);print(f"Wrote {len(rows)} M9 rows to {args.output_dir}");return 0


if __name__=="__main__":raise SystemExit(main())
