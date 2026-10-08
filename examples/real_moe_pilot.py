"""真实 MoE 尺寸的主模拟器 pilot；路由/预测是显式保存的合成激励。"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import sys
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from edge_moe_sim.advanced_schema import AdvancedExecutionConfig
from edge_moe_sim.batch_predictor import BatchPredictorOutput, PrecomputedBatchPredictor, TokenPrediction
from edge_moe_sim.batch_routing import BatchRoutingEvent
from edge_moe_sim.execution_rules import ExecutionRules
from edge_moe_sim.hardware_profile import HardwareProfile
from edge_moe_sim.real_model_profile import RealMoEProfile, load_real_moe_profiles
from edge_moe_sim.schema import CandidatePrefetch
from edge_moe_sim.unified_schema import UnifiedLayerSequence, UnifiedPredictionSpec
from edge_moe_sim.unified_sequence_simulator import UnifiedSequenceSimulator


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """写入非空、确定字段顺序的 CSV。"""
    if not rows:
        raise ValueError(f"empty result: {path}")
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def synthetic_inputs(profile: RealMoEProfile, layers: int, seed: int,
                     candidate_count: int, quality: str) -> tuple[UnifiedLayerSequence, dict[Any, Any], dict[str, Any]]:
    """生成一个 token 的固定路由；partial 的候选 recall 固定为 0.5，而非真实预测。"""
    if not 1 <= layers <= profile.num_moe_layers:
        raise ValueError("simulated layers must be within the actual model")
    if quality not in {"perfect", "partial"}:
        raise ValueError("unknown synthetic quality")
    model = profile.model
    rng = random.Random(seed)
    events = tuple(BatchRoutingEvent(1, 0, layer, (tuple(rng.sample(range(model.num_experts), model.top_k)),))
                   for layer in range(layers))
    outputs: dict[Any, Any] = {}
    rows: list[dict[str, Any]] = []
    for event in events[1:]:
        truth = event.routes[0]
        hits = model.top_k if quality == "perfect" else model.top_k // 2
        wrong = [expert for expert in range(model.num_experts) if expert not in truth]
        rng.shuffle(wrong)
        if candidate_count - hits > len(wrong):
            raise ValueError("candidate count cannot preserve requested recall")
        ids = truth[:hits] + tuple(wrong[:candidate_count - hits])
        scores = tuple((candidate_count - rank) / candidate_count for rank in range(candidate_count))
        output = BatchPredictorOutput(1, 0, event.layer_id, 0,
            (TokenPrediction(0, tuple(CandidatePrefetch(e, score) for e, score in zip(ids, scores))),))
        outputs[(event.layer_id - 1, event.layer_id)] = PrecomputedBatchPredictor({(1, 0, event.layer_id): output})
        rows.append({"source_layer_id": event.layer_id - 1, "target_layer_id": event.layer_id,
                     "candidate_ids": ids, "scores": scores, "static_recall": hits / model.top_k})
    trace = {"synthetic": True, "seed": seed, "quality": quality,
             "routes": [asdict(event) for event in events], "predictions": rows}
    return UnifiedLayerSequence(events), outputs, trace


def run_case(profile: RealMoEProfile, settings: dict[str, Any], base: dict[str, Any], *,
             capacity_bytes: int, ops_per_cycle: int, quality: str, policy: str,
             layers: int, diagnostics: bool = False) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """保持主模拟器算法不变；扩大 slot 尺寸及 burst 并明确记录 scale-study 假设。"""
    candidates = min(profile.model.num_experts, profile.model.top_k * settings["candidate_multiplier"])
    sequence, predictors, _ = synthetic_inputs(profile, layers, settings["seed"], candidates, quality)
    slot_bytes = (profile.model.input_projection_count * profile.model.hidden_size * min(settings["f_tile_size"], profile.model.intermediate_size)
                  * profile.model.weight_bits + 7) // 8
    banks = base["execution_rules"]["ocm_banks"]
    usable_bytes = capacity_bytes // (banks * slot_bytes) * banks * slot_bytes
    if usable_bytes < 4 * slot_bytes:
        raise ValueError("OCM cannot fit demand pair plus speculative pair")
    hardware = HardwareProfile(**base["hardware"], ocm_bytes=usable_bytes)
    hardware = replace(hardware, name=f"u200_scaled_ops{ops_per_cycle}",
        ops_per_cycle_per_aie=ops_per_cycle, burst_bytes=settings["burst_bytes"],
        hardware_tag="SCALED_STRUCTURAL_STUDY_NOT_MEASURED_OR_RTL_VALIDATED")
    rules = ExecutionRules(**{**base["execution_rules"], "ocm_slot_bytes": slot_bytes})
    execution = AdvancedExecutionConfig(settings["f_tile_size"], placement_policy="striped_tile",
        candidate_count=candidates, nonlinear_elements_per_cycle=settings["nonlinear_elements_per_cycle"],
        speculative_byte_budget=usable_bytes - 2 * slot_bytes, execution_rules=rules)
    predictions = () if policy == "no_prediction" else tuple(
        UnifiedPredictionSpec(event.layer_id - 1, event.layer_id, "source_router_done",
            settings["predictor_latency_cycles"], candidates, policy, settings["prefix_depth"],
            wait_for_previous_layer_weights=settings.get("wait_for_previous_layer_weights", False))
        for event in sequence.layers[1:])
    result = UnifiedSequenceSimulator(profile.model, hardware).simulate(sequence, execution,
        predictions=predictions, predictors=predictors, trace_enabled=False)
    if len(result.layer_results) != layers or result.peak_ocm_bytes > usable_bytes:
        raise AssertionError("incomplete sequence or OCM violation")
    run_id = f"{profile.model.name}_ocm{capacity_bytes}_ops{ops_per_cycle}_{quality}_{policy}"
    row = {"run_id": run_id, "model": profile.model.name, "simulated_layers": layers,
        "model_moe_layers": profile.num_moe_layers, "policy": policy, "quality": quality,
        "candidate_recall": 1.0 if quality == "perfect" else 0.5,
        "candidate_count": candidates, "ops_per_cycle": ops_per_cycle,
        "requested_ocm_bytes": capacity_bytes, "usable_ocm_bytes": usable_bytes,
        "slot_bytes": slot_bytes, "burst_bytes": hardware.burst_bytes,
        "total_cycles": result.inference_makespan_cycles,
        "simulated_moe_ms": result.inference_makespan_cycles * 1000 / hardware.clock_hz,
        "peak_ocm_bytes": result.peak_ocm_bytes,
        "aie_starvation_cycles": sum(layer.aie_starvation_cycles for layer in result.layer_results.values()),
        "speculative_bytes": result.memory_metrics["speculative_bytes"],
        "useful_speculative_bytes": result.memory_metrics["useful_speculative_bytes"],
        "wasted_speculative_bytes": result.memory_metrics["wasted_speculative_bytes"],
        "blocked_by_future_speculation_cycles": result.demand_blocked_by_future_speculation_cycles,
        "baseline_cycles": 0, "speedup_vs_no_prediction": 0.0}
    if diagnostics:
        tile_count = profile.geometry(settings["f_tile_size"])["tiles_per_expert"]
        completed: dict[tuple[int, int], set[tuple[str, int]]] = {}
        for request in result.speculative_request_records:
            end = request["completion_cycle"]
            target = int(request["target_layer"])
            if request["issue_cycle"] is not None and end is not None and (
                int(end) + rules.ocm_payload_visibility_cycles <= result.layer_results[target].router_resolve_cycle
            ):
                key = (target, int(request["expert_id"]))
                completed.setdefault(key, set()).add((str(request["matrix_type"]), int(request["tile_id"])))
        required = {(matrix, tile) for matrix in ("gate_up", "down") for tile in range(tile_count)}
        row.update({
            "planned_speculative_bytes": sum(p.planned_speculative_bytes for p in result.predictions),
            "speculative_payload_budget_per_target_bytes": usable_bytes - 2 * slot_bytes,
            "whole_expert_reserved_slot_bytes": tile_count * 2 * slot_bytes,
            "whole_expert_fits_ocm_with_demand_reserve": tile_count * 2 * slot_bytes <= usable_bytes - 2 * slot_bytes,
            "experts_all_spec_transfers_completed_by_router": sum(required <= keys for keys in completed.values()),
            "completed_speculative_bytes_before_router": sum(p.completed_speculative_bytes_before_router for p in result.predictions),
        })
    timing = [{"run_id": run_id, "layer_id": layer.layer_id,
               "layer_start_cycle": layer.layer_start_cycle, "router_resolve_cycle": layer.router_resolve_cycle,
               "layer_done_cycle": layer.layer_done_cycle, "aie_starvation_cycles": layer.aie_starvation_cycles}
              for layer in result.layer_results.values()]
    return row, timing


def plot_results(directory: Path, rows: list[dict[str, Any]]) -> None:
    """可选生成静态 PNG/PDF；缺失 matplotlib 时明确报错，不静默跳过。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    models = list(dict.fromkeys(row["model"] for row in rows))
    fig, axes = plt.subplots(1, len(models), figsize=(5 * len(models), 4), squeeze=False)
    for model, ax in zip(models, axes[0]):
        for policy in ("uniform_prefix", "ranked_prefix"):
            for quality in ("perfect", "partial"):
                selected = [r for r in rows if r["model"] == model and r["policy"] == policy
                            and r["quality"] == quality and r["requested_ocm_bytes"] == 8388608]
                if selected:
                    ax.plot([r["ops_per_cycle"] for r in selected],
                            [r["speedup_vs_no_prediction"] for r in selected], marker="o",
                            label=f"{policy}/{quality}")
        ax.axhline(1, color="gray", linestyle="--")
        ax.set_title(model)
        ax.set_xlabel("Configured operations/cycle")
        ax.set_ylabel("Speedup over no prediction")
        ax.legend(fontsize=7)
    fig.suptitle("Real dimensions / synthetic routes / 8 MiB requested OCM")
    fig.tight_layout()
    for extension in ("png", "pdf"):
        fig.savefig(directory / f"speedup.{extension}")
    plt.close(fig)


def main() -> int:
    """运行配置矩阵；保存每组输入、尺寸、CSV 和带限制声明的 summary。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/experiments/real_moe_pilot.yaml")
    parser.add_argument("--output", type=Path, default=ROOT / "results/real_moe_pilot")
    parser.add_argument("--models", nargs="+")
    parser.add_argument("--full-moe-depth", action="store_true")
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()
    settings = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    base = yaml.safe_load((ROOT / settings["alignment_base"]).read_text(encoding="utf-8"))
    profiles = load_real_moe_profiles(ROOT / settings["catalogue"], weight_bits=settings["weight_bits"],
                                      activation_bits=settings["activation_bits"])
    if args.models:
        if set(args.models) - {p.model.name for p in profiles}:
            raise ValueError("unknown model selection")
        profiles = tuple(p for p in profiles if p.model.name in args.models)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "summary.json").write_text(json.dumps({"status": "running"}) + "\n")
    rows: list[dict[str, Any]] = []
    timings: list[dict[str, Any]] = []
    geometry = [p.geometry(settings["f_tile_size"]) for p in profiles]
    write_csv(args.output / "model_geometry.csv", geometry)
    for profile in profiles:
        layers = profile.num_moe_layers if args.full_moe_depth else settings["simulated_layers"]
        candidates = min(profile.model.num_experts, profile.model.top_k * settings["candidate_multiplier"])
        for quality in settings["qualities"]:
            _, _, trace = synthetic_inputs(profile, layers, settings["seed"], candidates, quality)
            (args.output / f"{profile.model.name}_{quality}_input.json").write_text(json.dumps(trace, indent=2) + "\n")
        for capacity in settings["ocm_capacity_bytes"]:
            for rate in settings["ops_per_cycle"]:
                baseline, timing = run_case(profile, settings, base, capacity_bytes=capacity,
                    ops_per_cycle=rate, quality="perfect", policy="no_prediction", layers=layers)
                baseline["baseline_cycles"] = baseline["total_cycles"]
                baseline["speedup_vs_no_prediction"] = 1.0
                rows.append(baseline)
                timings.extend(timing)
                for quality in settings["qualities"]:
                    for policy in settings["policies"]:
                        row, timing = run_case(profile, settings, base, capacity_bytes=capacity,
                            ops_per_cycle=rate, quality=quality, policy=policy, layers=layers)
                        row["baseline_cycles"] = baseline["total_cycles"]
                        row["speedup_vs_no_prediction"] = baseline["total_cycles"] / row["total_cycles"]
                        rows.append(row)
                        timings.extend(timing)
                write_csv(args.output / "runs.csv", rows)
                print(f"completed {profile.model.name} OCM={capacity} ops/cycle={rate}", flush=True)
    write_csv(args.output / "per_layer.csv", timings)
    source_paths = [args.config, ROOT / settings["catalogue"], ROOT / settings["alignment_base"], Path(__file__)]
    source_paths.extend(sorted((ROOT / "src/edge_moe_sim").rglob("*.py")))
    summary = {"status": "complete", "simulator": "UnifiedSequenceSimulator", "simulation_count": len(rows),
        "settings": settings, "alignment_base": base, "model_geometry": geometry,
        "source_sha256": {str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p):
                           hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths},
        "limits": ["real architecture, synthetic routing and predictor; no model inference",
                   "INT8 is an assumed deployment, not accuracy validated",
                   "MoE-only sequence; attention/KV/norm/embedding/communication excluded",
                   "slot geometry and burst enlarged; not current RTL configuration or board measurement",
                   "single shared logical compute engine; aie field names represent PL compute on U200",
                   "DDR4 service calibration not applied; no claim of end-to-end token latency"]}
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    if args.plot:
        plot_results(args.output, rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
