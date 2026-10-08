"""导出收益谷值两侧的逐请求证据；不修改模拟器或DMA语义。"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
from typing import Any

import yaml

from prefetch_phase_sweep import (ROOT, AdvancedExecutionConfig, BatchPredictorOutput, BatchRoutingEvent,
    CandidatePrefetch, HardwareProfile, ModelConfig, PrecomputedBatchPredictor, TokenPrediction,
    UnifiedLayerSequence, UnifiedPredictionSpec, UnifiedSequenceSimulator, candidates, make_inputs, simulate_job)


def diagnose(settings: dict[str, Any], record: dict[str, Any], point: dict[str, Any]) -> dict[str, Any]:
    """重建同一配置开启trace，并检查makespan与扫描执行函数完全一致。"""
    h = settings["hidden_size"]
    f = settings["f_tile_size"] * point["tile_groups"]
    capacity = int(3 * h * f * point["ocm_expert_equivalents"])
    model = ModelConfig(name="synthetic_phase", hidden_size=h, intermediate_size=f,
        num_experts=settings["num_experts"], top_k=1, weight_bits=8, activation_bits=8,
        moe_layer_indices=tuple(range(settings["layers"])))
    hardware = HardwareProfile(name="synthetic_PE_not_board", clock_hz=settings["clock_hz"],
        num_memory_channels=settings["channels"], bandwidth_bytes_per_sec_per_channel=
            point["bytes_per_cycle_per_channel"] * settings["clock_hz"], bandwidth_efficiency=1,
        dma_setup_cycles=settings["dma_setup_cycles"], max_inflight_per_channel=1, burst_bytes=16384,
        supports_channel_striping=True, total_aie_units=point["pe_count"],
        ops_per_cycle_per_aie=settings["ops_per_cycle_per_pe"], aie_efficiency=1,
        aie_task_startup_cycles=settings["compute_startup_cycles"], ocm_bytes=capacity)
    sequence = UnifiedLayerSequence(tuple(BatchRoutingEvent(record["sample"], 0, i, ((e,),))
        for i, e in enumerate(record["truth"])))
    specs, predictors = [], {}
    if point["policy"] != "no_prediction":
        for target in record["targets"]:
            layer = target["target"]
            ids = candidates(record, target, point["quality"], settings["samples"] * (settings["layers"] - 1))
            output = BatchPredictorOutput(record["sample"], 0, layer, 0,
                (TokenPrediction(0, tuple(CandidatePrefetch(e, 1/(i+1)) for i,e in enumerate(ids))),))
            predictors[(layer-1,layer)] = PrecomputedBatchPredictor({(record["sample"],0,layer):output})
            specs.append(UnifiedPredictionSpec(layer-1,layer,"source_router_done",
                settings["predictor_latency_cycles"],settings["candidate_count"],"uniform_prefix",
                point["prefix_depth"],wait_for_previous_layer_weights=False))
    execution = AdvancedExecutionConfig(settings["f_tile_size"], placement_policy="striped_tile",
        aie_units_per_expert=point["pe_count"], router_aie_units=point["pe_count"],
        max_parallel_experts=1,candidate_count=settings["candidate_count"],speculative_byte_budget=capacity,
        nonlinear_elements_per_cycle=settings["nonlinear_elements_per_cycle"])
    result = UnifiedSequenceSimulator(model,hardware).simulate(sequence,execution,predictions=tuple(specs),
        predictors=predictors,trace_enabled=True)
    reference = simulate_job((settings,point,record))
    if reference["total_cycles"] != result.inference_makespan_cycles:
        raise AssertionError("trace-enabled execution changed makespan")
    return {"configuration":point,"metrics":reference,"layers":[asdict(r) for r in result.layer_results.values()],
        "speculative_requests":result.speculative_request_records,"blocking_records":result.blocking_records,
        "timeline":result.trace}


def main() -> None:
    """保存相同比较点的可复现JSON请求记录和timeline。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",type=Path,default=ROOT/"configs/experiments/prefetch_dense.yaml")
    parser.add_argument("--output",type=Path,default=ROOT/"results/prefetch_dense/diagnostics")
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    settings = yaml.safe_load((ROOT/config["base_config"]).read_text())
    settings.update(seed=config["seeds"][0],samples=config["samples_per_seed"])
    record = make_inputs(settings)[0]
    args.output.mkdir(parents=True,exist_ok=True)
    for ratio in [.9,1.]:
        for policy in ["no_prediction","uniform_prefix"]:
            point = dict(tile_groups=4,pe_count=1,bytes_per_cycle_per_channel=
                ratio*settings["ops_per_cycle_per_pe"]/(2*settings["channels"]),
                ocm_expert_equivalents=4,policy=policy,
                quality=0 if policy == "no_prediction" else config["fixed_quality"],
                prefix_depth=0 if policy == "no_prediction" else 1)
            report = {"seed":settings["seed"],"input":record,"settings":settings,"ratio":ratio,
                **diagnose(settings,record,point)}
            (args.output/f"ratio_{ratio:g}_{policy}.json").write_text(json.dumps(report,indent=2))


if __name__ == "__main__":
    main()
