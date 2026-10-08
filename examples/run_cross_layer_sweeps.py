"""Run focused one-dimensional Milestone 5 synthetic sensitivity sweeps."""

from __future__ import annotations

import csv
import json
import statistics
import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from edge_moe_sim.config import (  # noqa: E402
    load_hardware_config, load_inter_layer_config, load_model_config,
    load_simulation_config, load_synthetic_predictor_config, load_tile_config,
)
from edge_moe_sim.prefetch_policy import FixedDepthPolicy  # noqa: E402
from edge_moe_sim.sequence_predictor import SyntheticCrossLayerPredictor  # noqa: E402
from edge_moe_sim.sequence_simulator import SequenceScenario, SequenceSimulator  # noqa: E402
from edge_moe_sim.trace import load_layer_sequences_jsonl  # noqa: E402


def _percentile(values: list[int], fraction: float) -> float:
    ordered = sorted(values)
    index = (len(ordered) - 1) * fraction
    low, high = int(index), min(len(ordered) - 1, int(index) + 1)
    weight = index - low
    return ordered[low] * (1 - weight) + ordered[high] * weight


def main() -> None:
    """Generate focused CSVs, including equal-recall/different-hit-distribution."""
    model = load_model_config(ROOT / "configs/model.yaml")
    hardware = load_hardware_config(ROOT / "configs/hardware.yaml")
    simulation = load_simulation_config(ROOT / "configs/sequence_simulation.yaml")
    tile = load_tile_config(ROOT / "configs/tile.yaml")
    inter = load_inter_layer_config(ROOT / "configs/inter_layer.yaml")
    base = load_synthetic_predictor_config(ROOT / "configs/synthetic_predictor.yaml", model)
    sequence = load_layer_sequences_jsonl(ROOT / "examples/multilayer_routing_trace.jsonl", model)[0]
    simulator = SequenceSimulator(model, hardware)
    exact = simulator.simulate_layer_sequence(
        sequence, SequenceScenario.EXACT, simulation=simulation,
        tile_config=tile, inter_layer_config=inter,
    ).result.total_cycles

    def run(config, *, depth: int = 1, gap: int | None = None, seed: int = 1234):
        predictor = SyntheticCrossLayerPredictor(config, model)
        configured_inter = inter if gap is None else replace(inter, default_aie_cycles=gap, overrides={})
        return simulator.simulate_layer_sequence(
            sequence, SequenceScenario.SYNTHETIC, simulation=simulation,
            tile_config=tile, inter_layer_config=configured_inter,
            predictor=predictor, prefetch_policy=FixedDepthPolicy(depth, 0), seed=seed,
        )

    output_dir = ROOT / "results/cross_layer_sweeps"
    output_dir.mkdir(parents=True, exist_ok=True)

    def write(name: str, rows: list[dict[str, object]]) -> None:
        with (output_dir / name).open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    def row(axis: str, value: object, output) -> dict[str, object]:
        metrics = output.result.prediction_metrics
        return {
            axis: value,
            "sequence_cycles": output.result.total_cycles,
            "speedup_vs_exact": exact / output.result.total_cycles,
            "observed_recall": statistics.fmean(metric.static_recall for metric in metrics),
            "full_hit_rate": statistics.fmean(float(metric.full_hit) for metric in metrics),
            "mean_lead_to_resolve": statistics.fmean(metric.prediction_lead_to_resolve for metric in metrics),
            "target_memory_stall": output.result.target_layer_memory_stall_cycles,
            "future_speculative_bytes": output.result.future_speculative_bytes,
            "useful_future_speculative_bytes": output.result.useful_future_speculative_bytes,
            "wasted_future_speculative_bytes": output.result.wasted_future_speculative_bytes,
            "current_demand_blocked_cycles": output.result.total_demand_blocked_by_future_speculative_cycles,
        }

    write("recall.csv", [
        row("target_recall", value, run(replace(
            base, generation_mode="independent_recall", target_recall=value,
            candidate_count=4,
        ))) for value in (0.0, 0.25, 0.5, 0.75, 1.0)
    ])
    write("candidate_count.csv", [
        row("candidate_count", value, run(replace(
            base, generation_mode="hit_distribution", hit_distribution={2: 1.0},
            candidate_count=value,
        ))) for value in (model.top_k, model.top_k + 1, 2 * model.top_k, model.num_experts)
    ])
    write("source_point.csv", [
        row("source_point", value, run(replace(base, source_point=value)))
        for value in (
            "source_router_done", "source_resolve", "source_expert_start",
            "source_gate_up_done", "source_expert_done",
        )
    ])
    write("predictor_latency.csv", [
        row("predictor_latency_cycles", value, run(replace(base, predictor_latency_cycles=value)))
        for value in (0, 10, 50, 100, 500)
    ])
    write("prefetch_depth.csv", [
        row("gate_up_tiles_per_candidate", value, run(replace(
            base, gate_up_tiles_per_candidate=value
        ), depth=value)) for value in (1, 2, 4, 8)
    ])
    write("inter_layer_gap.csv", [
        row("inter_layer_aie_cycles", value, run(base, gap=value))
        for value in (0, 100, 500, 1000)
    ])

    distribution_rows = []
    for label, distribution in (("always_one", {1: 1.0}), ("zero_or_full", {0: 0.5, 2: 0.5})):
        outputs = [run(replace(
            base, generation_mode="hit_distribution", hit_distribution=distribution,
            candidate_count=4,
        ), seed=seed) for seed in range(1000)]
        recalls = [
            metric.static_recall for output in outputs for metric in output.result.prediction_metrics
        ]
        full_hits = [
            float(metric.full_hit) for output in outputs for metric in output.result.prediction_metrics
        ]
        latencies = [output.result.target_layer_cycles for output in outputs]
        distribution_rows.append({
            "distribution": label,
            "mean_recall": statistics.fmean(recalls),
            "full_hit_rate": statistics.fmean(full_hits),
            "mean_target_stall": statistics.fmean(
                output.result.target_layer_memory_stall_cycles for output in outputs
            ),
            "p50_target_latency": statistics.median(latencies),
            "p95_target_latency": _percentile(latencies, 0.95),
            "mean_sequence_cycles": statistics.fmean(output.result.total_cycles for output in outputs),
            "sequence_speedup": exact / statistics.fmean(output.result.total_cycles for output in outputs),
        })
    write("same_recall_hit_distribution.csv", distribution_rows)
    manifest = {
        "note": "oracle-informed synthetic placeholder; not measured VEK280 performance",
        "exact_cycles": exact,
        "files": sorted(path.name for path in output_dir.glob("*.csv")),
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
