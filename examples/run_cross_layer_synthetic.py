"""Run the bundled oracle-informed synthetic cross-layer sensitivity example."""

from __future__ import annotations

import json
import sys
from dataclasses import asdict, replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from edge_moe_sim.config import (  # noqa: E402
    load_hardware_config, load_inter_layer_config, load_model_config,
    load_simulation_config, load_synthetic_predictor_config, load_tile_config,
)
from edge_moe_sim.prefetch_policy import FixedDepthPolicy  # noqa: E402
from edge_moe_sim.sequence_oracle import optimize_cross_layer_oracle_depth  # noqa: E402
from edge_moe_sim.sequence_predictor import (  # noqa: E402
    CrossLayerOracleSetPredictor, SyntheticCrossLayerPredictor,
)
from edge_moe_sim.sequence_simulator import SequenceScenario, SequenceSimulator  # noqa: E402
from edge_moe_sim.timeline import write_timeline_csv  # noqa: E402
from edge_moe_sim.trace import load_layer_sequences_jsonl  # noqa: E402


def _write_output(name: str, output, output_root: Path) -> None:
    directory = output_root / name
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "summary.json").write_text(
        json.dumps(asdict(output.result), indent=2) + "\n", encoding="utf-8"
    )
    write_timeline_csv(output.timeline, directory / "timeline.csv")


def main() -> None:
    """Run all sequence baselines and controlled-recall prediction points."""
    model = load_model_config(ROOT / "configs/model.yaml")
    hardware = load_hardware_config(ROOT / "configs/hardware.yaml")
    simulation = load_simulation_config(ROOT / "configs/sequence_simulation.yaml")
    tile = load_tile_config(ROOT / "configs/tile.yaml")
    inter_layer = load_inter_layer_config(ROOT / "configs/inter_layer.yaml")
    base = load_synthetic_predictor_config(ROOT / "configs/synthetic_predictor.yaml", model)
    sequence = load_layer_sequences_jsonl(
        ROOT / "examples/multilayer_routing_trace.jsonl", model
    )[0]
    simulator = SequenceSimulator(model, hardware)
    output_root = ROOT / "results/synthetic_cross_layer"
    outputs = {}
    outputs["exact"] = simulator.simulate_layer_sequence(
        sequence, SequenceScenario.EXACT, simulation=simulation,
        tile_config=tile, inter_layer_config=inter_layer,
    )
    for label, recall in (("recall_0", 0.0), ("recall_50", 0.5), ("recall_75", 0.75), ("recall_100", 1.0)):
        config = replace(
            base, generation_mode="independent_recall", target_recall=recall,
            candidate_count=4, source_point="source_router_done",
            predictor_latency_cycles=20, gate_up_tiles_per_candidate=1,
        )
        outputs[label] = simulator.simulate_layer_sequence(
            sequence, SequenceScenario.SYNTHETIC, simulation=simulation,
            tile_config=tile, inter_layer_config=inter_layer,
            predictor=SyntheticCrossLayerPredictor(config, model),
            prefetch_policy=FixedDepthPolicy(1, 0), seed=config.seed,
        )
    oracle_predictor = CrossLayerOracleSetPredictor(base, model)
    outputs["oracle_set"] = simulator.simulate_layer_sequence(
        sequence, SequenceScenario.ORACLE_SET, simulation=simulation,
        tile_config=tile, inter_layer_config=inter_layer,
        predictor=oracle_predictor, prefetch_policy=FixedDepthPolicy(1, 0), seed=base.seed,
    )
    optimal = optimize_cross_layer_oracle_depth(
        sequence, model, hardware, simulation, tile, inter_layer, base,
        byte_budget=4 * tile.tile_size_bytes, seed=base.seed,
    )
    outputs["oracle_optimal"] = optimal.output
    outputs["ideal"] = simulator.simulate_layer_sequence(
        sequence, SequenceScenario.IDEAL, simulation=simulation,
        tile_config=tile, inter_layer_config=inter_layer,
    )
    exact_cycles = outputs["exact"].result.total_cycles
    exact_stall = outputs["exact"].result.target_layer_memory_stall_cycles
    oracle_cycles = outputs["oracle_set"].result.total_cycles
    comparison = {"note": "synthetic placeholder; not measured VEK280 performance", "scenarios": {}}
    for name, output in outputs.items():
        output.result.speedup_vs_sequence_exact = exact_cycles / output.result.total_cycles
        output.result.speedup_vs_cross_layer_oracle = oracle_cycles / output.result.total_cycles
        output.result.target_stall_reduction_cycles = (
            exact_stall - output.result.target_layer_memory_stall_cycles
        )
        _write_output(name, output, output_root)
        metrics = output.result.prediction_metrics
        comparison["scenarios"][name] = {
            "sequence_cycles": output.result.total_cycles,
            "target_layer_cycles": output.result.target_layer_cycles,
            "target_memory_stall": output.result.target_layer_memory_stall_cycles,
            "observed_recall": sum(metric.static_recall for metric in metrics) / len(metrics) if metrics else None,
            "full_hit": all(metric.full_hit for metric in metrics) if metrics else None,
            "ready_at_source_end": sum(metric.ready_at_source_layer_end for metric in metrics) / len(metrics) if metrics else None,
            "ready_at_target_start": sum(metric.ready_at_target_layer_start for metric in metrics) / len(metrics) if metrics else None,
            "ready_at_resolve": sum(metric.ready_at_target_resolve for metric in metrics) / len(metrics) if metrics else None,
            "future_speculative_bytes": output.result.future_speculative_bytes,
            "useful_bytes": output.result.useful_future_speculative_bytes,
            "wasted_bytes": output.result.wasted_future_speculative_bytes,
            "speedup_vs_exact": exact_cycles / output.result.total_cycles,
        }
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "comparison.json").write_text(
        json.dumps(comparison, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(comparison, indent=2))


if __name__ == "__main__":
    main()
