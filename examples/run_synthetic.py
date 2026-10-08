"""Run all five scenarios and write summaries, timelines, and comparison."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from edge_moe_sim.config import load_hardware_config, load_model_config
from edge_moe_sim.metrics import result_to_dict
from edge_moe_sim.predictor import ExternalTracePredictor, OraclePredictor
from edge_moe_sim.prefetch_policy import FixedDepthPolicy
from edge_moe_sim.oracle_optimal import optimize_oracle_depth
from edge_moe_sim.schema import SimulationConfig, TileConfig
from edge_moe_sim.simulator import Scenario, Simulator
from edge_moe_sim.synthetic import make_decode_event
from edge_moe_sim.timeline import write_timeline_csv
from edge_moe_sim.sweep import run_tile_sweep, write_sweep_csv


def main() -> None:
    """Execute the bundled deterministic five-scenario synthetic example."""
    root = Path(__file__).resolve().parents[1]
    output_dir = root / "results" / "synthetic"
    output_dir.mkdir(parents=True, exist_ok=True)
    model = load_model_config(root / "configs" / "model.yaml")
    simulator = Simulator(model, load_hardware_config(root / "configs" / "hardware.yaml"))
    event = make_decode_event((0, 2), gating_weights=(0.65, 0.35))
    predictor = ExternalTracePredictor(root / "examples" / "predictor_trace.jsonl", model)
    outputs = {
        Scenario.DEMAND_SERIAL.value: simulator.simulate(event, Scenario.DEMAND_SERIAL),
        Scenario.EXACT_PIPELINE.value: simulator.simulate(event, Scenario.EXACT_PIPELINE),
        Scenario.PREDICTIVE.value: simulator.simulate(
            event, Scenario.PREDICTIVE, predictor=predictor
        ),
        Scenario.ORACLE.value: simulator.simulate(
            event, Scenario.ORACLE, predictor=OraclePredictor(0)
        ),
        Scenario.IDEAL.value: simulator.simulate(event, Scenario.IDEAL),
    }
    demand_cycles = outputs[Scenario.DEMAND_SERIAL.value].result.total_cycles
    exact_cycles = outputs[Scenario.EXACT_PIPELINE.value].result.total_cycles
    ideal_cycles = outputs[Scenario.IDEAL.value].result.total_cycles
    comparison: dict[str, object] = {"scenarios": []}
    for name, output in outputs.items():
        cycles = output.result.total_cycles
        output.result.speedup_vs_demand_serial = demand_cycles / cycles
        output.result.speedup_vs_exact_pipeline = exact_cycles / cycles
        output.result.speedup_vs_ideal = ideal_cycles / cycles
        scenario_dir = output_dir / name
        scenario_dir.mkdir(parents=True, exist_ok=True)
        (scenario_dir / "summary.json").write_text(
            json.dumps(result_to_dict(output.result), indent=2) + "\n", encoding="utf-8"
        )
        write_timeline_csv(output.timeline, scenario_dir / "timeline.csv")
        comparison["scenarios"].append({  # type: ignore[union-attr]
            "scenario": name,
            "total_cycles": cycles,
            "speedup_vs_demand_serial": demand_cycles / cycles,
            "speedup_vs_exact_pipeline": exact_cycles / cycles,
            "speedup_vs_ideal": ideal_cycles / cycles,
        })
    (output_dir / "comparison.json").write_text(
        json.dumps(comparison, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(comparison, indent=2))

    tile_config = TileConfig(4096)
    tile_common = {"simulation": SimulationConfig("tile"), "tile_config": tile_config}
    tile_outputs = {
        "demand_serial": simulator.simulate(event, Scenario.DEMAND_SERIAL, **tile_common),
        "exact_pipeline": simulator.simulate(event, Scenario.EXACT_PIPELINE, **tile_common),
        "predictive": simulator.simulate(
            event, Scenario.PREDICTIVE, predictor=predictor,
            prefetch_policy=FixedDepthPolicy(2, 0), **tile_common,
        ),
        "oracle_set": simulator.simulate(
            event, Scenario.ORACLE, predictor=OraclePredictor(0),
            prefetch_policy=FixedDepthPolicy(2, 0), **tile_common,
        ),
        "ideal": simulator.simulate(event, Scenario.IDEAL, **tile_common),
    }
    optimal = optimize_oracle_depth(
        simulator, event, tile_config=tile_config, byte_budget=16_384,
    )
    tile_outputs["oracle_optimal"] = optimal.output
    tile_output_dir = root / "results" / "synthetic_tile"
    tile_output_dir.mkdir(parents=True, exist_ok=True)
    stage_exact_cycles = outputs[Scenario.EXACT_PIPELINE.value].result.total_cycles
    tile_exact_cycles = tile_outputs["exact_pipeline"].result.total_cycles
    tile_comparison: dict[str, object] = {
        "stage_exact_cycles": stage_exact_cycles,
        "oracle_optimal_depths": optimal.optimal_depths,
        "oracle_optimal_evaluated_combinations": optimal.evaluated_combinations,
        "scenarios": [],
    }
    for name, output in tile_outputs.items():
        result = output.result
        scenario_dir = tile_output_dir / name
        scenario_dir.mkdir(parents=True, exist_ok=True)
        summary = result_to_dict(result)
        if name == "oracle_optimal":
            summary["optimal_depths"] = optimal.optimal_depths
            summary["evaluated_combinations"] = optimal.evaluated_combinations
        (scenario_dir / "summary.json").write_text(
            json.dumps(summary, indent=2) + "\n", encoding="utf-8"
        )
        write_timeline_csv(output.timeline, scenario_dir / "timeline.csv")
        tile_comparison["scenarios"].append({  # type: ignore[union-attr]
            "scenario": name,
            "cycles": result.total_cycles,
            "speedup_vs_stage_exact": stage_exact_cycles / result.total_cycles,
            "speedup_vs_tile_exact": tile_exact_cycles / result.total_cycles,
            "memory_stall": result.aie_memory_stall_cycles,
            "first_tile_stall": result.aie_first_tile_stall_cycles,
            "inter_tile_stall": result.aie_inter_tile_stall_cycles,
            "speculative_bytes": result.speculative_bytes,
            "useful_bytes": result.useful_speculative_bytes,
            "wasted_bytes": result.wasted_speculative_bytes,
            "ready_recall": result.ready_recall_at_resolve,
            "resident_fraction_at_resolve": result.true_tile_resident_fraction_at_resolve,
        })
    (tile_output_dir / "comparison.json").write_text(
        json.dumps(tile_comparison, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(tile_comparison, indent=2))

    sweep_dir = root / "results" / "sweeps"
    write_sweep_csv(
        run_tile_sweep(
            simulator, [event],
            tile_sizes=(4096, 16384, 65536, 262144, 1048576),
            byte_budgets=(16384,), ready_offsets=(0,),
        ),
        sweep_dir / "tile_size.csv",
    )
    write_sweep_csv(
        run_tile_sweep(
            simulator, [event], tile_sizes=(4096,), byte_budgets=(16384,),
            ready_offsets=(0, 7, 14, 21, 28),
        ),
        sweep_dir / "ready_offset.csv",
    )
    write_sweep_csv(
        run_tile_sweep(
            simulator, [event], tile_sizes=(4096,),
            byte_budgets=(4096, 8192, 16384, 32768, 65536), ready_offsets=(0,),
        ),
        sweep_dir / "byte_budget.csv",
    )


if __name__ == "__main__":
    main()
