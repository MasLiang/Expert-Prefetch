"""Command-line interface for trace simulation and scenario comparison."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Sequence

from .config import (
    load_hardware_config, load_inter_layer_config, load_model_config,
    load_simulation_config, load_synthetic_predictor_config, load_tile_config,
)
from .dma_scheduler import (
    AllGateUpThenDownScheduler, CriticalPathAwareScheduler, ExpertContiguousScheduler,
    TileCriticalPathAwareScheduler,
)
from .metrics import result_to_dict
from .predictor import ExternalTracePredictor, OraclePredictor
from .prefetch_policy import FixedDepthPolicy, policy_from_config
from .scheduler import ReadyFirstScheduler, RouterOrderScheduler
from .schema import PrefetchConfig, SimulationConfig, TileConfig, ValidationError
from .simulator import Scenario, Simulator
from .timeline import write_timeline_csv
from .trace import aggregate_results, load_layer_sequences_jsonl, load_routing_jsonl
from .sweep import run_tile_sweep, write_sweep_csv
from .cross_layer_sweep import run_cross_layer_sweep, write_cross_layer_sweep_csv
from .sequence_metrics import aggregate_sequence_outputs
from .sequence_oracle import optimize_cross_layer_oracle_depth
from .sequence_predictor import CrossLayerOracleSetPredictor, SyntheticCrossLayerPredictor
from .sequence_simulator import SequenceScenario, SequenceSimulator


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="edge-moe-sim")
    subparsers = parser.add_subparsers(dest="command", required=True)
    run = subparsers.add_parser("run", help="simulate an independent routing trace")
    run.add_argument("--model", required=True)
    run.add_argument("--hardware", required=True)
    run.add_argument("--trace", required=True)
    run.add_argument("--scenario", required=True, choices=[scenario.value for scenario in Scenario])
    run.add_argument("--predictor")
    run.add_argument("--oracle-ready-offset", type=int, default=0)
    run.add_argument("--prefetch-scope", choices=["gate_up_only", "full_expert"], default="gate_up_only")
    run.add_argument("--max-candidates", type=int, default=4)
    run.add_argument(
        "--dma-scheduler",
        choices=["critical_path", "tile_critical_path", "all_gate_up_then_down", "expert_contiguous"],
        default=None,
    )
    run.add_argument("--expert-scheduler", choices=["ready_first", "router_order"], default="ready_first")
    run.add_argument("--allow-missing-predictor", action="store_true")
    run.add_argument("--output", required=True)
    run.add_argument("--granularity", choices=["stage", "tile"], default="stage")
    run.add_argument("--tile-size-bytes", type=int, default=65_536)
    run.add_argument(
        "--prefetch-policy",
        choices=["fixed_depth", "fixed_byte_budget", "progressive_rank", "predictor_specified"],
        default="fixed_depth",
    )
    run.add_argument("--prefetch-byte-budget", type=int)
    run.add_argument("--gate-up-tiles-per-candidate", type=int, default=2)
    run.add_argument("--down-tiles-per-candidate", type=int, default=0)
    run.add_argument("--include-down", action="store_true")

    compare = subparsers.add_parser("compare", help="compare aggregate summary JSON files")
    compare.add_argument("--inputs", nargs="+", required=True)
    compare.add_argument("--output", required=True)

    sweep = subparsers.add_parser("sweep", help="run a tile design-space grid")
    sweep.add_argument("--model", required=True)
    sweep.add_argument("--hardware", required=True)
    sweep.add_argument("--trace", required=True)
    sweep.add_argument("--scenario", choices=["oracle"], default="oracle")
    sweep.add_argument("--granularity", choices=["tile"], default="tile")
    sweep.add_argument("--tile-sizes", required=True)
    sweep.add_argument("--byte-budgets", required=True)
    sweep.add_argument("--ready-offsets", required=True)
    sweep.add_argument("--output", required=True)

    sequence = subparsers.add_parser("run-sequence", help="simulate continuous MoE layers")
    sequence.add_argument("--model", required=True)
    sequence.add_argument("--hardware", required=True)
    sequence.add_argument("--simulation", required=True)
    sequence.add_argument("--tile", required=True)
    sequence.add_argument("--inter-layer", required=True)
    sequence.add_argument("--trace", required=True)
    sequence.add_argument("--scenario", required=True, choices=[item.value for item in SequenceScenario])
    sequence.add_argument("--synthetic-predictor")
    sequence.add_argument("--oracle-byte-budget", type=int, default=65_536)
    sequence.add_argument("--seed", type=int)
    sequence.add_argument("--output", required=True)

    cross = subparsers.add_parser("sweep-cross-layer", help="run cross-layer synthetic grid")
    for name in ("model", "hardware", "simulation", "tile", "inter-layer", "trace", "synthetic-predictor"):
        cross.add_argument(f"--{name}", required=True)
    cross.add_argument("--candidate-counts", required=True)
    cross.add_argument("--recalls", required=True)
    cross.add_argument("--predictor-latencies", required=True)
    cross.add_argument("--source-points", required=True)
    cross.add_argument("--prefetch-depths", required=True)
    cross.add_argument("--inter-layer-cycles", required=True)
    cross.add_argument("--seeds", required=True)
    cross.add_argument("--output", required=True)
    return parser


def _run(args: argparse.Namespace) -> int:
    model = load_model_config(args.model)
    hardware = load_hardware_config(args.hardware)
    events = load_routing_jsonl(args.trace, model)
    scenario = Scenario(args.scenario)
    predictor = None
    if scenario is Scenario.PREDICTIVE:
        if not args.predictor:
            raise ValidationError("--predictor is required for scenario predictive")
        predictor = ExternalTracePredictor(
            args.predictor, model, allow_missing=args.allow_missing_predictor
        )
    elif scenario is Scenario.ORACLE:
        predictor = OraclePredictor(args.oracle_ready_offset)
    dma_scheduler_name = args.dma_scheduler or (
        "tile_critical_path" if args.granularity == "tile" else "critical_path"
    )
    dma_scheduler = {
        "critical_path": CriticalPathAwareScheduler,
        "tile_critical_path": TileCriticalPathAwareScheduler,
        "all_gate_up_then_down": AllGateUpThenDownScheduler,
        "expert_contiguous": ExpertContiguousScheduler,
    }[dma_scheduler_name]()
    expert_scheduler = {
        "ready_first": ReadyFirstScheduler,
        "router_order": RouterOrderScheduler,
    }[args.expert_scheduler]()
    simulator = Simulator(model, hardware)
    prefetch = PrefetchConfig(
        scope=args.prefetch_scope,
        max_candidates=args.max_candidates,
        policy=args.prefetch_policy,
        gate_up_tiles_per_candidate=args.gate_up_tiles_per_candidate,
        down_tiles_per_candidate=args.down_tiles_per_candidate,
        byte_budget=args.prefetch_byte_budget,
        include_down=args.include_down,
    )
    outputs = [
        simulator.simulate(
            event, scenario, predictor=predictor,
            expert_scheduler=expert_scheduler,
            dma_scheduler=dma_scheduler,
            prefetch=prefetch,
            simulation=SimulationConfig(args.granularity),
            tile_config=TileConfig(args.tile_size_bytes),
            prefetch_policy=policy_from_config(prefetch) if args.granularity == "tile" else None,
        )
        for event in events
    ]
    aggregate = aggregate_results([output.result for output in outputs])
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(
        json.dumps(aggregate.to_dict(), indent=2) + "\n", encoding="utf-8"
    )
    with (output_dir / "per_event.jsonl").open("w", encoding="utf-8") as stream:
        for output in outputs:
            stream.write(json.dumps(result_to_dict(output.result), sort_keys=True) + "\n")
    write_timeline_csv(
        [entry for output in outputs for entry in output.timeline],
        output_dir / "timeline.csv",
    )
    print(json.dumps(aggregate.to_dict(), indent=2))
    return 0


def _csv_ints(value: str, name: str) -> list[int]:
    try:
        values = [int(item) for item in value.split(",")]
    except ValueError as exc:
        raise ValidationError(f"{name} must be a comma-separated integer list") from exc
    if not values or any(item < 0 for item in values):
        raise ValidationError(f"{name} must contain non-negative integers")
    return values


def _sweep(args: argparse.Namespace) -> int:
    model = load_model_config(args.model)
    simulator = Simulator(model, load_hardware_config(args.hardware))
    events = load_routing_jsonl(args.trace, model)
    rows = run_tile_sweep(
        simulator,
        events,
        tile_sizes=_csv_ints(args.tile_sizes, "tile-sizes"),
        byte_budgets=_csv_ints(args.byte_budgets, "byte-budgets"),
        ready_offsets=_csv_ints(args.ready_offsets, "ready-offsets"),
    )
    output_dir = Path(args.output)
    write_sweep_csv(rows, output_dir / "sweep.csv")
    print(json.dumps({"points": len(rows), "output": str(output_dir / "sweep.csv")}, indent=2))
    return 0


def _compare(args: argparse.Namespace) -> int:
    summaries: dict[str, dict[str, Any]] = {}
    for raw_path in args.inputs:
        path = Path(raw_path)
        try:
            summary = json.loads(path.read_text(encoding="utf-8"))
            scenario = summary["scenario"]
            total_cycles = summary["total_cycles"]
        except (OSError, KeyError, TypeError, json.JSONDecodeError) as exc:
            raise ValidationError(f"invalid summary {path}: {exc}") from exc
        if scenario in summaries:
            raise ValidationError(f"duplicate scenario {scenario!r} in comparison inputs")
        if not isinstance(total_cycles, (int, float)) or total_cycles <= 0:
            raise ValidationError(f"summary {path} has invalid total_cycles")
        summaries[scenario] = summary
    demand = summaries.get(Scenario.DEMAND_SERIAL.value, {}).get("total_cycles")
    exact = summaries.get(Scenario.EXACT_PIPELINE.value, {}).get("total_cycles")
    ideal = summaries.get(Scenario.IDEAL.value, {}).get("total_cycles")
    rows: list[dict[str, Any]] = []
    for scenario in sorted(summaries):
        cycles = summaries[scenario]["total_cycles"]
        rows.append({
            "scenario": scenario,
            "total_cycles": cycles,
            "speedup_vs_demand_serial": demand / cycles if demand else None,
            "speedup_vs_exact_pipeline": exact / cycles if exact else None,
            "speedup_vs_ideal": ideal / cycles if ideal else None,
        })
    comparison = {"scenarios": rows}
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(comparison, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(comparison, indent=2))
    return 0


def _run_sequence(args: argparse.Namespace) -> int:
    model = load_model_config(args.model)
    hardware = load_hardware_config(args.hardware)
    simulation = load_simulation_config(args.simulation)
    tile = load_tile_config(args.tile)
    inter_layer = load_inter_layer_config(args.inter_layer)
    sequences = load_layer_sequences_jsonl(args.trace, model)
    scenario = SequenceScenario(args.scenario)
    predictor_config = (
        load_synthetic_predictor_config(args.synthetic_predictor, model)
        if args.synthetic_predictor else None
    )
    if scenario in {
        SequenceScenario.SYNTHETIC, SequenceScenario.ORACLE_SET,
        SequenceScenario.ORACLE_OPTIMAL,
    } and predictor_config is None:
        raise ValidationError("--synthetic-predictor is required for predictive sequence scenarios")
    simulator = SequenceSimulator(model, hardware)
    outputs = []
    oracle_allocations: list[dict[str, object]] = []
    for index, sequence in enumerate(sequences):
        seed = args.seed if args.seed is not None else (
            predictor_config.seed if predictor_config is not None else 0
        )
        if scenario is SequenceScenario.ORACLE_OPTIMAL:
            optimal = optimize_cross_layer_oracle_depth(
                sequence, model, hardware, simulation, tile, inter_layer,
                predictor_config, args.oracle_byte_budget, seed=seed + index,
            )
            outputs.append(optimal.output)
            oracle_allocations.append({
                "request_id": sequence.request_id,
                "token_position": sequence.token_position,
                "depths": optimal.gate_up_depths_by_target,
                "evaluated_allocations": optimal.evaluated_allocations,
            })
            continue
        predictor = None
        if scenario is SequenceScenario.SYNTHETIC:
            predictor = SyntheticCrossLayerPredictor(predictor_config, model)
        elif scenario is SequenceScenario.ORACLE_SET:
            predictor = CrossLayerOracleSetPredictor(predictor_config, model)
        depth = predictor_config.gate_up_tiles_per_candidate if predictor_config else 1
        down_depth = predictor_config.down_tiles_per_candidate if predictor_config else 0
        outputs.append(simulator.simulate_layer_sequence(
            sequence, scenario, simulation=simulation, tile_config=tile,
            inter_layer_config=inter_layer, predictor=predictor,
            prefetch_policy=FixedDepthPolicy(depth, down_depth), seed=seed + index,
        ))
    exact_outputs = outputs if scenario is SequenceScenario.EXACT else [
        simulator.simulate_layer_sequence(
            sequence, SequenceScenario.EXACT, simulation=simulation,
            tile_config=tile, inter_layer_config=inter_layer,
        ) for sequence in sequences
    ]
    for output, exact_output in zip(outputs, exact_outputs):
        output.result.speedup_vs_sequence_exact = (
            exact_output.result.total_cycles / output.result.total_cycles
        )
        output.result.target_stall_reduction_cycles = (
            exact_output.result.target_layer_memory_stall_cycles
            - output.result.target_layer_memory_stall_cycles
        )
    aggregate = aggregate_sequence_outputs(outputs)
    summary = {
        "scenario": scenario.value,
        **aggregate.to_dict(),
        "total_cycles": sum(output.result.total_cycles for output in outputs),
        "total_future_speculative_bytes": sum(output.result.future_speculative_bytes for output in outputs),
        "total_useful_future_speculative_bytes": sum(output.result.useful_future_speculative_bytes for output in outputs),
        "total_wasted_future_speculative_bytes": sum(output.result.wasted_future_speculative_bytes for output in outputs),
        "total_canceled_future_speculative_bytes": sum(output.result.canceled_future_speculative_bytes for output in outputs),
        "total_current_demand_blocked_cycles": sum(
            output.result.total_demand_blocked_by_future_speculative_cycles for output in outputs
        ),
        "mean_speedup_vs_sequence_exact": sum(
            float(output.result.speedup_vs_sequence_exact or 0.0) for output in outputs
        ) / len(outputs),
        "total_target_stall_reduction_cycles": sum(
            int(output.result.target_stall_reduction_cycles or 0) for output in outputs
        ),
        "oracle_allocations": oracle_allocations,
    }
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    with (output_dir / "per_sequence.jsonl").open("w", encoding="utf-8") as stream:
        for output in outputs:
            stream.write(json.dumps(asdict(output.result), sort_keys=True) + "\n")
    write_timeline_csv([entry for output in outputs for entry in output.timeline], output_dir / "timeline.csv")
    print(json.dumps(summary, indent=2))
    return 0


def _csv_floats(value: str, name: str) -> list[float]:
    try:
        values = [float(item) for item in value.split(",")]
    except ValueError as exc:
        raise ValidationError(f"{name} must be a comma-separated number list") from exc
    if not values:
        raise ValidationError(f"{name} must not be empty")
    return values


def _sweep_cross_layer(args: argparse.Namespace) -> int:
    model = load_model_config(args.model)
    rows = run_cross_layer_sweep(
        load_layer_sequences_jsonl(args.trace, model), model,
        load_hardware_config(args.hardware), load_simulation_config(args.simulation),
        load_tile_config(args.tile), load_synthetic_predictor_config(args.synthetic_predictor, model),
        candidate_counts=_csv_ints(args.candidate_counts, "candidate-counts"),
        recalls=_csv_floats(args.recalls, "recalls"),
        predictor_latencies=_csv_ints(args.predictor_latencies, "predictor-latencies"),
        source_points=args.source_points.split(","),
        prefetch_depths=_csv_ints(args.prefetch_depths, "prefetch-depths"),
        inter_layer_cycles=_csv_ints(args.inter_layer_cycles, "inter-layer-cycles"),
        seeds=_csv_ints(args.seeds, "seeds"),
    )
    output = Path(args.output) / "cross_layer_sweep.csv"
    write_cross_layer_sweep_csv(rows, output)
    print(json.dumps({"points": len(rows), "output": str(output)}, indent=2))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Parse CLI arguments, execute the command, and return an exit code."""
    args = _parser().parse_args(argv)
    try:
        if args.command == "run":
            return _run(args)
        if args.command == "sweep":
            return _sweep(args)
        if args.command == "run-sequence":
            return _run_sequence(args)
        if args.command == "sweep-cross-layer":
            return _sweep_cross_layer(args)
        return _compare(args)
    except ValidationError as exc:
        raise SystemExit(f"error: {exc}") from exc


if __name__ == "__main__":
    raise SystemExit(main())
