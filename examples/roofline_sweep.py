"""Deterministic Milestone 6 Roofline and prefetch DSE experiment."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from edge_moe_sim.advanced_schema import AdvancedExecutionConfig, AdvancedScenario  # noqa: E402
from edge_moe_sim.batch_predictor import (  # noqa: E402
    BatchPredictorOutput, PrecomputedBatchPredictor, TokenPrediction,
)
from edge_moe_sim.batch_routing import BatchRoutingEvent  # noqa: E402
from edge_moe_sim.config import load_hardware_profile  # noqa: E402
from edge_moe_sim.roofline_prefetch_dse import (  # noqa: E402
    run_roofline_prefetch_dse, write_dse_csv,
)
from edge_moe_sim.roofline_sweep import run_roofline_sweep, write_roofline_sweep_csv  # noqa: E402
from edge_moe_sim.schema import CandidatePrefetch, ModelConfig  # noqa: E402


def _ints(value: str) -> list[int]:
    return [int(item) for item in value.split(",")]


def _floats(value: str) -> list[float]:
    return [float(item) for item in value.split(",")]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full-grid", action="store_true")
    parser.add_argument("--batch-size", default="1,4,16,64")
    parser.add_argument("--top-k", default="1,2,4,6")
    parser.add_argument("--num-experts", type=int, default=64)
    parser.add_argument("--hidden-size", type=int, default=2048)
    parser.add_argument("--ffn-size", type=int, default=1408)
    parser.add_argument("--weight-bits", type=int, default=8)
    parser.add_argument("--memory-channels", type=int, default=3)
    parser.add_argument("--bandwidth-gbps", default="25,50,75,102.4")
    parser.add_argument("--bandwidth-efficiency", type=float, default=0.70)
    parser.add_argument("--dma-setup-cycles", type=int, default=200)
    parser.add_argument("--aie-total-units", type=int, default=64)
    parser.add_argument("--aie-units-per-expert", type=int, default=8)
    parser.add_argument("--aie-ops-per-cycle", type=float, default=8.0)
    parser.add_argument("--aie-efficiency", type=float, default=0.60)
    parser.add_argument("--max-parallel-experts", type=int, default=4)
    parser.add_argument("--f-tile-size", type=int, default=352)
    parser.add_argument("--candidate-count", type=int)
    parser.add_argument("--prefetch-depth", type=int, default=1)
    parser.add_argument("--prefetch-policy", default="uniform")
    parser.add_argument("--spec-byte-budget", type=int, default=16 * 1024 * 1024)
    parser.add_argument("--ocm-bytes", type=int, default=64 * 1024 * 1024)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--output", default=str(ROOT / "results/roofline"))
    return parser


def _event(batch: int, top_k: int) -> BatchRoutingEvent:
    # Controlled reuse: every token uses the same K experts, so n_e=B.
    routes = tuple(tuple(range(top_k)) for _ in range(batch))
    return BatchRoutingEvent(batch * 100 + top_k, 0, 0, routes)


def _predictor(events: list[BatchRoutingEvent], num_experts: int) -> PrecomputedBatchPredictor:
    outputs = {}
    for event in events:
        token_predictions = []
        for token_id, route in enumerate(event.routes):
            ids = list(route)
            false_count = min(7 * len(route), num_experts - len(route))
            ids.extend(range(len(route), len(route) + false_count))
            candidates = tuple(CandidatePrefetch(
                expert, 1.0 - rank * 0.01 if expert in route else 0.30 - rank * 0.001
            ) for rank, expert in enumerate(ids))
            token_predictions.append(TokenPrediction(token_id, candidates))
        output = BatchPredictorOutput(
            event.request_id, 0, 0, 0, tuple(token_predictions), "max"
        )
        outputs[(event.request_id, 0, 0)] = output
    return PrecomputedBatchPredictor(outputs)


def main(argv: list[str] | None = None) -> int:
    """Run the configured event-driven sweep and write plot-ready CSV files."""
    args = _parser().parse_args(argv)
    batches = _ints(args.batch_size)
    top_ks = _ints(args.top_k)
    bandwidths = _floats(args.bandwidth_gbps)
    if args.full_grid:
        batches = [1, 2, 4, 8, 16, 32, 64]
        top_ks = [1, 2, 4, 6]
        bandwidths = [25.0, 50.0, 75.0, 102.4]
    model = ModelConfig(
        "roofline_synthetic", args.hidden_size, args.ffn_size,
        args.num_experts, max(top_ks), args.weight_bits, 8,
    )
    base = load_hardware_profile(ROOT / "configs/hardware_profile.yaml")
    profile = replace(
        base, num_memory_channels=args.memory_channels,
        bandwidth_efficiency=args.bandwidth_efficiency,
        dma_setup_cycles=args.dma_setup_cycles,
        total_aie_units=args.aie_total_units,
        ops_per_cycle_per_aie=args.aie_ops_per_cycle,
        aie_efficiency=args.aie_efficiency, ocm_bytes=args.ocm_bytes,
    )
    events = [_event(batch, top_k) for batch in batches for top_k in top_ks]
    predictor = _predictor(events, model.num_experts)
    scenarios = (
        AdvancedScenario.EXACT_FULL_EXPERT_LOAD,
        AdvancedScenario.EXACT_TILE_STREAMING,
        AdvancedScenario.PREDICTIVE_FULL_EXPERT,
        AdvancedScenario.PREDICTIVE_UNIFORM_PREFIX,
        AdvancedScenario.PREDICTIVE_DSE,
    )
    rows = []
    for event in events:
        candidate_count = args.candidate_count or min(model.num_experts, 2 * event.top_k)
        config = AdvancedExecutionConfig(
            args.f_tile_size, "round_robin_expert", args.aie_units_per_expert,
            min(args.max_parallel_experts, event.top_k),
            candidate_count=candidate_count,
            uniform_gu_prefetch_depth=args.prefetch_depth,
            explicit_depths=tuple(
                max(0, args.prefetch_depth - rank // max(1, event.top_k))
                for rank in range(candidate_count)
            ),
            speculative_byte_budget=args.spec_byte_budget,
        )
        rows.extend(run_roofline_sweep(
            model, profile, [event], predictor, scenarios,
            bandwidths_gbps=bandwidths, config=config,
        ))
    # Low-reuse point makes AIE overprovisioning and breadth/depth tradeoffs visible.
    representative = _event(1, 2)
    dse_predictor = _predictor([representative], model.num_experts)
    dse_rows = run_roofline_prefetch_dse(
        model, profile, representative, dse_predictor,
        candidate_counts=(2, 4, 8),
        prefetch_policies=("uniform", "rank_based", "confidence_based"),
        depths=(1, 2, 4) if not args.full_grid else (0, 1, 2, 4, 8),
        aie_units_per_expert=(1, 2, 4, 8, 16),
        max_parallel_experts=(1, 2),
        f_tile_sizes=(args.f_tile_size,),
        placement_policies=("round_robin_expert",),
        effective_bandwidth_gbps=bandwidths,
        speculative_byte_budget=args.spec_byte_budget,
        ocm_capacity_bytes=args.ocm_bytes,
    )
    output = Path(args.output)
    write_roofline_sweep_csv(rows, output / "roofline_sweep_results.csv")
    write_dse_csv(dse_rows, output / "dse_results.csv")
    best = min((row for row in dse_rows if row.feasible), key=lambda row: row.makespan_cycles)
    summary = {
        "note": "synthetic/theoretical inputs; not measured VEK280 performance",
        "roofline_points": len(rows), "dse_points": len(dse_rows),
        "best_dse": {
            "cycles": best.makespan_cycles,
            "candidate_count": best.candidate_count,
            "depth": best.prefetch_depth,
            "aie_units_per_expert": best.aie_units_per_expert,
            "max_parallel_experts": best.max_parallel_experts,
            "bandwidth_gbps": best.effective_bandwidth_gbps,
            "speculative_bytes": best.speculative_bytes,
            "roofline_efficiency": best.board_roofline_efficiency,
        },
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    sample_batch, sample_k, sample_bw = batches[-1], top_ks[min(1, len(top_ks) - 1)], bandwidths[-1]
    print("\nscenario                         B  K   BW_GB/s       cycles      AI  regime")
    for row in rows:
        if row.batch_size == sample_batch and row.top_k == sample_k and row.bandwidth_gbps == sample_bw:
            print(
                f"{row.scenario:<32} {row.batch_size:>2} {row.top_k:>2} "
                f"{row.bandwidth_gbps:>9.1f} {row.layer_cycles:>12} "
                f"{row.arithmetic_intensity:>7.1f}  {row.regime}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
