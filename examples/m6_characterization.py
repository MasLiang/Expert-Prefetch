"""Run Milestone 6.5 synthetic/theoretical Roofline characterization.

Examples:
    python examples/m6_characterization.py --smoke
    python examples/m6_characterization.py --full --experiment reuse-bandwidth
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from edge_moe_sim.advanced_schema import AdvancedExecutionConfig, AdvancedScenario
from edge_moe_sim.characterization import controlled_routing, sanity_rows, simulate_row, write_rows
from edge_moe_sim.config import load_hardware_profile
from edge_moe_sim.roofline_prefetch_dse import profile_with_effective_bandwidth
from edge_moe_sim.schema import ModelConfig


EXPERIMENTS = ("reuse-bandwidth", "reuse-aie", "prefetch-window", "breadth-depth", "policy-comparison", "topk", "memory-channel", "tile-size", "dse")


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--smoke", action="store_true", help="small deterministic study")
    mode.add_argument("--full", action="store_true", help="full synthetic grid")
    parser.add_argument("--experiment", choices=EXPERIMENTS, help="run one study")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results" / "characterization")
    return parser.parse_args()


def _model(max_k: int = 8) -> ModelConfig:
    return ModelConfig("characterization_synthetic", 2048, 1408, 64, max_k, 8, 8)


def _point(name: str, model: ModelConfig, base, *, batch: int, top_k: int, pattern: str,
           bandwidth: float = 102.4, groups: int = 4, parallel: int = 2, tile: int = 352,
           scenario: AdvancedScenario = AdvancedScenario.EXACT_TILE_STREAMING,
           candidates: int = 1, depth: int = 1, ready: int = 0, quality: str = "medium",
           channels: int | None = None, placement: str = "round_robin_expert", ocm: int | None = None,
           seed: int = 0) -> dict:
    profile = profile_with_effective_bandwidth(base, bandwidth)
    if channels is not None:
        profile = replace(profile, num_memory_channels=channels)
        profile = profile_with_effective_bandwidth(profile, bandwidth)
    # Most studies isolate bandwidth/compute effects.  Keep a finite but roomy
    # default OCM so a deliberately small M6 profile does not turn unrelated
    # sweeps into admission deadlocks; the policy study overrides it explicitly.
    profile = replace(profile, ocm_bytes=ocm if ocm is not None else max(profile.ocm_bytes, 512 * 1024 * 1024))
    event = controlled_routing(batch_size=batch, top_k=top_k, num_experts=model.num_experts,
                               pattern=pattern, active_experts=min(model.num_experts, max(top_k, batch * top_k)), seed=seed)
    cfg = AdvancedExecutionConfig(tile, placement, groups, parallel, candidate_count=candidates,
                                  uniform_gu_prefetch_depth=depth,
                                  explicit_depths=tuple(depth for _ in range(candidates)),
                                  speculative_byte_budget=64 * 1024 * 1024)
    return simulate_row(experiment_name=name, run_id=f"{name}-{batch}-{top_k}-{bandwidth}-{groups}-{tile}-{seed}",
                        model=model, profile=profile, event=event, scenario=scenario, config=cfg,
                        routing_pattern=pattern, seed=seed, predictor_quality=quality, ready_cycle=ready)


def _run(name: str, full: bool, model: ModelConfig, base) -> list[dict]:
    reuse = (1, 2, 4, 8, 16, 32, 64, 128, 256) if full else (1, 4, 16)
    bandwidths = (25, 50, 75, 102.4, 200, 400, 800) if full else (25, 102.4)
    rows: list[dict] = []
    if name == "reuse-bandwidth":
        for weight_bits in ((16, 8) if full else (8,)):
            local = replace(model, weight_bits=weight_bits)
            for n in reuse:
                for bw in bandwidths:
                    rows.append(_point(name, local, base, batch=n, top_k=1, pattern="max_reuse", bandwidth=bw))
    elif name == "reuse-aie":
        for bw in ((25, 50, 102.4) if full else (25, 102.4)):
            for n in reuse[:7]:
                baseline = None
                for groups in ((1, 2, 4, 8, 16, 32) if full else (1, 4, 16)):
                    row = _point(name, model, base, batch=n, top_k=1, pattern="max_reuse", bandwidth=bw, groups=groups, parallel=1)
                    baseline = baseline or row["makespan_cycles"]
                    row["speedup_vs_1_group"] = baseline / row["makespan_cycles"]
                    rows.append(row)
    elif name == "prefetch-window":
        for batch in ((1, 2, 4) if full else (1, 4)):
            for top_k in ((1, 2, 4) if full else (2,)):
                for lead in ((0, 1000, 5000, 10000, 20000) if full else (0, 5000, 20000)):
                    for scenario in (AdvancedScenario.EXACT_TILE_STREAMING, AdvancedScenario.PREDICTIVE_FULL_EXPERT,
                                     AdvancedScenario.PREDICTIVE_UNIFORM_PREFIX, AdvancedScenario.PREDICTIVE_RANKED_PREFIX,
                                     AdvancedScenario.PREDICTIVE_DSE):
                        rows.append(_point(name, model, base, batch=batch, top_k=top_k, pattern="low_reuse", scenario=scenario,
                                           candidates=min(model.num_experts, 2 * top_k), depth=2, ready=max(0, 30000 - lead)))
    elif name == "breadth-depth":
        for top_k in ((1, 2, 4, 6) if full else (2, 6)):
            for batch in ((1, 4, 16) if full else (1, 4)):
                for breadth in tuple(min(model.num_experts, top_k * x) for x in ((1, 2, 4, 8) if full else (1, 4))):
                    for depth in ((1, 2, 4, 8, 16) if full else (1, 4)):
                        rows.append(_point(name, model, base, batch=batch, top_k=top_k, pattern="low_reuse", scenario=AdvancedScenario.PREDICTIVE_DSE, candidates=breadth, depth=depth))
    elif name == "policy-comparison":
        for pattern, batch in (("low_reuse", 1), ("balanced_reuse", 4), ("max_reuse", 16)):
            for top_k in ((1, 2, 4, 6) if full else (2, 6)):
                for quality in (("low", "medium", "high") if full else ("medium",)):
                    for ocm_factor in ((1, 2, 4, 8) if full else (2,)):
                        # Keep the policy comparison runnable with M6's current
                        # eager demand-window issuer.  The reported OCM field is
                        # still explicit; genuinely tiny OCM scheduling remains a
                        # model limitation rather than a silently dropped case.
                        ocm = max(ocm_factor, batch * top_k, 2 * top_k) * 3 * model.hidden_size * model.intermediate_size * model.weight_bits // 8
                        for scenario in (AdvancedScenario.EXACT_TILE_STREAMING, AdvancedScenario.PREDICTIVE_FULL_EXPERT,
                                         AdvancedScenario.PREDICTIVE_UNIFORM_PREFIX, AdvancedScenario.PREDICTIVE_RANKED_PREFIX,
                                         AdvancedScenario.PREDICTIVE_DSE):
                            rows.append(_point(name, model, base, batch=batch, top_k=top_k, pattern=pattern, scenario=scenario,
                                               candidates=min(model.num_experts, 2 * top_k), depth=2, quality=quality, ocm=ocm))
    elif name == "topk":
        for batch in ((1, 4, 16) if full else (1, 4)):
            for top_k in range(1, 9):
                for parallel in ((1, 2, 4, 8) if full else (1, 4)):
                    rows.append(_point(name, model, base, batch=batch, top_k=top_k, pattern="low_reuse", parallel=parallel,
                                       groups=min(8, base.total_aie_units // parallel)))
    elif name == "memory-channel":
        for channels in ((1, 2, 3) if full else (1, 3)):
            for placement in ("single_channel", "round_robin_expert", "striped_tile"):
                for top_k in ((1, 2, 4, 6) if full else (2, 6)):
                    for pattern, batch in (("low_reuse", 1), ("max_reuse", 16)):
                        rows.append(_point(name, model, base, batch=batch, top_k=top_k, pattern=pattern, channels=channels, placement=placement))
    elif name == "tile-size":
        tiles = (16, 32, 64, 128, 256, 352, 704, model.intermediate_size) if full else (64, 352, model.intermediate_size)
        for tile in tiles:
            for scenario in (AdvancedScenario.EXACT_TILE_STREAMING, AdvancedScenario.PREDICTIVE_UNIFORM_PREFIX):
                rows.append(_point(name, model, base, batch=4, top_k=2, pattern="balanced_reuse", tile=tile, scenario=scenario, candidates=4, depth=2))
    elif name == "dse":
        for pattern, batch, top_k in (("low_reuse", 1, 2), ("max_reuse", 16, 2), ("low_reuse", 4, 6)):
            for candidates in ((top_k, 2 * top_k, 4 * top_k) if full else (top_k, 2 * top_k)):
                for groups in ((1, 4, 8) if full else (1, 4)):
                    for tile in ((128, 352, 704) if full else (352,)):
                        rows.append(_point(name, model, base, batch=batch, top_k=top_k, pattern=pattern,
                                           scenario=AdvancedScenario.PREDICTIVE_DSE, candidates=min(64, candidates), depth=2,
                                           groups=groups, tile=tile))
    return rows


def main() -> int:
    args = _args()
    model, base = _model(), load_hardware_profile(ROOT / "configs" / "hardware_profile.yaml")
    raw = args.output_dir / "raw"
    sanity = sanity_rows(model, base)
    write_rows(sanity, raw / "characterization_sanity.csv")
    names = (args.experiment,) if args.experiment else EXPERIMENTS
    filenames = {"reuse-bandwidth": "reuse_bandwidth_phase.csv", "reuse-aie": "reuse_aie_allocation.csv",
                 "prefetch-window": "prefetch_window_study.csv", "breadth-depth": "breadth_depth_phase.csv",
                 "policy-comparison": "policy_comparison.csv", "topk": "topk_scalability.csv",
                 "memory-channel": "memory_channel_study.csv", "tile-size": "tile_size_study.csv", "dse": "selected_dse_results.csv"}
    for name in names:
        rows = _run(name, args.full, model, base)
        write_rows(rows, raw / filenames[name])
        if name == "dse":
            write_rows([row for row in rows if row["makespan_cycles"] == min(item["makespan_cycles"] for item in rows)], raw / "selected_dse_pareto.csv")
    print(f"Wrote characterization CSVs under {raw}; regenerate figures with examples/m6_make_figures.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
