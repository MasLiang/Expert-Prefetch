"""Deterministic acceptance tests for the M7.2 policy-study adapter."""
from __future__ import annotations

import sys
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "examples"))

import m71_unified_characterization as m71
import m72_policy_phase as m72
from edge_moe_sim.unified_policy_search import UnifiedBreadthDepthSearch, rank_policy_rows


@lru_cache(maxsize=2)
def _rows(mode: str) -> tuple[list[dict[str, object]], dict[str, object]]:
    cfg = m72.make_config("medium", m72.OPPS[1], 65_536, 2)
    budget = 65_536 if mode == "fixed_budget" else None
    rows, _, _, _ = m72.evaluate_configuration(
        cfg, [0, 1_000], comparison_mode=mode, budget_per_prediction=budget
    )
    return rows, m72.winner(rows)


def test_main_rows_use_unified_real_timing_and_common_inputs() -> None:
    """Policies in a cell share truth, ranking, workload, and real timing."""
    rows, _ = _rows("unconstrained")
    assert {row["simulator_scope"] for row in rows} == {"unified_cross_layer"}
    assert {row["synthetic_lead_time"] for row in rows} == {False}
    for field in ("predictor_trace_id", "workload_trace_id", "true_expert_rank_vector",
                  "source_milestone", "predictor_latency_cycles", "hardware_profile"):
        assert len({row[field] for row in rows}) == 1


def test_full_prefix_only_change_policy_granularity() -> None:
    """The controlled full/prefix pair keeps all configuration identities fixed."""
    rows, _ = _rows("unconstrained")
    full = next(row for row in rows if row["policy"] == "full_expert_prefetch")
    prefix = next(row for row in rows if row["policy"] == "ranked_prefix")
    for field in ("configuration_id", "workload_trace_id", "predictor_trace_id",
                  "memory_channels", "ocm_bytes", "top_k", "batch_size"):
        assert full[field] == prefix[field]
    assert full["policy_plan_signature"] != prefix["policy_plan_signature"]


def test_budget_modes_are_explicit_and_enforced() -> None:
    """Fixed budget bounds total two-edge allocation; natural mode is unbudgeted."""
    fixed, _ = _rows("fixed_budget")
    natural, _ = _rows("unconstrained")
    assert all(int(row["initial_allocated_speculative_bytes"]) <= 131_072 for row in fixed)
    assert all(row["speculative_budget_bytes"] == "" for row in natural)
    assert any(int(row["initial_allocated_speculative_bytes"]) > 131_072 for row in natural)


def test_search_budget_deduplication_and_depth_domain() -> None:
    """Search prunes duplicate/zero/over-budget plans before simulation."""
    search = UnifiedBreadthDepthSearch(4096)
    plans, stats = search.plans_with_stats(tuple(range(32)), 65_536)
    assert stats["duplicate_pruned"] > 0
    assert stats["static_pruned"] > 0
    assert len({plan.signature for plan in plans}) == len(plans)
    assert all(sum(plan.depths) * 4096 <= 65_536 for plan in plans)
    assert all(set(plan.depths) <= {1, 2, 4, 8} for plan in plans)


def test_winner_regret_and_near_sets_are_consistent() -> None:
    """Makespan selects the winner and all reported regrets are non-negative."""
    rows, selected = _rows("unconstrained")
    assert selected["best_makespan"] == min(row["sequence_makespan"] for row in rows)
    assert min(float(row["regret_percent"]) for row in rows) == 0
    assert all(float(row["regret_percent"]) >= 0 for row in rows)
    assert selected["winning_policy"] in str(selected["policies_within_5pct"])


def test_tie_break_is_deterministic_without_erasing_latency_tie() -> None:
    """Cost ordering chooses a preferred exact tie and preserves both raw winners."""
    base = {"sequence_makespan": 100, "peak_ocm_bytes": 10,
            "initial_allocated_speculative_bytes": 10,
            "physical_speculative_bytes_transferred_by_router": 10,
            "policy_plan_signature": "b"}
    rows = rank_policy_rows([
        dict(base, policy="full_expert_prefetch", wasted_physical_speculative_bytes=8),
        dict(base, policy="ranked_prefix", wasted_physical_speculative_bytes=2,
             policy_plan_signature="a"),
    ])
    assert rows[0]["policy"] == "ranked_prefix"
    assert rows[0]["regret_cycles"] == rows[1]["regret_cycles"] == 0


def test_configuration_ids_and_workload_derived_ocm_reference() -> None:
    """Canonical IDs change with configuration and OCM references are measured."""
    a = m72.make_config("perfect", m72.OPPS[0], 65_536, 1)
    b = m72.make_config("perfect", m72.OPPS[1], 65_536, 1)
    assert a["configuration_id"] != b["configuration_id"]
    reference_b1 = m72.demand_reference(4, 1, 1)
    reference_b16 = m72.demand_reference(4, 16, 1)
    assert reference_b1 in (8192, 16384, 32768, 49152, 65536, 98304, 131072, 262144, 1048576)
    assert reference_b1 != reference_b16


def test_m72_removes_deprecated_speculative_aliases() -> None:
    """New analysis rows expose only M7.1b canonical speculative accounting."""
    rows, _ = _rows("unconstrained")
    assert all(not (set(row) & m72.DEPRECATED_SPEC_FIELDS) for row in rows)


def test_complete_ranking_is_fixed_while_candidate_count_truncates() -> None:
    """Changing M takes a prefix of one deterministic complete ranking."""
    model, _, _, layers = m71.machine()
    p4, _ = m71.predictors(layers, model, "medium", 4, True, m72.RANKS["medium"])
    p16, _ = m71.predictors(layers, model, "medium", 16, True, m72.RANKS["medium"])
    ranking4 = tuple(x.expert_id for x in p4[(0, 1)].predict(layers[1]).aggregate_candidates())
    ranking16 = tuple(x.expert_id for x in p16[(0, 1)].predict(layers[1]).aggregate_candidates())
    assert ranking4 == ranking16
    # M is applied by UnifiedPredictionSpec; predictor identity/ranking stays complete.
    row4, _ = m71.run(policy="uniform_prefix", m=4, ranked=True,
                       true_ranks=m72.RANKS["medium"])
    row16, _ = m71.run(policy="uniform_prefix", m=16, ranked=True,
                        true_ranks=m72.RANKS["medium"])
    assert row4["predictor_trace_id"] == row16["predictor_trace_id"]


def test_router_truth_is_policy_authoritative() -> None:
    """Prediction changes requests, never the target Router's true expert set."""
    _, no = m71.run(policy="no_prediction", ranked=True,
                     true_ranks=m72.RANKS["wrong"])
    _, predicted = m71.run(policy="full_expert_prefetch", ranked=True,
                            true_ranks=m72.RANKS["wrong"])
    assert [layer.true_expert_ids for layer in no.layer_results.values()] == [
        layer.true_expert_ids for layer in predicted.layer_results.values()
    ]
