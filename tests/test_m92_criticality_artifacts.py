"""Artifact-level validation for M9.2 criticality scheduling analysis."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "m92_criticality_scheduler"


def read(name: str) -> list[dict[str, str]]:
    """Load one deterministic M9.2 CSV artifact."""
    with (OUT / name).open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def test_m92_matrix_and_predictor_identity_are_complete() -> None:
    """Every requested point has nine schedulers and one predictor trace."""
    rows = read("scheduler_comparison.csv")
    assert len(rows) == 3 * 3 * 4 * 4 * 9 == 1296
    assert {row["scheduler"] for row in rows} == {
        "no_prefetch", "full_expert", "uniform_tile", "ranked_tile",
        "budget_proportional", "hardware_greedy", "criticality_aware",
        "fixed_prediction_oracle", "full_oracle",
    }
    groups: dict[tuple[str, ...], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        key = tuple(row[name] for name in (
            "workload", "hardware", "prediction_quality", "ocm_scale"
        ))
        groups[key].append(row)
    assert len(groups) == 144
    for group in groups.values():
        assert len(group) == 9
        assert len({row["true_expert_ranks"] for row in group}) == 1
        assert len({row["candidate_count"] for row in group}) == 1


def test_m92_oracle_bounds_conservation_and_resource_limits() -> None:
    """Oracle ordering, signed gaps, conservation, and resource limits hold."""
    groups: dict[tuple[str, ...], list[dict[str, str]]] = defaultdict(list)
    for row in read("scheduler_comparison.csv"):
        assert int(row["allocated_bytes"]) == (
            int(row["useful_bytes"])
            + int(row["wasted_bytes"])
            + int(row["cancelled_bytes"])
        )
        assert int(row["allocated_tile_transfers"]) == (
            int(row["useful_tile_transfers"])
            + int(row["wasted_tile_transfers"])
            + int(row["cancelled_tile_transfers"])
        )
        assert int(row["allocated_bytes"]) <= int(row["plan_budget_bytes"])
        assert int(row["peak_ocm_bytes"]) <= int(row["ocm_capacity_bytes"])
        groups[tuple(row[name] for name in (
            "workload", "hardware", "prediction_quality", "ocm_scale"
        ))].append(row)

    for group in groups.values():
        by_name = {row["scheduler"]: row for row in group}
        full = int(by_name["full_oracle"]["latency_cycles"])
        fixed = int(by_name["fixed_prediction_oracle"]["latency_cycles"])
        assert full <= fixed
        assert all(full <= int(row["latency_cycles"]) for row in group)
        assert all(
            fixed <= int(by_name[name]["latency_cycles"])
            for name in (
                "no_prefetch", "full_expert", "uniform_tile", "ranked_tile",
                "budget_proportional", "hardware_greedy", "criticality_aware",
            )
        )
        for row in group:
            latency = int(row["latency_cycles"])
            assert int(row["prediction_gap_cycles_full_minus_fixed"]) == full - fixed
            assert int(row["allocation_gap_cycles_fixed_minus_scheduler"]) == fixed - latency
            assert int(row["total_loss_to_full_oracle_cycles"]) == latency - full


def test_m92_required_csv_figures_summary_and_claim_boundary() -> None:
    """All requested deliverables exist in both human and machine formats."""
    for name in (
        "scheduler_comparison.csv", "oracle_decomposition.csv",
        "criticality_analysis.csv", "summary.json", "SUMMARY.md",
        "artifact_validation.json",
    ):
        path = OUT / name
        assert path.is_file() and path.stat().st_size > 0
    stems = (
        "oracle_gap_decomposition", "scheduler_latency_comparison",
        "latency_reduction_per_byte", "criticality_improvement",
        "expert_size_tile_benefit",
    )
    for index, stem in enumerate(stems, start=1):
        for suffix in ("png", "pdf"):
            path = OUT / f"fig{index:02d}_{stem}.{suffix}"
            assert path.is_file() and path.stat().st_size > 1000

    summary = json.loads((OUT / "summary.json").read_text(encoding="utf-8"))
    validation = json.loads(
        (OUT / "artifact_validation.json").read_text(encoding="utf-8")
    )
    assert summary["scheduler_rows"] == 1296
    assert summary["configurations"] == 144
    assert "RTL implementation" in summary["not_claimed"]
    assert validation["status"] == "PASS"
    assert validation["fixed_prediction_trace_identity"] is True
    assert validation["full_oracle_dominance"] is True
    assert validation["hardware_validation"] is False


def test_m92_report_documents_formula_scope_and_negative_result() -> None:
    """The research report states formula, Oracle scope, and honest findings."""
    text = (ROOT / "docs" / "M92_CRITICALITY_AWARE_TILE_SCHEDULING.md").read_text(
        encoding="utf-8"
    )
    for phrase in (
        "Fixed Prediction Oracle",
        "confidence × depth_priority × execution_urgency × transfer_cycles",
        "same candidate set",
        "does not outperform",
        "SUPPORTED",
        "NOT CLAIMED",
        "python examples/m92_criticality_scheduler.py",
    ):
        assert phrase in text
