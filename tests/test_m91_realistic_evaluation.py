"""Artifact-level validation for the checked-in M9.1 characterization."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "m91_realistic_evaluation"


def _rows() -> list[dict[str, str]]:
    """Load the deterministic scheduler result matrix."""
    with (RESULTS / "scheduler_results.csv").open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def test_m91_matrix_is_complete() -> None:
    """The checked-in matrix covers every requested evaluation dimension."""
    rows = _rows()
    assert len(rows) == 3 * 3 * 4 * 4 * 3 * 7 == 3024
    assert {row["workload"] for row in rows} == {
        "small_edge_moe", "medium_moe", "large_moe"
    }
    assert len({row["hardware"] for row in rows}) == 3
    assert {row["prediction_quality"] for row in rows} == {
        "perfect", "high", "medium", "low"
    }
    assert {row["ocm_scale"] for row in rows} == {"0.25", "0.5", "1.0", "2.0"}
    assert {row["bandwidth_scale"] for row in rows} == {"0.5", "1.0", "2.0"}
    assert {row["scheduler"] for row in rows} == {
        "no_prefetch", "full_expert", "uniform_tile", "ranked_tile",
        "budget_proportional", "hardware_greedy", "oracle",
    }


def test_m91_conservation_capacity_and_oracle_invariants() -> None:
    """Every row conserves traffic, respects OCM, and is bounded by Oracle."""
    groups: dict[tuple[str, ...], list[dict[str, str]]] = defaultdict(list)
    for row in _rows():
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
        key = tuple(row[name] for name in (
            "workload", "hardware", "prediction_quality", "ocm_scale",
            "bandwidth_scale",
        ))
        groups[key].append(row)

    assert len(groups) == 3 * 3 * 4 * 4 * 3 == 432
    for group in groups.values():
        assert len(group) == 7
        oracle = next(
            int(row["latency_cycles"])
            for row in group if row["scheduler"] == "oracle"
        )
        assert all(oracle <= int(row["latency_cycles"]) for row in group)


def test_m91_required_tables_figures_and_claim_boundary_exist() -> None:
    """Required CSV, figure, report, and claim-boundary artifacts are present."""
    for name in (
        "workload_summary.csv", "hardware_summary.csv", "scheduler_results.csv",
        "oracle_gap.csv", "summary.json", "SUMMARY.md", "artifact_validation.json",
    ):
        path = RESULTS / name
        assert path.is_file() and path.stat().st_size > 0
    for index, stem in enumerate((
        "speedup_vs_ocm", "speedup_vs_experts", "scheduler_by_workload",
        "oracle_gap_vs_workload", "hardware_comparison", "useful_byte_fraction",
    ), start=1):
        for suffix in ("png", "pdf"):
            path = RESULTS / f"fig{index:02d}_{stem}.{suffix}"
            assert path.is_file() and path.stat().st_size > 0

    summary = json.loads((RESULTS / "summary.json").read_text(encoding="utf-8"))
    assert summary["reported_scheduler_rows"] == 3024
    assert summary["formal_configurations"] == 432
    assert summary["supported_claims"] == [
        "relative scheduler comparison", "scaling trends", "resource tradeoffs"
    ]
    assert "real VEK280 performance" in summary["not_claimed"]
    assert "real V80 performance" in summary["not_claimed"]


def test_m91_documentation_names_profiles_and_reproduction_command() -> None:
    """The report identifies provenance, limitations, and how to reproduce it."""
    report = (ROOT / "docs" / "M91_REALISTIC_WORKLOAD_EVALUATION.md").read_text(
        encoding="utf-8"
    )
    for phrase in (
        "representative synthetic derived from MoE scaling",
        "VEK280 theoretical dataflow",
        "V80-like theoretical streaming",
        "Generic FPGA streaming",
        "python examples/m91_realistic_evaluation.py",
        "SUPPORTED CLAIMS",
        "NOT CLAIMED",
    ):
        assert phrase in report
