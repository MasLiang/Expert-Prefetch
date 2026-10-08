"""Artifact-level checks for the complete M9.3 sensitivity study."""

from __future__ import annotations

import ast
import csv
import json
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "m93_predictor_sensitivity"


def read(name: str) -> list[dict[str, str]]:
    """Load one checked-in M9.3 CSV."""
    with (OUT / name).open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def test_m93_matrices_cover_all_requested_axes() -> None:
    """Quality, calibration, and robustness tables have exact coverage."""
    quality = read("quality_sweep.csv")
    calibration = read("calibration_sweep.csv")
    robustness = read("scheduler_robustness.csv")
    assert len(quality) == 3 * 3 * 4 * 3 * 9 * 5 == 4860
    assert len(calibration) == 3 * 3 * 4 * 3 * 9 * 4 * 5 == 19440
    assert len(robustness) == 3 * 3 * 4 * 3 * 9 * 5 == 4860
    assert {float(row["prediction_quality"]) for row in calibration} == {
        1.0, .9, .8, .7, .6, .5, .4, .3, .2
    }
    assert {row["calibration_mode"] for row in calibration} == {
        "well_calibrated", "overconfident_wrong",
        "underconfident_correct", "uniform",
    }
    assert {row["scheduler"] for row in calibration} == {
        "no_prefetch", "full_expert", "budget_proportional",
        "hardware_greedy", "ranked_tile",
    }


def test_m93_calibration_preserves_trace_and_quality_ranks_are_monotonic() -> None:
    """Calibration changes scores only; quality moves truth monotonically down."""
    rows = read("calibration_sweep.csv")
    groups: dict[tuple[str, ...], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        key = tuple(row[name] for name in (
            "workload", "hardware", "ocm_scale", "bandwidth_scale",
            "prediction_quality",
        ))
        groups[key].append(row)
    for group in groups.values():
        assert len({row["candidate_ids"] for row in group}) == 1
        assert len({row["ranking_trace_id"] for row in group}) == 1
        assert len({row["true_expert_ranks"] for row in group}) == 1

    for workload in {row["workload"] for row in rows}:
        sample = {
            float(row["prediction_quality"]): ast.literal_eval(row["true_expert_ranks"])
            for row in rows if row["workload"] == workload
        }
        ordered = [sample[value] for value in (1.0, .9, .8, .7, .6, .5, .4, .3, .2)]
        for left, right in zip(ordered, ordered[1:]):
            assert all(a <= b for a, b in zip(left, right))


def test_m93_scheduler_resource_and_conservation_invariants() -> None:
    """Every simulation respects allocation accounting and resource envelopes."""
    for row in read("calibration_sweep.csv"):
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


def test_m93_required_figures_summary_and_claim_boundary() -> None:
    """All deliverables and the explicit research boundary are checked in."""
    for name in (
        "quality_sweep.csv", "calibration_sweep.csv", "scheduler_robustness.csv",
        "summary.json", "SUMMARY.md", "artifact_validation.json",
    ):
        path = OUT / name
        assert path.is_file() and path.stat().st_size > 0
    stems = (
        "quality_vs_latency", "quality_vs_speedup", "calibration_robustness",
        "useful_fraction_vs_quality", "scheduler_rank_vs_quality",
        "quality_ocm_phase_diagram",
    )
    for index, stem in enumerate(stems, start=1):
        for suffix in ("png", "pdf"):
            path = OUT / f"fig{index:02d}_{stem}.{suffix}"
            assert path.is_file() and path.stat().st_size > 1000

    summary = json.loads((OUT / "summary.json").read_text(encoding="utf-8"))
    validation = json.loads(
        (OUT / "artifact_validation.json").read_text(encoding="utf-8")
    )
    assert summary["calibration_rows"] == 19440
    assert summary["robustness_rows"] == 4860
    assert "real predictor accuracy" in summary["not_claimed"]
    assert validation["status"] == "PASS"
    assert validation["calibration_candidate_identity"] is True
    assert validation["monotonic_ranking_quality"] is True
    assert validation["hardware_validation"] is False


def test_m93_report_documents_results_and_reproduction() -> None:
    """The report answers all five questions without real-predictor claims."""
    text = (ROOT / "docs" / "M93_PREDICTOR_SENSITIVITY_ANALYSIS.md").read_text(
        encoding="utf-8"
    )
    for phrase in (
        "Prediction quality threshold",
        "better predictor quality",
        "Budget Proportional",
        "Hardware Greedy",
        "Full Expert",
        "SUPPORTED CLAIMS",
        "NOT CLAIMED",
        "python examples/m93_predictor_sensitivity.py",
    ):
        assert phrase in text
