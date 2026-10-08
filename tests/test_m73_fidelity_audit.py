"""Artifact and analytical checks for the M7.3 fidelity audit."""
from __future__ import annotations

import csv
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/m73_fidelity_audit"


def rows(name: str) -> list[dict[str, str]]:
    """Read one generated M7.3 CSV."""
    with (OUT / name).open(encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def test_m73_required_artifacts_and_provenance_classes() -> None:
    """All required audit artifacts exist and use only declared provenance classes."""
    required = (
        "parameter_provenance.csv", "bandwidth_calibration.csv",
        "compute_calibration.csv", "ocm_fidelity.csv",
        "sensitivity_results.csv", "end_to_end_validation.csv",
        "artifact_validation.json",
    )
    assert all((OUT / name).is_file() and (OUT / name).stat().st_size > 0 for name in required)
    allowed = {"MEASURED", "HARDWARE_SPECIFIED", "DERIVED", "CALIBRATED", "ASSUMED", "SYNTHETIC_WORKLOAD_PARAMETER"}
    provenance = rows("parameter_provenance.csv")
    assert len(provenance) >= 30
    assert {row["source_type"] for row in provenance} <= allowed
    assert all(row["parameter"] and row["simulator_location"] and row["notes"] for row in provenance)


def test_m73_bandwidth_units_and_setup_dominance() -> None:
    """Bandwidth calibration preserves configured rate units and request duration."""
    data = rows("bandwidth_calibration.csv")
    controlled = [row for row in data if row["profile"] == "controlled_synthetic"]
    aggregates = {
        int(row["memory_channels"]): float(row["aggregate_effective_GB_per_s"])
        for row in controlled
    }
    assert aggregates == {1: 12.8, 2: 25.6, 3: 38.4}
    for row in data:
        assert int(row["total_dma_cycles"]) == int(row["setup_cycles"]) + int(row["payload_cycles"])
    small = next(row for row in controlled if row["memory_channels"] == "2" and row["transfer_bytes"] == "512")
    tile = next(row for row in controlled if row["memory_channels"] == "2" and row["transfer_bytes"] == "4096")
    assert float(small["setup_fraction"]) == 0.5
    assert tile["dominant_term"] == "payload_bandwidth"


def test_m73_sensitivity_matrix_and_fixed_bds() -> None:
    """Sensitivity has the requested axes and never silently re-optimizes BDS."""
    data = rows("sensitivity_results.csv")
    assert len(data) == 52
    expected = {
        "memory_bandwidth": {0.5, 0.75, 1.0, 1.5, 2.0},
        "compute_speed": {0.5, 0.75, 1.0, 1.5, 2.0},
        "ocm_capacity": {0.5, 1.0, 2.0},
    }
    for axis, scales in expected.items():
        subset = [row for row in data if row["sensitivity_axis"] == axis]
        assert {float(row["sensitivity_scale"]) for row in subset} == scales
        assert all(len([r for r in subset if r["sensitivity_scale"] == row["sensitivity_scale"]]) == 4 for row in subset)
    assert all(row["full_m72_search_rerun"] == "False" and row["bds_plan_reoptimized"] == "False" for row in data)
    bds = [row for row in data if row["policy"] == "BREADTH_DEPTH_SEARCH"]
    assert all(row["candidate_count"] == "4" and row["candidate_depth_vector"] == "(1, 1, 1, 1)" for row in bds)


def test_m73_ocm_and_no_fabricated_hardware_validation() -> None:
    """OCM occupancy is bounded and missing hardware validation is explicit."""
    policy_rows = [row for row in rows("ocm_fidelity.csv") if row["check"].startswith("policy accounting")]
    assert len(policy_rows) == 4
    assert all(int(row["value"]) <= 49_152 and row["status"] == "PASS" for row in policy_rows)
    validation = rows("end_to_end_validation.csv")
    assert validation == [validation[0]]
    assert validation[0]["validation_status"] == "NOT_AVAILABLE"
    assert validation[0]["hardware_measurement_cycles"] == ""
    assert validation[0]["notes"] == "No direct hardware validation available."


def test_m73_validation_record_and_markdown_links() -> None:
    """Machine validation passes and all local links in the two audit documents resolve."""
    record = json.loads((OUT / "artifact_validation.json").read_text(encoding="utf-8"))
    assert record["status"] == "PASS"
    assert record["unique_unified_simulator_runs"] == 44
    assert record["full_m72_search_rerun"] is False
    assert record["simulator_semantics_modified"] is False
    documents = (ROOT / "M73_SIMULATOR_FIDELITY_AUDIT_REPORT.md", ROOT / "docs/M73_SIMULATOR_FIDELITY.md")
    for document in documents:
        text = document.read_text(encoding="utf-8")
        for target in re.findall(r"\[[^]]+\]\(([^)]+)\)", text):
            if "://" not in target and not target.startswith("#"):
                assert (document.parent / target).resolve().is_file(), f"broken link in {document}: {target}"
