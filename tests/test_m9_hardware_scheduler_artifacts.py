"""Validate M9 hardware-friendly scheduler experiment artifacts."""
from __future__ import annotations

import csv
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/m9_hardware_scheduler"


def read(name: str) -> list[dict[str, str]]:
    """Read one M9 CSV artifact."""
    with (OUT / name).open(encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def test_m9_matrix_conservation_and_resource_limits() -> None:
    """All 48x7 rows obey allocation conservation and planning limits."""
    rows = read("experiment.csv")
    assert len(rows) == 336
    assert {row["scheduler"] for row in rows} == {
        "no_prefetch", "full_expert", "uniform_tile", "ranked_tile",
        "budget_proportional", "hardware_greedy", "oracle",
    }
    assert {float(row["ocm_scale"]) for row in rows} == {0.25, 0.5, 1.0, 2.0}
    assert {float(row["bandwidth_scale"]) for row in rows} == {0.5, 1.0, 2.0}
    assert {row["prediction_quality"] for row in rows} == {"perfect", "high", "medium", "low"}
    for row in rows:
        assert int(row["allocated_bytes"]) == int(row["useful_bytes"]) + int(row["wasted_bytes"]) + int(row["cancelled_bytes"])
        assert int(row["allocated_tile_transfers"]) == int(row["useful_tile_transfers"]) + int(row["wasted_tile_transfers"]) + int(row["cancelled_tile_transfers"])
        assert int(row["allocated_bytes"]) <= int(row["plan_budget_bytes"])
        assert int(row["peak_ocm_bytes"]) <= int(row["ocm_capacity_bytes"])


def test_m9_oracle_dominates_every_reported_heuristic() -> None:
    """The event-simulated offline Oracle is a valid latency upper bound."""
    groups: dict[tuple[str, str, str], list[dict[str, str]]] = {}
    for row in read("experiment.csv"):
        groups.setdefault((row["prediction_quality"], row["ocm_scale"], row["bandwidth_scale"]), []).append(row)
    assert len(groups) == 48
    for group in groups.values():
        oracle = next(int(row["latency_cycles"]) for row in group if row["scheduler"] == "oracle")
        assert all(oracle <= int(row["latency_cycles"]) for row in group)
        assert next(row for row in group if row["scheduler"] == "oracle")["oracle_gap_cycles"] == "0"


def test_m9_complexity_marks_online_and_offline_boundaries() -> None:
    """Greedy has no divider and the exhaustive Oracle is never hardware friendly."""
    complexity = {row["scheduler"]: row for row in read("complexity_comparison.csv")}
    assert complexity["hardware_greedy"]["divisions"] == "0"
    assert complexity["hardware_greedy"]["hardware_friendly"] == "True"
    assert complexity["budget_proportional"]["hardware_friendly"] == "True"
    assert int(complexity["budget_proportional"]["divisions"]) > 0
    assert complexity["oracle"]["hardware_friendly"] == "False"
    assert complexity["oracle"]["estimated_control_cycles"] == "offline"


def test_m9_summaries_figures_and_document_links() -> None:
    """Validation passes, plots exist, and all local documentation links resolve."""
    validation = json.loads((OUT / "artifact_validation.json").read_text(encoding="utf-8"))
    summary = json.loads((OUT / "summary.json").read_text(encoding="utf-8"))
    assert validation["status"] == "PASS" and validation["oracle_dominance"] is True
    assert validation["hardware_validation"] is False
    assert summary["fpga_rtl_or_hls_implemented"] is False
    for name in summary["plots"]:
        assert (OUT / name).is_file() and (OUT / name).stat().st_size > 1000
    document = ROOT / "docs/HARDWARE_FRIENDLY_TILE_SCHEDULER.md"
    text = document.read_text(encoding="utf-8")
    for target in re.findall(r"\[[^]]+\]\(([^)]+)\)", text):
        if "://" not in target and not target.startswith("#"):
            assert (document.parent / target).resolve().exists(), target
    assert "not near the truth-oracle" in text
