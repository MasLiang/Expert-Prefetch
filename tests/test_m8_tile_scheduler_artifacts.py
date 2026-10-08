"""Validate the checked-in M8 experiment products and documentation."""
from __future__ import annotations

import csv
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/m8_tile_scheduler"


def test_m8_experiment_matrix_and_conservation() -> None:
    """The artifact contains the complete requested sweep and valid accounting."""
    with (OUT / "experiment.csv").open(encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 1296
    assert {row["scheduler"] for row in rows} == {
        "no_prefetch", "full_expert", "uniform_tile", "ranked_tile",
    }
    assert {float(row["ocm_scale"]) for row in rows} == {0.5, 1.0, 2.0}
    assert {float(row["bandwidth_scale"]) for row in rows} == {0.5, 1.0, 2.0}
    assert {int(row["dma_channels"]) for row in rows} == {1, 2, 4}
    assert {row["prediction_quality"] for row in rows} == {"perfect", "high", "medium", "low"}
    assert {row["tile_size"] for row in rows} == {"small", "medium", "large"}
    for row in rows:
        assert int(row["planned_bytes"]) <= int(row["plan_budget_bytes"])
        assert int(row["peak_ocm_bytes"]) <= int(row["ocm_capacity_bytes"])
        assert int(row["prefetched_tiles"]) == int(row["useful_prefetched_tiles"]) + int(row["wasted_prefetched_tiles"])
        assert int(row["prefetched_bytes"]) == int(row["useful_prefetched_bytes"]) + int(row["wasted_prefetched_bytes"])


def test_m8_summary_figures_and_validation() -> None:
    """Summaries and all PNG/PDF plots are non-empty and validation passed."""
    validation = json.loads((OUT / "artifact_validation.json").read_text(encoding="utf-8"))
    summary = json.loads((OUT / "summary.json").read_text(encoding="utf-8"))
    assert validation["status"] == "PASS" and validation["rows"] == 1296
    assert validation["hardware_validation"] is False
    assert summary["configuration_count"] == 324 and summary["simulation_count"] == 1296
    assert summary["hardware_validation"] is False
    for name in summary["plots"]:
        assert (OUT / name).is_file() and (OUT / name).stat().st_size > 1000


def test_tile_scheduling_document_links_resolve() -> None:
    """All local links in the M8 design document resolve."""
    document = ROOT / "docs/TILE_LEVEL_SCHEDULING.md"
    text = document.read_text(encoding="utf-8")
    for target in re.findall(r"\[[^]]+\]\(([^)]+)\)", text):
        if "://" not in target and not target.startswith("#"):
            assert (document.parent / target).resolve().exists(), target
    assert "do not constitute" in text
