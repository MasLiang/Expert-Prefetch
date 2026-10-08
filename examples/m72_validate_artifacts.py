"""Validate completeness and row-level integrity of an M7.2 artifact tree."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

DEPRECATED = {
    "planned_speculative_bytes", "speculative_bytes",
    "useful_speculative_bytes", "wasted_speculative_bytes",
    "actual_speculative_bytes_before_router", "useful_true_bytes_before_router",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    """Read a required non-empty CSV file."""
    assert path.is_file() and path.stat().st_size > 0, f"missing/empty CSV: {path}"
    with path.open(encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert rows, f"CSV has no data rows: {path}"
    return rows


def validate(root: Path) -> dict[str, object]:
    """Assert M7.2 datasets, accounting, figures, and traces are coherent."""
    required = (
        "raw/m72_policy_evaluations.csv", "raw/m72_policy_winners.csv",
        "search/m72_breadth_depth_search.csv",
        "phase/m72_phase_quality_opportunity.csv", "phase/m72_phase_quality_ocm.csv",
        "phase/m72_phase_opportunity_channels.csv", "raw/m72_topk_sensitivity.csv",
        "raw/m72_batch_reuse_sensitivity.csv", "raw/m72_heuristic_regret.csv",
        "traces/m72_representative_cases.csv",
        "vek280_theoretical/m72_vek280_theoretical.csv",
    )
    datasets = {name: read_csv(root / name) for name in required}
    evaluations = datasets["raw/m72_policy_evaluations.csv"]
    winners = datasets["raw/m72_policy_winners.csv"]
    assert not (set(evaluations[0]) & DEPRECATED)
    assert {row["simulator_scope"] for row in evaluations} == {"unified_cross_layer"}
    assert {row["synthetic_lead_time"] for row in evaluations} == {"False"}
    eval_keys = {(row["comparison_id"], row["policy_plan_signature"]) for row in evaluations}
    assert len(eval_keys) == len(evaluations)
    assert len({row["comparison_id"] for row in winners}) == len(winners)
    assert {float(row["ocm_ratio"]) for row in winners} == {0.75, 1.0, 1.5, 4.0}
    assert max(int(row["initial_allocated_speculative_bytes"]) for row in evaluations
               if row["comparison_mode"] == "fixed_budget") <= 131_072
    for row in evaluations:
        assert float(row["regret_percent"]) >= 0
        assert "total_true_expert_memory_starvation_cycles" in row
        assert "critical_expert_starvation_cycles" in row
    for row in winners:
        assert row["winning_policy"] in row["policies_within_5pct"]
    manifest_path = root / "figure_manifest_m72.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert len(manifest) == 10
    for number in range(1, 11):
        stem = f"fig{number:02d}_"
        matches = [item for item in manifest if item["figure_id"].startswith(stem)]
        assert len(matches) == 1
        item = matches[0]
        for key in ("png", "pdf", "plot_data"):
            path = Path(item[key])
            assert path.is_file() and path.stat().st_size > 0
    trace_rows = datasets["traces/m72_representative_cases.csv"]
    assert 4 <= len(trace_rows) <= 6
    for row in trace_rows:
        trace_path = root / row["trace_file"]
        assert trace_path.is_file() and trace_path.stat().st_size > 0
    return {
        "status": "passed", "evaluation_rows": len(evaluations),
        "winner_rows": len(winners), "figure_count": len(manifest),
        "representative_trace_count": len(trace_rows),
    }


def main() -> int:
    """CLI entry point for artifact validation."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    args = parser.parse_args()
    result = validate(args.input_dir)
    output = args.input_dir / "diagnostics" / "artifact_validation.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
