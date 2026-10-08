"""Regression tests for M10's isolated event/resource alignment adapter."""
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import pytest
import yaml

from edge_moe_sim.rtl_alignment import M10AlignmentSimulator
from edge_moe_sim.schema import ValidationError

ROOT = Path(__file__).resolve().parents[1]


def parameters() -> dict:
    """Load the independent service-time fixture used by RTL comparison."""
    return yaml.safe_load((ROOT/"configs/hardware/u200_rtl_alignment.yaml").read_text())["alignment"]


def cases() -> list[dict]:
    """Load common routing/predictor stimulus rows."""
    return [json.loads(line) for line in (ROOT/"hardware/m10_u200/traces/alignment_cases.jsonl").read_text().splitlines()]


@pytest.mark.parametrize("case", cases(), ids=lambda case: case["name"])
def test_alignment_resources_and_conservation(case: dict) -> None:
    """All cases conserve bytes/slots, serialize MAC/DDR and preserve prefix order."""
    simulator = M10AlignmentSimulator(parameters(), ddr_gaps=bool(case["ddr_gaps"]), allow_historical=True)
    result = simulator.simulate(case)
    assert asdict(result) == asdict(simulator.simulate(case))
    assert result.peak_ocm_slots <= 4
    assert result.demand_bytes+result.useful_speculative_bytes == 3072
    assert result.speculative_bytes == result.useful_speculative_bytes+result.wasted_speculative_bytes
    assert result.allocations == ([(case["predicted_expert"], 0)] if case["budget_bytes"] else [])
    assert sorted(result.compute_order) == [(3, 0), (3, 1), (4, 0), (4, 1)]
    for expert in (3, 4):
        assert [t for e, t in result.compute_order if e == expert] == [0, 1]
    if case["budget_bytes"] and case["predicted_expert"] == 4:
        assert result.compute_order[0] == (4, 0)
    for resource in ("MAC", "DDR_STRIPED"):
        spans = sorted((r["start_cycle"], r["end_cycle"]) for r in result.timeline if r["resource"] == resource)
        assert all(a[1] <= b[0] for a, b in zip(spans, spans[1:]))
    for row in result.timeline:
        if row["resource"] == "DDR_STRIPED":
            assert row["bytes"] <= (row["end_cycle"]-row["start_cycle"])*64
    for e, t in result.compute_order:
        gu = next(r for r in result.timeline if r["resource"] == "MAC" and r["operation"] == "gate_up" and
                  r["expert_id"] == e and r["tile_id"] == t)
        weights = [r for r in result.timeline if r["resource"] == "DDR_STRIPED" and r["expert_id"] == e and r["tile_id"] == t]
        assert len(weights) == 2 and max(r["end_cycle"] for r in weights) <= gu["start_cycle"]


def test_prefetch_benefit_without_truth_change() -> None:
    """Ready true prefix reduces initial waiting; false prefix only wastes bandwidth."""
    results = {c["name"]: M10AlignmentSimulator(parameters(), ddr_gaps=bool(c["ddr_gaps"]), allow_historical=True).simulate(c) for c in cases()}
    for mode in ("fast", "gaps"):
        assert results[f"hit_{mode}"].total_cycles < results[f"demand_{mode}"].total_cycles
        assert results[f"hit_{mode}"].weight_wait_cycles == 0
        assert results[f"wrong_{mode}"].wasted_speculative_bytes == 768


@pytest.mark.parametrize("field,value", [("ocm_slots", 2), ("physical_memory_channels", 1)])
def test_alignment_rejects_unsupported_constraints(field: str, value: int) -> None:
    """Do not silently simulate a materially different machine."""
    p = parameters()
    p[field] = value
    with pytest.raises(ValidationError):
        M10AlignmentSimulator(p, ddr_gaps=False, allow_historical=True)
