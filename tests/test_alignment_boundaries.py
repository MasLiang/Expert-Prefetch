"""Backend-independent functional-boundary causality and accounting tests."""
from __future__ import annotations

from copy import deepcopy
from typing import Any

import pytest

from edge_moe_sim.alignment_boundaries import analyze_boundaries


def event(cycle: int, kind: str, expert: int = -1, tile: int = -1,
          matrix: str = "none") -> dict[str, Any]:
    """Construct a typed-unit one-layer boundary fixture."""
    return dict(cycle=cycle,event=kind,expert_id=expert,tile_id=tile,matrix=matrix,
                bytes=0,request_type="none",value=0)


def fixture() -> list[dict[str, Any]]:
    """Use two ready-first Experts; second's weights arrive before its window."""
    rows = [event(10,"ROUTE_RESOLVED"),event(10,"TILE_ISSUE_WINDOW_OPEN"),
            event(43,"TILE_ISSUE_WINDOW_OPEN")]
    for expert, ready in ((3,20),(4,15)):
        for matrix in ("gate_up","down"):
            rows.extend([event(1,"REQUEST_QUEUED",expert,0,matrix),
                         event(2,"MEMORY_SERVICE_START",expert,0,matrix),
                         event(ready,"PAYLOAD_READY",expert,0,matrix)])
    for expert, times in ((4,(21,23,30,31,39,40)),(3,(45,46,51,52,59,60))):
        for kind,cycle in zip(("COMPUTE_REQUEST_ACCEPTED","GU_START","GU_DONE", "DOWN_START","DOWN_DONE","TILE_RETIRED"),times):
            rows.append(event(cycle,kind,expert,0))
        rows.extend([event(times[-1],"EXPERT_COMBINE_START",expert,0),
                     event(times[-1]+3,"EXPERT_COMBINE_DONE",expert,0)])
    return rows


def test_functional_boundaries_ready_first_and_conservation() -> None:
    """Compute waiting on any eligible pair, not only Router-order first Expert."""
    result = analyze_boundaries(fixture(),(3,4),1)
    first,second = result["tile_records"]
    assert first["expert_id"] == 4
    assert first["memory_wait_cycles"] == 5
    assert first["ready_to_accept_cycles"] == 6
    assert first["accept_to_gu_cycles"] == 2
    assert second["memory_wait_cycles"] == 0
    assert second["ready_to_accept_cycles"] == 2
    assert result["totals"]["ready_to_gu_cycles"] == 11
    assert result["totals"]["expert_combine_cycles"] == 6
    assert len(result["stage_records"]) == 4
    assert result["stage_records"][0]["queue_wait_cycles"] == 1
    assert result["stage_records"][0]["service_to_payload_cycles"] == 18
    assert sum(r["end_cycle"]-r["start_cycle"] for r in result["timeline"]) == 16
    assert result == analyze_boundaries(list(reversed(fixture())),(3,4),1)


@pytest.mark.parametrize("failure",["missing","duplicate","negative","fractional","early_accept",
                                    "overlap","missing_payload","memory_order","combine_order"])
def test_invalid_boundary_lifecycle_rejected(failure: str) -> None:
    """Reject malformed observations instead of manufacturing latency agreement."""
    rows = deepcopy(fixture())
    if failure == "missing":
        rows = [r for r in rows if r["event"] != "GU_DONE"]
    elif failure == "duplicate":
        rows.append(next(r for r in rows if r["event"] == "PAYLOAD_READY"))
    elif failure in ("negative","fractional"):
        rows[0]["cycle"] = -1 if failure == "negative" else 1.5
    elif failure == "early_accept":
        next(r for r in rows if r["event"] == "COMPUTE_REQUEST_ACCEPTED")["cycle"] = 11
    elif failure == "overlap":
        rows[2]["cycle"] = 39
    elif failure == "missing_payload":
        rows = [r for r in rows if not (r["event"] == "PAYLOAD_READY" and r["expert_id"] == 3)]
    elif failure == "memory_order":
        next(r for r in rows if r["event"] == "MEMORY_SERVICE_START")["cycle"] = 50
    else:
        next(r for r in rows if r["event"] == "EXPERT_COMBINE_DONE")["cycle"] = 39
    with pytest.raises(ValueError):
        analyze_boundaries(rows,(3,4),1)


def test_collapsed_acceptance_stage_is_zero_duration() -> None:
    """A backend may expose an explicitly unmodeled zero stage, never fabricate it."""
    rows = fixture()
    for row in rows:
        if row["event"] == "COMPUTE_REQUEST_ACCEPTED":
            row["cycle"] = next(r["cycle"] for r in rows if r["event"] == "GU_START" and r["expert_id"] == row["expert_id"])
    result = analyze_boundaries(rows,(3,4),1)
    assert result["totals"]["accept_to_gu_cycles"] == 0
    assert result["totals"]["ready_to_gu_cycles"] == 11


def test_gu_early_rule_still_enforces_down_dependency() -> None:
    """GU may precede DOWN residency, but DOWN compute never may."""
    rows = fixture()
    next(r for r in rows if r["event"] == "PAYLOAD_READY" and r["expert_id"] == 4 and r["matrix"] == "down")["cycle"] = 35
    next(r for r in rows if r["event"] == "DOWN_START" and r["expert_id"] == 4)["cycle"] = 36
    result = analyze_boundaries(rows,(3,4),1,tile_start="gu_only_ready")
    first = result["tile_records"][0]
    assert first["startup_payload_ready_cycle"] == 15
    assert first["pair_payload_ready_cycle"] == 35
    assert first["gu_start_cycle"] == 23
    with pytest.raises(ValueError):
        analyze_boundaries(rows,(3,4),1)  # Old pair-ready rule rejects early acceptance.
    next(r for r in rows if r["event"] == "DOWN_START" and r["expert_id"] == 4)["cycle"] = 34
    with pytest.raises(ValueError):
        analyze_boundaries(rows,(3,4),1,tile_start="gu_only_ready")
    with pytest.raises(ValueError):
        analyze_boundaries(rows,(3,4),1,tile_start="invalid")
