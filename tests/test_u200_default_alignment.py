"""Full-default M10 alignment geometry, invariants and shared manifest validation."""
from __future__ import annotations

import importlib
import hashlib
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pytest
import yaml

from edge_moe_sim.rtl_alignment import M10AlignmentSimulator
from edge_moe_sim.alignment_boundaries import analyze_boundaries
from edge_moe_sim.schema import ValidationError

ROOT = Path(__file__).resolve().parents[1]


def parameters() -> dict[str, Any]:
    """Read the full-default constraints and independent structural service estimates."""
    return yaml.safe_load((ROOT/"configs/hardware/u200_rtl_alignment_default.yaml").read_text())["alignment"]


def cases() -> list[dict[str, Any]]:
    """Read the actual shared eight-case predictor/DDR matrix."""
    return [json.loads(line) for line in (ROOT/"hardware/m10_u200/traces/alignment_cases.jsonl").read_text().splitlines()]


@pytest.mark.parametrize("case", cases(), ids=lambda case: case["name"])
def test_full_default_resources(case: dict[str, Any]) -> None:
    """Conserve payload, slot capacity and prefix execution without RTL timing input."""
    p = parameters()
    simulator = M10AlignmentSimulator(p, ddr_gaps=bool(case["ddr_gaps"]), allow_historical=True)
    result = simulator.simulate(case)
    assert asdict(result) == asdict(simulator.simulate(case))
    assert result.demand_bytes+result.useful_speculative_bytes == 24576
    assert result.speculative_bytes == result.useful_speculative_bytes+result.wasted_speculative_bytes
    assert result.peak_ocm_slots <= 128
    assert len(result.peak_ocm_slots_per_bank) == 4
    assert max(result.peak_ocm_slots_per_bank) <= 32
    assert sorted(result.compute_order) == [(e,t) for e in (3,4) for t in range(16)]
    for expert in (3,4):
        assert [t for e,t in result.compute_order if e == expert] == list(range(16))
    for resource_name in ("MAC", "DDR_STRIPED"):
        spans = sorted((r["start_cycle"],r["end_cycle"]) for r in result.timeline if r["resource"] == resource_name)
        assert all(a[1] <= b[0] for a,b in zip(spans,spans[1:]))
    assert result.allocations == ([(case["predicted_expert"],0)] if case["budget_bytes"] else [])
    assert result.memory_wait_cycles == result.weight_wait_cycles
    assert result.ready_control_wait_cycles == 32*p["tile_dispatch_cycles"]
    assert result.expert_issue_idle_cycles == result.memory_wait_cycles+result.ready_control_wait_cycles
    waits = sorted((r["start_cycle"], r["end_cycle"]) for r in result.timeline if r["resource"] == "EXPERT_ISSUE")
    assert all(a[1] <= b[0] for a,b in zip(waits,waits[1:]))
    assert sum(end-start for start,end in waits) == result.expert_issue_idle_cycles
    canonical = analyze_boundaries(result.boundary_events,case["true_expert_ids"],16)
    assert len(canonical["tile_records"]) == 32
    assert canonical["totals"]["memory_wait_cycles"] == result.weight_wait_cycles
    assert canonical["totals"]["ready_to_gu_cycles"] == 256
    assert canonical["totals"]["accept_to_gu_cycles"] == 0
    if case["predicted_expert"] == 5 and case["budget_bytes"]:
        assert result.wasted_speculative_bytes == 768
        assert result.demand_bytes == 24576


def test_shared_manifest_defaults_and_oracle(monkeypatch: pytest.MonkeyPatch) -> None:
    """Reject default mismatches and generate E64/F256 numerical oracle from config."""
    monkeypatch.setattr(sys,"path",[str(ROOT/"hardware/m10_u200/scripts"),*sys.path])
    legacy = importlib.import_module("compare_u200_alignment")
    driver = importlib.import_module("compare_u200_default_alignment")
    p = parameters()
    assert len(legacy.read_cases(ROOT/"hardware/m10_u200/traces/alignment_cases.jsonl",p)) == 8
    driver.verify_default_parameters(p)
    fixture = legacy.numerical_fixture(p)
    assert len(fixture["router_weights_output_major"]) == 64
    assert fixture["expert_tile_count"] == 16
    assert fixture["true_expert_ids"] == [3,4]
    assert fixture["expected_result_y"] == [192]*16
    changed = {**p,"ocm_slots":4}
    with pytest.raises(ValueError):
        legacy.read_cases(ROOT/"hardware/m10_u200/traces/alignment_cases.jsonl",changed)
    with pytest.raises(ValueError):
        driver.verify_default_parameters({**p,"dma_fifo_depth":4})


def test_unconditional_gu_start_rejects_legacy_adapter(monkeypatch: pytest.MonkeyPatch) -> None:
    """No compatibility parameter remains; old pair-ready comparisons cannot silently run."""
    monkeypatch.setattr(sys,"path",[str(ROOT/"hardware/m10_u200/scripts"),*sys.path])
    driver = importlib.import_module("compare_u200_alignment")
    driver.require_current_tile_start("gu_only_ready")
    with pytest.raises(ValueError, match="M10AlignmentSimulator 已失效"):
        driver.require_current_tile_start("gu_and_down_ready")
    for name in ("tile_execution_scheduler", "tile_prefetch_predictive_axi_top", "u200_moe_top"):
        source = (ROOT/f"hardware/m10_u200/rtl/{name}.sv").read_text()
        assert "GU_EARLY_START" not in source


@pytest.mark.parametrize("value", [-1, 1.5, True])
def test_invalid_structural_delay(value: Any) -> None:
    """Reject invalid control-cycle units rather than silently rounding or fitting."""
    with pytest.raises(ValidationError):
        M10AlignmentSimulator({**parameters(),"demand_release_delay_cycles":value},ddr_gaps=False,allow_historical=True)


def test_default_bank_geometry_rejected() -> None:
    """Never reinterpret 128 slots as a differently banked hardware configuration."""
    with pytest.raises(ValidationError):
        M10AlignmentSimulator({**parameters(),"ocm_banks":2,"slots_per_bank":64},ddr_gaps=False,allow_historical=True)


def test_netlist_provenance_rejects_stale_config_and_source(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """A substituted netlist or stale RTL/config must fail before any simulation."""
    monkeypatch.setattr(sys,"path",[str(ROOT/"hardware/m10_u200/scripts"),*sys.path])
    driver = importlib.import_module("compare_u200_default_alignment")
    netlist = tmp_path/"tiny.v"
    netlist.write_text("module u200_moe_top; endmodule\n")
    proof = dict(top="u200_moe_top",part="xcu200-fsgd2104-2-e",parameters=parameters(),
                 netlist_sha256=hashlib.sha256(netlist.read_bytes()).hexdigest(),
                 rtl_source_sha256={path.name:hashlib.sha256(path.read_bytes()).hexdigest()
                                    for path in (ROOT/"hardware/m10_u200/rtl").glob("*.sv")})
    metadata = tmp_path/"manifest.json"
    metadata.write_text(json.dumps(proof))
    assert driver.validate_netlist(netlist,metadata,parameters()) == proof
    with pytest.raises(ValueError,match="parameter mismatch"):
        driver.validate_netlist(netlist,metadata,{**parameters(),"num_experts":6})
    proof["rtl_source_sha256"]["u200_moe_top.sv"] = "0"*64
    metadata.write_text(json.dumps(proof))
    with pytest.raises(ValueError,match="RTL sources differ"):
        driver.validate_netlist(netlist,metadata,parameters())
    netlist.write_text("module changed; endmodule\n")
    with pytest.raises(ValueError,match="SHA256 mismatch"):
        driver.validate_netlist(netlist,metadata,parameters())


def test_wait_metric_partitions_and_timeline(monkeypatch: pytest.MonkeyPatch) -> None:
    """Require disjoint samples, conserved legacy/new windows and strict input parsing."""
    monkeypatch.setattr(sys,"path",[str(ROOT/"hardware/m10_u200/scripts"),*sys.path])
    driver = importlib.import_module("compare_u200_default_alignment")
    events = [dict(cycle=c,event=k) for c,k in [(1,"WAIT_MEMORY"),(2,"WAIT_MEMORY"),
                                              (3,"WAIT_CONTROL"),(5,"WAIT_CONTROL")]]
    text = "WAIT_METRICS 4 2 2 2 1 1 1\n"
    result = driver.parse_wait_metrics(text,events,3)
    assert result["expert_issue_idle_cycles"] == 4
    assert [(r["start_cycle"],r["end_cycle"]) for r in driver.wait_intervals(events)] == [(1,3),(3,4),(5,6)]
    for bad in ("",text+text,"WAIT_METRICS 5 2 2 2 1 1 1", "WAIT_METRICS 4 -2 6 2 1 1 1",
                "WAIT_METRICS 4 2 2 2 0 1 1", "WAIT_METRICS 4 2 2 2 1 5 1"):
        with pytest.raises(ValueError):
            driver.parse_wait_metrics(bad,events,3)
    with pytest.raises(ValueError,match="timeline/counter mismatch"):
        driver.parse_wait_metrics(text,events[:-1],3)
    with pytest.raises(ValueError,match="duplicate/overlapping"):
        driver.parse_wait_metrics(text,events+[events[0]],3)


def test_wait_observation_does_not_change_performance() -> None:
    """New observability must preserve the independently frozen pre-split model cycles."""
    expected = dict(demand_fast=14010,hit_fast=13900,second_hit_fast=13900,wrong_fast=14010,
                    demand_gaps=14019,hit_gaps=13900,second_hit_gaps=13900,wrong_gaps=14019)
    for case in cases():
        result = M10AlignmentSimulator(parameters(),ddr_gaps=bool(case["ddr_gaps"]),allow_historical=True).simulate(case)
        assert result.total_cycles == expected[case["name"]]


def test_acceptance_partition_preserves_timeline() -> None:
    """Splitting the same 8-cycle dispatch does not change the modeled machine."""
    p = parameters()
    case = cases()[0]
    baseline = M10AlignmentSimulator(p,ddr_gaps=False,allow_historical=True).simulate(case)
    split = M10AlignmentSimulator({**p,"request_accept_to_gu_cycles":2},ddr_gaps=False,allow_historical=True).simulate(case)
    assert split.total_cycles == baseline.total_cycles
    assert split.timeline == baseline.timeline
    result = analyze_boundaries(split.boundary_events,case["true_expert_ids"],16)
    assert result["totals"]["ready_to_accept_cycles"] == 192
    assert result["totals"]["accept_to_gu_cycles"] == 64
    assert result["totals"]["ready_to_gu_cycles"] == 256
    for bad in (-1,9,True,1.5):
        with pytest.raises(ValidationError):
            M10AlignmentSimulator({**p,"request_accept_to_gu_cycles":bad},ddr_gaps=False,allow_historical=True)
