"""Canonical M7.1b speculative lifecycle and startup metric tests."""
from __future__ import annotations
import importlib.util
from pathlib import Path
SCRIPT=Path(__file__).parents[1]/'examples'/'m71_unified_characterization.py'
SPEC=importlib.util.spec_from_file_location('m71b_characterization',SCRIPT);assert SPEC and SPEC.loader
M71=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(M71)

def breadth(m:int):
    return M71.run(policy='uniform_prefix',m=m,ranked=True,budget=65536,candidate_depths=tuple(16//m for _ in range(m)))

def test_fixed_budget_uses_initial_allocation_and_conserves_lifecycle() -> None:
    for m in (2,4,8,16):
        row,_=breadth(m)
        assert row['initial_allocated_speculative_bytes']==131072
        assert row['initial_allocated_speculative_bytes']==row['cancelled_before_issue_speculative_bytes']+row['pending_unissued_speculative_bytes']+row['issued_speculative_bytes']
        assert row['completed_speculative_bytes']<=row['issued_speculative_bytes']

def test_router_physical_bytes_split_completed_and_partial_exactly() -> None:
    for m in (2,4,8,16):
        row,_=breadth(m)
        assert row['physical_speculative_bytes_transferred_by_router']==row['completed_speculative_bytes_before_router']+row['partial_inflight_speculative_bytes_at_router']
        assert 0<=row['partial_inflight_speculative_bytes_at_router']<=row['issued_speculative_bytes']

def test_cancelled_false_requests_are_not_physical_waste() -> None:
    row,_=breadth(16)
    assert row['cancelled_before_issue_speculative_bytes']>0
    assert row['wasted_physical_speculative_bytes']==0

def test_partial_progress_is_nonexecutable_but_reduces_gap() -> None:
    _,base=M71.run(policy='no_prediction',ranked=True,m=4)
    _,prefetch=M71.run(policy='ranked_prefix',milestone='source_layer_start',ranked=True,m=4)
    no=next(x for x in base.layer_results[2].expert_timings if x['expert_id']==16)
    hit=next(x for x in prefetch.layer_results[2].expert_timings if x['expert_id']==16)
    assert hit['startup_readiness']==0 and 0<hit['startup_transfer_progress_at_router']<1
    assert hit['startup_gap_cycles']<no['startup_gap_cycles']

def test_critical_expert_metrics_match_realized_completion() -> None:
    row,result=breadth(4)
    critical=max((x for layer in (result.layer_results[1],result.layer_results[2]) for x in layer.expert_timings),key=lambda x:x['finish_cycle'])
    assert row['critical_expert_id']==critical['expert_id']
    assert row['critical_expert_startup_gap_cycles']==critical['startup_gap_cycles']
    assert row['critical_path_demand_blocking_cycles']<=row['aggregate_demand_request_blocking_cycles']
