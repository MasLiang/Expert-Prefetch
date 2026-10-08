"""Deterministic M7.1 characterization-plumbing checks."""
from __future__ import annotations
import importlib.util
from pathlib import Path

SCRIPT=Path(__file__).parents[1]/'examples'/'m71_unified_characterization.py'
SPEC=importlib.util.spec_from_file_location('m71_characterization',SCRIPT)
assert SPEC and SPEC.loader
M71=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(M71)

def test_m71_rows_are_unified_and_lead_is_derived() -> None:
    early,_=M71.run(policy='uniform_prefix',milestone='source_layer_start')
    late,_=M71.run(policy='uniform_prefix',milestone='source_expert_done')
    assert early['simulator_scope']=='unified_cross_layer'
    assert early['synthetic_lead_time'] is False
    assert early['realized_lead_cycles']>late['realized_lead_cycles']

def test_m71_ranked_breadth_and_budget_are_deterministic() -> None:
    rows=[M71.run(policy='ranked_prefix',m=m,ranked=True,budget=16384)[0] for m in (2,4,8,16)]
    assert [row['Recall_at_M'] for row in rows]==[.25,.5,.75,1.0]
    assert all(row['planned_speculative_bytes']<=32768 for row in rows)
    assert len({row['predictor_trace_id'] for row in rows})==1

def test_m71_explicit_fixed_budget_depth_decreases_and_changes_requests() -> None:
    rows=[]
    for m in (2,4,8,16):
        depths=tuple(16//m for _ in range(m))
        row,result=M71.run(policy='uniform_prefix',m=m,ranked=True,budget=65536,candidate_depths=depths,trace_enabled=True)
        signature={(x.get('layer_id'),x.get('expert_id'),x.get('tile_id')) for x in result.trace if x.get('event_type')=='dma_start' and x.get('request_state')=='speculative'}
        rows.append((depths,signature,row))
    assert sum(rows[0][0])/len(rows[0][0]) > sum(rows[-1][0])/len(rows[-1][0])
    assert len({repr(x[1]) for x in rows})>1
    assert all(x[2]['planned_speculative_bytes']<=131072 for x in rows)

def test_m71_blocking_records_do_not_double_charge_multichannel_burst() -> None:
    _,result=M71.run(policy='ranked_prefix',quality='perfect',channels=3,trace_enabled=True)
    keys=[(x['blocked_request_id'],x['blocking_request_id'],x['block_start_cycle'],x['block_end_cycle']) for x in result.blocking_records]
    assert len(keys)==len(set(keys))
    assert sum(x['blocked_cycles'] for x in result.blocking_records)==result.demand_blocked_by_future_speculation_cycles
    assert result.critical_path_future_speculation_blocking_cycles<=result.demand_blocked_by_future_speculation_cycles
