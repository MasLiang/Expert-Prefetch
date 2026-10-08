"""M7.1c causal and metamorphic invariants for recall/readiness changes."""
from __future__ import annotations
import importlib.util
from pathlib import Path
SCRIPT=Path(__file__).parents[1]/'examples'/'m71_unified_characterization.py'
SPEC=importlib.util.spec_from_file_location('m71c_characterization',SCRIPT);assert SPEC and SPEC.loader
M71=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(M71)

def test_true_only_nested_coverage_is_no_contention_monotonic() -> None:
    values=[]
    for m in (1,2,3,4):values.append(M71.run(policy='uniform_prefix',quality='perfect',m=m,channels=4)[0]['sequence_makespan'])
    assert all(right<=left for left,right in zip(values,values[1:]))

def test_earlier_ready_and_more_true_depth_cannot_hurt_control() -> None:
    late,late_result=M71.run(policy='uniform_prefix',quality='perfect',m=4,channels=16,milestone='source_expert_done')
    early,early_result=M71.run(policy='uniform_prefix',quality='perfect',m=4,channels=16,milestone='source_layer_start')
    assert early['sequence_makespan']<=late['sequence_makespan']
    late_expert=next(x for x in late_result.layer_results[2].expert_timings if x['expert_id']==16);early_expert=next(x for x in early_result.layer_results[2].expert_timings if x['expert_id']==16)
    assert early_expert['start_cycle']<=late_expert['start_cycle'] and early_expert['finish_cycle']<=late_expert['finish_cycle']
    depth=[M71.run(policy='uniform_prefix',quality='perfect',m=4,channels=16,milestone='source_layer_start',candidate_depths=(d,)*4)[0]['sequence_makespan'] for d in (0,1,2,4)]
    assert all(right<=left for left,right in zip(depth,depth[1:]))

def test_oracle_memory_bounds_and_router_authority() -> None:
    results=[]
    for policy in ('no_prediction','uniform_prefix','full_expert_prefetch'):
        bandwidth=1e12 if policy=='full_expert_prefetch' else None
        row,result=M71.run(policy=policy,quality='perfect',m=4,channels=16,milestone='source_layer_start',trace_enabled=True,bandwidth_per_channel=bandwidth);results.append((row,result))
        assert all(x['start_cycle']>=layer.router_resolve_cycle for layer in result.layer_results.values() for x in layer.expert_timings)
    assert results[2][0]['sequence_makespan']<=results[1][0]['sequence_makespan']<=results[0][0]['sequence_makespan']
    assert results[1][0]['max_startup_gap_cycles']==0
    assert all(p.completed_speculative_bytes_before_router==p.initial_allocated_speculative_bytes for p in results[2][1].predictions)
    starts=[(e.get('layer_id'),e.get('expert_id'),e.get('matrix_type'),e.get('tile_id')) for e in results[2][1].trace if e.get('event_type')=='dma_start' and e.get('layer_id') in (1,2)]
    assert len(starts)==len(set(starts))

def test_e18_decomposition_and_critical_tie_are_consistent() -> None:
    starts=[];finishes=[]
    for m in (4,8,16):
        row,result=M71.run(policy='uniform_prefix',m=m,ranked=True,budget=65536,candidate_depths=tuple(16//m for _ in range(m)))
        layer=result.layer_results[2];x=next(x for x in layer.expert_timings if x['expert_id']==18)
        assert x['start_cycle']==max(layer.router_resolve_cycle,x['earliest_memory_ready_cycle'],x['aie_available_cycle'])
        assert x['aie_wait_cycles']==0 and x['finish_cycle']==layer.layer_done_cycle
        assert layer.critical_completion_expert_ids==(18,19)
        assert layer.layer_done_cycle-x['finish_cycle']>=0
        starts.append(x['start_cycle']);finishes.append(x['finish_cycle'])
    assert starts[2]<starts[1]<starts[0] and len(set(finishes))==1
