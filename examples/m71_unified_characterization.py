"""Reproducible M7.1 unified cross-layer characterization (CSV-first)."""
from __future__ import annotations
import argparse, csv, json, sys
from dataclasses import replace
from pathlib import Path
from statistics import fmean

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from edge_moe_sim.advanced_schema import AdvancedExecutionConfig
from edge_moe_sim.batch_routing import BatchRoutingEvent
from edge_moe_sim.characterization import controlled_predictor, rank_controlled_predictor
from edge_moe_sim.hardware_profile import HardwareProfile
from edge_moe_sim.schema import ModelConfig
from edge_moe_sim.unified_schema import UnifiedLayerSequence, UnifiedPredictionSpec
from edge_moe_sim.unified_sequence_simulator import UnifiedSequenceSimulator

OUT=ROOT/'results/m71_unified_characterization'

def write(path:Path, rows:list[dict])->None:
    path.parent.mkdir(parents=True,exist_ok=True)
    if not rows: raise ValueError(f'empty M7.1 dataset: {path.name}')
    with path.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=sorted({k for r in rows for k in r}));w.writeheader();w.writerows(rows)

def machine(*,channels:int=2,ocm:int=1<<20,top_k:int=4,hardware_mode:str='controlled_synthetic_hardware',batch_size:int=1,routing_shape:str='high_reuse',placement_policy:str='round_robin_expert'):
    model=ModelConfig('m71_controlled',64,256,32,top_k,8,8)
    profile=HardwareProfile(name='controlled_synthetic_hardware',clock_hz=1_000_000_000,
        num_memory_channels=channels,bandwidth_bytes_per_sec_per_channel=16e9,
        bandwidth_efficiency=.8,dma_setup_cycles=40,max_inflight_per_channel=1,
        burst_bytes=4096,supports_channel_striping=True,total_aie_units=16,
        ops_per_cycle_per_aie=32,aie_efficiency=.75,aie_task_startup_cycles=20,
        ocm_bytes=ocm,hardware_tag='synthetic_placeholder')
    if hardware_mode=='vek280_theoretical':
        profile=HardwareProfile(name='vek280_theoretical_not_measured',clock_hz=1_000_000_000,num_memory_channels=channels,bandwidth_bytes_per_sec_per_channel=102.4e9/channels,bandwidth_efficiency=1.0,dma_setup_cycles=40,max_inflight_per_channel=1,burst_bytes=4096,supports_channel_striping=True,total_aie_units=304,ops_per_cycle_per_aie=202e12/304/1e9,aie_efficiency=1.0,aie_task_startup_cycles=20,ocm_bytes=ocm,physical_aie_tiles=304,hardware_tag='theoretical_not_measured')
    config=AdvancedExecutionConfig(32,placement_policy=placement_policy,aie_units_per_expert=4,max_parallel_experts=top_k,candidate_count=16,uniform_gu_prefetch_depth=1)
    routes=tuple(tuple(range(i*8,i*8+top_k)) for i in range(3))
    def token_routes(layer_id,route):
        if routing_shape=='high_reuse':return tuple(route for _ in range(batch_size))
        if routing_shape=='balanced':return tuple(tuple((layer_id*8+t*top_k+j)%model.num_experts for j in range(top_k)) for t in range(batch_size))
        if routing_shape=='skewed':
            alternate=tuple((x+top_k)%model.num_experts for x in route)
            return tuple(route if t<max(1,batch_size-1) else alternate for t in range(batch_size))
        raise ValueError(f'unknown routing_shape: {routing_shape}')
    layers=tuple(BatchRoutingEvent(7100+batch_size,0,i,token_routes(i,route)) for i,route in enumerate(routes))
    return model,profile,config,layers

def predictors(layers, model, quality:str, m:int, ranked:bool=False,true_ranks:tuple[int,...]|None=None,seed:int=17):
    result={}; trace=[]
    for source,target in ((0,1),(1,2)):
        if ranked:
            p,_,tid=rank_controlled_predictor(layers[target],model.num_experts,true_ranks_one_based=true_ranks or (1,3,7,13),seed=seed)
        else:
            p,tid,*_=controlled_predictor(layers[target],model.num_experts,profile=quality,candidate_count=m,ready_cycle=0,seed=seed)
        result[(source,target)]=p;trace.append(tid)
    return result,'|'.join(trace)

def run(*,policy:str, milestone:str='source_router_done', quality:str='perfect', m:int=4, channels:int=2, ocm:int=1<<20, top_k:int=4, budget:int|None=None, ranked:bool=False, hardware_mode:str='controlled_synthetic_hardware', trace_enabled:bool=False, candidate_depths:tuple[int,...]=(), bandwidth_per_channel:float|None=None,predictor_latency:int=20,true_ranks:tuple[int,...]|None=None,batch_size:int=1,routing_shape:str='high_reuse',placement_policy:str='round_robin_expert',predictor_seed:int=17):
    model,profile,config,layers=machine(channels=channels,ocm=ocm,top_k=top_k,hardware_mode=hardware_mode,batch_size=batch_size,routing_shape=routing_shape,placement_policy=placement_policy)
    if bandwidth_per_channel is not None:profile=replace(profile,bandwidth_bytes_per_sec_per_channel=bandwidth_per_channel)
    config=replace(config,candidate_count=m,speculative_byte_budget=budget)
    ps,trace_id=predictors(layers,model,quality,m,ranked,true_ranks,predictor_seed)
    specs=() if policy=='no_prediction' else tuple(UnifiedPredictionSpec(s,t,milestone,predictor_latency,m,policy,1 if policy!='ranked_prefix' else 8,candidate_depths=candidate_depths) for s,t in ((0,1),(1,2)))
    result=UnifiedSequenceSimulator(model,profile).simulate(UnifiedLayerSequence(layers),config,predictions=specs,predictors=ps,trace_enabled=trace_enabled)
    assert all(not p.synthetic_lead_time for p in result.predictions)
    predictions=result.predictions; targets=[result.layer_results[i] for i in (1,2)]
    true=set().union(*(set(x.true_expert_ids) for x in targets)); candidates=set().union(*(set(p.candidate_ids) for p in predictions))
    spec=sum(p.speculative_bytes for p in predictions); useful=sum(p.useful_bytes for p in predictions); waste=sum(p.wasted_bytes for p in predictions)
    row=dict(simulator_scope='unified_cross_layer',hardware_mode=hardware_mode,hardware_profile=profile.name,workload_trace_id=f'm72-controlled-3layer-k{top_k}-b{batch_size}-{routing_shape}',predictor_trace_id=trace_id,synthetic_lead_time=False,policy=policy,predictor_quality=quality,source_milestone=milestone,predictor_latency_cycles=predictor_latency,candidate_count=m,memory_channels=channels,ocm_bytes=ocm,top_k=top_k,batch_size=batch_size,routing_shape=routing_shape,memory_placement_policy=placement_policy,speculative_byte_budget=budget if budget is not None else '',sequence_makespan=result.inference_makespan_cycles,planned_speculative_bytes=spec,issued_speculative_bytes=spec,speculative_bytes=spec,useful_speculative_bytes=useful,wasted_speculative_bytes=waste,canceled_speculative_bytes=sum(p.canceled_bytes for p in predictions),useful_true_bytes_before_router=sum(p.useful_true_bytes_before_router for p in predictions),actual_speculative_bytes_before_router=sum(p.actual_speculative_bytes_before_router for p in predictions),realized_lead_cycles=fmean(p.realized_lead_cycles for p in predictions) if predictions else 0,prefetch_opportunity_expert_equivalents=fmean(p.prefetch_opportunity_expert_equivalents for p in predictions) if predictions else 0,weighted_startup_readiness=fmean(x.weighted_startup_readiness for x in targets),StartableRecall=fmean(x.startable_recall for x in targets),WeightedStartableRecall=fmean(x.weighted_startable_recall for x in targets),router_to_first_true_expert_start=fmean(x.first_true_expert_start-x.router_resolve_cycle for x in targets),router_to_last_true_expert_start=fmean(x.last_true_expert_start-x.router_resolve_cycle for x in targets),demand_blocked_by_future_speculation_cycles=result.demand_blocked_by_future_speculation_cycles,peak_ocm_bytes=result.peak_ocm_bytes,rejected_speculative_bytes=result.rejected_future_prefetch_bytes,evicted_speculative_bytes=result.evicted_future_speculative_bytes,memory_utilization=fmean(result.memory_metrics['per_channel_utilization']),aie_utilization=result.aie_metrics['board_aie_utilization'],Recall_at_M=len(candidates&true)/len(true) if true else 0,AnyHit=bool(candidates&true),FullHit=true<=candidates)
    timings=[dict(x,layer_id=layer.layer_id) for layer in targets for x in layer.expert_timings];critical=max(timings,key=lambda x:int(x['finish_cycle'] or 0));critical_prediction=next((p for p in predictions if p.target_layer_id==critical['layer_id']),None)
    critical_rank=(critical_prediction.candidate_ids.index(critical['expert_id'])+1) if critical_prediction and critical['expert_id'] in critical_prediction.candidate_ids else None
    critical_depth=0
    if critical_rank:
        if candidate_depths:critical_depth=candidate_depths[critical_rank-1]
        elif policy=='ranked_prefix':critical_depth=max(0,8-(critical_rank-1))
        elif policy=='uniform_prefix':critical_depth=1
        elif policy=='full_expert_prefetch':critical_depth=len(__import__('edge_moe_sim.f_tile',fromlist=['build_f_tile_groups']).build_f_tile_groups(model,critical['expert_id'],int(critical['token_count']),config.f_tile_size))
    total_tokens=sum(int(x['token_count']) for x in timings)
    row.update(speculative_budget_bytes=budget if budget is not None else '',initial_allocated_speculative_bytes=sum(p.initial_allocated_speculative_bytes for p in predictions),cancelled_before_issue_speculative_bytes=sum(p.cancelled_before_issue_speculative_bytes for p in predictions),pending_unissued_speculative_bytes=sum(p.pending_unissued_speculative_bytes for p in predictions),issued_speculative_bytes=sum(p.issued_speculative_bytes for p in predictions),completed_speculative_bytes=sum(p.completed_speculative_bytes for p in predictions),completed_speculative_bytes_before_router=sum(p.completed_speculative_bytes_before_router for p in predictions),partial_inflight_speculative_bytes_at_router=sum(p.partial_inflight_speculative_bytes_at_router for p in predictions),physical_speculative_bytes_transferred_by_router=sum(p.physical_speculative_bytes_transferred_by_router for p in predictions),useful_completed_speculative_bytes_before_router=sum(p.useful_completed_speculative_bytes_before_router for p in predictions),useful_physical_speculative_bytes_by_router=sum(p.useful_physical_speculative_bytes_by_router for p in predictions),wasted_physical_speculative_bytes=sum(p.wasted_physical_speculative_bytes for p in predictions),theoretical_prefetch_opportunity_bytes=sum(p.theoretical_bytes_in_lead_window for p in predictions),physical_opportunity_utilization=(sum(p.physical_speculative_bytes_transferred_by_router for p in predictions)/sum(p.theoretical_bytes_in_lead_window for p in predictions)) if predictions and sum(p.theoretical_bytes_in_lead_window for p in predictions) else 0,useful_physical_opportunity_utilization=(sum(p.useful_physical_speculative_bytes_by_router for p in predictions)/sum(p.theoretical_bytes_in_lead_window for p in predictions)) if predictions and sum(p.theoretical_bytes_in_lead_window for p in predictions) else 0,mean_startup_transfer_progress=fmean(float(x['startup_transfer_progress_at_router']) for x in timings),weighted_startup_transfer_progress=sum(float(x['startup_transfer_progress_at_router'])*int(x['token_count']) for x in timings)/total_tokens,mean_startup_gap_cycles=fmean(float(x['startup_gap_cycles'] or 0) for x in timings),weighted_startup_gap_cycles=sum(float(x['startup_gap_cycles'] or 0)*int(x['token_count']) for x in timings)/total_tokens,max_startup_gap_cycles=max(float(x['startup_gap_cycles'] or 0) for x in timings),total_true_expert_memory_starvation_cycles=sum(int(x['memory_wait_cycles']) for x in timings),critical_expert_starvation_cycles=int(critical['memory_wait_cycles']),weight_induced_stalls_after_expert_start='not_derivable_from_current_events',critical_layer_id=critical['layer_id'],critical_expert_id=critical['expert_id'],critical_expert_predictor_rank=critical_rank if critical_rank is not None else '',critical_expert_predicted=critical_rank is not None,critical_expert_prefix_depth=critical_depth,critical_expert_startup_readiness=critical['startup_readiness'],critical_expert_startup_transfer_progress=critical['startup_transfer_progress_at_router'],critical_expert_startup_gap_bytes=critical['startup_gap_bytes'],critical_expert_startup_gap_cycles=critical['startup_gap_cycles'],critical_expert_router_to_start_cycles=critical['router_to_start_cycles'],critical_expert_finish_cycle=critical['finish_cycle'],aggregate_demand_request_blocking_cycles=result.demand_blocked_by_future_speculation_cycles,critical_path_demand_blocking_cycles=result.critical_path_future_speculation_blocking_cycles)
    return row,result

def main()->int:
    ap=argparse.ArgumentParser();ap.add_argument('--smoke',action='store_true');ap.add_argument('--full',action='store_true');ap.add_argument('--experiment',choices=['source','quality','breadth','ocm','memory','topk']);ap.add_argument('--output-dir',type=Path,default=OUT);a=ap.parse_args();raw=a.output_dir/'raw'; tables=a.output_dir/'tables';raw.mkdir(parents=True,exist_ok=True)
    selected={a.experiment} if a.experiment else {'source','quality','breadth','ocm','memory','topk'}
    if a.smoke:selected={'source','quality','breadth','ocm'}
    datasets={}
    if 'source' in selected:
        datasets['source_milestone_study.csv']=[run(policy='uniform_prefix',milestone=x)[0] for x in ('source_layer_start','source_router_done','source_expert_start','source_gate_up_done','source_expert_done')]
    if 'quality' in selected:
        datasets['perfect_prediction_policy_baseline.csv']=[run(policy=x,quality='perfect')[0] for x in ('no_prediction','full_expert_prefetch','uniform_prefix','ranked_prefix')]
        qs=('perfect','high') if a.smoke else ('perfect','high','medium','low','wrong')
        datasets['predictor_quality_policy_study.csv']=[run(policy=p,quality=q)[0] for q in qs for p in ('no_prediction','full_expert_prefetch','uniform_prefix','ranked_prefix')]
    if 'breadth' in selected:
        ms=(2,4) if a.smoke else (2,4,8,16)
        datasets['unified_breadth_fixed_depth.csv']=[dict(run(policy='uniform_prefix',m=m,ranked=True)[0],candidate_depth_vector='['+','.join(['1']*m)+']') for m in ms]
        # One Gate-Up F tile is 2 * H * F_tile * INT8 bytes = 4096 bytes.
        budget=2*16*2*64*32
        # The unified engine enforces a budget per source->target prediction;
        # split the study-level budget evenly across its two prediction edges.
        def budget_row(m):
            depths=tuple(16//m for _ in range(m))
            row,_=run(policy='uniform_prefix',m=m,ranked=True,budget=budget//2,candidate_depths=depths)
            return dict(row,speculative_byte_budget=budget,candidate_depth_vector=str(list(depths)),unused_budget_bytes=max(0,budget-row['planned_speculative_bytes']))
        datasets['unified_breadth_fixed_budget.csv']=[budget_row(m) for m in ms]
    if 'ocm' in selected:
        caps=(8192,) if a.smoke else (8192,16384,65536,1<<20)
        datasets['unified_ocm_pressure.csv']=[run(policy=p,ocm=c)[0] for c in caps for p in ('no_prediction','full_expert_prefetch','uniform_prefix','ranked_prefix')]
    if 'memory' in selected: datasets['unified_memory_channel_study.csv']=[run(policy=p,channels=c)[0] for c in (1,2,3) for p in ('no_prediction','full_expert_prefetch','uniform_prefix','ranked_prefix')]
    if 'topk' in selected: datasets['unified_topk_study.csv']=[run(policy=p,top_k=k,m=min(16,max(4,k)))[0] for k in (1,2,4,6) for p in ('no_prediction','full_expert_prefetch','ranked_prefix')]
    for name,rows in datasets.items():write(raw/name,rows);write(tables/name,rows)
    (a.output_dir/'traces').mkdir(exist_ok=True)
    trace_runs=[('perfect_full_expert','full_expert_prefetch','perfect'),('imperfect_full_expert','full_expert_prefetch','wrong'),('imperfect_ranked_prefix','ranked_prefix','wrong')]
    exemplar=None
    for name,policy,quality_name in trace_runs:
        _,traced=run(policy=policy,quality=quality_name,trace_enabled=True)
        (a.output_dir/'traces'/(name+'.json')).write_text(json.dumps(traced.trace,indent=2)+'\n')
        if name=='imperfect_ranked_prefix':exemplar=traced
    assert exemplar is not None;result=exemplar
    timings=[]
    for layer in result.layer_results.values():
        for timing in layer.expert_timings:timings.append(dict(simulator_scope='unified_cross_layer',policy='ranked_prefix',predictor_quality='wrong',layer_id=layer.layer_id,**timing))
    write(raw/'per_expert_timing.csv',timings);write(tables/'per_expert_timing.csv',timings)
    vek=[run(policy=p,hardware_mode='vek280_theoretical')[0] for p in ('no_prediction','full_expert_prefetch','uniform_prefix','ranked_prefix')]
    write(raw/'vek280_theoretical_policy_baseline.csv',vek);write(tables/'vek280_theoretical_policy_baseline.csv',vek)
    # Derived explanation and expert timing are emitted whenever quality data exists.
    quality=datasets.get('predictor_quality_policy_study.csv',[])
    if quality:
        by={(r['predictor_quality'],r['policy']):r for r in quality}; expl=[]
        for q in sorted({r['predictor_quality'] for r in quality}):
            full,prefix,no=(by[(q,x)] for x in ('full_expert_prefetch','ranked_prefix','no_prediction'))
            label='full_best_exact_prediction' if full['sequence_makespan']<=prefix['sequence_makespan'] and q=='perfect' else ('prefix_bandwidth_advantage' if prefix['sequence_makespan']<full['sequence_makespan'] else 'mixed / inconclusive')
            expl.append(dict(configuration=q,full_makespan=full['sequence_makespan'],prefix_makespan=prefix['sequence_makespan'],no_prediction_makespan=no['sequence_makespan'],latency_delta_prefix_vs_full=prefix['sequence_makespan']-full['sequence_makespan'],byte_delta=prefix['speculative_bytes']-full['speculative_bytes'],waste_delta=prefix['wasted_speculative_bytes']-full['wasted_speculative_bytes'],OCM_delta=prefix['peak_ocm_bytes']-full['peak_ocm_bytes'],blocking_delta=prefix['demand_blocked_by_future_speculation_cycles']-full['demand_blocked_by_future_speculation_cycles'],interpretation=label))
        write(raw/'full_vs_prefix_explanation.csv',expl);write(tables/'full_vs_prefix_explanation.csv',expl)
    (a.output_dir/'summaries').mkdir(exist_ok=True);(a.output_dir/'summaries'/'run_metadata.json').write_text(json.dumps({'simulator_scope':'unified_cross_layer','synthetic_prediction_ready_override':False,'datasets':sorted(datasets)},indent=2)+'\n')
    import subprocess;subprocess.run([sys.executable,str(ROOT/'examples/m71_make_figures.py'),'--input-dir',str(raw)],check=True)
    print(f'Wrote M7.1 artifacts to {a.output_dir}');return 0
if __name__=='__main__':raise SystemExit(main())
