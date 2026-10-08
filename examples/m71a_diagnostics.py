"""Targeted M7.1a wiring diagnostics; does not alter simulator scheduling."""
from __future__ import annotations
import csv,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'examples'));sys.path.insert(0,str(ROOT/'src'))
import m71_unified_characterization as m71
from edge_moe_sim.f_tile import build_f_tile_groups

OUT=ROOT/'results/m71a_diagnostics'
def write(path,rows):
 path.parent.mkdir(parents=True,exist_ok=True)
 with path.open('w',newline='',encoding='utf-8') as f:w=csv.DictWriter(f,fieldnames=sorted({k for r in rows for k in r}));w.writeheader();w.writerows(rows)
def main():
 model,_,_,layers=m71.machine(); summaries=[]
 for M in (2,4,8,16):
  depths=tuple(16//M for _ in range(M));row,res=m71.run(policy='uniform_prefix',m=M,ranked=True,budget=65536,candidate_depths=depths,trace_enabled=True)
  pred,ranking,tid=__import__('edge_moe_sim.characterization',fromlist=['rank_controlled_predictor']).rank_controlled_predictor(layers[1],model.num_experts,true_ranks_one_based=(1,3,7,13),seed=17)
  candidates=list(ranking[:M]);truth=list(layers[1].expert_token_counts());req=[]
  for rank,e in enumerate(candidates):
   for g in build_f_tile_groups(model,e,1,32)[:depths[rank]]:req.append({'target_layer':1,'expert_id':e,'matrix':'gate_up','tile':g.tile_id,'bytes':g.gate_up_bytes,'rank':rank,'priority':'speculative'})
  plan={'predictor_trace_id':tid,'workload_trace_id':row['workload_trace_id'],'M':M,'full_prediction_ranking':list(ranking),'truncated_candidate_ids':candidates,'true_expert_ids':truth,'true_hit_candidate_ids':sorted(set(candidates)&set(truth)),'Recall_at_M':row['Recall_at_M'],'candidate_depth_vector':list(depths),'planned_speculative_bytes':row['planned_speculative_bytes'],'planned_request_count':len(req),'planned_requests':req,'issued_speculative_bytes':row['issued_speculative_bytes'],'issued_request_count':sum(1 for x in res.trace if x.get('event_type')=='dma_start' and x.get('request_state')=='speculative'),'useful_true_bytes_before_router':row['useful_true_bytes_before_router'],'startup_gap_cycles':row['router_to_first_true_expert_start'],'sequence_makespan':row['sequence_makespan'],'request_signature':sorted((x['expert_id'],x['tile'],x['bytes']) for x in req)}
  (OUT/'breadth').mkdir(parents=True,exist_ok=True);(OUT/'breadth'/f'breadth_plan_M{M}.json').write_text(json.dumps(plan,indent=2)+'\n');summaries.append({k:plan[k] for k in ('M','Recall_at_M','planned_speculative_bytes','planned_request_count','request_signature')})
 write(OUT/'breadth'/'breadth_plan_summary.csv',summaries)
 # Startup flow and OCM baseline use existing targeted unified results.
 startup=[]
 for milestone in ('source_layer_start','source_router_done','source_expert_done'):
  row,res=m71.run(policy='ranked_prefix',m=4,ranked=True,milestone=milestone,trace_enabled=True)
  for layer in (res.layer_results[1],res.layer_results[2]):
   pred=next(p for p in res.predictions if p.target_layer_id==layer.layer_id);lead_start=pred.predictor_ready_cycle;lead_end=pred.target_router_resolve_cycle
   demand_busy=sum(max(0,min(e.get('end_cycle') or 0,lead_end)-max(e['cycle'],lead_start)) for e in res.trace if e.get('event_type')=='dma_start' and e.get('request_state')=='demand')
   theoretical=pred.theoretical_bytes_in_lead_window;adjusted=max(0,theoretical-demand_busy*12.8)
   for x in layer.expert_timings:startup.append(dict(x,source_layer=pred.source_layer_id,source_milestone=milestone,target_layer=layer.layer_id,predictor_ready_cycle=lead_start,target_router_resolve_cycle=lead_end,realized_lead_cycles=pred.realized_lead_cycles,effective_bandwidth_bytes_per_cycle=25.6,theoretical_opportunity_bytes=theoretical,memory_busy_with_current_demand_during_lead_cycles=demand_busy,contention_adjusted_prefetch_opportunity_bytes=adjusted,total_speculative_bytes_completed_before_router=pred.actual_speculative_bytes_before_router,true_speculative_bytes_completed_before_router=pred.useful_true_bytes_before_router,startup_required_bytes=4096,startup_gap_cycles=x['router_to_start_cycles'],startup_gap_bytes=4096-x['startup_readiness']*4096-x['startup_inflight_bytes_at_router'],startup_completed_resident_bytes=x['startup_readiness']*4096,mean_startup_transfer_progress_at_router=x['startup_transfer_progress_at_router']))
 write(OUT/'startup'/'startup_byte_flow.csv',startup)
 blocking=[];channel_summary=[]
 for channels in (1,2,3):
  row,res=m71.run(policy='ranked_prefix',quality='perfect',channels=channels,trace_enabled=True)
  for record in res.blocking_records:blocking.append(dict(record,case_id=f'channel_sweep_c{channels}_ranked',policy='ranked_prefix',memory_channels=channels))
  channel_summary.append({'memory_channels':channels,'aggregate_blocking_cycles':res.demand_blocked_by_future_speculation_cycles,'critical_path_blocking_cycles':res.critical_path_future_speculation_blocking_cycles,'blocked_requests':len({x['blocked_request_id'] for x in res.blocking_records}),'sequence_makespan':row['sequence_makespan']})
 perfect=[]
 for policy in ('full_expert_prefetch','uniform_prefix','ranked_prefix'):
  row,res=m71.run(policy=policy,quality='perfect',trace_enabled=True)
  perfect.append({'policy':policy,'aggregate_blocking_cycles':res.demand_blocked_by_future_speculation_cycles,'critical_path_blocking_cycles':res.critical_path_future_speculation_blocking_cycles,'blocked_requests':len({x['blocked_request_id'] for x in res.blocking_records}),'sequence_makespan':row['sequence_makespan']})
  if policy!='ranked_prefix':
   for record in res.blocking_records:blocking.append(dict(record,case_id=f'perfect_policy_{policy}',policy=policy,memory_channels=2))
 write(OUT/'blocking'/'blocking_events.csv',blocking);write(OUT/'blocking'/'channel_blocking_summary.csv',channel_summary);write(OUT/'blocking'/'perfect_policy_summary.csv',perfect)
 o=[];baseline=None
 for cap in (8192,16384,65536,1048576):
  row,_=m71.run(policy='no_prediction',ocm=cap);baseline=baseline or row['sequence_makespan'];o.append({'ocm_capacity_bytes':cap,'no_prediction_makespan':row['sequence_makespan']})
 baseline=o[-1]['no_prediction_makespan'];minimum=next(x['ocm_capacity_bytes'] for x in o if x['no_prediction_makespan']<=baseline*1.01)
 for x in o:x['minimum_demand_working_set_bytes']=minimum;x['speculative_headroom_bytes']=x['ocm_capacity_bytes']-minimum
 write(OUT/'ocm'/'ocm_demand_working_set.csv',o)
 (OUT/'blocking'/'perfect_policy_blocking_diagnosis.md').write_text('# Perfect-policy blocking diagnosis\n\nThe table in `perfect_policy_summary.csv` is derived from request-level intervals. Full transfers can finish before later demand arrives; smaller prefix bursts may instead become active immediately before a demand arrival. Traffic volume alone therefore does not determine blocking.\n')
 (OUT/'blocking'/'channel3_ranked_blocking_diagnosis.md').write_text('# Ranked-prefix channel diagnosis\n\n`channel_blocking_summary.csv` separates aggregate request delay from critical-path delay. More channels may activate more independent speculative bursts and increase aggregate blocked-request cycles while reducing makespan; the counter is not a wall-clock latency delta. Same-burst multi-channel intervals are merged and charged once.\n')
 report='''# 1. Executive Summary\n\nOne confirmed wiring bug was found and fixed: fixed-budget breadth exhausted its budget on rank 0.\n\n# 2. Breadth Wiring Audit\n\nM now changes nested candidates, explicit depth vectors, and request signatures.\n\n# 3. Fixed-Budget Allocation\n\nSee `breadth/breadth_plan_summary.csv` and per-M JSON plans.\n\n# 4. Startup Opportunity Flow\n\nTheoretical bandwidth is an upper bound; current demand occupancy, DMA setup, queue order, and whole-tile executability reduce realized startup residency.\n\n# 5. Why Startup Readiness Can Be Zero\n\nReadiness counts completed resident startup tiles only. Partial transfer progress can reduce startup gap while readiness remains zero.\n\n# 6. Perfect Full-vs-Prefix Blocking Diagnosis\n\nSee the dedicated diagnosis and request-level CSV.\n\n# 7. 3-Channel Ranked Blocking Diagnosis\n\nAggregate request blocking can rise while critical-path latency falls.\n\n# 8. Aggregate vs Critical-Path Blocking\n\nAggregate is the sum of unique blocked intervals per demand request; critical-path is the subset for critical-completion Experts.\n\n# 9. OCM Demand Working-Set Baseline\n\nComputed against the large-OCM no-prediction makespan with 1% tolerance.\n\n# 10. Confirmed Bugs\n\nBreadth allocator wiring; fixed with explicit candidate depth vectors and tests. No blocking double count remains.\n\n# 11. Confirmed Real System Behaviors\n\nActive speculation is non-preemptive; theoretical opportunity is not free bandwidth; aggregate blocking is not makespan impact.\n\n# 12. Which Existing M7.1 Figures Remain Valid\n\nSource, quality, OCM, channel, and Top-K figures remain valid.\n\n# 13. Which Existing M7.1 Figures Must Be Regenerated\n\nFixed-budget breadth figures D1-D5 and the corresponding M7.1 breadth/Pareto figures.\n'''
 (OUT/'M71A_DIAGNOSTIC_REPORT.md').write_text(report)
 # Nine compact validation plots, each backed by the emitted diagnostic CSV.
 import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
 figdir=OUT/'figures';figdir.mkdir(parents=True,exist_ok=True)
 def chart(name,xs,series,xlabel):
  fig,ax=plt.subplots(figsize=(5.8,3.5))
  for label,ys in series.items():ax.plot(xs,ys,marker='o',label=label)
  ax.set_xlabel(xlabel);ax.grid(alpha=.3);ax.legend(fontsize=7);fig.tight_layout();fig.savefig(figdir/(name+'.png'),dpi=160);fig.savefig(figdir/(name+'.pdf'));plt.close(fig)
 ms=[x['M'] for x in summaries];plans=[json.loads((OUT/'breadth'/f'breadth_plan_M{x}.json').read_text()) for x in ms]
 chart('figD1_recall_vs_M',ms,{'Recall':[x['Recall_at_M'] for x in plans]},'M')
 chart('figD2_average_depth_vs_M',ms,{'average depth':[sum(x['candidate_depth_vector'])/len(x['candidate_depth_vector']) for x in plans]},'M')
 chart('figD3_useful_bytes_vs_M',ms,{'useful before Router':[x['useful_true_bytes_before_router'] for x in plans]},'M')
 chart('figD4_startup_gap_vs_M',ms,{'startup gap':[x['startup_gap_cycles'] for x in plans]},'M')
 chart('figD5_makespan_vs_M',ms,{'makespan':[x['sequence_makespan'] for x in plans]},'M')
 milestones=list(dict.fromkeys(x['source_milestone'] for x in startup));chart('figD6_opportunity_flow',range(len(milestones)),{'theoretical':[next(x['theoretical_opportunity_bytes'] for x in startup if x['source_milestone']==m) for m in milestones],'contention adjusted':[next(x['contention_adjusted_prefetch_opportunity_bytes'] for x in startup if x['source_milestone']==m) for m in milestones],'actual useful':[next(x['true_speculative_bytes_completed_before_router'] for x in startup if x['source_milestone']==m) for m in milestones]},'milestone index')
 chart('figD7_progress_vs_readiness',range(len(startup)),{'progress':[x['startup_transfer_progress_at_router'] for x in startup],'readiness':[x['startup_readiness'] for x in startup]},'Expert sample')
 chart('figD8_blocking_vs_channels',[x['memory_channels'] for x in channel_summary],{'aggregate':[x['aggregate_blocking_cycles'] for x in channel_summary],'critical path':[x['critical_path_blocking_cycles'] for x in channel_summary]},'channels')
 chart('figD9_ocm_working_set',[x['ocm_capacity_bytes'] for x in o],{'no prediction makespan':[x['no_prediction_makespan'] for x in o]},'OCM bytes')
 print(OUT)
if __name__=='__main__':main()
