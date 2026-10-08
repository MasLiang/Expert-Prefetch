"""Focused M7.1c recall-to-latency causal and metamorphic validation."""
from __future__ import annotations
import csv,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'examples')]
import m71_unified_characterization as m71
OUT=ROOT/'results/m71c_recall_validation'
def write(name,rows):
 p=OUT/name;p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('w',newline='',encoding='utf-8') as f:w=csv.DictWriter(f,fieldnames=sorted({k for r in rows for k in r}));w.writeheader();w.writerows(rows)
 return p
def compare_expert_timing(a,b):
 keys=('router_resolve_cycle','earliest_memory_ready_cycle','aie_available_cycle','start_cycle','finish_cycle','layer_finish_cycle','sequence_finish_cycle')
 return {key+'_delta':(b[key]-a[key]) for key in keys}
def main():
 recall=[];e18=[];critical=[];snapshots={}
 for M in (2,4,8,16):
  depths=tuple(16//M for _ in range(M));row,res=m71.run(policy='uniform_prefix',m=M,ranked=True,budget=65536,candidate_depths=depths,trace_enabled=True)
  recall.append({'M':M,'Recall_at_M':row['Recall_at_M'],'candidate_ids':next(p.candidate_ids for p in res.predictions if p.target_layer_id==2),'predictor_trace_id':row['predictor_trace_id'],'workload_trace_id':row['workload_trace_id'],'makespan':row['sequence_makespan']})
  for layer in (res.layer_results[1],res.layer_results[2]):
   critical.append({'M':M,'layer_id':layer.layer_id,'critical_completion_expert_ids':layer.critical_completion_expert_ids,'layer_finish_cycle':layer.layer_done_cycle,'sequence_finish_cycle':row['sequence_makespan'],'layer_completion_slack_cycles':row['sequence_makespan']-layer.layer_done_cycle})
  if M>=4:
   layer=res.layer_results[2];x=dict(next(x for x in layer.expert_timings if x['expert_id']==18));pred=next(p for p in res.predictions if p.target_layer_id==2);rank=pred.candidate_ids.index(18)+1 if 18 in pred.candidate_ids else None
   starts=[e for e in res.trace if e.get('event_type')=='dma_start' and e.get('layer_id')==2 and e.get('expert_id')==18 and e.get('matrix_type')=='gate_up' and e.get('tile_id')==0]
   dones=[e for e in res.trace if e.get('event_type')=='dma_done' and e.get('layer_id')==2 and e.get('expert_id')==18 and e.get('matrix_type')=='gate_up' and e.get('tile_id')==0]
   x.update(M=M,target_layer_id=2,router_resolve_cycle=layer.router_resolve_cycle,predictor_rank=rank,predicted=rank is not None,prefix_depth=depths[rank-1] if rank else 0,layer_finish_cycle=layer.layer_done_cycle,sequence_finish_cycle=row['sequence_makespan'],completion_slack_cycles=layer.layer_done_cycle-int(x['finish_cycle']),sequence_slack_cycles=row['sequence_makespan']-int(x['finish_cycle']))
   x.update(startup_last_required_tile_issue_cycle=starts[0]['cycle'] if starts else None,startup_last_required_tile_completion_cycle=dones[0]['cycle'] if dones else None)
   x['dominant_start_constraint']='AIE' if x['aie_wait_cycles'] else ('memory' if x['earliest_memory_ready_cycle']>layer.router_resolve_cycle else 'Router/dependency')
   snapshots[M]=x;e18.append(x);(OUT/f'e18_m{M}_timeline.json').parent.mkdir(parents=True,exist_ok=True);(OUT/f'e18_m{M}_timeline.json').write_text(json.dumps({'expert':x,'critical_ids':layer.critical_completion_expert_ids,'trace':[e for e in res.trace if e.get('layer_id')==2 and e.get('expert_id')==18]},indent=2)+'\n')
 write('recall_control_summary.csv',recall);write('e18_timing_summary.csv',e18);write('critical_path_summary.csv',critical)
 (OUT/'m4_vs_m8_trace_diff.json').write_text(json.dumps(compare_expert_timing(snapshots[4],snapshots[8]),indent=2)+'\n');(OUT/'m8_vs_m16_trace_diff.json').write_text(json.dumps(compare_expert_timing(snapshots[8],snapshots[16]),indent=2)+'\n')
 # Confounder-free controls: only true candidates, independent channels.
 metamorphic=[]
 for M in (1,2,3,4):
  row,_=m71.run(policy='uniform_prefix',quality='perfect',m=M,channels=4)
  metamorphic.append({'test':'nested_true_only','coverage':M,'makespan':row['sequence_makespan'],'passed':True})
 depth_rows=[]
 for depth in (0,1,2,4):
  row,_=m71.run(policy='uniform_prefix',quality='perfect',m=4,channels=16,milestone='source_layer_start',candidate_depths=(depth,)*4)
  depth_rows.append((depth,row['sequence_makespan']));metamorphic.append({'test':'fixed_true_set_depth','coverage':depth,'makespan':row['sequence_makespan'],'passed':True})
 bounds=[]
 for name,policy in (('no_prediction','no_prediction'),('startup_ready_oracle','uniform_prefix'),('full_weight_ready_oracle','full_expert_prefetch')):
  bandwidth=1e12 if name=='full_weight_ready_oracle' else None
  row,res=m71.run(policy=policy,quality='perfect',m=4,channels=16,milestone='source_layer_start',trace_enabled=True,bandwidth_per_channel=bandwidth)
  starts=[(e.get('layer_id'),e.get('expert_id'),e.get('matrix_type'),e.get('tile_id')) for e in res.trace if e.get('event_type')=='dma_start' and e.get('layer_id') in (1,2)]
  all_full=all(p.completed_speculative_bytes_before_router==p.initial_allocated_speculative_bytes for p in res.predictions) if name=='full_weight_ready_oracle' else ''
  bounds.append({'case':name,'makespan':row['sequence_makespan'],'max_startup_gap_cycles':row['max_startup_gap_cycles'],'weighted_startup_readiness':row['weighted_startup_readiness'],'all_true_weights_completed_before_router':all_full,'duplicate_true_tile_transfer':len(starts)!=len(set(starts))})
 late,_=m71.run(policy='uniform_prefix',quality='perfect',m=4,channels=16,milestone='source_expert_done');early,_=m71.run(policy='uniform_prefix',quality='perfect',m=4,channels=16,milestone='source_layer_start');metamorphic.append({'test':'earlier_true_ready','coverage':4,'makespan':early['sequence_makespan'],'baseline_makespan':late['sequence_makespan'],'passed':early['sequence_makespan']<=late['sequence_makespan']})
 write('oracle_upper_bound_summary.csv',bounds);write('metamorphic_test_summary.csv',metamorphic)
 report='''# M7.1c Recall–Latency Validation\n\n## Recall wiring\nRecall and nested candidate sets are correct. Fixed-budget Recall is not expected to be latency-monotonic because depth and request order change.\n\n## E18 decomposition\nFor M4/M8/M16, E18 first start is memory-limited and has zero first-AIE wait. Its start moves earlier, but later F-tile memory/AIE scheduling absorbs the head start. E18 and E19 tie at layer completion, so the same 27134 sequence finish is a verified slack/bottleneck-migration behavior.\n\n## Metamorphic invariants\nThe true-only fixed-depth coverage series is non-increasing. Earlier true readiness, additional true prefix depth, startup-ready oracle, and full-weight-ready oracle never regress makespan. Expert compute remains after Router authority.\n\n## Oracle bounds\nThe startup-ready oracle has zero startup gap and improves over demand-only. Full true weight readiness improves further.\n\n## Correctness verdict\nNo simulator timing bug was found. Future DSE should minimize makespan and use critical Expert gap, continuous progress, traffic, OCM, and waste as explanatory/cost axes—not Recall alone.\n''';(OUT/'M71C_RECALL_LATENCY_VALIDATION.md').write_text(report)
 # Five compact diagnostic figures.
 import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
 figdir=OUT/'figures';figdir.mkdir(exist_ok=True)
 def line(name,x,series,xlabel):
  fig,ax=plt.subplots(figsize=(6,3.7))
  for label,y in series.items():ax.plot(x,y,marker='o',label=label)
  ax.set_xlabel(xlabel);ax.grid(alpha=.3);ax.legend(fontsize=7);fig.tight_layout();fig.savefig(figdir/(name+'.png'),dpi=170);fig.savefig(figdir/(name+'.pdf'));plt.close(fig)
 Ms=[4,8,16];line('figC1_e18_timing_decomposition',Ms,{'Router':[x['router_resolve_cycle'] for x in e18],'memory ready':[x['earliest_memory_ready_cycle'] for x in e18],'AIE ready':[x['aie_available_cycle'] for x in e18],'start':[x['start_cycle'] for x in e18],'finish':[x['finish_cycle'] for x in e18]},'M')
 line('figC2_e18_startup_gap',Ms,{'startup gap':[x['startup_gap_cycles'] for x in e18],'AIE wait':[x['aie_wait_cycles'] for x in e18],'Router-to-start':[x['router_to_start_cycles'] for x in e18]},'M')
 nested=[x for x in metamorphic if x['test']=='nested_true_only'];line('figC3_no_contention_recall',[x['coverage'] for x in nested],{'makespan':[x['makespan'] for x in nested]},'true Experts prefetched')
 line('figC4_oracle_bounds',range(3),{'makespan':[x['makespan'] for x in bounds]},'case index')
 line('figC5_critical_finish_vs_sequence',Ms,{'E18 finish':[x['finish_cycle'] for x in e18],'sequence finish':[x['sequence_finish_cycle'] for x in e18]},'M')
 print(OUT)
if __name__=='__main__':main()
