"""Generate only M7.1b metric-cleanup datasets and validation figures."""
from __future__ import annotations
import csv,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'examples')]
import m71_unified_characterization as m71
OUT=ROOT/'results/m71b_metric_cleanup'
def write(name,rows):
 p=OUT/name;p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('w',newline='',encoding='utf-8') as f:w=csv.DictWriter(f,fieldnames=sorted({k for r in rows for k in r}));w.writeheader();w.writerows(rows)
 return p
def main():
 breadth=[];expert=[]
 for M in (2,4,8,16):
  depths=tuple(16//M for _ in range(M));row,res=m71.run(policy='uniform_prefix',m=M,ranked=True,budget=65536,candidate_depths=depths)
  row.update(speculative_budget_bytes=131072,candidate_depth_vector=str(list(depths)),unused_allocated_budget_bytes=131072-row['initial_allocated_speculative_bytes']);breadth.append(row)
  for layer in (res.layer_results[1],res.layer_results[2]):
   for x in layer.expert_timings:expert.append(dict(x,experiment='breadth',M=M,layer_id=layer.layer_id,policy='uniform_prefix'))
 source=[m71.run(policy='ranked_prefix',m=4,ranked=True,milestone=x)[0] for x in ('source_layer_start','source_router_done','source_expert_start','source_gate_up_done','source_expert_done')]
 quality=[m71.run(policy=p,quality=q)[0] for q in ('perfect','high','medium','low','wrong') for p in ('no_prediction','full_expert_prefetch','uniform_prefix','ranked_prefix')]
 memory=[m71.run(policy='ranked_prefix',channels=c)[0] for c in (1,2,3)]
 ocm=[m71.run(policy=p,ocm=c)[0] for c in (8192,16384,65536,1048576) for p in ('no_prediction','full_expert_prefetch','uniform_prefix','ranked_prefix')]
 large=next(r['sequence_makespan'] for r in ocm if r['policy']=='no_prediction' and r['ocm_bytes']==1048576);ref=next(c for c in (8192,16384,65536,1048576) if next(r['sequence_makespan'] for r in ocm if r['policy']=='no_prediction' and r['ocm_bytes']==c)<=large*1.01)
 for r in ocm:r.update(demand_working_set_reference_bytes=ref,ocm_capacity_over_demand_reference=r['ocm_bytes']/ref,speculative_headroom_bytes=max(0,r['ocm_bytes']-ref))
 write('m71b_breadth_fixed_budget.csv',breadth);write('m71b_source_milestone_metrics.csv',source);write('m71b_predictor_quality_metrics.csv',quality);write('m71b_memory_channel_metrics.csv',memory);write('m71b_ocm_metrics.csv',ocm);write('m71b_per_expert_startup.csv',expert)
 definitions=[
 ('initial_allocated_speculative_bytes','prediction','policy-selected bytes before admission/cancellation','byte','no','no','no','no','budget allocation','planned_speculative_bytes'),
 ('cancelled_before_issue_speculative_bytes','prediction','queued speculative bytes cancelled before service','byte','no','yes','no','no','cancellation cost','canceled_speculative_bytes'),
 ('pending_unissued_speculative_bytes','prediction','allocated bytes neither cancelled nor physically issued at drain','byte','no','no','no','no','conservation'),
 ('issued_speculative_bytes','prediction','whole request bytes counted once when memory service starts','byte','no','no','no','yes','physical service commitment','speculative_bytes'),
 ('completed_speculative_bytes_before_router','prediction','whole requests completed by target Router resolve','byte','no','no','no','yes','completed Router-time traffic','actual_speculative_bytes_before_router'),
 ('partial_inflight_speculative_bytes_at_router','prediction','payload bytes already transferred for active requests at Router','byte','yes','no','no','yes','partial progress',''),
 ('physical_speculative_bytes_transferred_by_router','prediction','completed-before-Router plus partial active payload','byte','yes','no','no','yes','opportunity utilization',''),
 ('useful_physical_speculative_bytes_by_router','prediction','physical Router-time bytes later selected true','byte','yes','yes','no','yes','useful progress',''),
 ('wasted_physical_speculative_bytes','prediction','physically transferred bytes later determined false','byte','no','yes','no','yes','waste axis','wasted_speculative_bytes'),
 ('startup_readiness','expert','completed resident startup bytes / required bytes','ratio','no','yes','yes','no','immediate executability',''),
 ('startup_transfer_progress','expert','physical startup bytes transferred by Router / required','ratio','yes','yes','no','yes','continuous progress',''),
 ('startup_gap_cycles','expert','event-derived Expert start minus Router resolve','cycle','yes','yes','no','no','startup delay',''),
 ('aggregate_demand_request_blocking_cycles','sequence','sum of unique blocked intervals per demand request','cycle','no','no','no','no','interference explanation','demand_blocked_by_future_speculation_cycles'),
 ('critical_path_demand_blocking_cycles','sequence','aggregate blocking subset for critical completion Experts','cycle','no','yes','no','no','critical-path explanation','critical_path_future_speculation_blocking_cycles')]
 write('m71b_metric_definitions.csv',[dict(zip(('metric_name','level','exact_definition','unit','includes_partial_transfer','requires_router_truth','executable_metric','physical_traffic_metric','recommended_use','deprecated_alias'),x)) for x in definitions])
 # Narrow M2/M4 causal diagnosis.
 (OUT/'m2_vs_m4_causal_diagnosis.md').write_text(f"# M2 versus M4\n\nM2 completes {breadth[0]['useful_completed_speculative_bytes_before_router']} useful bytes before Router but makespan is {breadth[0]['sequence_makespan']}; M4 completes {breadth[1]['useful_completed_speculative_bytes_before_router']} and makespan is {breadth[1]['sequence_makespan']}. Completion bytes are not ordered by critical-Expert rank/startup gap; the per-Expert CSV identifies the realized critical completion Expert.\n")
 figures=OUT/'figures';figures.mkdir(exist_ok=True)
 import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
 def plot(name,x,series,xlabel):
  fig,ax=plt.subplots(figsize=(6,3.7))
  for label,y in series.items():ax.plot(x,y,marker='o',label=label)
  ax.set_xlabel(xlabel);ax.grid(alpha=.3);ax.legend(fontsize=7);fig.tight_layout();fig.savefig(figures/(name+'.png'),dpi=170);fig.savefig(figures/(name+'.pdf'));plt.close(fig)
 Ms=[r['candidate_count'] for r in breadth];plot('fig1_allocation_lifecycle',Ms,{'budget':[r['speculative_budget_bytes'] for r in breadth],'allocated':[r['initial_allocated_speculative_bytes'] for r in breadth],'issued':[r['issued_speculative_bytes'] for r in breadth],'physical by Router':[r['physical_speculative_bytes_transferred_by_router'] for r in breadth]},'M')
 plot('fig2_readiness_vs_progress',[x['startup_transfer_progress_at_router'] for x in expert],{'readiness':[x['startup_readiness'] for x in expert]},'startup transfer progress')
 plot('fig3_gap_vs_progress',[x['startup_transfer_progress_at_router'] for x in expert],{'gap cycles':[x['startup_gap_cycles'] for x in expert]},'startup transfer progress')
 configs=breadth+source;plot('fig4_critical_gap_vs_makespan',[r['critical_expert_startup_gap_cycles'] for r in configs],{'makespan':[r['sequence_makespan'] for r in configs]},'critical Expert startup gap cycles')
 plot('fig5_blocking_by_channels',[r['memory_channels'] for r in memory],{'aggregate':[r['aggregate_demand_request_blocking_cycles'] for r in memory],'critical':[r['critical_path_demand_blocking_cycles'] for r in memory]},'memory channels')
 print(OUT)
if __name__=='__main__':main()
