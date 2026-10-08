"""Regenerate M7.2 PNG/PDF figures from CSV only."""
from __future__ import annotations
import argparse,csv,json
from pathlib import Path
def read(p):
 with p.open(encoding='utf-8') as f:return list(csv.DictReader(f))
def main():
 p=argparse.ArgumentParser();p.add_argument('--input-dir',type=Path,required=True);a=p.parse_args();root=a.input_dir;fig=root/'figures';data=fig/'data';fig.mkdir(parents=True,exist_ok=True);data.mkdir(exist_ok=True)
 import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
 manifest=[];colors={'NO_PREFETCH':'black','FULL_EXPERT':'tab:blue','UNIFORM_PREFIX':'tab:orange','RANKED_PREFIX':'tab:green','BREADTH_DEPTH_SEARCH':'tab:red'}
 def save(name,rows,x,y,group,source,scatter=True):
  if not rows:return
  fields=sorted({key for row in rows for key in row})
  with (data/(name+'.csv')).open('w',newline='',encoding='utf-8') as f:w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
  figp,ax=plt.subplots(figsize=(6.4,4));groups={}
  for r in rows:groups.setdefault(r.get(group,'all'),[]).append(r)
  for label,g in groups.items():
   xs=[float(r[x]) for r in g];ys=[float(r[y]) for r in g]
   if scatter:ax.scatter(xs,ys,label=label,color=colors.get(label))
   else:
    order=sorted(range(len(xs)),key=xs.__getitem__);ax.plot([xs[i] for i in order],[ys[i] for i in order],marker='o',label=label)
  ax.set(xlabel=x.replace('_',' '),ylabel=y.replace('_',' '));ax.grid(alpha=.25);ax.legend(fontsize=7);figp.tight_layout();figp.savefig(fig/(name+'.png'),dpi=180);figp.savefig(fig/(name+'.pdf'));plt.close(figp)
  manifest.append({'figure_id':name,'source_csv':source,'hardware_mode':rows[0].get('hardware_mode','controlled_synthetic_hardware'),'x':x,'y':y,'grouping':group,'fixed_filters':'see exact plotting CSV','policy_set':'P0-P4','normalization_baseline':'no_prediction where applicable','trace_ids':sorted(set(r.get('predictor_trace_id','') for r in rows)),'png':str(fig/(name+'.png')),'pdf':str(fig/(name+'.pdf')),'plot_data':str(data/(name+'.csv'))})
 qopp=read(root/'phase/m72_phase_quality_opportunity.csv');save('fig01_policy_phase_quality_vs_opportunity',qopp,'opportunity','predictor_quality_scalar','winning_policy','phase/m72_phase_quality_opportunity.csv')
 qocm=read(root/'phase/m72_phase_quality_ocm.csv');save('fig02_policy_phase_quality_vs_ocm',qocm,'ocm_ratio','predictor_quality_scalar','winning_policy','phase/m72_phase_quality_ocm.csv')
 och=read(root/'phase/m72_phase_opportunity_channels.csv');save('fig03_policy_phase_opportunity_vs_channels',och,'opportunity','channels','winning_policy','phase/m72_phase_opportunity_channels.csv')
 searches=read(root/'search/m72_breadth_depth_search.csv');competitive=[r for r in searches if r.get('comparison_mode')=='unconstrained' and float(r['regret_percent'])<=.05];save('fig04_breadth_depth_strategy_map',competitive,'configuration_opportunity_expert_equivalents','predictor_quality_scalar','candidate_count','search/m72_breadth_depth_search.csv')
 evals=read(root/'raw/m72_policy_evaluations.csv');subset=[r for r in evals if r.get('comparison_mode')=='unconstrained' and abs(float(r['ocm_capacity_over_demand_reference'])-1.5)<1e-9 and r['memory_channels']=='2' and r['predictor_quality'] in ('perfect','medium') and r['policy'] in ('full_expert_prefetch','ranked_prefix','breadth_depth_search') and (r['policy']!='breadth_depth_search' or r['search_rank']=='1')]
 for r in subset:
  base=next(float(x['sequence_makespan']) for x in evals if x['comparison_id']==r['comparison_id'] and x['policy']=='no_prediction');r['speedup_vs_no_prefetch']=base/float(r['sequence_makespan']);r['plot_group']=r['predictor_quality']+' / '+r['policy']
 save('fig05_policy_speedup',subset,'configuration_opportunity_expert_equivalents','speedup_vs_no_prefetch','plot_group','raw/m72_policy_evaluations.csv',False)
 save('fig06_waste_vs_performance',subset,'wasted_physical_speculative_bytes','sequence_makespan','policy','raw/m72_policy_evaluations.csv')
 ocm=[r for r in evals if r.get('comparison_mode')=='unconstrained' and r['opportunity_setting']=='medium' and r['predictor_quality']=='medium' and r['memory_channels']=='2' and r['policy'] in ('no_prediction','full_expert_prefetch','ranked_prefix')];save('fig07_ocm_cost_vs_latency',ocm,'peak_ocm_bytes','sequence_makespan','policy','raw/m72_policy_evaluations.csv')
 save('fig08_startup_vs_streaming',subset,'critical_expert_startup_gap_cycles','sequence_makespan','policy','raw/m72_policy_evaluations.csv')
 regret=read(root/'tables/m72_heuristic_regret.csv');save('fig09_static_policy_regret',regret,'mean_regret_percent','max_regret_percent','heuristic','tables/m72_heuristic_regret.csv')
 vek=read(root/'vek280_theoretical/m72_vek280_theoretical.csv');save('fig10_vek280_theoretical_projection',vek,'predictor_quality_scalar','speedup_vs_no_prefetch','policy','vek280_theoretical/m72_vek280_theoretical.csv')
 (root/'figure_manifest_m72.json').write_text(json.dumps(manifest,indent=2)+'\n');print(f'Wrote {len(manifest)} M7.2 figures')
if __name__=='__main__':main()
