"""Strict, CSV-only V4 figure regeneration with duplicate-x protection."""
from __future__ import annotations
import argparse,csv,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from edge_moe_sim.plot_validation import assert_unique_x_per_line,require_plot_data
def read(p):
 with p.open(encoding='utf-8') as f:return list(csv.DictReader(f))
def line(rows,x,y,name,out,debug,group='predictor_profile',annotation=None):
 import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
 require_plot_data(rows,[x,y,group]);assert_unique_x_per_line(rows,x_column=x,line_group_columns=[group],figure_name=name,debug_dir=debug)
 out.mkdir(parents=True,exist_ok=True);fig,ax=plt.subplots(figsize=(6.4,4));groups={}
 for r in rows:groups.setdefault(r[group],[]).append(r)
 for label,g in groups.items():
  g.sort(key=lambda r:float(r[x]));ax.plot([float(r[x]) for r in g],[float(r[y]) for r in g],marker='o',label=label)
  if annotation:
   for r in g:ax.annotate(str(r.get(annotation,'')),(float(r[x]),float(r[y])),fontsize=7)
 ax.set_xlabel(x.replace('_',' ') + (' (count)' if x=='candidate_count' else ''));ax.set_ylabel(y.replace('_',' '));ax.grid(alpha=.3);ax.legend();fig.tight_layout();fig.savefig(out/(name+'.png'),dpi=180);fig.savefig(out/(name+'.pdf'));plt.close(fig)
def scatter(rows,x,y,name,out,debug,annotation):
 import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
 require_plot_data(rows,[x,y]);out.mkdir(parents=True,exist_ok=True);fig,ax=plt.subplots(figsize=(6.4,4));ax.scatter([float(r[x]) for r in rows],[float(r[y]) for r in rows])
 for r in rows:ax.annotate(str(r.get(annotation,'')),(float(r[x]),float(r[y])),fontsize=7)
 ax.set(xlabel=x.replace('_',' '),ylabel=y.replace('_',' '));ax.grid(alpha=.3);fig.tight_layout();fig.savefig(out/(name+'.png'),dpi=180);fig.savefig(out/(name+'.pdf'));plt.close(fig)
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--input-dir',type=Path,default=ROOT/'results/characterization_v4/raw');a=p.parse_args();raw=a.input_dir;root=raw.parent;fig=root/'figures';debug=root/'debug';manifest=[]
 breadth=read(raw/'breadth_fixed_depth.csv');line(breadth,'candidate_count','Recall_at_M','fig01_breadth_recall_progressive',fig,debug);manifest.append({'filename_png':'fig01_breadth_recall_progressive.png','filename_pdf':'fig01_breadth_recall_progressive.pdf','source_csv':'breadth_fixed_depth.csv','x_column':'candidate_count','y_column':'Recall_at_M','line_group_columns':['predictor_profile'],'fixed_filters':'one fixed progressive ranking','normalization_baseline':'none','hardware_profile':breadth[0]['hardware_profile_id'],'workload_trace_id':breadth[0]['workload_trace_id'],'predictor_trace_id':breadth[0]['predictor_trace_id'],'description':'Fixed ranking [1,3,7,13], truncated by M.'})
 line(breadth,'candidate_count','weighted_startup_readiness','fig02_breadth_startup_readiness',fig,debug)
 budget=read(raw/'breadth_fixed_budget.csv');base=min(float(r['makespan_cycles']) for r in budget)
 for r in budget:r['normalized_makespan']=float(r['makespan_cycles'])/base
 line(budget,'candidate_count','normalized_makespan','fig03_breadth_fixed_budget_latency',fig,debug,group='depth_policy',annotation='candidate_depth_vector')
 scatter(budget,'speculative_bytes','makespan_cycles','fig04_breadth_depth_pareto',fig,debug,'candidate_depth_vector')
 # Top-K clean subsets: placement is explicitly fixed before any channel line.
 topk_path=raw/'topk_round_robin_clean.csv'
 if topk_path.exists():
  topk=read(topk_path); rr=[r for r in topk if r['memory_placement_policy']=='round_robin_expert']; striped=[r for r in topk if r['memory_placement_policy']=='striped_tile']
  line(rr,'top_k','makespan_cycles','fig05_topk_round_robin_latency',fig,debug,group='memory_channels')
  line(rr,'top_k','memory_transfer_wave_count','fig06_topk_memory_transfer_waves',fig,debug,group='memory_channels')
  c3=[r for r in rr if r['memory_channels']=='3']
  for r in c3:r['wave_kind']='memory_transfer'; r['wave_count']=r['memory_transfer_wave_count']
  execution=[dict(r,wave_kind='expert_execution',wave_count=r['expert_execution_wave_count']) for r in c3]
  line(c3+execution,'top_k','wave_count','fig07_memory_vs_execution_waves',fig,debug,group='wave_kind')
  c3striped=[r for r in striped if r['memory_channels']=='3']
  for r in c3:r['placement']='round_robin_expert'
  for r in c3striped:r['placement']='striped_tile'
  line(c3+c3striped,'top_k','makespan_cycles','fig08_topk_striping_comparison',fig,debug,group='placement')
 tile_path=raw/'tile_size_clean_source.csv'
 if tile_path.exists():
  tile=read(tile_path); exact=[r for r in tile if r['scenario']=='exact_tile_streaming']
  line(exact,'F_tile_size','makespan_cycles','fig09_tile_size_exact_clean',fig,debug)
  pred=[r for r in tile if r['scenario']=='predictive_uniform_prefix']
  if pred:
   exact_by_tile={r['F_tile_size']:float(r['makespan_cycles']) for r in exact}
   for r in pred:r['normalized_latency']=float(r['makespan_cycles'])/exact_by_tile[r['F_tile_size']]
   line(pred,'F_tile_size','normalized_latency','fig10_tile_size_predictor_quality_clean',fig,debug)
 (root/'figure_manifest_v4.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8');print(f'Wrote V4 figures to {fig}')
if __name__=='__main__':main()
