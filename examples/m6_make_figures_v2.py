"""Validated regeneration of M6.6 figures from CSV only."""
from __future__ import annotations
import argparse,csv,json,sys
from collections import defaultdict
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from edge_moe_sim.plot_validation import require_plot_data,validate_line_group
def read(p):
 with p.open() as f:return list(csv.DictReader(f))
def plot(rows,x,y,name,out,group='predictor_profile',scatter=False):
 import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
 require_plot_data(rows,[x,y]);out.mkdir(parents=True,exist_ok=True);fig,ax=plt.subplots(figsize=(6,4));gs=defaultdict(list)
 for r in rows:gs['|'.join(r.get(part,'') for part in group.split('|'))].append(r)
 for label,g in gs.items():
  g.sort(key=lambda z:float(z[x]))
  if not scatter: validate_line_group(g,sweep=x,fixed=['experiment_name','hardware_profile_id','scenario'] if len(g)>1 else [])
  fun=ax.scatter if scatter else ax.plot; fun([float(z[x]) for z in g],[float(z[y]) for z in g],label=label)
 ax.set(xlabel=x,ylabel=y);ax.grid(alpha=.3);ax.legend(fontsize=7);fig.tight_layout();fig.savefig(out/(name+'.png'),dpi=160);fig.savefig(out/(name+'.pdf'));plt.close(fig)
def main():
 p=argparse.ArgumentParser();p.add_argument('--input-dir',type=Path,default=ROOT/'results/characterization_v2/raw');a=p.parse_args();raw=a.input_dir;root=raw.parent;fig=root/'figures';tab=root/'tables';tab.mkdir(parents=True,exist_ok=True)
 datasets={x.stem:read(x) for x in raw.glob('*.csv')};manifest=[]
 # Fixed-context datasets have one intended x dimension; DSE/breadth use scatter.
 for file,x,y,name,group,sc in [('prefetch_window_controlled','prefetch_window_cycles','useful_bytes_during_prefetch_window','fig_prefetch_window_useful_bytes','scenario|predictor_profile',False),('prefetch_window_controlled','prefetch_window_cycles','router_to_first_true_expert_start','fig_prefetch_window_start_latency','scenario|predictor_profile',False),('prefetch_window_controlled','physical_prefetch_window_utilization','useful_prefetch_window_utilization','fig_prefetch_window_efficiency','scenario|predictor_profile',True),('topk_memory_wave','top_k','makespan_cycles','fig_topk_channels_latency','memory_channels|memory_placement_policy',False),('topk_memory_wave','top_k','expert_execution_wave_count','fig_topk_channels_waves','memory_channels|memory_placement_policy',False),('tile_size_controlled','F_tile_size','makespan_cycles','fig_tile_size_latency','scenario|predictor_profile',False),('predictor_policy_comparison','expected_Recall_at_M','normalized_latency','fig_policy_latency_vs_predictor_quality','scenario|mean_tokens_per_expert',False),('breadth_depth_controlled','speculative_bytes','makespan_cycles','fig_breadth_depth_pareto','predictor_profile',True)]:
  rows=datasets.get(file,[])
  if rows and (not any(not r.get(y) for r in rows)):
   plot(rows,x,y,name,fig,group,sc);manifest.append({'figure_filename':name+'.png','source_csv':file+'.csv','x_variable':x,'y_variable':y,'line_or_scatter':'scatter' if sc else 'validated fixed-context lines'})
 # concise tables directly from raw
 for source,target in [('hardware_profile_sanity','table_hardware_profiles'),('aie_allocation_controlled','table_aie_sufficient'),('topk_memory_wave','table_topk_memory_waves'),('predictor_policy_comparison','table_full_prefix_dse'),('breadth_depth_controlled','table_breadth_depth_pareto'),('negative_controls','table_negative_controls')]:
  rows=datasets.get(source,[])
  if rows:
   with (tab/(target+'.csv')).open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)
 (root/'figure_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');(root/'CHARACTERIZATION_V2.md').write_text('# Characterization V2\n\nAll results use synthetic/theoretical hardware inputs, not measurements. Figures were regenerated from CSV with fixed-context line validation.\n');print(f'Wrote {len(manifest)} figures')
if __name__=='__main__':main()
