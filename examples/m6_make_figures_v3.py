"""Create clean M6.7 figures from V3 CSVs without rerunning simulation."""
from __future__ import annotations
import argparse,csv,json,sys
from collections import defaultdict
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from edge_moe_sim.plot_validation import require_plot_data
def load(p):
 with p.open() as f:return list(csv.DictReader(f))
def draw(rows,x,y,name,out,group):
 import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
 require_plot_data(rows,[x,y]);out.mkdir(parents=True,exist_ok=True);fig,ax=plt.subplots(figsize=(6,4));gs=defaultdict(list)
 for r in rows:gs[r[group]].append(r)
 for n,g in gs.items():g.sort(key=lambda r:float(r[x]));ax.plot([float(r[x]) for r in g],[float(r[y]) for r in g],marker='o',label=n)
 ax.set(xlabel=x,ylabel=y);ax.grid(alpha=.3);ax.legend(fontsize=7);fig.tight_layout();fig.savefig(out/(name+'.png'),dpi=160);fig.savefig(out/(name+'.pdf'));plt.close(fig)
def main():
 p=argparse.ArgumentParser();p.add_argument('--input-dir',type=Path,default=ROOT/'results/characterization_v3/raw');a=p.parse_args();raw=a.input_dir;root=raw.parent;fig=root/'figures';manifest=[]
 specs=[('topk_memory_wave_corrected.csv','top_k','makespan_cycles','fig_topk_round_robin_latency','memory_channels'),('topk_memory_wave_corrected.csv','top_k','memory_transfer_wave_count','fig_topk_memory_transfer_waves','memory_channels'),('topk_memory_wave_corrected.csv','top_k','expert_execution_wave_count','fig_topk_execution_waves','memory_channels'),('breadth_depth_fixed_ranking.csv','candidate_count','Recall_at_M','fig_breadth_depth_fixed_ranking','predictor_profile'),('tile_size_corrected.csv','F_tile_size','makespan_cycles','fig_tile_size_predictor_quality','predictor_profile')]
 for f,x,y,n,g in specs:
  path=raw/f
  if path.exists():draw(load(path),x,y,n,fig,g);manifest.append({'figure_filename':n+'.png','source_csv':f,'x_variable':x,'y_variable':y,'group':g})
 (root/'figure_manifest_v3.json').write_text(json.dumps(manifest,indent=2)+'\n');print(f'Wrote {len(manifest)} V3 figures')
if __name__=='__main__':main()
