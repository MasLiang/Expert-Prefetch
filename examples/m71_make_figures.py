"""Generate M7.1 PNG/PDF figures strictly from existing unified CSV files."""
from __future__ import annotations
import argparse,csv,json,sys
from collections import defaultdict
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))

def read(path:Path):
    with path.open(encoding='utf-8') as f:return list(csv.DictReader(f))
def plot(rows,x,y,name,out,data,group='policy',scatter=False):
    from edge_moe_sim.plot_validation import assert_unique_x_per_line,require_plot_data
    import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
    require_plot_data(rows,[x,y]);out.mkdir(parents=True,exist_ok=True);data.mkdir(parents=True,exist_ok=True)
    if not scatter: assert_unique_x_per_line(rows,x_column=x,line_group_columns=[group],figure_name=name,debug_dir=out.parent/'diagnostics')
    with (data/(name+'.csv')).open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)
    fig,ax=plt.subplots(figsize=(6.4,4)); groups=defaultdict(list)
    for r in rows:groups[r.get(group,'all')].append(r)
    def number(value):
        try:return float(value)
        except ValueError:return None
    categorical=number(rows[0][x]) is None
    labels=list(dict.fromkeys(r[x] for r in rows)); positions={label:index for index,label in enumerate(labels)}
    for label,g in groups.items():
        if not categorical:g.sort(key=lambda r:float(r[x]))
        fun=ax.scatter if scatter else ax.plot
        xs=[positions[r[x]] if categorical else float(r[x]) for r in g]
        fun(xs,[float(r[y]) for r in g],label=label,marker='o')
    if categorical:ax.set_xticks(range(len(labels)),labels,rotation=20,ha='right')
    ax.set(xlabel=x.replace('_',' '),ylabel=y.replace('_',' '));ax.grid(alpha=.3);ax.legend(fontsize=7);fig.tight_layout();fig.savefig(out/(name+'.png'),dpi=180);fig.savefig(out/(name+'.pdf'));plt.close(fig)
    return {'figure_id':name,'source_csv':'','simulator_scope':'unified_cross_layer','x':x,'y':y,'grouping':group,'output_png':str(out/(name+'.png')),'output_pdf':str(out/(name+'.pdf')),'plot_data':str(data/(name+'.csv'))}
def main():
    p=argparse.ArgumentParser();p.add_argument('--input-dir',type=Path,required=True);a=p.parse_args();raw=a.input_dir;root=raw.parent;manifest=[]
    choices=[('source_milestone_study.csv','source_milestone','realized_lead_cycles','fig01_realized_lead_by_source_milestone','source_milestone',False),('source_milestone_study.csv','source_milestone','prefetch_opportunity_expert_equivalents','fig02_prefetch_opportunity_by_source_milestone','source_milestone',False),('source_milestone_study.csv','source_milestone','weighted_startup_readiness','fig03_startup_readiness_by_source_milestone','source_milestone',False),('source_milestone_study.csv','source_milestone','sequence_makespan','fig04_makespan_by_source_milestone','source_milestone',False),('perfect_prediction_policy_baseline.csv','policy','sequence_makespan','fig05_perfect_prediction_policy_baseline','policy',False),('predictor_quality_policy_study.csv','predictor_quality','sequence_makespan','fig06_predictor_quality_latency','policy',False),('predictor_quality_policy_study.csv','predictor_quality','wasted_speculative_bytes','fig07_predictor_quality_waste','policy',False),('predictor_quality_policy_study.csv','predictor_quality','demand_blocked_by_future_speculation_cycles','fig08_predictor_quality_blocking','policy',False),('unified_breadth_fixed_budget.csv','candidate_count','planned_speculative_bytes','fig09_fixed_budget_depth_vs_breadth','policy',False),('unified_breadth_fixed_budget.csv','candidate_count','weighted_startup_readiness','fig10_fixed_budget_startup_readiness','policy',False),('unified_breadth_fixed_budget.csv','candidate_count','sequence_makespan','fig11_fixed_budget_makespan','policy',False),('unified_breadth_fixed_budget.csv','candidate_count','demand_blocked_by_future_speculation_cycles','fig12_fixed_budget_blocking','policy',False),('unified_breadth_fixed_budget.csv','speculative_bytes','sequence_makespan','fig13_breadth_depth_pareto','policy',True),('unified_ocm_pressure.csv','ocm_bytes','sequence_makespan','fig14_ocm_capacity_latency','policy',False),('unified_ocm_pressure.csv','ocm_bytes','rejected_speculative_bytes','fig15_ocm_rejection_eviction','policy',False),('unified_ocm_pressure.csv','ocm_bytes','StartableRecall','fig16_ocm_startable_recall','policy',False),('unified_memory_channel_study.csv','memory_channels','sequence_makespan','fig17_memory_channels_latency','policy',False),('unified_memory_channel_study.csv','memory_channels','demand_blocked_by_future_speculation_cycles','fig18_memory_channels_blocking','policy',False),('unified_topk_study.csv','top_k','sequence_makespan','fig19_topk_unified_latency','policy',False),('predictor_quality_policy_study.csv','prefetch_opportunity_expert_equivalents','sequence_makespan','fig20_prefetch_opportunity_vs_latency','policy',True)]
    for file,x,y,name,g,scatter in choices:
        path=raw/file
        if path.exists(): manifest.append(plot(read(path),x,y,name,root/'figures',root/'figures/data',g,scatter))
    # Interpretation timelines consume recorded traces only; they never rerun simulation.
    for stem in ('perfect_full_expert','imperfect_full_expert','imperfect_ranked_prefix'):
        path=root/'traces'/(stem+'.json')
        if not path.exists():continue
        import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
        events=json.loads(path.read_text()); intervals=[e for e in events if e.get('end_cycle') is not None and e.get('event_type') in {'dma_start','aie_start'}]
        fig,ax=plt.subplots(figsize=(8,3)); labels=[]
        for i,e in enumerate(intervals[:24]):
            label=f"{e.get('resource')} L{e.get('layer_id')} {e.get('request_state') or e.get('matrix_type')}";labels.append(label);ax.barh(i,e['end_cycle']-e['cycle'],left=e['cycle'])
        ax.set(yticks=range(len(labels)),yticklabels=labels,xlabel='cycle',title=stem.replace('_',' '));fig.tight_layout();name='timeline_'+stem;fig.savefig(root/'figures'/(name+'.png'),dpi=180);fig.savefig(root/'figures'/(name+'.pdf'));plt.close(fig);manifest.append({'figure_id':name,'source_csv':str(path),'simulator_scope':'unified_cross_layer','output_png':str(root/'figures'/(name+'.png')),'output_pdf':str(root/'figures'/(name+'.pdf'))})
    (root/'figure_manifest_m71.json').write_text(json.dumps(manifest,indent=2)+'\n');print(f'Wrote {len(manifest)} figures')
if __name__=='__main__':main()
