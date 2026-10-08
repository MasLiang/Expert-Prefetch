"""Regenerate M7.2a corrected paper figures from plotting CSVs only."""
from __future__ import annotations
import argparse,csv,json
from pathlib import Path

POLICY_COLORS={'NO_PREFETCH':'black','FULL_EXPERT':'#377eb8','UNIFORM_PREFIX':'#ff7f00','RANKED_PREFIX':'#4daf4a','BREADTH_DEPTH_SEARCH':'#e41a1c'}

def read(path:Path)->list[dict[str,str]]:
    """Read plotting rows."""
    with path.open(encoding='utf-8') as stream:return list(csv.DictReader(stream))

def make_figures(root:Path)->list[dict[str,object]]:
    """Create ten PNG/PDF figures and return their provenance manifest."""
    import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
    figures=root/'figures';figures.mkdir(parents=True,exist_ok=True);manifest=[]
    def finish(name,fig,source,x,y,group,filters,ties='not applicable',hardware='controlled_synthetic_hardware'):
        fig.tight_layout();png=figures/f'{name}.png';pdf=figures/f'{name}.pdf';fig.savefig(png,dpi=200);fig.savefig(pdf);plt.close(fig)
        manifest.append({'figure_id':name,'source_file':f'data/{source}','filters':filters,'comparison_mode':'unconstrained','x':x,'y':y,'grouping':group,'tie_semantics':ties,'hardware_mode':hardware,'png':str(png),'pdf':str(pdf),'plotting_csv':str(root/'data'/source)})
    def phase(name,source,x,y):
        rows=read(root/'data'/source);fig,ax=plt.subplots(figsize=(7,4.5))
        for row in rows:
            policy=row['preferred_policy_under_tie'];tie=row['latency_tie']=='True';robust=row['robust_latency_winner']=='True';marker='s' if tie else ('o' if robust else '^');face='none' if tie else POLICY_COLORS[policy]
            ax.scatter(float(row[x]),float(row[y]),marker=marker,s=75,facecolors=face,edgecolors=POLICY_COLORS[policy],linewidths=1.4)
        for policy,color in POLICY_COLORS.items():ax.scatter([],[],color=color,label=policy)
        ax.scatter([],[],marker='o',color='gray',label='Robust latency winner');ax.scatter([],[],marker='^',color='gray',label='Near-tie (<1%)');ax.scatter([],[],marker='s',facecolors='none',edgecolors='gray',label='Cost-preferred exact tie')
        ax.set(xlabel=x.replace('_',' '),ylabel=y.replace('_',' '));ax.grid(alpha=.2);ax.legend(fontsize=6,ncol=2)
        finish(name,fig,source,x,y,'preferred_policy_under_tie','actual discrete configurations; no interpolation','robust / near / exact-cost-tie marker')
    phase('fig01a_policy_phase_quality_vs_opportunity_confidence','fig01a_policy_phase_quality_vs_opportunity_confidence.csv','opportunity','predictor_quality_scalar')
    phase('fig02a_policy_phase_quality_vs_ocm_confidence','fig02a_policy_phase_quality_vs_ocm_confidence.csv','ocm_ratio','predictor_quality_scalar')
    phase('fig03a_policy_phase_lead_vs_channels','fig03a_policy_phase_lead_vs_channels.csv','reference_bandwidth_opportunity_expert_equivalents','channels')
    for name,source,value,label in (('fig04a_best_breadth_map','fig04a_best_breadth_map.csv','candidate_count','Best breadth M'),('fig04b_best_depth_map','fig04b_best_depth_map.csv','mean_depth','Best mean prefix depth')):
        rows=read(root/'data'/source);fig,ax=plt.subplots(figsize=(6.5,4.2));sc=ax.scatter([float(r['configuration_opportunity_expert_equivalents']) for r in rows],[float(r['predictor_quality_scalar']) for r in rows],c=[float(r[value]) for r in rows],cmap='viridis',s=75);fig.colorbar(sc,ax=ax,label=label);ax.set(xlabel='prefetch opportunity expert equivalents',ylabel='predictor quality scalar');ax.grid(alpha=.2);finish(name,fig,source,'configuration_opportunity_expert_equivalents','predictor_quality_scalar',value,'search_rank == 1 and BDS cost-preferred','only canonical best BDS plan')
    rows=read(root/'data/fig09a_static_policy_feasibility.csv');fig,ax=plt.subplots(figsize=(6,4));ax.bar([r['heuristic'] for r in rows],[100*float(r['feasible_fraction']) for r in rows],color='#4c78a8');ax.set(ylabel='feasible configurations (%)',ylim=(0,105));ax.tick_params(axis='x',rotation=15);finish('fig09a_static_policy_feasibility',fig,'fig09a_static_policy_feasibility.csv','heuristic','feasible_fraction','heuristic','main unconstrained grid','feasibility shown separately')
    rows=read(root/'data/fig09b_static_policy_regret_cdf.csv');fig,ax=plt.subplots(figsize=(6,4));groups={}
    for r in rows:groups.setdefault(r['heuristic'],[]).append(float(r['regret_percent']))
    for label,values in groups.items():
        values=sorted(values);ax.step(values,[(i+1)/len(values) for i in range(len(values))],where='post',label=label)
    ax.set(xlabel='regret among feasible configurations (%)',ylabel='empirical CDF');ax.grid(alpha=.2);ax.legend();finish('fig09b_static_policy_regret_cdf',fig,'fig09b_static_policy_regret_cdf.csv','regret_percent','empirical_cdf','heuristic','feasible configurations only','infeasible fraction is in Fig. 9A')
    rows=read(root/'data/fig10a_vek280_theoretical_policy_comparison.csv');fig,ax=plt.subplots(figsize=(7,4.3));qualities=['perfect','medium','wrong'];policies=['No Prediction','Full','Uniform Prefix','Ranked Prefix','Best BDS']
    for j,policy in enumerate(policies):
        subset=[r for r in rows if r['canonical_policy']==policy and r['feasible']=='True'];ax.plot(range(3),[float(next(r['speedup_vs_no_prediction'] for r in subset if r['predictor_quality']==q)) for q in qualities],marker='o',label=policy)
    ax.set_xticks(range(3),qualities);ax.set(ylabel='speedup vs no prediction',title='VEK280 THEORETICAL MODEL — NOT MEASURED');ax.axhline(1,color='gray',lw=.8);ax.grid(alpha=.2);ax.legend(fontsize=7);finish('fig10a_vek280_theoretical_policy_comparison',fig,'fig10a_vek280_theoretical_policy_comparison.csv','predictor_quality','speedup_vs_no_prediction','canonical_policy','exactly one row per quality × policy','best BDS canonicalized',hardware='vek280_theoretical_not_measured')
    rows=read(root/'data/fig11_search_vs_best_fixed_gain.csv');fig,ax=plt.subplots(figsize=(6,4));ax.hist([float(r['search_improvement_percent']) for r in rows],bins=30,color='#e41a1c');ax.set(xlabel='BDS improvement over best fixed (%)',ylabel='configurations');ax.axvline(0,color='black',lw=.8);finish('fig11_search_vs_best_fixed_gain',fig,'fig11_search_vs_best_fixed_gain.csv','search_improvement_percent','configuration_count','all compared cells','best feasible fixed predictive policy','zero denotes latency tie')
    rows=read(root/'data/fig12_search_tie_resource_efficiency.csv');fig,ax=plt.subplots(figsize=(6,4));ax.scatter([-float(r['allocation_delta_bytes']) for r in rows],[-float(r['physical_traffic_delta_bytes']) for r in rows],alpha=.65);ax.axvline(0,color='gray',lw=.8);ax.axhline(0,color='gray',lw=.8);ax.set(xlabel='BDS allocated-byte saving',ylabel='BDS physical-byte saving');ax.grid(alpha=.2);finish('fig12_search_tie_resource_efficiency',fig,'fig12_search_tie_resource_efficiency.csv','allocation saving','physical traffic saving','latency-tie cells','BDS makespan == best fixed makespan','exact latency ties only')
    return manifest

def main()->int:
    """CLI entry point."""
    parser=argparse.ArgumentParser();parser.add_argument('--input-dir',type=Path,required=True);args=parser.parse_args();manifest=make_figures(args.input_dir);(args.input_dir/'figure_manifest_m72a.json').write_text(json.dumps(manifest,indent=2)+'\n');print(f'Wrote {len(manifest)} M7.2a figures');return 0
if __name__=='__main__':raise SystemExit(main())
