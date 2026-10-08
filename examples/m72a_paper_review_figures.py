"""Regenerate paper-review figures from existing M7.2a CSV artifacts only."""
from __future__ import annotations
import argparse,csv,json,math
from collections import Counter
from pathlib import Path
from statistics import median

POLICY_NAME={'NO_PREFETCH':'No Prefetch','FULL_EXPERT':'Full Expert','UNIFORM_PREFIX':'Uniform Prefix','RANKED_PREFIX':'Ranked Prefix','BREADTH_DEPTH_SEARCH':'Breadth–Depth Search'}
POLICY_COLOR={'NO_PREFETCH':'#333333','FULL_EXPERT':'#377eb8','UNIFORM_PREFIX':'#ff9d00','RANKED_PREFIX':'#4daf4a','BREADTH_DEPTH_SEARCH':'#e41a1c'}

def read_csv(path:Path)->list[dict[str,str]]:
    """Read a required non-empty CSV."""
    assert path.is_file() and path.stat().st_size>0,f'missing source: {path}'
    with path.open(encoding='utf-8') as stream:rows=list(csv.DictReader(stream))
    assert rows,f'empty source: {path}';return rows

def write_csv(path:Path,rows:list[dict[str,object]])->None:
    """Write exact plotting rows deterministically."""
    assert rows;path.parent.mkdir(parents=True,exist_ok=True);fields=sorted({key for row in rows for key in row})
    with path.open('w',newline='',encoding='utf-8') as stream:writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader();writer.writerows(rows)

def percentile(values:list[float],fraction:float)->float:
    """Compute a deterministic linearly interpolated percentile."""
    values=sorted(values);position=(len(values)-1)*fraction;lo=math.floor(position);hi=math.ceil(position)
    return values[lo] if lo==hi else values[lo]+(values[hi]-values[lo])*(position-lo)

def generate(source:Path,review:Path)->list[dict[str,object]]:
    """Validate source statistics and generate the review collection without simulation."""
    import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
    data=review/'data';review.mkdir(parents=True,exist_ok=True);data.mkdir(exist_ok=True)
    source_data=source/'data';manifest=[]
    inventory=[]
    old_manifest=json.loads((source/'figure_manifest_m72a.json').read_text())
    for item in old_manifest:
        record={'figure_id':item['figure_id']}
        for kind,key in (('plotting_csv','plotting_csv'),('png','png'),('pdf','pdf')):
            path=Path(item[key]);record[f'{kind}_path']=str(path);record[f'{kind}_exists']=path.exists();record[f'{kind}_bytes']=path.stat().st_size if path.exists() else 0
        inventory.append(record)
    write_csv(review/'source_figure_inventory.csv',inventory)
    winners=read_csv(source_data/'m72a_corrected_winners.csv');uncon=[row for row in winners if row['comparison_mode']=='unconstrained']
    tie_count=sum(row['latency_tie']=='True' for row in uncon);robust_count=sum(row['robust_latency_winner']=='True' for row in uncon);near=len(uncon)-tie_count-robust_count
    assert (len(uncon),tie_count,robust_count,near)==(240,125,46,69)
    print(f'Global confidence: robust={robust_count}/240, near={near}/240, exact ties={tie_count}/240')
    def finish(figure_id:str,fig,source_name:str,filters:str,x:str,y:str,grouping:str,tie_semantics:str)->None:
        fig.tight_layout();png=review/f'{figure_id}.png';pdf=review/f'{figure_id}.pdf';fig.savefig(png,dpi=220,bbox_inches='tight');fig.savefig(pdf,bbox_inches='tight');plt.close(fig)
        manifest.append({'figure_id':figure_id,'source_csv':str(source_data/source_name),'filters':filters,'x':x,'y':y,'grouping':grouping,'confidence_tie_semantics':tie_semantics,'output_png':str(png),'output_pdf':str(pdf),'plotting_csv':str(data/f'{figure_id}.csv')})
    def phase(figure_id:str,source_name:str,x:str,y:str,xlabel:str)->None:
        rows=read_csv(source_data/source_name);assert len(rows)==len({row['configuration_id'] for row in rows});write_csv(data/f'{figure_id}.csv',rows)
        subset_ties=sum(row['latency_tie']=='True' for row in rows);subset_robust=sum(row['robust_latency_winner']=='True' for row in rows);subset_near=len(rows)-subset_ties-subset_robust
        print(f'{figure_id}: rows={len(rows)}, robust={subset_robust}, near={subset_near}, ties={subset_ties}')
        fig,ax=plt.subplots(figsize=(7.2,4.5))
        for row in rows:
            policy=row['preferred_policy_under_tie'];tie=row['latency_tie']=='True';robust=row['robust_latency_winner']=='True';marker='s' if tie else ('o' if robust else '^');face='none' if tie else POLICY_COLOR[policy]
            ax.scatter(float(row[x]),float(row[y]),marker=marker,s=82,facecolors=face,edgecolors=POLICY_COLOR[policy],linewidths=1.5,zorder=3)
        for policy,color in POLICY_COLOR.items():ax.scatter([],[],s=45,color=color,label=POLICY_NAME[policy])
        ax.scatter([],[],s=60,marker='o',color='#777777',label='Robust latency winner ≥1%');ax.scatter([],[],s=60,marker='^',color='#777777',label='Unique winner <1%');ax.scatter([],[],s=60,marker='s',facecolors='none',edgecolors='#777777',label='Cost-preferred exact tie')
        ax.set(xlabel=xlabel,ylabel='Memory channels' if y=='channels' else 'Predictor quality scalar');ax.grid(alpha=.2);ax.legend(fontsize=7,ncol=2,loc='best')
        finish(figure_id,fig,source_name,'unconstrained representative slice; actual grid points only; no interpolation',x,y,'preferred_policy_under_tie','fill/triangle/hollow-square distinguish robust/near/exact tie')
    phase('fig01a_policy_phase_quality_vs_opportunity_confidence','fig01a_policy_phase_quality_vs_opportunity_confidence.csv','opportunity','predictor_quality_scalar','Prefetch opportunity (Expert equivalents)')
    channel_rows=read_csv(source_data/'fig03a_policy_phase_lead_vs_channels.csv')
    for row in channel_rows:assert abs(float(row['reference_bandwidth_opportunity_expert_equivalents'])-float(row['opportunity'])/int(row['channels']))<1e-12
    phase('fig03a_policy_phase_lead_vs_channels','fig03a_policy_phase_lead_vs_channels.csv','reference_bandwidth_opportunity_expert_equivalents','channels','Fixed-reference temporal opportunity (Expert equivalents)')
    breadth=read_csv(source_data/'fig04a_best_breadth_map.csv');depth=read_csv(source_data/'fig04b_best_depth_map.csv')
    assert breadth==depth;assert all(row['search_rank']=='1' for row in breadth);assert len(breadth)==len({row['configuration_id'] for row in breadth})
    distribution=Counter(int(row['candidate_count']) for row in breadth);expected={1:35,2:22,4:30,8:2,16:0};assert {m:distribution.get(m,0) for m in expected}==expected
    depth_values=[float(row['mean_depth']) for row in breadth];print(f'fig04: rows={len(breadth)}, M={expected}, mean_depth=[{min(depth_values)}, {max(depth_values)}]')
    for figure_id,value,label,cmap in (('fig04a_best_breadth_map','candidate_count','Best candidate breadth M','viridis'),('fig04b_best_depth_map','mean_depth','Best mean GU-prefix depth','plasma')):
        write_csv(data/f'{figure_id}.csv',breadth);fig,ax=plt.subplots(figsize=(6.6,4.4));plot=ax.scatter([float(r['configuration_opportunity_expert_equivalents']) for r in breadth],[float(r['predictor_quality_scalar']) for r in breadth],c=[float(r[value]) for r in breadth],cmap=cmap,s=82,edgecolors='white',linewidths=.35);fig.colorbar(plot,ax=ax,label=label);ax.set(xlabel='Prefetch opportunity (Expert equivalents)',ylabel='Predictor quality scalar');ax.grid(alpha=.2);finish(figure_id,fig,f'{figure_id}.csv','search_rank == 1; BDS is final cost-preferred policy; one row/configuration','configuration_opportunity_expert_equivalents','predictor_quality_scalar',value,'canonical best BDS plan only')
    regret_points=read_csv(source_data/'fig09b_static_policy_regret_cdf.csv');regret_summary=read_csv(source_data/'m72a_static_policy_regret_summary.csv');summary={row['heuristic']:row for row in regret_summary}
    expected_regret={'Always Full':(.8041666667,.3943274727,9.5703082505,12.3065585851),'Always Prefix':(.8083333333,.4432690234,6.4874219663,13.5187914517),'Always No Prefetch':(1.0,0.0,20.1061301645,34.4978590891)}
    for label,(feasible,med,p95,maximum) in expected_regret.items():
        row=summary[label];assert abs(float(row['feasible_fraction'])-feasible)<1e-8 and abs(float(row['median_regret_percent'])-med)<1e-8 and abs(float(row['p95_regret_percent'])-p95)<1e-8 and abs(float(row['max_regret_percent'])-maximum)<1e-8
    write_csv(data/'fig09a_static_policy_feasibility.csv',regret_summary);fig,ax=plt.subplots(figsize=(6.2,4.1));labels=list(expected_regret);values=[100*float(summary[x]['feasible_fraction']) for x in labels];bars=ax.bar(labels,values,color=['#377eb8','#4daf4a','#333333']);ax.bar_label(bars,fmt='%.2f%%',padding=3);ax.set(ylabel='Feasible configurations (%)',ylim=(0,108));ax.tick_params(axis='x',rotation=12);finish('fig09a_static_policy_feasibility',fig,'m72a_static_policy_regret_summary.csv','unconstrained main grid','heuristic','feasible_fraction','static heuristic','infeasibility is not zero regret')
    write_csv(data/'fig09b_static_policy_regret_cdf.csv',regret_points);fig,ax=plt.subplots(figsize=(6.5,4.2));colors={'Always Full':'#377eb8','Always Prefix':'#4daf4a','Always No Prefetch':'#333333'}
    for label in labels:
        values=sorted(float(row['regret_percent']) for row in regret_points if row['heuristic']==label);feasible=100*float(summary[label]['feasible_fraction']);ax.step(values,[(i+1)/len(values) for i in range(len(values))],where='post',lw=1.8,color=colors[label],label=f'{label} ({feasible:.2f}% feasible)')
    ax.set(xlabel='Regret among feasible configurations (%)',ylabel='Empirical cumulative fraction',ylim=(0,1.02));ax.set_xlim(left=0);ax.grid(alpha=.2);ax.legend(fontsize=8);finish('fig09b_static_policy_regret_cdf',fig,'fig09b_static_policy_regret_cdf.csv','feasible configurations only','regret_percent','empirical_cdf','static heuristic','infeasible configurations excluded, feasibility shown in legend/Fig. 9A')
    vek=read_csv(source_data/'fig10a_vek280_theoretical_policy_comparison.csv');assert len(vek)==15 and len({(row['predictor_quality'],row['canonical_policy']) for row in vek})==15;assert sum(row['canonical_policy']=='Best BDS' for row in vek)==3
    write_csv(data/'fig10a_vek280_theoretical_policy_comparison.csv',vek);fig,ax=plt.subplots(figsize=(7.2,4.4));qualities=('perfect','medium','wrong');policies=('No Prediction','Full','Uniform Prefix','Ranked Prefix','Best BDS')
    for policy in policies:
        points=[next(row for row in vek if row['predictor_quality']==quality and row['canonical_policy']==policy) for quality in qualities];feasible=[row['feasible']=='True' for row in points]
        if all(feasible):ax.plot(range(3),[float(row['speedup_vs_no_prediction']) for row in points],marker='o',lw=1.6,label=policy)
        else:
            for x,row,ok in zip(range(3),points,feasible):ax.scatter(x,float(row['speedup_vs_no_prediction']) if ok else 0,marker='o' if ok else 'x',label=policy if x==0 else None)
    ax.set_xticks(range(3),[x.capitalize() for x in qualities]);ax.set(ylabel='Speedup vs No Prediction',title='VEK280 THEORETICAL MODEL — NOT MEASURED');ax.axhline(1,color='#777777',lw=.8);ax.grid(alpha=.2);ax.legend(fontsize=8,ncol=2);finish('fig10a_vek280_theoretical_policy_comparison',fig,'fig10a_vek280_theoretical_policy_comparison.csv','3 qualities × 5 canonical policies; one best BDS/quality','predictor_quality','speedup_vs_no_prediction','canonical_policy','infeasible cells explicitly marked')
    gains=read_csv(source_data/'fig11_search_vs_best_fixed_gain.csv');compared=len(gains);strict=sum(float(row['search_improvement_percent'])>0 for row in gains);gt1=sum(float(row['search_improvement_percent'])>1 for row in gains);gt5=sum(float(row['search_improvement_percent'])>5 for row in gains);maximum=max(float(row['search_improvement_percent']) for row in gains);assert (compared,strict,gt1,gt5)==(209,50,36,11) and abs(maximum-18.3437221728)<1e-8
    write_csv(data/'fig11_search_vs_best_fixed_gain.csv',gains);fig,ax=plt.subplots(figsize=(6.5,4.1));ax.hist([float(row['search_improvement_percent']) for row in gains],bins=30,color='#e41a1c',alpha=.8);ax.axvline(0,color='black',lw=.8);ax.set(xlabel='BDS improvement over best fixed predictive policy (%)',ylabel='Configurations');ax.text(.98,.96,'50 strict wins\n36 >1%\n11 >5%',ha='right',va='top',transform=ax.transAxes);finish('fig11_search_vs_best_fixed_gain',fig,'fig11_search_vs_best_fixed_gain.csv','209 mutually feasible comparisons','search_improvement_percent','configuration_count','BDS vs best fixed','zero is same latency')
    ties=read_csv(source_data/'fig12_search_tie_resource_efficiency.csv');assert len(ties)==155;allocation=[-int(row['allocation_delta_bytes']) for row in ties];assert sum(x>0 for x in allocation)==155 and median(allocation)==122880;lower_ocm=sum(int(row['peak_ocm_delta_bytes'])<0 for row in ties);lower_physical=sum(int(row['physical_traffic_delta_bytes'])<0 for row in ties);assert (lower_ocm,lower_physical)==(43,26)
    write_csv(data/'fig12_search_tie_resource_efficiency.csv',ties);fig,(ax1,ax2)=plt.subplots(1,2,figsize=(9,4));ordered=sorted(x/1024 for x in allocation);ax1.step(ordered,[(i+1)/len(ordered) for i in range(len(ordered))],where='post',color='#e41a1c',lw=2);ax1.axvline(120,color='#555555',ls='--',label='Median: 120 KiB');ax1.set(xlabel='BDS speculative-allocation saving (KiB)',ylabel='Empirical cumulative fraction',title='Same latency, lower allocation');ax1.grid(alpha=.2);ax1.legend(fontsize=8)
    names=('Lower allocation','Lower peak OCM','Lower physical traffic');fractions=(1,lower_ocm/155,lower_physical/155);bars=ax2.bar(names,[100*x for x in fractions],color=('#e41a1c','#984ea3','#377eb8'));ax2.bar_label(bars,fmt='%.1f%%',padding=3);ax2.set(ylabel='Latency-tie configurations (%)',ylim=(0,110));ax2.tick_params(axis='x',rotation=20);finish('fig12_search_tie_resource_efficiency',fig,'fig12_search_tie_resource_efficiency.csv','BDS makespan == best fixed predictive makespan; 155 cells','allocation_saving_bytes','empirical_cdf_and_fraction','resource metric','exact same-latency comparisons only')
    readme='''# M7.2a Paper-Review Figures\n\nAll figures were regenerated from existing M7.2a CSV artifacts. New simulation count: **0**.\n\n- **fig01a** — source: corrected quality/opportunity phase CSV; unconstrained representative slice, actual grid points only. Question: Which policy is cost-preferred, and is it a robust, near, or exact-tie result?\n- **fig03a** — source: corrected lead/channel phase CSV; X uses fixed single-channel reference bandwidth. Question: Does additional memory parallelism expose speculative slack?\n- **fig04a** — source: canonical best BDS rows; `search_rank == 1` and BDS final cost-preferred. Question: What breadth does the best BDS plan choose?\n- **fig04b** — same rows/filter as fig04a. Question: What mean prefix depth accompanies the selected breadth?\n- **fig09a** — static-policy feasibility summary. Question: How often can each heuristic execute under the modeled constraints?\n- **fig09b** — feasible-only regret samples. Question: What regret distribution remains after feasibility is handled separately?\n- **fig10a** — complete 3×5 VEK280 theoretical matrix. Question: How do canonical policies compare across quality? **Not measured.**\n- **fig11** — 209 mutually feasible BDS/fixed comparisons. Question: How often and by how much does search strictly improve latency?\n- **fig12** — 155 same-latency comparisons. Question: How much speculative resource does BDS save when latency ties?\n''';(review/'README.md').write_text(readme,encoding='utf-8')
    (review/'figure_manifest_paper_review.json').write_text(json.dumps(manifest,indent=2)+'\n');(review/'generation_summary.json').write_text(json.dumps({'new_simulations':0,'source_figures_found':len(old_manifest),'figures_regenerated':len(manifest),'global_confidence':{'total':240,'robust':46,'near':69,'exact_tie':125},'fig04_M_distribution':expected},indent=2)+'\n')
    return manifest

def main()->int:
    """CLI entry point."""
    parser=argparse.ArgumentParser();parser.add_argument('--source-dir',type=Path,required=True);parser.add_argument('--output-dir',type=Path,required=True);args=parser.parse_args();manifest=generate(args.source_dir,args.output_dir);print(f'Regenerated {len(manifest)} review figures; new simulations: 0');return 0
if __name__=='__main__':raise SystemExit(main())
