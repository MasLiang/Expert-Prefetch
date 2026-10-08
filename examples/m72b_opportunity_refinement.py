"""M7.2b genuine cross-layer opportunity boundary refinement."""
from __future__ import annotations
import argparse,csv,json,sys
from collections import defaultdict
from pathlib import Path
from statistics import fmean

ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'examples')]
import m71_unified_characterization as m71
import m72_policy_phase as m72
from edge_moe_sim.unified_policy_search import UnifiedBreadthDepthSearch,rank_policy_rows

OUT=ROOT/'results/m72b_opportunity_refinement';LATENCIES=(7167,4767,2847);QUALITIES=('perfect','high','medium','low','wrong')
POLICY_LABEL={'no_prediction':'NO_PREFETCH','full_expert_prefetch':'FULL_EXPERT','ranked_prefix':'RANKED_PREFIX','breadth_depth_search':'BREADTH_DEPTH_SEARCH'}

def write(path:Path,rows:list[dict[str,object]])->None:
    """Write deterministic non-empty CSV rows."""
    assert rows;path.parent.mkdir(parents=True,exist_ok=True);fields=sorted({key for row in rows for key in row})
    with path.open('w',newline='',encoding='utf-8') as stream:writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader();writer.writerows(rows)

def read(path:Path)->list[dict[str,str]]:
    """Read an existing CSV."""
    with path.open(encoding='utf-8') as stream:return list(csv.DictReader(stream))

def base_config(quality:str,latency:int,opportunity:float)->dict[str,object]:
    """Build the fixed Fig. 1 machine/workload configuration for one timing point."""
    base={'quality':quality,'opportunity_setting':f'm72b_router_latency_{latency}','milestone':'source_router_done','latency':latency,'ocm':49152,'channels':2,'top_k':4,'batch_size':1,'routing_shape':'high_reuse','placement_policy':'round_robin_expert','predictor_seed':17,'true_ranks':m72.RANKS[quality],'quality_scalar':fmean(1/rank for rank in m72.RANKS[quality]),'demand_reference':32768}
    base['configuration_id']=m72.config_id(base);base['opportunity']=opportunity;return base

def calibrate(latency:int)->tuple[dict[str,object],int]:
    """Run one zero-request unified prediction to measure genuine opportunity."""
    cfg=base_config('perfect',latency,0.0);row,_=m72.evaluate(cfg,'uniform_prefix',16,(0,)*16,None,'controlled_synthetic_hardware',False,'unconstrained')
    opportunity=float(row['prefetch_opportunity_expert_equivalents']);lead=float(row['realized_lead_cycles'])
    assert row['simulator_scope']=='unified_cross_layer' and row['synthetic_lead_time'] is False and .381<opportunity<4.546
    return {'source_milestone':'source_router_done','predictor_latency_cycles':latency,'realized_lead_cycles':lead,'opportunity_expert_equivalents':opportunity,'hardware_mode':row['hardware_mode'],'workload_trace_id':row['workload_trace_id'],'predictor_trace_id':row['predictor_trace_id'],'memory_channels':row['memory_channels'],'ocm_bytes':row['ocm_bytes'],'top_k':row['top_k'],'batch_size':row['batch_size'],'synthetic_lead_time':False},1

def evaluate_point(cfg:dict[str,object],counter:list[int])->list[dict[str,object]]:
    """Evaluate three simple policies and the unchanged bounded BDS plan set."""
    rows=[]
    def run(policy:str,m:int=16,depths:tuple[int,...]=(),family:str='')->None:
        counter[0]+=1
        try:row,_=m72.evaluate(cfg,policy,m,depths,None,'controlled_synthetic_hardware',False,'unconstrained',family)
        except RuntimeError as exc:
            rows.append({'configuration_id':cfg['configuration_id'],'predictor_quality':cfg['quality'],'source_milestone':cfg['milestone'],'predictor_latency_cycles':cfg['latency'],'opportunity_expert_equivalents':cfg['opportunity'],'policy':policy,'candidate_count':m,'candidate_depth_vector':str(depths),'feasible':False,'infeasible_reason':str(exc),'synthetic_lead_time':False});return
        row.update(feasible=True,infeasible_reason='',opportunity_expert_equivalents=cfg['opportunity'],point_realized_lead_cycles=cfg.get('point_realized_lead_cycles',''),policy_realized_opportunity_expert_equivalents=row['prefetch_opportunity_expert_equivalents'],policy_realized_lead_cycles=row['realized_lead_cycles']);rows.append(row)
    for policy in ('no_prediction','full_expert_prefetch','ranked_prefix'):run(policy)
    search=UnifiedBreadthDepthSearch(4096)
    for plan in search.plans(tuple(range(32))):run('breadth_depth_search',plan.candidate_count,plan.depths,plan.family)
    feasible=[row for row in rows if row['feasible'] is True];ranked=rank_policy_rows(feasible);rank_by_signature={row['policy_plan_signature']:row for row in ranked}
    for index,row in enumerate(rows):
        if row['feasible'] is True:rows[index]=rank_by_signature[row['policy_plan_signature']]
    baseline=next(int(row['sequence_makespan']) for row in rows if row['policy']=='no_prediction' and row['feasible'] is True)
    for row in rows:
        row['speedup_vs_no_prediction']=baseline/int(row['sequence_makespan']) if row['feasible'] is True else ''
    return rows

def select_winner(rows:list[dict[str,object]])->dict[str,object]:
    """Collapse BDS to its best plan, then apply canonical exact-tie semantics."""
    feasible=[row for row in rows if row['feasible'] is True];bds=[row for row in feasible if row['policy']=='breadth_depth_search'];best_bds=min(bds,key=lambda row:int(row['search_rank']));canonical=[row for row in feasible if row['policy']!='breadth_depth_search']+[best_bds]
    minimum=min(int(row['sequence_makespan']) for row in canonical);raw=[row for row in canonical if int(row['sequence_makespan'])==minimum];distinct=sorted({int(row['sequence_makespan']) for row in canonical});unique=len(raw)==1;second=distinct[1] if len(distinct)>1 else None;margin=(second/minimum-1)*100 if unique and second is not None else 0.0;preferred=min(canonical,key=lambda row:int(row['search_rank']))
    return {'configuration_id':preferred['configuration_id'],'predictor_quality':preferred['predictor_quality'],'predictor_quality_scalar':preferred['predictor_quality_scalar'],'source_milestone':preferred['source_milestone'],'predictor_latency_cycles':preferred['predictor_latency_cycles'],'realized_lead_cycles':preferred['point_realized_lead_cycles'],'opportunity_expert_equivalents':preferred['opportunity_expert_equivalents'],'raw_latency_winners':'|'.join(sorted(POLICY_LABEL[row['policy']] for row in raw)),'latency_tie':not unique,'strict_winner_margin_percent':margin,'robust_latency_winner':unique and margin>=1.0,'preferred_policy_under_tie':POLICY_LABEL[preferred['policy']],'preferred_makespan':minimum,'speedup_vs_no_prediction':preferred['speedup_vs_no_prediction'],'bds_M':best_bds['candidate_count'],'bds_depth_vector':best_bds['candidate_depth_vector'],'bds_makespan':best_bds['sequence_makespan'],'workload_trace_id':preferred['workload_trace_id'],'predictor_trace_id':preferred['predictor_trace_id'],'hardware_mode':preferred['hardware_mode'],'synthetic_lead_time':False}

def make_figures(output:Path,winners:list[dict[str,object]],evaluations:list[dict[str,object]])->None:
    """Combine old/new phase points and draw a policy-speedup sensitivity figure."""
    import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
    old=read(ROOT/'results/m72a_paper_figure_cleanup/data/fig01a_policy_phase_quality_vs_opportunity_confidence.csv');combined=[dict(row,point_source='M7.2a coarse') for row in old]
    for row in winners:combined.append(dict(row,opportunity=row['opportunity_expert_equivalents'],quality=row['predictor_quality'],point_source='M7.2b refinement'))
    write(output/'fig_m72b_quality_vs_opportunity_refined.csv',combined)
    colors={'NO_PREFETCH':'#333333','FULL_EXPERT':'#377eb8','RANKED_PREFIX':'#4daf4a','BREADTH_DEPTH_SEARCH':'#e41a1c'};fig,ax=plt.subplots(figsize=(8,4.8))
    for row in combined:
        policy=row['preferred_policy_under_tie'];tie=str(row['latency_tie'])=='True';robust=str(row.get('robust_latency_winner',row.get('robust_winner','False')))=='True';marker='s' if tie else ('o' if robust else '^');face='none' if tie else colors[policy];size=95 if row['point_source']=='M7.2b refinement' else 55
        ax.scatter(float(row['opportunity']),float(row['predictor_quality_scalar']),marker=marker,s=size,facecolors=face,edgecolors=colors[policy],linewidths=1.5)
    for label,color in colors.items():ax.scatter([],[],color=color,label=label.replace('_',' ').title())
    ax.scatter([],[],marker='o',color='gray',label='Robust ≥1%');ax.scatter([],[],marker='^',color='gray',label='Unique <1%');ax.scatter([],[],marker='s',facecolors='none',edgecolors='gray',label='Exact tie / cost-preferred');ax.set(xlabel='Realized prefetch opportunity (Expert equivalents)',ylabel='Predictor quality scalar');ax.grid(alpha=.2);ax.legend(fontsize=7,ncol=2);fig.tight_layout();fig.savefig(output/'fig_m72b_quality_vs_opportunity_refined.png',dpi=220);fig.savefig(output/'fig_m72b_quality_vs_opportunity_refined.pdf');plt.close(fig)
    best=[]
    for cfg in sorted({row['configuration_id'] for row in evaluations}):
        group=[row for row in evaluations if row['configuration_id']==cfg and row['feasible'] is True];bds=min((row for row in group if row['policy']=='breadth_depth_search'),key=lambda row:int(row['search_rank']))
        for policy in ('full_expert_prefetch','ranked_prefix'):
            row=next((row for row in group if row['policy']==policy),None)
            if row:best.append(dict(row,canonical_policy=POLICY_LABEL[policy]))
        best.append(dict(bds,canonical_policy='BREADTH_DEPTH_SEARCH'))
    write(output/'fig_m72b_speedup_vs_opportunity.csv',best);fig,axes=plt.subplots(2,3,figsize=(11,6),sharex=True,sharey=True)
    for ax,quality in zip(axes.flat,QUALITIES):
        for policy,color in (('FULL_EXPERT','#377eb8'),('RANKED_PREFIX','#4daf4a'),('BREADTH_DEPTH_SEARCH','#e41a1c')):
            points=sorted((row for row in best if row['predictor_quality']==quality and row['canonical_policy']==policy),key=lambda row:float(row['opportunity_expert_equivalents']));ax.plot([float(row['opportunity_expert_equivalents']) for row in points],[float(row['speedup_vs_no_prediction']) for row in points],marker='o',label=policy.replace('_',' ').title(),color=color)
        ax.axhline(1,color='gray',lw=.7);ax.set_title(quality.capitalize());ax.grid(alpha=.2)
    axes.flat[-1].axis('off');axes[0,0].legend(fontsize=7);fig.supxlabel('Realized opportunity (Expert equivalents)');fig.supylabel('Speedup vs No Prediction');fig.tight_layout();fig.savefig(output/'fig_m72b_speedup_vs_opportunity.png',dpi=220);fig.savefig(output/'fig_m72b_speedup_vs_opportunity.pdf');plt.close(fig)

def main()->int:
    """Run exactly three timing controls and the bounded refinement matrix."""
    parser=argparse.ArgumentParser();parser.add_argument('--output-dir',type=Path,default=OUT);parser.add_argument('--resume',action='store_true');args=parser.parse_args();out=args.output_dir
    if args.resume and (out/'m72b_policy_evaluations.csv').exists():
        evaluations=[dict(row,feasible=row['feasible']=='True') for row in read(out/'m72b_policy_evaluations.csv')];winners=read(out/'m72b_policy_winners.csv');points=read(out/'m72b_opportunity_points.csv');winner_by={row['configuration_id']:row for row in winners};lead_by={row['predictor_latency_cycles']:row['realized_lead_cycles'] for row in points}
        for row in evaluations:
            win=winner_by[row['configuration_id']];row.update(point_realized_lead_cycles=lead_by[row['predictor_latency_cycles']],raw_latency_winners=win['raw_latency_winners'],latency_tie=win['latency_tie'],robust_latency_winner=win['robust_latency_winner'],preferred_policy_under_tie=win['preferred_policy_under_tie'],bds_M=win['bds_M'],bds_depth_vector=win['bds_depth_vector'])
        write(out/'m72b_policy_evaluations.csv',evaluations);make_figures(out,winners,evaluations);print('M7.2b resumed from CSV; new simulator runs: 0');return 0
    controls=[];runs=0
    for latency in LATENCIES:control,count=calibrate(latency);controls.append(control);runs+=count
    print('Calibrated opportunities:',[(row['predictor_latency_cycles'],row['opportunity_expert_equivalents']) for row in controls])
    evaluations=[];winners=[];counter=[0]
    for control in controls:
        for quality in QUALITIES:
            cfg=base_config(quality,int(control['predictor_latency_cycles']),float(control['opportunity_expert_equivalents']));cfg['point_realized_lead_cycles']=control['realized_lead_cycles'];rows=evaluate_point(cfg,counter);winner=select_winner(rows)
            for row in rows:row.update(point_realized_lead_cycles=control['realized_lead_cycles'],raw_latency_winners=winner['raw_latency_winners'],latency_tie=winner['latency_tie'],robust_latency_winner=winner['robust_latency_winner'],preferred_policy_under_tie=winner['preferred_policy_under_tie'],bds_M=winner['bds_M'],bds_depth_vector=winner['bds_depth_vector'])
            evaluations.extend(rows);winners.append(winner)
    runs+=counter[0]
    old=read(ROOT/'results/m72a_paper_figure_cleanup/data/fig01a_policy_phase_quality_vs_opportunity_confidence.csv');old_trace={row['quality']:next(x['predictor_trace_id'] for x in read(ROOT/'results/m72_policy_phase/raw/m72_policy_evaluations.csv') if x['configuration_id']==row['configuration_id']) for row in old}
    assert all(row['synthetic_lead_time'] is False and row['simulator_scope']=='unified_cross_layer' for row in evaluations if row['feasible'] is True)
    assert all(winner['predictor_trace_id']==old_trace[winner['predictor_quality']] for winner in winners)
    assert all(.381<float(row['opportunity_expert_equivalents'])<4.546 for row in controls)
    write(out/'m72b_policy_evaluations.csv',evaluations);write(out/'m72b_policy_winners.csv',winners);write(out/'m72b_opportunity_points.csv',controls);make_figures(out,winners,evaluations)
    (out/'run_summary.json').write_text(json.dumps({'new_simulator_runs':runs,'timing_control_runs':3,'policy_plan_runs':counter[0],'full_m72_search_rerun':False,'source_milestone':'source_router_done','latencies':LATENCIES},indent=2)+'\n');print(f'M7.2b wrote {out}; new simulator runs: {runs}');return 0
if __name__=='__main__':raise SystemExit(main())
