"""M7.2 paper-oriented unified policy DSE and phase characterization."""
from __future__ import annotations
import argparse,ast,csv,hashlib,json,sys
from pathlib import Path
from statistics import fmean,median
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'examples')]
import m71_unified_characterization as m71
from edge_moe_sim.unified_policy_search import UnifiedBreadthDepthSearch,rank_policy_rows
OUT=ROOT/'results/m72_policy_phase'
RANKS={'perfect':(1,2,3,4),'high':(1,2,4,7),'medium':(1,3,7,13),'low':(3,7,13,25),'wrong':(17,21,25,29)}
OPPS=(('very_low','source_expert_done',500),('low','source_expert_done',20),('medium','source_gate_up_done',20),('high','source_layer_start',20))
DEPRECATED_SPEC_FIELDS={'planned_speculative_bytes','speculative_bytes','useful_speculative_bytes','wasted_speculative_bytes','actual_speculative_bytes_before_router','useful_true_bytes_before_router'}
_REFERENCE_CACHE={}
def hardware_fields(mode,channels):
 """Return explicit study inputs and Roofline identities for every output row."""
 if mode=='vek280_theoretical':
  return {'hardware_evidence':'theoretical_not_measured','physical_aie_ml_tiles':304,'logical_aie_groups':304,'physical_aie_tiles_per_compute_group':1,'theoretical_memory_bandwidth_bytes_per_sec':102.4e9,'effective_memory_bandwidth_bytes_per_sec':102.4e9,'theoretical_compute_peak_ops_per_sec':202e12,'theoretical_ridge_ops_per_byte':202e12/102.4e9,'mac_operation_convention':'1 MAC = 2 operations'}
 return {'hardware_evidence':'synthetic_placeholder','physical_aie_ml_tiles':'','logical_aie_groups':16,'physical_aie_tiles_per_compute_group':'','theoretical_memory_bandwidth_bytes_per_sec':channels*16e9,'effective_memory_bandwidth_bytes_per_sec':channels*16e9*.8,'theoretical_compute_peak_ops_per_sec':16*32*1e9,'theoretical_ridge_ops_per_byte':(16*32*1e9)/(channels*16e9),'mac_operation_convention':'configured operations; not a VEK280 claim'}
def write(path,rows):
 path.parent.mkdir(parents=True,exist_ok=True)
 if not rows:path.write_text('',encoding='utf-8');return
 with path.open('w',newline='',encoding='utf-8') as f:w=csv.DictWriter(f,fieldnames=sorted({k for r in rows for k in r}));w.writeheader();w.writerows(rows)
def read(path):
 with path.open(encoding='utf-8') as f:return list(csv.DictReader(f))
def config_id(d):return hashlib.sha256(json.dumps(d,sort_keys=True,separators=(',',':')).encode()).hexdigest()[:16]
def evaluate(cfg,policy,m=16,depths=(),budget=None,hardware_mode='controlled_synthetic_hardware',trace=False,comparison_mode='unconstrained',search_family=''):
 engine_policy='uniform_prefix' if policy=='breadth_depth_search' else policy
 row,result=m71.run(policy=engine_policy,milestone=cfg['milestone'],predictor_latency=cfg['latency'],quality=cfg['quality'],m=m,channels=cfg['channels'],ocm=cfg['ocm'],top_k=cfg.get('top_k',4),ranked=True,true_ranks=cfg['true_ranks'],candidate_depths=depths,budget=budget,hardware_mode=hardware_mode,batch_size=cfg.get('batch_size',1),routing_shape=cfg.get('routing_shape','high_reuse'),placement_policy=cfg.get('placement_policy','round_robin_expert'),predictor_seed=cfg.get('predictor_seed',17),trace_enabled=trace)
 assert row['simulator_scope']=='unified_cross_layer' and row['synthetic_lead_time'] is False
 for key in DEPRECATED_SPEC_FIELDS:row.pop(key,None)
 candidate_ids=tuple(result.predictions[0].candidate_ids[:m]) if result.predictions else ()
 total_depth=sum(depths)
 row.update(configuration_id=cfg['configuration_id'],comparison_id=f"{cfg['configuration_id']}:{comparison_mode}",opportunity_setting=cfg['opportunity_setting'],configuration_opportunity_expert_equivalents=cfg['opportunity'],predictor_quality_scalar=cfg['quality_scalar'],true_expert_rank_vector=str(cfg['true_ranks']),ocm_capacity_over_demand_reference=cfg['ocm']/cfg['demand_reference'],demand_working_set_reference_bytes=cfg['demand_reference'],speculative_headroom_bytes=max(0,cfg['ocm']-cfg['demand_reference']),comparison_mode=comparison_mode,policy_plan_signature=f'{policy}:M{m}:I{candidate_ids}:D{depths}:B{budget}',candidate_ids=str(candidate_ids),candidate_depth_vector=str(depths),search_family=search_family or policy,total_depth=total_depth,mean_depth=total_depth/len(depths) if depths else 0,depth_concentration=max(depths,default=0)/total_depth if total_depth else 0,policy=policy,sequence_makespan_cycles=row['sequence_makespan'],**hardware_fields(hardware_mode,cfg['channels']))
 if comparison_mode=='fixed_budget':row['speculative_budget_bytes']=(budget or 0)*2
 return row,result
def demand_reference(top_k=4,batch_size=1,channels=2,hardware_mode='controlled_synthetic_hardware',routing_shape='high_reuse',placement_policy='round_robin_expert'):
 """Derive the smallest OCM capacity within 1% of the large-OCM demand run."""
 key=(top_k,batch_size,channels,hardware_mode,routing_shape,placement_policy)
 if key in _REFERENCE_CACHE:return _REFERENCE_CACHE[key]
 caps=(8192,16384,32768,49152,65536,98304,131072,262144,1048576);latencies=[]
 truth_count=top_k if routing_shape=='high_reuse' or batch_size==1 else (min(32,batch_size*top_k) if routing_shape=='balanced' else min(32,2*top_k))
 ranks=tuple(range(1,truth_count+1))
 for cap in caps:
  row,_=m71.run(policy='no_prediction',milestone='source_gate_up_done',predictor_latency=20,quality='perfect',m=max(16,top_k),channels=channels,ocm=cap,top_k=top_k,ranked=True,true_ranks=ranks,hardware_mode=hardware_mode,batch_size=batch_size,routing_shape=routing_shape,placement_policy=placement_policy)
  latencies.append((cap,row['sequence_makespan']))
 large=latencies[-1][1];reference=next(cap for cap,value in latencies if value<=large*1.01)
 _REFERENCE_CACHE[key]=reference;return reference
def make_config(quality,opp,ocm,channels,top_k=4,batch_size=1,hardware_mode='controlled_synthetic_hardware',routing_shape='high_reuse',placement_policy='round_robin_expert',true_ranks=None,predictor_seed=17):
 label,milestone,latency=opp;ranks=tuple(true_ranks) if true_ranks is not None else (RANKS[quality] if top_k==4 else tuple(range(1,top_k+1)));base={'quality':quality,'opportunity_setting':label,'milestone':milestone,'latency':latency,'ocm':ocm,'channels':channels,'top_k':top_k,'batch_size':batch_size,'routing_shape':routing_shape,'placement_policy':placement_policy,'predictor_seed':predictor_seed,'true_ranks':ranks,'quality_scalar':fmean(1/r for r in ranks),'demand_reference':demand_reference(top_k,batch_size,channels,hardware_mode,routing_shape,placement_policy)}
 base['configuration_id']=config_id(base);control,_=evaluate(dict(base,opportunity=0), 'uniform_prefix',m=16 if top_k==4 else top_k,depths=(0,)*(16 if top_k==4 else top_k),hardware_mode=hardware_mode);base['opportunity']=control['prefetch_opportunity_expert_equivalents'];return base
def evaluate_configuration(cfg,max_counter,hardware_mode='controlled_synthetic_hardware',comparison_mode='unconstrained',budget_per_prediction=None):
 rows=[]
 dynamic_pruned=0
 def add(policy,m=16,depths=(),budget=None,search_family=''):
  nonlocal dynamic_pruned
  if max_counter[0]>=max_counter[1]:raise RuntimeError('max simulations reached')
  effective_budget=None if policy=='no_prediction' else budget
  try:row,_=evaluate(cfg,policy,m,depths,effective_budget,hardware_mode,comparison_mode=comparison_mode,search_family=search_family)
  except RuntimeError:
   max_counter[0]+=1;dynamic_pruned+=1;return
  max_counter[0]+=1;rows.append(row)
 add('no_prediction',m=16);add('full_expert_prefetch',m=16,budget=budget_per_prediction);add('uniform_prefix',m=16,budget=budget_per_prediction);add('ranked_prefix',m=16,budget=budget_per_prediction)
 search=UnifiedBreadthDepthSearch(4096);plans,stats=search.plans_with_stats(tuple(range(32)),budget_per_prediction)
 seen=set()
 for plan in plans:
  signature=f'{plan.candidate_count}:{plan.depths}'
  if signature in seen:continue
  seen.add(signature);add('breadth_depth_search',plan.candidate_count,plan.depths,budget_per_prediction,plan.family)
 ranked=rank_policy_rows(rows)
 if comparison_mode=='fixed_budget':
  total_budget=(budget_per_prediction or 0)*2
  assert all(int(r['initial_allocated_speculative_bytes'])<=total_budget for r in ranked)
 return ranked,stats,len(seen),dynamic_pruned
def winner(cfg_rows):
 best=cfg_rows[0];best_latency=best['sequence_makespan'];raw=[r['policy'] for r in cfg_rows if r['sequence_makespan']==best_latency];second=next((r for r in cfg_rows if r['sequence_makespan']>best_latency),best);margin=second['sequence_makespan']/best_latency-1 if second is not best else 0
 labels={'no_prediction':'NO_PREFETCH','full_expert_prefetch':'FULL_EXPERT','uniform_prefix':'UNIFORM_PREFIX','ranked_prefix':'RANKED_PREFIX','breadth_depth_search':'BREADTH_DEPTH_SEARCH'}
 return dict(configuration_id=best['configuration_id'],comparison_id=best['comparison_id'],comparison_mode=best['comparison_mode'],winning_policy=labels[best['policy']],preferred_policy_under_tie=labels[best['policy']],raw_latency_winners='|'.join(sorted(set(raw))),latency_tie=len(set(raw))>1,best_makespan=best_latency,second_best_policy=labels[second['policy']],winner_margin_percent=margin*100,robust_winner=margin>=.01,policies_within_1pct='|'.join(sorted(set(labels[r['policy']] for r in cfg_rows if r['sequence_makespan']/best_latency<=1.01))),policies_within_5pct='|'.join(sorted(set(labels[r['policy']] for r in cfg_rows if r['sequence_makespan']/best_latency<=1.05))),winning_M=best['candidate_count'],winning_depth_vector=best['candidate_depth_vector'],speedup_vs_no_prefetch=next(r['sequence_makespan'] for r in cfg_rows if r['policy']=='no_prediction')/best_latency,quality=best['predictor_quality'],predictor_quality_scalar=best['predictor_quality_scalar'],opportunity=best['configuration_opportunity_expert_equivalents'],opportunity_setting=best['opportunity_setting'],ocm_bytes=best['ocm_bytes'],ocm_ratio=best['ocm_capacity_over_demand_reference'],channels=best['memory_channels'],top_k=best['top_k'],batch_size=best['batch_size'],routing_shape=best.get('routing_shape','high_reuse'),memory_placement_policy=best.get('memory_placement_policy','round_robin_expert'),critical_gap=best['critical_expert_startup_gap_cycles'],physical_bytes=best['physical_speculative_bytes_transferred_by_router'],wasted_bytes=best['wasted_physical_speculative_bytes'],peak_ocm=best['peak_ocm_bytes'],explanation='near_tie' if margin<.01 else ('insufficient_opportunity' if best['policy']=='no_prediction' else ('full_prefetch_useful' if best['policy']=='full_expert_prefetch' else ('prefix_bandwidth_efficient' if best['policy']!='breadth_depth_search' else 'mixed_or_unclassified'))))

def full_prefix_table(evaluations,winners):
 rows=[]
 for w in winners:
  group=[r for r in evaluations if r['comparison_id']==w['comparison_id']]
  full=next((r for r in group if r['policy']=='full_expert_prefetch'),None);prefix=next((r for r in group if r['policy']=='ranked_prefix'),None);searches=[r for r in group if r['policy']=='breadth_depth_search'];search=min(searches,key=lambda r:float(r['sequence_makespan'])) if searches else None
  if full is None or prefix is None or search is None:
   rows.append({'configuration':w['comparison_id'],'comparison_mode':w['comparison_mode'],'full_latency':full['sequence_makespan'] if full else 'infeasible','prefix_latency':prefix['sequence_makespan'] if prefix else 'infeasible','search_latency':search['sequence_makespan'] if search else 'infeasible','full_physical_bytes':full['physical_speculative_bytes_transferred_by_router'] if full else 'infeasible','prefix_physical_bytes':prefix['physical_speculative_bytes_transferred_by_router'] if prefix else 'infeasible','full_waste':full['wasted_physical_speculative_bytes'] if full else 'infeasible','prefix_waste':prefix['wasted_physical_speculative_bytes'] if prefix else 'infeasible','full_peak_ocm':full['peak_ocm_bytes'] if full else 'infeasible','prefix_peak_ocm':prefix['peak_ocm_bytes'] if prefix else 'infeasible','interpretation':'search_infeasible' if search is None else ('full_infeasible' if full is None else 'prefix_infeasible')});continue
  full_latency=float(full['sequence_makespan']);prefix_latency=float(prefix['sequence_makespan'])
  rows.append({'configuration':w['comparison_id'],'comparison_mode':w['comparison_mode'],'full_latency':full['sequence_makespan'],'prefix_latency':prefix['sequence_makespan'],'search_latency':search['sequence_makespan'],'full_physical_bytes':full['physical_speculative_bytes_transferred_by_router'],'prefix_physical_bytes':prefix['physical_speculative_bytes_transferred_by_router'],'full_waste':full['wasted_physical_speculative_bytes'],'prefix_waste':prefix['wasted_physical_speculative_bytes'],'full_peak_ocm':full['peak_ocm_bytes'],'prefix_peak_ocm':prefix['peak_ocm_bytes'],'interpretation':'prefix_latency_advantage' if prefix_latency<full_latency else ('full_latency_advantage' if full_latency<prefix_latency else 'latency_tie')})
 return rows

def export_representative_traces(root,representatives,evaluations):
 """Rerun only selected cells with tracing and export compact causal evidence."""
 (root/'traces').mkdir(parents=True,exist_ok=True)
 timing_rows=[];case_rows=[]
 for index,w in enumerate(representatives,1):
  row=next(r for r in evaluations if r['comparison_id']==w['comparison_id'] and int(r['search_rank'])==1)
  cfg={'configuration_id':row['configuration_id'],'milestone':row['source_milestone'],'latency':int(row['predictor_latency_cycles']),'quality':row['predictor_quality'],'true_ranks':tuple(ast.literal_eval(row['true_expert_rank_vector'])),'channels':int(row['memory_channels']),'ocm':int(row['ocm_bytes']),'top_k':int(row['top_k']),'batch_size':int(row['batch_size']),'routing_shape':row.get('routing_shape','high_reuse'),'placement_policy':row.get('memory_placement_policy','round_robin_expert'),'opportunity_setting':row['opportunity_setting'],'opportunity':float(row['configuration_opportunity_expert_equivalents']),'quality_scalar':float(row['predictor_quality_scalar']),'demand_reference':int(row['demand_working_set_reference_bytes'])}
  depths=tuple(ast.literal_eval(row['candidate_depth_vector']));budget=row['speculative_budget_bytes'];budget_per_edge=(int(budget)//2 if str(budget) not in ('','0') else None)
  _,result=evaluate(cfg,row['policy'],int(row['candidate_count']),depths,budget_per_edge,row['hardware_mode'],True,row['comparison_mode'],row.get('search_family',''))
  trace_name=f'case{index:02d}_{w["winning_policy"].lower()}.json';(root/'traces'/trace_name).write_text(json.dumps(result.trace,indent=2)+'\n')
  for layer in result.layer_results.values():
   for timing in layer.expert_timings:
    record=dict(timing);record.update(case_id=index,configuration_id=w['configuration_id'],policy=row['policy'],layer_id=layer.layer_id,selected_critical_expert=(int(timing['expert_id'])==int(row['critical_expert_id']) and layer.layer_id in (1,2)),critical_path_demand_blocking_cycles=row['critical_path_demand_blocking_cycles']);timing_rows.append(record)
  case_rows.append(dict(w,trace_file=f'traces/{trace_name}',policy_plan_signature=row['policy_plan_signature']))
 write(root/'traces/m72_representative_cases.csv',case_rows);write(root/'traces/m72_representative_expert_timing.csv',timing_rows)
 return case_rows
def main():
 p=argparse.ArgumentParser();p.add_argument('--smoke',action='store_true');p.add_argument('--full',action='store_true');p.add_argument('--experiment',choices=('phase-quality-opportunity','phase-quality-ocm','phase-opportunity-channels','breadth-depth','topk','reuse','regret','vek280'));p.add_argument('--max-simulations',type=int,default=20000);p.add_argument('--resume',action='store_true');p.add_argument('--output-dir',type=Path,default=OUT);a=p.parse_args();root=a.output_dir
 if a.resume and (root/'raw/m72_policy_evaluations.csv').exists():
  evaluations=read(root/'raw/m72_policy_evaluations.csv');winners=read(root/'raw/m72_policy_winners.csv');regret=read(root/'raw/m72_heuristic_regret.csv');vek=read(root/'vek280_theoretical/m72_vek280_theoretical.csv')
  for row in evaluations:row.update(hardware_fields(row['hardware_mode'],int(row['memory_channels'])))
  for row in vek:row.update(hardware_fields('vek280_theoretical',int(row['memory_channels'])))
  write(root/'raw/m72_policy_evaluations.csv',evaluations);write(root/'search/m72_breadth_depth_search.csv',[row for row in evaluations if row['policy']=='breadth_depth_search']);write(root/'vek280_theoretical/m72_vek280_theoretical.csv',vek)
  eligible=[w for w in winners if w['comparison_mode']=='unconstrained'];representative=[]
  for label in ('NO_PREFETCH','FULL_EXPERT','UNIFORM_PREFIX','RANKED_PREFIX','BREADTH_DEPTH_SEARCH'):
   if any(w['winning_policy']==label for w in eligible):representative.append(next(w for w in eligible if w['winning_policy']==label))
  for extra in (min(eligible,key=lambda w:float(w['winner_margin_percent'])),next((w for w in eligible if w['quality']=='wrong'),None)):
   if extra and all(x['configuration_id']!=extra['configuration_id'] for x in representative):representative.append(extra)
  representative=representative[:6];write(root/'tables/table1_policy_regions.csv',winners);write(root/'tables/table2_full_vs_prefix.csv',full_prefix_table(evaluations,winners));write(root/'tables/table3_breadth_depth.csv',[r for r in evaluations if r['policy']=='breadth_depth_search' and int(r['search_rank'])<=3]);write(root/'tables/table4_static_policy_regret.csv',regret);write(root/'tables/table5_vek280_theoretical.csv',vek);export_representative_traces(root,representative,evaluations)
  configs=len(winners)//2;uncon_stats=UnifiedBreadthDepthSearch(4096).plans_with_stats(tuple(range(32)),None)[1];budget_stats=UnifiedBreadthDepthSearch(4096).plans_with_stats(tuple(range(32)),65536)[1];stats={key:configs*(uncon_stats[key]+budget_stats[key]) for key in uncon_stats};refined=read(root/'phase/m72_refined_boundary.csv');total_attempts=configs*(8+uncon_stats['unique']+budget_stats['unique']);secondary_points=4+5+2+3+4+len(refined);summary={'configurations':configs,'comparison_modes':['unconstrained','fixed_budget'],'fixed_budget_bytes':131072,'main_grid_simulation_attempts':total_attempts,'total_simulation_attempts':total_attempts+secondary_points*(4+uncon_stats['unique']),'main_grid_feasible_rows':len(evaluations),'search_plan_statistics':stats,'dynamically_infeasible_policy_plans':total_attempts-len(evaluations),'refined_boundary_points':len(refined),'representative_traces':len(representative),'resume_mode':'CSV post-processing; no simulation rerun'};(root/'summaries').mkdir(parents=True,exist_ok=True);(root/'summaries/run_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
  (root/'configs').mkdir(parents=True,exist_ok=True);(root/'configs'/'m72_hardware_modes.json').write_text(json.dumps({'controlled_synthetic_hardware':hardware_fields('controlled_synthetic_hardware',2),'vek280_theoretical':hardware_fields('vek280_theoretical',3)},indent=2)+'\n')
  import subprocess;subprocess.run([sys.executable,str(ROOT/'examples/m72_make_figures.py'),'--input-dir',str(root)],check=True);print(f'Resumed M7.2 post-processing from {root}');return 0
 qualities=('perfect','medium') if a.smoke else tuple(RANKS);opps=OPPS[:2] if a.smoke else OPPS;ocm_ratios=(1.0,2.0) if a.smoke else (.75,1.0,1.5,4.0);channels=(1,2) if a.smoke else (1,2,3)
 configs=[]
 for q in qualities:
  for opp in opps:
   for ratio in ocm_ratios:
    for ch in channels:
     reference=demand_reference(4,1,ch);ocm=max(4096,int(reference*ratio));configs.append(make_config(q,opp,ocm,ch))
 estimate=len(configs);search_count=len(UnifiedBreadthDepthSearch(4096).plans(tuple(range(32))));print(f'Estimate: {estimate} configurations, {4+search_count} plans/configuration, <= {estimate*(4+search_count)} simulations')
 counter=[0,a.max_simulations];evaluations=[];winners=[];search_stats={'generated':0,'unique':0,'static_pruned':0,'duplicate_pruned':0};dynamic_pruned=0
 for cfg in configs:
  for mode,budget in (('unconstrained',None),('fixed_budget',65536)):
   rows,stats,u,d=evaluate_configuration(cfg,counter,comparison_mode=mode,budget_per_prediction=budget)
   for key in search_stats:search_stats[key]+=stats[key]
   assert u==stats['unique'];dynamic_pruned+=d;evaluations.extend(rows);w=winner(rows);winners.append(w)
   for r in rows:r['best_policy_for_configuration']=w['winning_policy']
 # Reduced sensitivities.
 topk=[]
 for k in (1,2,4,6):
  cfg=make_config('perfect',OPPS[2],131072,2,top_k=k);rows,_,_,_=evaluate_configuration(cfg,counter);w=winner(rows);w.update(mean_tokens_per_expert=1,reuse_factor=1,unique_active_experts=k);topk.append(w)
 reuse=[]
 for b in (1,4,16):
  cfg=make_config('medium',OPPS[2],131072,2,batch_size=b);rows,_,_,_=evaluate_configuration(cfg,counter);w=winner(rows);w.update(mean_tokens_per_expert=b,reuse_factor=b,unique_active_experts=4,workload_class='synthetic_batched_high_reuse');reuse.append(w)
 # Balanced/skewed B=4 routing and selected placement sensitivity.
 shape_rows=[]
 for shape,count in (('balanced',16),('skewed',8)):
  cfg=make_config('perfect',OPPS[2],262144,2,batch_size=4,routing_shape=shape,true_ranks=tuple(range(1,count+1)));rows,_,_,_=evaluate_configuration(cfg,counter);w=winner(rows);w.update(mean_tokens_per_expert=16/count,reuse_factor=16/count,unique_active_experts=count,workload_class='synthetic_batched_routing_shape');shape_rows.append(w)
 placement=[]
 for place in ('round_robin_expert','striped_tile'):
  cfg=make_config('medium',OPPS[2],131072,1,placement_policy=place);rows,_,_,_=evaluate_configuration(cfg,counter);placement.append(winner(rows))
 # Small theoretical subset.
 vek=[]
 for q in ('perfect','medium','wrong'):
  cfg=make_config(q,OPPS[2],1<<20,3,hardware_mode='vek280_theoretical');rows,_,_,_=evaluate_configuration(cfg,counter,'vek280_theoretical')
  base=next(r['sequence_makespan'] for r in rows if r['policy']=='no_prediction')
  for r in rows:
   if r['search_rank']<=5:r.update(hardware_evidence='theoretical_not_measured',speedup_vs_no_prefetch=base/r['sequence_makespan']);vek.append(r)
 # One-level boundary refinement: only the real-latency interval sharing the
 # source_expert_done milestone is interpolated, never target-local lead time.
 refined=[]
 if not a.smoke:
  coarse=[w for w in winners if w['comparison_mode']=='unconstrained']
  for q in RANKS:
   pairs=[(x,y) for x in coarse for y in coarse if x['quality']==q==y['quality'] and x['opportunity_setting']=='very_low' and y['opportunity_setting']=='low' and x['ocm_bytes']==y['ocm_bytes'] and x['channels']==y['channels'] and (x['winning_policy']!=y['winning_policy'] or min(x['winner_margin_percent'],y['winner_margin_percent'])<5)]
   for x,_ in pairs[:2]:
    cfg=make_config(q,('refined_very_low_low','source_expert_done',260),int(x['ocm_bytes']),int(x['channels']));rows,_,_,_=evaluate_configuration(cfg,counter);refined.append(winner(rows))
 # Phase projections and regret.
 representative_ratio=1.5;representative_opp=opps[len(opps)//2][0];representative_channel=2 if 2 in channels else channels[-1];representative_ocm=int(demand_reference(4,1,representative_channel)*representative_ratio)
 quality_opp=[w for w in winners if w.get('comparison_mode','unconstrained')=='unconstrained' and abs(float(w['ocm_ratio'])-representative_ratio)<1e-9 and w['channels']==representative_channel];quality_ocm=[w for w in winners if w.get('comparison_mode','unconstrained')=='unconstrained' and w['opportunity_setting']==representative_opp and w['channels']==representative_channel];opp_ch=[w for w in winners if w.get('comparison_mode','unconstrained')=='unconstrained' and w['quality']=='medium' and abs(float(w['ocm_ratio'])-representative_ratio)<1e-9]
 regret=[]
 main_winners=[w for w in winners if w['comparison_mode']=='unconstrained']
 for heuristic,policy in [('always_full','full_expert_prefetch'),('always_prefix','ranked_prefix'),('always_no_prefetch','no_prediction')]:
  vals=[]
  for w in main_winners:
   row=next((r for r in evaluations if r['comparison_id']==w['comparison_id'] and r['policy']==policy),None)
   if row is not None:vals.append(row['sequence_makespan']/w['best_makespan']-1)
  regret.append({'heuristic':heuristic,'feasible_configurations':len(vals),'total_configurations':len(main_winners),'coverage_fraction':len(vals)/len(main_winners),'infeasible_fraction':1-len(vals)/len(main_winners),'mean_regret_percent':fmean(vals)*100,'median_regret_percent':median(vals)*100,'max_regret_percent':max(vals)*100,'fraction_within_1pct':sum(v<=.01 for v in vals)/len(vals),'fraction_within_5pct':sum(v<=.05 for v in vals)/len(vals)})
 # Deterministic ranking variants quantify nominal-winner portability.
 robustness=[]
 nominal_cfg=make_config('medium',OPPS[2],representative_ocm,representative_channel)
 nominal_rows,_,_,_=evaluate_configuration(nominal_cfg,counter);nominal=winner(nominal_rows);nominal_policy=next(r['policy'] for r in nominal_rows if r['search_rank']==1);nominal_depths=tuple(ast.literal_eval(next(r['candidate_depth_vector'] for r in nominal_rows if r['search_rank']==1)));nominal_m=int(next(r['candidate_count'] for r in nominal_rows if r['search_rank']==1))
 for seed in (17,23,31):
  cfg=make_config('medium',OPPS[2],representative_ocm,representative_channel,predictor_seed=seed);rows,_,_,_=evaluate_configuration(cfg,counter);best=winner(rows);chosen=next((r for r in rows if r['policy']==nominal_policy and int(r['candidate_count'])==nominal_m and tuple(ast.literal_eval(r['candidate_depth_vector']))==nominal_depths),next(r for r in rows if r['policy']==nominal_policy));robustness.append({'variant_seed':seed,'winner':best['winning_policy'],'nominal_policy':nominal['winning_policy'],'winner_consistent':best['winning_policy']==nominal['winning_policy'],'nominal_policy_regret_percent':(chosen['sequence_makespan']/best['best_makespan']-1)*100})
 representative=[];eligible=[w for w in winners if w['comparison_mode']=='unconstrained']
 for label in ('NO_PREFETCH','FULL_EXPERT','UNIFORM_PREFIX','RANKED_PREFIX','BREADTH_DEPTH_SEARCH'):
  if any(w['winning_policy']==label for w in eligible):representative.append(next(w for w in eligible if w['winning_policy']==label))
 near=min(eligible,key=lambda w:w['winner_margin_percent']);negative=next((w for w in eligible if w['quality']=='wrong'),None)
 for extra in (near,negative):
  if extra and all(x['configuration_id']!=extra['configuration_id'] for x in representative):representative.append(extra)
 representative=representative[:6]
 write(root/'raw/m72_policy_evaluations.csv',evaluations);write(root/'raw/m72_policy_winners.csv',winners);write(root/'search/m72_breadth_depth_search.csv',[r for r in evaluations if r['policy']=='breadth_depth_search']);write(root/'phase/m72_phase_quality_opportunity.csv',quality_opp);write(root/'phase/m72_phase_quality_ocm.csv',quality_ocm);write(root/'phase/m72_phase_opportunity_channels.csv',opp_ch);write(root/'phase/m72_refined_boundary.csv',refined);write(root/'raw/m72_topk_sensitivity.csv',topk);write(root/'raw/m72_batch_reuse_sensitivity.csv',reuse+shape_rows);write(root/'raw/m72_memory_placement_sensitivity.csv',placement);write(root/'raw/m72_trace_robustness.csv',robustness);write(root/'raw/m72_heuristic_regret.csv',regret);write(root/'tables/m72_heuristic_regret.csv',regret);write(root/'vek280_theoretical/m72_vek280_theoretical.csv',vek)
 table2=full_prefix_table(evaluations,winners);write(root/'tables/table1_policy_regions.csv',winners);write(root/'tables/table2_full_vs_prefix.csv',table2);write(root/'tables/table3_breadth_depth.csv',[r for r in evaluations if r['policy']=='breadth_depth_search' and r['search_rank']<=3]);write(root/'tables/table4_static_policy_regret.csv',regret);write(root/'tables/table5_vek280_theoretical.csv',vek)
 representative=export_representative_traces(root,representative,evaluations)
 (root/'configs').mkdir(parents=True,exist_ok=True);(root/'configs'/'m72_experiment.json').write_text(json.dumps({'qualities':qualities,'opportunities':OPPS,'ocm_capacity_ratios':ocm_ratios,'ocm_reference_method':'smallest demand-only capacity within 1% of 1 MiB capacity','memory_channels':channels,'fixed_budget_bytes':131072,'controlled_hardware':hardware_fields('controlled_synthetic_hardware',2),'vek280_theoretical_hardware':hardware_fields('vek280_theoretical',3),'hardware_modes':['controlled_synthetic_hardware','vek280_theoretical']},indent=2)+'\n')
 (root/'diagnostics').mkdir(parents=True,exist_ok=True);write(root/'diagnostics/coarse_grid.csv',[w for w in winners if w['comparison_mode']=='unconstrained']);write(root/'diagnostics/refined_boundary.csv',refined);write(root/'diagnostics/trace_robustness.csv',robustness)
 (root/'summaries').mkdir(parents=True,exist_ok=True);(root/'summaries/run_summary.json').write_text(json.dumps({'configurations':len(configs),'comparison_modes':['unconstrained','fixed_budget'],'fixed_budget_bytes':131072,'simulations':counter[0],'search_plan_statistics':search_stats,'dynamically_infeasible_plans':dynamic_pruned,'refined_boundary_points':len(refined),'representative_traces':len(representative)},indent=2)+'\n')
 import subprocess;subprocess.run([sys.executable,str(ROOT/'examples/m72_make_figures.py'),'--input-dir',str(root)],check=True)
 print(root);return 0
if __name__=='__main__':raise SystemExit(main())
