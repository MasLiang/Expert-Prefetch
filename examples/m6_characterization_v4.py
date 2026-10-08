"""M6.8 clean fixed-ranking breadth smoke generator."""
from __future__ import annotations
import argparse,json,sys,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from edge_moe_sim.advanced_schema import AdvancedExecutionConfig,AdvancedScenario
from edge_moe_sim.advanced_simulator import AdvancedSimulator
from edge_moe_sim.characterization import _row,controlled_routing,rank_controlled_predictor,write_rows
from edge_moe_sim.config import load_hardware_profile
from edge_moe_sim.schema import ModelConfig
def main():
 p=argparse.ArgumentParser();p.add_argument('--smoke',action='store_true');p.add_argument('--output-dir',type=Path,default=ROOT/'results/characterization_v4');a=p.parse_args();m=ModelConfig('m68',2048,1408,64,4,8,8);h=load_hardware_profile(ROOT/'configs/hardware_profile_synthetic_balanced.yaml');e=controlled_routing(batch_size=1,top_k=4,num_experts=64,pattern='max_reuse');pred,ranking,tid=rank_controlled_predictor(e,64,true_ranks_one_based=(1,3,7,13),seed=19);rows=[]
 for width in (2,4,8,16):
  cfg=AdvancedExecutionConfig(352,'round_robin_expert',16,4,candidate_count=width,uniform_gu_prefetch_depth=1,speculative_byte_budget=64*1024*1024)
  r=AdvancedSimulator(m,h).simulate(e,AdvancedScenario.PREDICTIVE_UNIFORM_PREFIX,cfg,predictor=pred);row=_row(r,experiment_name='breadth_fixed_depth',run_id=f'progressive-M{width}',seed=19,model=m,profile=h,config=cfg,routing_pattern='max_reuse',predictor_profile='breadth_progressive');c=pred.predict(e).aggregate_candidates(width);row.update(workload_trace_id='breadth-progressive-workload',predictor_trace_id=tid,hardware_profile_id=h.name,true_expert_rank_vector='[1,3,7,13]',full_prediction_ranking=json.dumps(ranking),true_expert_ranks='[1,3,7,13]',truncated_candidates=json.dumps([x.expert_id for x in c]),Recall_at_M=len({x.expert_id for x in c}&set(e.expert_token_counts()))/4,candidate_depth_vector=json.dumps([1]*width));rows.append(row)
 out=a.output_dir;write_rows(rows,out/'raw/breadth_fixed_depth.csv')
 budget_rows=[]; tile_bytes=2*m.hidden_size*352*m.weight_bits//8; budget=16*tile_bytes
 for width in (2,4,8,16):
  uniform=tuple(min(4,16//width) for _ in range(width))
  weighted=[1 for _ in range(width)]
  remaining=16-width
  for rank in range(width):
   extra=min(3,remaining);weighted[rank]+=extra;remaining-=extra
   if remaining==0:break
  rank_weighted=tuple(weighted)
  for policy,depths in (('uniform_depth',uniform),('rank_weighted_depth',rank_weighted),('DSE_optimal',rank_weighted)):
   used=sum(depths)*tile_bytes
   if used>budget: raise RuntimeError('fixed budget planning exceeded')
   cfg=AdvancedExecutionConfig(352,'round_robin_expert',16,4,candidate_count=width,explicit_depths=depths,speculative_byte_budget=budget)
   r=AdvancedSimulator(m,h).simulate(e,AdvancedScenario.PREDICTIVE_DSE,cfg,predictor=pred);row=_row(r,experiment_name='breadth_fixed_budget',run_id=f'{policy}-M{width}',seed=19,model=m,profile=h,config=cfg,routing_pattern='max_reuse',predictor_profile='breadth_progressive');c=pred.predict(e).aggregate_candidates(width);row.update(workload_trace_id='breadth-progressive-workload',predictor_trace_id=tid,hardware_profile_id=h.name,true_expert_rank_vector='[1,3,7,13]',full_prediction_ranking=json.dumps(ranking),truncated_candidates=json.dumps([x.expert_id for x in c]),Recall_at_M=len({x.expert_id for x in c}&set(e.expert_token_counts()))/4,depth_policy=policy,candidate_depth_vector=json.dumps(depths),speculative_budget_bytes=budget,planned_speculative_bytes=used,unused_budget_bytes=budget-used,mean_prefix_depth=sum(depths)/width)
   if row['issued_speculative_bytes']>budget:raise RuntimeError('strict issued budget exceeded')
   budget_rows.append(row)
 write_rows(budget_rows,out/'raw/breadth_fixed_budget.csv')
 # V3 owns the already controlled Top-K/tile sweeps.  Copy their immutable
 # CSV artifacts into V4 so the final plotting stage has one input directory.
 v3=ROOT/'results/characterization_v3/raw'
 for source,target in (('topk_memory_wave_corrected.csv','topk_round_robin_clean.csv'),('tile_size_corrected.csv','tile_size_clean_source.csv')):
  if (v3/source).exists():
   (out/'raw').mkdir(parents=True,exist_ok=True);shutil.copy2(v3/source,out/'raw'/target)
 (out/'characterization_v4_summary.json').write_text(json.dumps({'recall_progression':[x['Recall_at_M'] for x in rows],'fixed_budget_bytes':budget,'note':'Synthetic input; fixed ranking truncated by M.'},indent=2)+'\n');print(f'Wrote {out}/raw/breadth_fixed_depth.csv')
if __name__=='__main__':main()
