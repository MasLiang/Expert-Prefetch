"""Targeted M6.7 validation: fixed rankings, strict budgets, and DMA waves."""
from __future__ import annotations
import argparse,json,sys
from dataclasses import replace
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from edge_moe_sim.advanced_schema import AdvancedExecutionConfig,AdvancedScenario
from edge_moe_sim.advanced_simulator import AdvancedSimulator
from edge_moe_sim.characterization import _row,controlled_predictor,controlled_routing,write_rows
from edge_moe_sim.config import load_hardware_profile
from edge_moe_sim.schema import ModelConfig

def parse():
 p=argparse.ArgumentParser(description=__doc__);g=p.add_mutually_exclusive_group(required=True);g.add_argument('--smoke',action='store_true');g.add_argument('--full',action='store_true');p.add_argument('--experiment',choices=['topk-memory-wave','prefetch-window','breadth-depth','tile-size','policy']);p.add_argument('--output-dir',type=Path,default=ROOT/'results/characterization_v3');return p.parse_args()
def one(m,p,*,b=1,k=2,c=3,tile=352,scenario=AdvancedScenario.EXACT_TILE_STREAMING,quality='perfect',mwidth=4,depth=2,budget=64*1024*1024,seed=7,placement='round_robin_expert'):
 p=replace(p,num_memory_channels=c,ocm_bytes=max(p.ocm_bytes,512*1024*1024));e=controlled_routing(batch_size=b,top_k=k,num_experts=m.num_experts,pattern='max_reuse' if b>1 else 'low_reuse',seed=seed)
 # One complete ranking is generated once at M=16; scenario only truncates it.
 pred,tid,rec,anyhit,fullhit=controlled_predictor(e,m.num_experts,profile=quality,candidate_count=16,ready_cycle=0,seed=seed)
 tile_bytes=2*m.hidden_size*min(tile,m.intermediate_size)*m.weight_bits//8
 planned=min(budget,mwidth*depth*tile_bytes);actual_depth=min(depth,planned//max(1,mwidth*tile_bytes))
 cfg=AdvancedExecutionConfig(tile,placement,min(16,p.total_aie_units),min(k,4),candidate_count=mwidth,uniform_gu_prefetch_depth=actual_depth,explicit_depths=tuple(actual_depth for _ in range(mwidth)),speculative_byte_budget=budget)
 r=AdvancedSimulator(m,p).simulate(e,scenario,cfg,predictor=pred if 'predictive' in scenario.value else None)
 row=_row(r,experiment_name='m67',run_id=f'B{b}K{k}C{c}M{mwidth}D{actual_depth}{quality}',seed=seed,model=m,profile=p,config=cfg,routing_pattern='max_reuse' if b>1 else 'low_reuse',predictor_profile=quality)
 row.update(workload_trace_id=f'fixed-route:B{b}:K{k}:seed{seed}',predictor_trace_id=f'fixed-ranking:{quality}:B{b}:K{k}:seed{seed}',hardware_profile_id=p.name,Recall_at_M=rec if mwidth>=16 else len(set(x.expert_id for x in pred.predict(e).aggregate_candidates(mwidth))&set(e.expert_token_counts()))/len(e.expert_token_counts()),AnyHit=anyhit,FullHit=fullhit,speculative_budget_bytes=budget,planned_speculative_bytes=planned,strict_budget=True,analytical_ceil_K_over_C=(len(e.expert_token_counts())+c-1)//c)
 if row['issued_speculative_bytes']>budget or planned>budget:raise RuntimeError('strict speculative budget exceeded')
 if row['useful_speculative_bytes']>row['speculative_bytes']:raise RuntimeError('speculative byte accounting violated')
 return row
def main():
 a=parse();m=ModelConfig('m67',2048,1408,64,12,8,8);s=load_hardware_profile(ROOT/'configs/hardware_profile_synthetic_balanced.yaml');v=load_hardware_profile(ROOT/'configs/hardware_profile_vek280_theoretical.yaml');out=a.output_dir;raw=out/'raw';tr=out/'traces'; names=[a.experiment] if a.experiment else ['topk-memory-wave','prefetch-window','breadth-depth','tile-size','policy'];data={}
 if 'topk-memory-wave' in names:
  rows=[]
  for placement in ('round_robin_expert','striped_tile'):
   for c in (1,2,3):
    for k in range(1,13 if a.full else 8):rows.append(one(m,s,k=k,c=c,placement=placement))
  data['topk_memory_wave_corrected.csv']=rows
  tr.mkdir(parents=True,exist_ok=True)
  for c,k in ((1,4),(2,3),(3,4),(3,7)):
   row=next(x for x in rows if x['memory_channels']==c and x['top_k']==k and x['memory_placement_policy']=='round_robin_expert');(tr/f'memory_wave_C{c}_K{k}.json').write_text(json.dumps({'memory_transfer_wave_count':row['memory_transfer_wave_count'],'waves':json.loads(row['memory_wave_records'])},indent=2)+'\n')
 if 'prefetch-window' in names:
  data['prefetch_window_corrected.csv']=[one(m,s,scenario=sc,quality=q,mwidth=4,depth=2,seed=5) for q in ('perfect','medium','wrong') for sc in (AdvancedScenario.PREDICTIVE_FULL_EXPERT,AdvancedScenario.PREDICTIVE_UNIFORM_PREFIX,AdvancedScenario.PREDICTIVE_RANKED_PREFIX,AdvancedScenario.PREDICTIVE_DSE)]
 if 'breadth-depth' in names:
  data['breadth_depth_fixed_ranking.csv']=[one(m,s,scenario=AdvancedScenario.PREDICTIVE_DSE,quality='medium',mwidth=w,depth=d,budget=8*2*m.hidden_size*352*m.weight_bits//8,seed=11) for w,d in ((2,8),(4,4),(8,2),(16,1))]
 if 'tile-size' in names:
  data['tile_size_corrected.csv']=[one(m,s,b=4,scenario=sc,quality=q,tile=t,mwidth=4) for t in ((16,32,64,128,256,352,704,1408) if a.full else (64,352,1408)) for q,sc in (('perfect',AdvancedScenario.EXACT_TILE_STREAMING),('perfect',AdvancedScenario.PREDICTIVE_UNIFORM_PREFIX),('medium',AdvancedScenario.PREDICTIVE_UNIFORM_PREFIX),('wrong',AdvancedScenario.PREDICTIVE_UNIFORM_PREFIX))]
 if 'policy' in names:
  data['policy_comparison_corrected.csv']=[one(m,s,b=b,k=k,scenario=sc,quality=q,mwidth=8,depth=2,seed=13) for b in (1,4,16) for k in (1,2,4,6) for q in ('perfect','high','medium','low','wrong') for sc in (AdvancedScenario.EXACT_TILE_STREAMING,AdvancedScenario.PREDICTIVE_FULL_EXPERT,AdvancedScenario.PREDICTIVE_UNIFORM_PREFIX,AdvancedScenario.PREDICTIVE_RANKED_PREFIX,AdvancedScenario.PREDICTIVE_DSE)]
 profiles=[]
 for p,tag in ((s,'SYNTHETIC'),(v,'THEORETICAL — NOT MEASURED')):profiles.append({'profile_name':p.name,'hardware_tag':tag,'physical_aie_tiles':p.physical_aie_tiles or p.total_aie_units,'physical_tiles_per_compute_group':p.physical_aie_tiles_per_compute_group,'total_compute_groups':p.total_aie_units,'theoretical_memory_bw_gbps':p.theoretical_memory_bandwidth_bytes_per_sec/1e9,'effective_memory_bw_gbps':p.effective_memory_bandwidth_bytes_per_sec/1e9,'theoretical_board_peak_tops':p.theoretical_compute_peak_ops_per_sec/1e12,'effective_board_peak_tops':p.effective_compute_peak_ops_per_sec/1e12,'theoretical_ridge_ai':p.theoretical_compute_peak_ops_per_sec/p.theoretical_memory_bandwidth_bytes_per_sec,'effective_ridge_ai':p.effective_compute_peak_ops_per_sec/p.effective_memory_bandwidth_bytes_per_sec})
 data['corrected_hardware_profiles.csv']=profiles
 for file,rows in data.items():write_rows(rows,raw/file)
 out.mkdir(parents=True,exist_ok=True);(out/'characterization_v3_summary.json').write_text(json.dumps({'note':'Synthetic/theoretical only; VEK280 profile is NOT MEASURED','files':{k:len(v) for k,v in data.items()}},indent=2)+'\n');(out/'CHARACTERIZATION_V3.md').write_text('# Characterization V3\n\nAll results are simulated with synthetic/theoretical inputs. Memory transfer waves are derived from scheduled demand-DMA event starts, not from `ceil(K/C)`.\n');print(f'Wrote V3 raw data to {raw}')
if __name__=='__main__':main()
