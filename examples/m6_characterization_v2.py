"""Controlled, validated M6.6 characterization pipeline (synthetic/theoretical)."""
from __future__ import annotations
import argparse, json, sys
from dataclasses import replace
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/"src"))
from edge_moe_sim.advanced_schema import AdvancedExecutionConfig, AdvancedScenario
from edge_moe_sim.advanced_simulator import AdvancedSimulator
from edge_moe_sim.characterization import _row, controlled_predictor, controlled_routing, write_rows
from edge_moe_sim.config import load_hardware_profile
from edge_moe_sim.schema import ModelConfig

EXPS=("predictor","prefetch-window","roofline","aie-allocation","topk-memory-wave","tile-size","policy","breadth-depth")
def args():
 p=argparse.ArgumentParser(description=__doc__); g=p.add_mutually_exclusive_group(required=True);g.add_argument('--smoke',action='store_true');g.add_argument('--full',action='store_true');p.add_argument('--experiment',choices=EXPS);p.add_argument('--output-dir',type=Path,default=ROOT/'results/characterization_v2');return p.parse_args()
def model(): return ModelConfig('m66_synthetic',2048,1408,64,12,8,8)
def run(name,m,p,*,b=1,k=2,pattern='low_reuse',groups=16,channels=None,tile=352,scenario=AdvancedScenario.EXACT_TILE_STREAMING,quality='perfect',candidate=None,ready=0,seed=1,placement='round_robin_expert'):
 if channels: p=replace(p,num_memory_channels=channels)
 e=controlled_routing(batch_size=b,top_k=k,num_experts=m.num_experts,pattern=pattern,seed=seed)
 candidate=candidate or min(m.num_experts,max(2*k,k))
 cfg=AdvancedExecutionConfig(tile,placement,groups,min(k,4),candidate_count=candidate,uniform_gu_prefetch_depth=2,explicit_depths=tuple(2 for _ in range(candidate)),speculative_byte_budget=64*1024*1024)
 pred,tid,rec,anyhit,fullhit=controlled_predictor(e,m.num_experts,profile=quality,candidate_count=candidate,ready_cycle=ready,seed=seed)
 r=AdvancedSimulator(m,p).simulate(e,scenario,cfg,predictor=pred if 'predictive' in scenario.value else None)
 row=_row(r,experiment_name=name,run_id=f'{name}-{b}-{k}-{quality}-{ready}-{groups}-{channels or p.num_memory_channels}-{tile}',seed=seed,model=m,profile=p,config=cfg,routing_pattern=pattern,predictor_profile=quality)
 row.update(workload_trace_id=f'route:{pattern}:B={b}:K={k}:seed={seed}',predictor_trace_id=tid if 'predictive' in scenario.value else 'none',hardware_profile_id=p.name,expected_Recall_at_M=rec,expected_AnyHit=anyhit,expected_FullHit=fullhit)
 if 'predictive' in scenario.value and abs(row['StartableRecall'])>1: raise RuntimeError('invalid recall')
 return row
def study(exp,full,m,synth,vek):
 rows=[]; qs=('perfect','high','medium','low','wrong'); reuse=(1,2,4,8,16,32) if full else (1,4)
 if exp=='predictor':
  for q in qs: rows.append(run(exp,m,synth,quality=q,scenario=AdvancedScenario.PREDICTIVE_UNIFORM_PREFIX,candidate=4))
 elif exp=='prefetch-window':
  # Router is ~55k cycles here; choose absolute causal early availability.
  for q in qs:
   for lead in ((0,10000,30000,50000) if full else (0,30000,50000)):
    for s in (AdvancedScenario.PREDICTIVE_FULL_EXPERT,AdvancedScenario.PREDICTIVE_UNIFORM_PREFIX,AdvancedScenario.PREDICTIVE_RANKED_PREFIX,AdvancedScenario.PREDICTIVE_DSE): rows.append(run(exp,m,synth,quality=q,ready=max(0,55000-lead),scenario=s,candidate=4))
 elif exp=='roofline':
  for prof in (synth,vek):
   for b in reuse: rows.append(run(exp,m,prof,b=b,k=1,pattern='max_reuse',scenario=AdvancedScenario.EXACT_TILE_STREAMING,groups=1))
 elif exp=='aie-allocation':
  for b in reuse:
   for g in ((1,2,4,8,16,32) if full else (1,4,16)): rows.append(run(exp,m,synth,b=b,k=1,pattern='max_reuse',groups=g))
 elif exp=='topk-memory-wave':
  for c in (1,2,3):
   for k in range(1,13 if full else 7): rows.append(run(exp,m,synth,k=k,channels=c,groups=32,placement='round_robin_expert'))
  for k in (1,4,7): rows.append(run(exp,m,synth,k=k,channels=3,groups=32,placement='striped_tile'))
 elif exp=='tile-size':
  for t in ((16,32,64,128,256,352,704,1408) if full else (64,352,1408)):
   for q,s in (('perfect',AdvancedScenario.EXACT_TILE_STREAMING),('perfect',AdvancedScenario.PREDICTIVE_UNIFORM_PREFIX),('wrong',AdvancedScenario.PREDICTIVE_UNIFORM_PREFIX)): rows.append(run(exp,m,synth,b=4,k=2,pattern='max_reuse',tile=t,quality=q,scenario=s,candidate=4))
 elif exp=='policy':
  for b in (reuse if full else (1,4,16)):
   for q in qs:
    base=run(exp,m,synth,b=b,k=2,pattern='max_reuse',quality=q,scenario=AdvancedScenario.EXACT_TILE_STREAMING,candidate=4); rows.append(base)
    for s in (AdvancedScenario.PREDICTIVE_FULL_EXPERT,AdvancedScenario.PREDICTIVE_UNIFORM_PREFIX,AdvancedScenario.PREDICTIVE_RANKED_PREFIX,AdvancedScenario.PREDICTIVE_DSE):
     row=run(exp,m,synth,b=b,k=2,pattern='max_reuse',quality=q,scenario=s,candidate=4);row['normalized_latency']=row['makespan_cycles']/base['makespan_cycles'];rows.append(row)
 elif exp=='breadth-depth':
  for q in ('high','medium'):
   for breadth,depth in ((2,8),(4,4),(8,2),(16,1)):
    rows.append(run(exp,m,synth,quality=q,scenario=AdvancedScenario.PREDICTIVE_DSE,candidate=breadth))
    rows[-1]['logical_prefix_depth']=depth;rows[-1]['allocated_speculative_bytes']=breadth*depth*rows[-1]['demand_bytes']//max(1,rows[-1]['number_of_tiles']*3*rows[-1]['unique_active_experts'])
 return rows
def main():
 a=args();m=model();s=load_hardware_profile(ROOT/'configs/hardware_profile_synthetic_balanced.yaml');v=load_hardware_profile(ROOT/'configs/hardware_profile_vek280_theoretical.yaml');out=a.output_dir;raw=out/'raw'; names=(a.experiment,) if a.experiment else EXPS
 files={'predictor':'predictor_profiles.csv','prefetch-window':'prefetch_window_controlled.csv','roofline':'board_vs_configured_roofline.csv','aie-allocation':'aie_allocation_controlled.csv','topk-memory-wave':'topk_memory_wave.csv','tile-size':'tile_size_controlled.csv','policy':'predictor_policy_comparison.csv','breadth-depth':'breadth_depth_controlled.csv'}
 allrows={}
 for n in names: allrows[n]=write_rows(study(n,a.full,m,s,v),raw/files[n])
 hp=[]
 for p,tag in ((s,'synthetic'),(v,'theoretical')): hp.append({'profile_name':p.name,'classification_tag':tag,'memory_channels':p.num_memory_channels,'theoretical_peak_BW_gbps':p.theoretical_memory_bandwidth_bytes_per_sec/1e9,'effective_BW_gbps':p.effective_memory_bandwidth_bytes_per_sec/1e9,'total_aie_compute_groups':p.total_aie_units,'per_group_compute_peak_ops_s':p.ops_per_cycle_per_aie*p.clock_hz*p.aie_efficiency,'total_board_compute_peak_ops_s':p.effective_compute_peak_ops_per_sec,'board_ridge_ai':p.effective_compute_peak_ops_per_sec/p.effective_memory_bandwidth_bytes_per_sec})
 write_rows(hp,raw/'hardware_profile_sanity.csv')
 neg=[r for r in allrows.get('policy',[]) if r['predictor_profile']=='wrong'];write_rows(neg,raw/'negative_controls.csv')
 out.mkdir(parents=True,exist_ok=True);(out/'characterization_v2_summary.json').write_text(json.dumps({'input_tag':'SYNTHETIC/THEORETICAL, not measured','experiments':{k:len(x) for k,x in allrows.items()}},indent=2)+'\n');print(f'Wrote controlled V2 CSVs to {raw}')
if __name__=='__main__':main()
