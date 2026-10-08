"""Deterministic three-layer genuine one-layer-ahead M7 demonstration."""
from __future__ import annotations
import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from edge_moe_sim.advanced_schema import AdvancedExecutionConfig
from edge_moe_sim.batch_routing import BatchRoutingEvent
from edge_moe_sim.characterization import controlled_predictor
from edge_moe_sim.hardware_profile import HardwareProfile
from edge_moe_sim.schema import ModelConfig
from edge_moe_sim.unified_schema import UnifiedLayerSequence,UnifiedPredictionSpec
from edge_moe_sim.unified_sequence_simulator import UnifiedSequenceSimulator

def main()->int:
    model=ModelConfig('unified_demo',64,128,8,2,8,8)
    profile=HardwareProfile('demo_synthetic',1_000_000_000,3,16e9,0.8,40,1,4096,True,16,32,0.75,20,2*1024*1024)
    layers=(BatchRoutingEvent(1,0,0,((0,1),)),BatchRoutingEvent(1,0,1,((3,5),)),BatchRoutingEvent(1,0,2,((2,7),)))
    sequence=UnifiedLayerSequence(layers,{(0,1):100,(1,2):100})
    predictors={};
    for source,target in ((0,1),(1,2)):
        predictors[(source,target)]=controlled_predictor(layers[target],8,profile='perfect',candidate_count=2,ready_cycle=999,seed=7)[0]
    config=AdvancedExecutionConfig(32,aie_units_per_expert=4,max_parallel_experts=2,candidate_count=2)
    simulator=UnifiedSequenceSimulator(model,profile)
    outputs={}
    for policy in ('no_prediction','full_expert_prefetch','uniform_prefix','ranked_prefix'):
        specs=() if policy=='no_prediction' else tuple(UnifiedPredictionSpec(source,target,'source_router_done',20,2,policy,2) for source,target in ((0,1),(1,2)))
        result=simulator.simulate(sequence,config,predictions=specs,predictors=predictors,trace_enabled=policy=='uniform_prefix')
        outputs[policy]={'makespan':result.inference_makespan_cycles,'speculative_bytes':result.memory_metrics['speculative_bytes'],'useful_speculative_bytes':result.memory_metrics['useful_speculative_bytes'],'wasted_speculative_bytes':result.memory_metrics['wasted_speculative_bytes'],'demand_blocking':result.demand_blocked_by_future_speculation_cycles,'mean_realized_lead':sum(p.realized_lead_cycles for p in result.predictions)/len(result.predictions) if result.predictions else 0,'mean_startup_readiness':sum(x.weighted_startup_readiness for x in result.layer_results.values())/len(result.layer_results)}
        if policy=='uniform_prefix':(ROOT/'results/unified_demo').mkdir(parents=True,exist_ok=True);(ROOT/'results/unified_demo/event_trace.json').write_text(json.dumps(result.trace,indent=2)+'\n')
    print(json.dumps(outputs,indent=2));return 0
if __name__=='__main__':raise SystemExit(main())
