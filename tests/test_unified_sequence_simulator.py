"""Milestone 7 causal and shared-resource acceptance micro-tests."""
from __future__ import annotations
from edge_moe_sim.advanced_schema import AdvancedExecutionConfig,AdvancedScenario
from edge_moe_sim.advanced_simulator import AdvancedSimulator
from edge_moe_sim.batch_routing import BatchRoutingEvent
from edge_moe_sim.batch_predictor import BatchPredictorOutput,PrecomputedBatchPredictor,TokenPrediction
from edge_moe_sim.characterization import controlled_predictor
from edge_moe_sim.hardware_profile import HardwareProfile
from edge_moe_sim.schema import ModelConfig
from edge_moe_sim.schema import CandidatePrefetch
from edge_moe_sim.unified_schema import UnifiedLayerSequence,UnifiedPredictionSpec
from edge_moe_sim.unified_sequence_simulator import UnifiedSequenceSimulator

def setup(channels:int=2,bw:int=8000,ocm:int=4096):
    model=ModelConfig('unified',4,8,8,2,8,8)
    profile=HardwareProfile('unified',1000,channels,bw,1,1,1,32,True,4,4,1,1,ocm)
    config=AdvancedExecutionConfig(4,aie_units_per_expert=2,max_parallel_experts=2,candidate_count=2)
    return model,profile,config

def test_real_one_layer_ahead_prefix_overlaps_source_and_commits() -> None:
    model,profile,config=setup();source=BatchRoutingEvent(1,0,0,((0,),));target=BatchRoutingEvent(1,0,1,((3,),))
    predictor,*_=controlled_predictor(target,8,profile='perfect',candidate_count=1,ready_cycle=999)
    result=UnifiedSequenceSimulator(model,profile).simulate(UnifiedLayerSequence((source,target)),config,predictions=(UnifiedPredictionSpec(0,1,'source_router_done',1,1,'uniform_prefix',1),),predictors={(0,1):predictor},trace_enabled=True)
    trigger=next(x for x in result.trace if x.get('event_type')=='prediction_trigger');spec=next(x for x in result.trace if x.get('event_type')=='dma_start' and x.get('layer_id')==1 and x.get('request_state')=='speculative');target_router=next(x for x in result.trace if x.get('event_type')=='router_done' and x.get('layer_id')==1)
    assert trigger['cycle']+1==spec['cycle']<target_router['cycle']
    assert spec['end_cycle']>spec['cycle'] and spec['cycle']<result.layer_results[0].layer_done_cycle
    source_compute=next(x for x in result.trace if x.get('event_type')=='aie_start' and x.get('layer_id')==0 and x.get('expert_id') is not None)
    assert spec['cycle']<source_compute['end_cycle'] and spec['end_cycle']>source_compute['cycle']
    assert result.layer_results[1].startable_recall==1 and result.predictions[0].useful_bytes>0
    assert result.layer_results[1].demand_bytes < 3*model.hidden_size*model.intermediate_size

def test_wrong_future_prediction_is_waste_and_router_is_authoritative() -> None:
    model,profile,config=setup(1,1000);source=BatchRoutingEvent(2,0,0,((0,),));target=BatchRoutingEvent(2,0,1,((3,),))
    predictor,*_=controlled_predictor(target,8,profile='wrong',candidate_count=2,ready_cycle=0)
    result=UnifiedSequenceSimulator(model,profile).simulate(UnifiedLayerSequence((source,target)),config,predictions=(UnifiedPredictionSpec(0,1,'source_router_done',0,2,'full_expert_prefetch',1),),predictors={(0,1):predictor},trace_enabled=True)
    assert result.predictions[0].useful_bytes==0
    assert result.predictions[0].wasted_bytes+result.predictions[0].canceled_bytes>0
    assert result.layer_results[1].true_expert_ids==(3,) and result.layer_results[1].demand_bytes>0

def test_batch_reuse_loads_one_weight_copy_and_scales_compute() -> None:
    model,profile,config=setup();one=BatchRoutingEvent(3,0,0,((3,),));four=BatchRoutingEvent(4,0,0,((3,),(3,),(3,),(3,)))
    sim=UnifiedSequenceSimulator(model,profile)
    r1=sim.simulate(UnifiedLayerSequence((one,)),config);r4=sim.simulate(UnifiedLayerSequence((four,)),config)
    assert r1.layer_results[0].demand_bytes==r4.layer_results[0].demand_bytes
    assert r4.inference_makespan_cycles>r1.inference_makespan_cycles
    assert r4.layer_results[0].per_expert_token_counts=={3:4}

def test_topk_commits_two_prefetched_startup_tiles() -> None:
    model,profile,config=setup();source=BatchRoutingEvent(5,0,0,((0,1),));target=BatchRoutingEvent(5,0,1,((3,7),))
    predictor,*_=controlled_predictor(target,8,profile='perfect',candidate_count=2,ready_cycle=0)
    result=UnifiedSequenceSimulator(model,profile).simulate(UnifiedLayerSequence((source,target)),config,predictions=(UnifiedPredictionSpec(0,1,candidate_count=2),),predictors={(0,1):predictor})
    assert result.layer_results[1].startable_recall==1
    assert result.layer_results[1].weighted_startable_recall==1
    assert float(result.aie_metrics['peak_active_aie_units'])<=profile.total_aie_units

def test_single_layer_unified_matches_advanced_demand_bytes() -> None:
    model,profile,config=setup();event=BatchRoutingEvent(6,0,0,((2,3),))
    unified=UnifiedSequenceSimulator(model,profile).simulate(UnifiedLayerSequence((event,)),config)
    advanced=AdvancedSimulator(model,profile).simulate(event,AdvancedScenario.EXACT_TILE_STREAMING,config)
    assert unified.layer_results[0].demand_bytes==advanced.memory_metrics['demand_bytes']
    assert unified.layer_results[0].per_expert_token_counts==advanced.per_expert_token_counts
    assert unified.layer_results[0].router_resolve_cycle==advanced.router_resolve_cycle
    assert unified.layer_results[0].first_true_expert_start==advanced.first_true_expert_start
    assert unified.inference_makespan_cycles==advanced.layer_makespan_cycles

def test_partial_topk_hit_mixes_commit_and_demand() -> None:
    model,profile,config=setup();source=BatchRoutingEvent(7,0,0,((0,1),));target=BatchRoutingEvent(7,0,1,((3,7),))
    predictor,*_=controlled_predictor(target,8,profile='medium',candidate_count=2,ready_cycle=0,seed=4)
    result=UnifiedSequenceSimulator(model,profile).simulate(UnifiedLayerSequence((source,target)),config,predictions=(UnifiedPredictionSpec(0,1,candidate_count=2),),predictors={(0,1):predictor})
    assert 0<result.layer_results[1].weighted_startable_recall<1
    assert result.predictions[0].useful_bytes>0 and result.layer_results[1].demand_bytes>0

def test_small_finite_ocm_rejects_future_prefetch_without_overflow() -> None:
    model,profile,config=setup(1,8000,32);source=BatchRoutingEvent(8,0,0,((0,),));target=BatchRoutingEvent(8,0,1,((3,),))
    predictor,*_=controlled_predictor(target,8,profile='perfect',candidate_count=1,ready_cycle=0)
    result=UnifiedSequenceSimulator(model,profile).simulate(UnifiedLayerSequence((source,target)),config,predictions=(UnifiedPredictionSpec(0,1,prefetch_policy='full_expert_prefetch'),),predictors={(0,1):predictor})
    assert result.peak_ocm_bytes<=profile.ocm_bytes
    assert result.rejected_future_prefetch_bytes>0

def test_prediction_timing_is_derived_and_trace_metadata_is_cross_layer() -> None:
    model,profile,config=setup();source=BatchRoutingEvent(9,0,0,((0,),));target=BatchRoutingEvent(9,0,1,((3,),))
    predictor,*_=controlled_predictor(target,8,profile='perfect',candidate_count=1,ready_cycle=999)
    result=UnifiedSequenceSimulator(model,profile).simulate(UnifiedLayerSequence((source,target)),config,predictions=(UnifiedPredictionSpec(0,1,'source_router_done',7),),predictors={(0,1):predictor},trace_enabled=True)
    record=result.predictions[0]
    assert record.predictor_ready_cycle==record.source_milestone_cycle+7
    assert not record.synthetic_lead_time and record.realized_lead_cycles>0
    trigger=next(x for x in result.trace if x.get('event_type')=='prediction_trigger')
    assert trigger['source_layer_id']==0 and trigger['target_layer_id']==1

def test_wrong_queued_requests_cancel_and_trace_is_json_serializable() -> None:
    import json
    model,profile,config=setup(1,1000);source=BatchRoutingEvent(10,0,0,((0,),));target=BatchRoutingEvent(10,0,1,((3,),))
    predictor,*_=controlled_predictor(target,8,profile='wrong',candidate_count=2,ready_cycle=0)
    result=UnifiedSequenceSimulator(model,profile).simulate(UnifiedLayerSequence((source,target)),config,predictions=(UnifiedPredictionSpec(0,1,candidate_count=2,prefetch_policy='full_expert_prefetch'),),predictors={(0,1):predictor},trace_enabled=True)
    assert result.predictions[0].canceled_bytes>0
    assert any(x.get('event_type')=='dma_cancel' for x in result.trace)
    json.dumps(result.trace)

def test_independent_channel_avoids_false_speculation_blocking() -> None:
    source=BatchRoutingEvent(11,0,0,((1,),));target=BatchRoutingEvent(11,0,1,((3,),))
    output=BatchPredictorOutput(11,0,1,0,(TokenPrediction(0,(CandidatePrefetch(2,1.0),)),))
    predictor=PrecomputedBatchPredictor({(11,0,1):output})
    blocked=[]
    for channels in (1,2):
        model,profile,config=setup(channels,200)
        result=UnifiedSequenceSimulator(model,profile).simulate(UnifiedLayerSequence((source,target)),config,predictions=(UnifiedPredictionSpec(0,1,candidate_count=1,prefetch_policy='full_expert_prefetch'),),predictors={(0,1):predictor})
        blocked.append(result.demand_blocked_by_future_speculation_cycles)
    assert blocked[0]>0 and blocked[1]==0

def test_unified_strict_speculative_budget_is_enforced_before_enqueue() -> None:
    model,profile,base=setup();source=BatchRoutingEvent(12,0,0,((0,1),));target=BatchRoutingEvent(12,0,1,((3,7),))
    predictor,*_=controlled_predictor(target,8,profile='perfect',candidate_count=2,ready_cycle=0)
    budget=32
    from dataclasses import replace
    config=replace(base,speculative_byte_budget=budget)
    result=UnifiedSequenceSimulator(model,profile).simulate(UnifiedLayerSequence((source,target)),config,predictions=(UnifiedPredictionSpec(0,1,candidate_count=2,prefetch_policy='full_expert_prefetch'),),predictors={(0,1):predictor})
    assert result.predictions[0].speculative_bytes<=budget

def test_topk_experts_can_overlap_without_exceeding_shared_aie_pool() -> None:
    model,profile,config=setup(2);event=BatchRoutingEvent(13,0,0,((2,3),))
    result=UnifiedSequenceSimulator(model,profile).simulate(UnifiedLayerSequence((event,)),config)
    assert result.aie_metrics['peak_active_aie_units']==profile.total_aie_units
    assert result.aie_metrics['peak_active_aie_units']<=profile.total_aie_units

def test_startup_readiness_honors_configured_tile_depth() -> None:
    from dataclasses import replace
    model,profile,base=setup();config=replace(base,startup_gu_tiles=2)
    source=BatchRoutingEvent(14,0,0,((0,),));target=BatchRoutingEvent(14,0,1,((3,),))
    predictor,*_=controlled_predictor(target,8,profile='perfect',candidate_count=1,ready_cycle=0)
    result=UnifiedSequenceSimulator(model,profile).simulate(
        UnifiedLayerSequence((source,target)),config,
        predictions=(UnifiedPredictionSpec(0,1,prefix_depth=1),),
        predictors={(0,1):predictor})
    assert result.layer_results[1].startable_recall==0
    assert result.layer_results[1].weighted_startup_readiness==0.5

def test_synthetic_inter_layer_work_uses_shared_resources_and_delays_target() -> None:
    from edge_moe_sim.schema import InterLayerConfig
    model,profile,config=setup(1,1000)
    source=BatchRoutingEvent(15,0,0,((0,),));target=BatchRoutingEvent(15,0,1,((3,),))
    sequence=UnifiedLayerSequence((source,target),inter_layer_config=InterLayerConfig(
        default_aie_cycles=11,default_nonlinear_cycles=7,
        default_controller_cycles=5,default_background_dma_bytes=16))
    result=UnifiedSequenceSimulator(model,profile).simulate(sequence,config,trace_enabled=True)
    assert result.layer_results[1].layer_start_cycle>=result.layer_results[0].layer_done_cycle+11
    assert result.memory_metrics['background_bytes']==16
    assert any(x.get('event_type')=='background_dma_done' for x in result.trace)
