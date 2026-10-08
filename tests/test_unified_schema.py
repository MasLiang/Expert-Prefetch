"""Construction invariants for the unified M7 cross-layer path."""
from __future__ import annotations
from edge_moe_sim.batch_routing import BatchRoutingEvent
from edge_moe_sim.hardware.nonlinear import NonlinearEngine
from edge_moe_sim.hardware_profile import HardwareProfile
from edge_moe_sim.schema import ModelConfig, NonlinearConfig
from edge_moe_sim.unified_schema import UnifiedLayerSequence,UnifiedPredictionSpec
from edge_moe_sim.unified_sequence_simulator import UnifiedSequenceSimulator

def test_unified_state_owns_one_shared_m6_resource_set() -> None:
    model=ModelConfig('u',4,8,8,1,8,8)
    profile=HardwareProfile('u',1000,1,8000,1,1,1,32,False,2,4,1,1,4096)
    seq=UnifiedLayerSequence((BatchRoutingEvent(1,0,0,((0,),)),BatchRoutingEvent(1,0,1,((1,),))))
    sim=UnifiedSequenceSimulator(model,profile,NonlinearEngine(NonlinearConfig(8,1)))
    state=sim.new_resource_state(seq,(UnifiedPredictionSpec(0,1),))
    assert state.memory.profile is profile and state.aie.profile is profile and state.ocm.capacity_bytes==profile.ocm_bytes
