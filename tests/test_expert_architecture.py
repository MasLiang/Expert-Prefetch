"""Selectable Expert geometry, rounding, and main-simulator acceptance tests."""
from dataclasses import replace
from pathlib import Path

import pytest

from edge_moe_sim.advanced_schema import AdvancedExecutionConfig
from edge_moe_sim.batch_routing import BatchRoutingEvent
from edge_moe_sim.config import load_model_config
from edge_moe_sim.f_tile import build_f_tile_groups
from edge_moe_sim.hardware_profile import HardwareProfile
from edge_moe_sim.schema import ModelConfig, ValidationError
from edge_moe_sim.simulator import Scenario, Simulator
from edge_moe_sim.unified_schema import UnifiedLayerSequence
from edge_moe_sim.unified_sequence_simulator import UnifiedSequenceSimulator
from edge_moe_sim.work_model import down_ops, expert_weight_bytes, gate_up_ops, nonlinear_elements

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("architecture,factor", [("swiglu", 2), ("relu_ffn", 1)])
def test_geometry_and_uneven_subbyte_partition(architecture: str, factor: int) -> None:
    """Both variants conserve exact stage bytes and operations, even with packed weights."""
    model = ModelConfig("tiny", 3, 7, 4, 1, 3, 8, expert_architecture=architecture)
    groups = build_f_tile_groups(model, 0, 2, 3)
    assert sum(g.gate_up_ops for g in groups) == gate_up_ops(model, 2) == 2 * factor * 2 * 3 * 7
    assert sum(g.down_ops for g in groups) == down_ops(model, 2)
    assert sum(g.nonlinear_elements for g in groups) == nonlinear_elements(model, 2)
    assert sum(g.gate_up_bytes + g.down_bytes for g in groups) == expert_weight_bytes(model)


@pytest.mark.parametrize("experts", [8, 16])
def test_switch_yaml(experts: int) -> None:
    """Official Switch dimensions give 4.5 MiB per dense INT8 Expert."""
    model = load_model_config(ROOT / f"configs/models/switch_base_{experts}.yaml")
    assert model.input_projection_count == 1
    assert model.nonlinear_operation == "relu"
    assert expert_weight_bytes(model) == 4_718_592
    assert model.num_experts == experts and model.top_k == 1
    assert len(model.moe_layer_indices) == 6


def test_validation_and_legacy_default() -> None:
    """Unsupported architectures fail; old configs remain fused SwiGLU."""
    model = ModelConfig("tiny", 4, 8, 4, 1, 8, 8)
    assert model.expert_architecture == "swiglu"
    assert expert_weight_bytes(model) == 96
    with pytest.raises(ValidationError, match="expert_architecture"):
        replace(model, expert_architecture="gelu")
    with pytest.raises(ValidationError, match="fused"):
        replace(model, fuse_gate_up=False)


def test_main_simulator_multilayer_relu_conservation() -> None:
    """Main unified engine runs both architectures deterministically without scheduler changes."""
    hardware = HardwareProfile("synthetic", 1000, 2, 8000, 1, 1, 1, 32, True, 4, 4, 1, 1, 4096)
    config = AdvancedExecutionConfig(3, aie_units_per_expert=1, max_parallel_experts=1, candidate_count=1)
    sequence = UnifiedLayerSequence(tuple(BatchRoutingEvent(1, 0, layer, ((0,),)) for layer in (0, 1)))
    results = []
    for architecture in ("swiglu", "relu_ffn"):
        model = ModelConfig("tiny", 4, 8, 4, 1, 8, 8, expert_architecture=architecture)
        simulator = UnifiedSequenceSimulator(model, hardware)
        first = simulator.simulate(sequence, config, trace_enabled=True)
        assert first == simulator.simulate(sequence, config, trace_enabled=True)
        assert all(layer.demand_bytes == expert_weight_bytes(model) for layer in first.layer_results.values())
        results.append(first)
    assert results[1].inference_makespan_cycles <= results[0].inference_makespan_cycles


@pytest.mark.parametrize("scenario", [Scenario.DEMAND_SERIAL, Scenario.EXACT_PIPELINE])
def test_relu_stage_baselines(small_model, small_hardware, two_expert_event, scenario: Scenario) -> None:
    """Stage-level baselines load two matrices, not an extra Gate matrix."""
    model = replace(small_model, expert_architecture="relu_ffn", fuse_gate_up=False)
    output = Simulator(model, small_hardware).simulate(two_expert_event, scenario)
    assert output.result.demand_bytes == 2 * 64
    assert output.result.total_cycles <= Simulator(small_model, small_hardware).simulate(
        two_expert_event, scenario
    ).result.total_cycles
