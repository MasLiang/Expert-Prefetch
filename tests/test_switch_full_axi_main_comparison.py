"""完整六层共同夹具的配置守恒；不把 cycle 拟合设为验收标准。"""
from __future__ import annotations

import importlib
from pathlib import Path

import pytest


@pytest.mark.parametrize('case', ['A_whole_demand', 'B_stream_demand',
                                  'C_whole_full_prefetch', 'D_stream_tile_prefetch'])
def test_full_fixture(case: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """检查真实尺寸、全命中、候选数量、冷首层及 RTL 整 Expert 保留 credit。"""
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]/'examples'))
    module = importlib.import_module('switch_full_axi_main_comparison')
    model, hw, config, sequence, specs, _, outputs = module.fixture(case)
    assert (model.hidden_size, model.intermediate_size, model.num_experts, model.top_k) == (768, 3072, 8, 1)
    assert model.expert_architecture == 'relu_ffn'
    assert len(sequence.layers) == 6
    assert [r.routes for r in sequence.layers] == [((l,),) for l in range(6)]
    assert hw.ocm_bytes == 128*config.execution_rules.ocm_slot_bytes == 16*1024*1024
    assert config.speculative_byte_budget == 9*1024*1024
    assert config.execution_rules.reserved_demand_slots == (48 if 'whole' in case else 2)
    assert config.execution_rules.wait_full_expert == ('whole' in case)
    assert config.nonlinear_elements_per_cycle == 1
    if 'prefetch' in case:
        count = 2 if 'whole' in case else 8
        assert len(specs) == len(outputs) == 5
        for target, (spec, output) in enumerate(zip(specs, outputs), 1):
            assert (spec.source_layer_id, spec.target_layer_id) == (target-1, target)
            assert not spec.wait_for_previous_layer_weights
            assert spec.predictor_latency_cycles == 4+count
            candidates = output.token_predictions[0].candidates
            assert len(candidates) == count
            assert candidates[0].expert_id == target
            assert [c.score for c in candidates][:2] == [3104, 352]
    else:
        assert not specs and not outputs
