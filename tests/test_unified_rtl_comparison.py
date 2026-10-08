"""共同激励诊断和旧模型失效保护。"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

import pytest
import yaml
from edge_moe_sim.rtl_alignment import M10AlignmentSimulator
from edge_moe_sim.schema import ValidationError


def load_runner() -> Any:
    """载入诊断入口，不运行 XSim。"""
    scripts = Path(__file__).resolve().parents[1] / 'hardware/m10_u200/scripts'
    sys.path.insert(0, str(scripts))
    try:
        spec = importlib.util.spec_from_file_location('unified_comparison_test', scripts / 'compare_unified_alignment.py')
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.remove(str(scripts))


def test_obsolete_model_refuses_current_use() -> None:
    """默认不得悄悄使用历史 pair-ready 性能模型。"""
    with pytest.raises(ValidationError, match='已失效'):
        M10AlignmentSimulator({}, ddr_gaps=False)
    assert M10AlignmentSimulator.compatibility_status == 'INVALID_FOR_CURRENT_RTL_HISTORICAL_ONLY'


def test_common_fixture_deterministic(tmp_path: Path) -> None:
    """同一事先声明的输入产生相同主模型结果及 trace。"""
    runner = load_runner()
    config = yaml.safe_load((runner.PROJECT / 'configs/hardware/u200_unified_alignment.yaml').read_text())
    left, right = tmp_path / 'left', tmp_path / 'right'
    left.mkdir()
    right.mkdir()
    result = runner.simulate_fixture(config, config['cases'][1], left)
    assert result == runner.simulate_fixture(config, config['cases'][1], right)
    assert (left / 'routing.jsonl').read_bytes() == (right / 'routing.jsonl').read_bytes()
    assert len(result['layer_results']) == 3
    assert result['peak_ocm_bytes'] <= 2048


def test_unsupported_fixture_rejected(tmp_path: Path) -> None:
    """不能把固定 RTL 夹具冒充任意模型参数测试。"""
    runner = load_runner()
    config = yaml.safe_load((runner.PROJECT / 'configs/hardware/u200_unified_alignment.yaml').read_text())
    config['fixture']['hidden_size'] = 32
    with pytest.raises(ValueError, match='共同夹具'):
        runner.simulate_fixture(config, config['cases'][0], tmp_path)


@pytest.mark.parametrize('field,value', [('ocm_slot_bytes', 256), ('ocm_banks', 1),
    ('reserved_demand_slots', 0), ('atomic_tile_pair', False),
    ('compute_policy', 'expert_locked'), ('retry_future_prefetch', False), ('request_order', 'legacy')])
def test_common_fixture_rejects_different_rules(tmp_path: Path, field: str, value: object) -> None:
    """配置必须对应实际 RTL，不能修改字段后仍标为资源一致。"""
    runner = load_runner()
    config = yaml.safe_load((runner.PROJECT / 'configs/hardware/u200_unified_alignment.yaml').read_text())
    config['execution_rules'][field] = value
    with pytest.raises(ValueError, match='当前 RTL 夹具'):
        runner.simulate_fixture(config, config['cases'][0], tmp_path)
