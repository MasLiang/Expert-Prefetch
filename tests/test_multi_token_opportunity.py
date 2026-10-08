"""初步 batch 机会实验的配对、复用、守恒和确定性检查。"""
from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

import pytest
import yaml


@pytest.fixture
def study(monkeypatch: pytest.MonkeyPatch) -> tuple[Any, dict[str, Any]]:
    """加载实际实验驱动与 YAML。"""
    root = Path(__file__).resolve().parents[1]
    monkeypatch.syspath_prepend(str(root/'examples'))
    spec = importlib.util.spec_from_file_location('multi_opportunity', root/'examples/multi_token_opportunity.py')
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, yaml.safe_load((root/'configs/experiments/multi_token_opportunity.yaml').read_text())


def test_validate(study: tuple[Any, dict[str, Any]]) -> None:
    """拒绝不能产生真正 dispersed 路由的 batch。"""
    m, c = study
    m.validate(c)
    with pytest.raises(ValueError):
        m.validate({**c, 'batch_sizes': [65]})


@pytest.mark.parametrize('pattern,unique', [('dispersed', 8), ('balanced', 4), ('hot', 3)])
def test_routing_and_renaming(study: tuple[Any, dict[str, Any]], pattern: str, unique: int) -> None:
    """ID 顺序变更不能改变 token sharing 或逐 token 命中。"""
    m, c = study
    record = m.make_record(c, 8, pattern, 1)
    assert record == m.make_record(c, 8, pattern, 1)
    truth, predicted = m.ordered_routes(record, 'hot_first', c['num_experts'])
    for old, new, old_p, new_p in zip(record['truth'], truth, record['predicted'], predicted):
        assert len(set(new)) == unique
        assert sorted(old.count(e) for e in set(old)) == sorted(new.count(e) for e in set(new))
        assert [a == b for a, b in zip(old, old_p)] == [a == b for a, b in zip(new, new_p)]


def test_weight_reuse_and_perfect(study: tuple[Any, dict[str, Any]]) -> None:
    """8 tokens/4 experts 仅搬运4份权重每层，Oracle 不能产生错误字节。"""
    m, c = study
    r = m.make_record(c, 8, 'balanced', 1)
    b = m.simulate(c, r, .5, 64, 'native', 'OFF')
    assert b['transferred_bytes'] == 3*4*3*128*256
    assert b['reuse_factor'] == 2
    assert b['extra_bytes'] == b['speculative_bytes'] == 0
    perfect = m.simulate(c, r, .5, 64, 'native', 'ORACLE', 1)
    assert perfect['wasted_speculative_bytes'] == perfect['extra_bytes'] == 0
    assert perfect == m.simulate(c, r, .5, 64, 'native', 'ORACLE', 1)


def test_single_token_and_tested_envelopes(study: tuple[Any, dict[str, Any]]) -> None:
    """Batch1共享无收益；OFF 包含在事后包络，不掩盖原始负收益。"""
    m, c = study
    raw, result = m.run_cell((c, m.make_record(c, 1, 'hot', 1), 1.))
    assert result['sharing_speedup'] == 1
    assert result['weight_reduction_factor'] == 1
    assert result['prediction_oracle_speedup'] >= 1
    assert result['prediction_synthetic_speedup'] >= 1
    assert len(raw) == 10
    assert all(set(row) == set(raw[0]) for row in raw)


def test_disperse_no_weight_reduction(study: tuple[Any, dict[str, Any]]) -> None:
    """Batch 增大不等于共享；每 token Expert 不同时 bytes 完全不减少。"""
    m, c = study
    _, result = m.run_cell((c, m.make_record(c, 4, 'dispersed', 1), .5))
    assert result['weight_reduction_factor'] == result['reuse_factor'] == 1
    singles = [r for r in m.run_cell((c, m.make_record(c, 4, 'dispersed', 1), .5))[0]
               if r['policy'] == 'A_SINGLE_COLD_REFERENCE']
    assert result['required_compute_ops'] == 4*singles[0]['required_compute_ops']


def test_attachment_eight_token_example(study: tuple[Any, dict[str, Any]]) -> None:
    """附件的4/2/1/1分配，核对逐层4份而非8份Expert权重。"""
    m, c = study
    ids = [0, 0, 1, 0, 2, 1, 0, 3]
    record = dict(batch_size=8, pattern='attachment_example', seed=1,
                  truth=[ids]*c['layers'], predicted=[ids]*c['layers'])
    result = m.simulate(c, record, .5, 64, 'native', 'OFF')
    assert result['reuse_factor'] == 2
    assert result['max_tokens_per_expert'] == 4
    assert result['transferred_bytes'] == c['layers']*4*3*128*256
