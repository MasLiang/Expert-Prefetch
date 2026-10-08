"""旧/新模型实验必须使用完整身份配对，不混用不同 seed、OCM 或 baseline。"""
from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any

import pytest


def driver(monkeypatch: pytest.MonkeyPatch) -> Any:
    """导入轻量汇总驱动，不启动实验进程。"""
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]/'examples'))
    return importlib.import_module('retest_real_models')


def test_paired_deltas(monkeypatch: pytest.MonkeyPatch) -> None:
    """差值为当前减旧，负值代表节省；byte 差不能冒充 cycle 差。"""
    module = driver(monkeypatch)
    old = [dict(model='switch', sample='0', total_cycles='100', speculative_bytes='64')]
    new = [dict(model='switch', sample='0', total_cycles='90', speculative_bytes='32')]
    row = module.paired_deltas(old, new, ('model', 'sample'), 'test')[0]
    assert row['delta_cycles'] == -10
    assert row['delta_percent'] == pytest.approx(-10)
    assert row['delta_speculative_bytes'] == -32


@pytest.mark.parametrize('fault', ['duplicate', 'identity', 'missing'])
def test_reject_invalid_sample_pairing(fault: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """不能把不同实验点合并或静默丢弃缺失样本。"""
    module = driver(monkeypatch)
    old = [dict(model='switch', sample='0', total_cycles='100', speculative_bytes='64')]
    new = [dict(old[0])]
    if fault == 'duplicate':
        new.append(dict(new[0]))
    elif fault == 'identity':
        new[0]['sample'] = '1'
    else:
        new = []
    with pytest.raises(ValueError):
        module.paired_deltas(old, new, ('model', 'sample'), 'test')
