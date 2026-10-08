"""MIG-backed Switch evidence gates; synthetic logs are not simulation proof."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from typing import Any

import pytest


def load_module(monkeypatch: pytest.MonkeyPatch) -> Any:
    """Load the standalone verification driver without running Vivado."""
    scripts = Path(__file__).resolve().parents[1]/'hardware/m10_u200/scripts'
    monkeypatch.syspath_prepend(str(scripts))
    spec = importlib.util.spec_from_file_location('switch_ddr4_evidence', scripts/'run_switch_ddr4.py')
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize('fault', ['none', 'backend', 'calibration', 'preload', 'bytes', 'commands'])
def test_evidence(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fault: str) -> None:
    """Only initialized MIG paths with matching physical payload may pass."""
    module = load_module(monkeypatch)
    lines = [f'DDR_CALIBRATED bank={b} cycle=100' for b in range(4)]
    lines += [f'DDR_INITIALIZED bank={b} cycle=200' for b in range(4)]
    lines += ['FULL_BACKEND MIG_RDIMM initialized_cycle=201', 'FULL_LAYER_START layer=0 cycle=210']
    lines += [f'DDR_DRAM_SUMMARY bank={b} reads=1 activates=1 refreshes=0 precharges=0' for b in range(4)]
    text = '\n'.join(lines)
    if fault == 'backend':
        text = text.replace('MIG_RDIMM', 'BEHAVIORAL_AXI')
    if fault == 'calibration':
        text = text.replace('DDR_CALIBRATED bank=3 cycle=100', '')
    if fault == 'commands':
        text = text.replace('bank=3 reads=1', 'bank=3 reads=0')
    (tmp_path/'simulate.log').write_text(text, encoding='utf-8')
    for bank in range(4):
        (tmp_path/f'ddr_bank_{bank}.csv').write_text(
            'bank,address_bytes,beats,ar_cycle,first_r_cycle,last_r_cycle\n'
            f'{bank},0,4,{190 if fault == "preload" else 220},250,253\n', encoding='utf-8')
    result = dict(banks=[dict(bank=b, bursts=1, bytes=65 if fault == 'bytes' else 64) for b in range(4)])
    if fault == 'none':
        evidence = module.ddr_evidence(tmp_path, result)
        assert all(s['first_latency_cycles_mean'] == 30 for s in evidence['service'])
    else:
        with pytest.raises(ValueError):
            module.ddr_evidence(tmp_path, result)


def test_bounded_command(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A long simulation cannot silently continue beyond its requested bound."""
    module = load_module(monkeypatch)
    with pytest.raises(RuntimeError, match='wall-time timeout'):
        module.command_run([sys.executable, '-c', 'import time; time.sleep(30)'],
                           tmp_path, tmp_path/'timeout.log', 1)
