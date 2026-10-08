"""请求级诊断必须拒绝顺序错误、byte 不守恒及没有 terminal PASS 的日志。"""
from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any

import pytest


def fixture(monkeypatch: pytest.MonkeyPatch) -> tuple[Any, dict[str, Any], dict[str, Any], str, list[Any]]:
    """构造与完整尺寸格式一致的独立两请求观测，不依赖生成结果文件。"""
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]/'hardware/m10_u200/scripts'))
    module = importlib.import_module('diagnose_switch_full_axi')
    trace = [dict(event_type=event, layer_id=0, expert_id=1, tile_id=0, matrix_type=matrix, cycle=cycle)
             for event in ('dma_start', 'aie_start') for matrix, cycle in (('gate_up', 4), ('down', 10))]
    main = dict(trace=trace, makespan_cycles=20, peak_ocm_bytes=131072,
        memory_metrics=dict(demand_bytes=2, speculative_bytes=0, useful_speculative_bytes=0,
                            wasted_speculative_bytes=0, canceled_speculative_bytes=0))
    rtl = dict(functional_pass=True, metrics=dict(cycles=20, demand=2, spec=0, useful=0,
                                                 wasted=0, cancelled=0, peak_slots=1))
    log = ('ALIGN_PAIR cycle=0 layer=0 expert=1 tile=0 class=1 slots=0\n'
           'ALIGN_CLAIM cycle=4 layer=0 expert=1 tile=0 matrix=0 class=1\n'
           'ALIGN_CLAIM cycle=10 layer=0 expert=1 tile=0 matrix=1 class=1\n'
           'ALIGN_ROUTE cycle=1 layer=0 expert=1 tile=0 matrix=0 truth=1\n'
           'TEST_PASS full Switch MoE\n$finish called\n')
    timeline = [dict(cycle='0', event='LAYER_START', layer_id='0', expert_id='-1', tile_id='-1'),
                *[dict(cycle=str(c), event=e, layer_id='0', expert_id='1', tile_id='0')
                  for c,e in ((4,'GU_START'),(10,'DOWN_START'))]]
    return module, main, rtl, log, timeline


def test_request_diagnostic_pass(monkeypatch: pytest.MonkeyPatch) -> None:
    """重复 claim 是合法恢复，不能伪造额外首发请求。"""
    module, main, rtl, log, timeline = fixture(monkeypatch)
    log = log.replace('ALIGN_ROUTE', 'ALIGN_CLAIM cycle=11 layer=0 expert=1 tile=0 matrix=0 class=1\nALIGN_ROUTE')
    result = module.compare(main, rtl, log, timeline)
    assert result['status'] == 'REQUEST_ORDER_AND_BYTES_PASS'
    assert result['rtl_start_count'] == 2 and result['cycle_exact']


@pytest.mark.parametrize('fault', ['order', 'bytes', 'compute', 'cancel', 'peak'])
def test_request_diagnostic_rejects_mismatch(fault: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """总周期相同仍须拒绝其他验收不一致。"""
    module, main, rtl, log, timeline = fixture(monkeypatch)
    if fault == 'order':
        main['trace'][0]['expert_id'] = 2
    elif fault == 'bytes':
        main['memory_metrics']['speculative_bytes'] = 1
    elif fault == 'compute':
        timeline[1]['tile_id'] = '1'
    elif fault == 'cancel':
        main['memory_metrics']['canceled_speculative_bytes'] = 1
    else:
        main['peak_ocm_bytes'] *= 2
    result = module.compare(main, rtl, log, timeline)
    assert result['status'] == 'MISMATCH' and result['cycle_exact']


@pytest.mark.parametrize('fault', ['no_pass', 'no_debug', 'fatal', 'not_finished'])
def test_request_diagnostic_rejects_invalid_log(fault: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """不能从不完整、没有观测或有 FATAL 的仿真宣称 PASS。"""
    module, main, rtl, log, timeline = fixture(monkeypatch)
    if fault == 'no_pass':
        log = log.replace('TEST_PASS full Switch MoE', '')
    elif fault == 'no_debug':
        log = 'TEST_PASS full Switch MoE\n$finish called\n'
    elif fault == 'fatal':
        log += 'Fatal: protocol failure\n'
    else:
        log = log.replace('$finish called', '')
    with pytest.raises(ValueError):
        module.compare(main, rtl, log, timeline)
