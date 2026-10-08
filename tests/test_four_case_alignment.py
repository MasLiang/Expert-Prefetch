"""四 case 主模型运行与 RTL 性能日志的严格验收。"""
from __future__ import annotations

import importlib
from pathlib import Path
import sys
from typing import Any

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]


def driver(monkeypatch: pytest.MonkeyPatch) -> Any:
    """导入使用主 UnifiedSequenceSimulator 的实验驱动。"""
    monkeypatch.syspath_prepend(str(ROOT / 'hardware/m10_u200/scripts'))
    return importlib.import_module('compare_four_cases')


@pytest.mark.parametrize('architecture', ['swiglu', 'relu_ffn'])
@pytest.mark.parametrize('case', ['A_whole_demand', 'B_stream_demand',
                                 'C_whole_prefetch', 'D_stream_tile_prefetch'])
def test_four_case_determinism(case: str, architecture: str, tmp_path: Path,
                               monkeypatch: pytest.MonkeyPatch) -> None:
    """每种模式重复运行必须得到完全相同的主模型指标与 timeline。"""
    module = driver(monkeypatch)
    config = yaml.safe_load((ROOT / 'configs/hardware/u200_four_case_validation.yaml').read_text())
    config['expert_architecture'] = architecture
    result = module.simulate(config, config['bandwidth_points'][0], 'hit', case, tmp_path)
    assert result == module.simulate(config, config['bandwidth_points'][0], 'hit', case, tmp_path)
    assert result['makespan_cycles'] > 0


@pytest.mark.parametrize('fault', ['duplicate', 'missing', 'early', 'fatal'])
def test_reject_invalid_performance_log(fault: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """数值 PASS 本身不足以掩盖重复计算、缺失阶段或整 Expert 提前启动。"""
    module = driver(monkeypatch)
    lines = ['PASS: numerical checks', '$finish called', 'LAYER_START layer=0 cycle=0',
             'LAYER_RESULT layer=0 cycle=100',
             'MAIN_TRUTH layer=0 cycle=1 expert=1 weight=32768',
             'WEIGHTS_READY layer=0 cycle=10']
    stages = [f'MAIN_STAGE layer=0 cycle={20+t*20+i} expert=1 tile={t} stage={stage}'
              for t in (0, 1)
              for i, stage in enumerate(('GU_START', 'GU_DONE', 'DOWN_START', 'DOWN_DONE'))]
    if fault == 'duplicate':
        stages.append(stages[0])
    elif fault == 'missing':
        stages.pop()
    elif fault == 'early':
        stages[0] = stages[0].replace('cycle=20', 'cycle=5')
    else:
        lines.append('FATAL: invalid payload')
    with pytest.raises(ValueError):
        module.verify_rtl('\n'.join(lines + stages), 1, True)


@pytest.mark.parametrize('switch', [False, True])
@pytest.mark.parametrize('fault', ['none', 'byte_count', 'no_prefetch'])
def test_expert_payload_and_prefetch_contract(switch: bool, fault: str,
                                             monkeypatch: pytest.MonkeyPatch) -> None:
    """不能仅凭 ReLU 数值结果将错误 future 大小或零预取认作成功。"""
    module = driver(monkeypatch)
    size = 256 if switch else 512
    if fault == 'byte_count':
        size = 512 if switch else 256
    kind = 1 if fault == 'no_prefetch' else 0
    lines = ['PASS: numerical checks', '$finish called', 'LAYER_START layer=0 cycle=0',
             'LAYER_RESULT layer=0 cycle=100',
             'MAIN_TRUTH layer=0 cycle=1 expert=1 weight=32768',
             'WEIGHTS_READY layer=0 cycle=10',
             f'MAIN_ADMIT layer=0 cycle=2 expert=1 tile=0 matrix=0 bytes={size} class={kind}']
    lines.extend(f'MAIN_STAGE layer=0 cycle={20+t*20+i} expert=1 tile={t} stage={stage}'
                 for t in (0, 1)
                 for i, stage in enumerate(('GU_START', 'GU_DONE', 'DOWN_START', 'DOWN_DONE')))
    if fault == 'none':
        assert module.verify_rtl('\n'.join(lines), 1, True, switch=switch, prefetch=True)[0] == 100
    else:
        with pytest.raises(ValueError):
            module.verify_rtl('\n'.join(lines), 1, True, switch=switch, prefetch=True)
