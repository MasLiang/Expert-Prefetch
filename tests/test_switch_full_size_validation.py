"""Complete-size RTL log acceptance must reject scaled or incomplete evidence."""
from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest


@pytest.fixture
def validator() -> ModuleType:
    """Load the verification-only driver without executing XSim."""
    path = Path(__file__).resolve().parents[1]/'hardware/m10_u200/scripts/run_switch_full_size.py'
    spec = importlib.util.spec_from_file_location('switch_full_validator', path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def complete_log(tmp_path: Path) -> tuple[Path, dict[str, int]]:
    """Build synthetic parser evidence; this fixture is NOT an RTL simulation."""
    geometry = dict(H=768, F=3072, FT=128, EXPERTS=8, LAYERS=6)
    activation = 16
    lines = []
    for layer in range(6):
        output = 24*activation*(1+layer%2)
        lines.append(f'FULL_LAYER_RESULT layer={layer} cycle={1000*(layer+1)} y={output} first_gu={1000*layer+100}')
        activation = min(127, output>>5)
    payload = 2*768*3072*6
    lines.append(f'FULL_METRICS cycles=6000 wait_gu=20 wait_down=10 router=60 expert=3000 peak_slots=48 spec=0 demand={payload} useful=0 wasted=0 cancelled=0')
    lines.extend(f'FULL_BANK bank={b} bytes={payload//4} bursts=100' for b in range(4))
    lines.extend(['TEST_PASS full Switch MoE H=768 F=3072 E=8 layers=6 whole=1 prefetch=0 pattern=0', '$finish called'])
    log = tmp_path/'simulation.log'
    log.write_text('\n'.join(lines), encoding='utf-8')
    return log, geometry


def test_complete_log(validator: ModuleType, complete_log: tuple[Path, dict[str, int]]) -> None:
    """Require all six independent numerical references and actual payload counts."""
    log, geometry = complete_log
    result = validator.parse_result(log, geometry, 1, 0)
    assert result['functional_pass']
    assert len(result['layers']) == 6


@pytest.mark.parametrize('old,new', [
    ('H=768 F=3072 E=8', 'H=16 F=32 E=8'),
    ('TEST_PASS', 'INCOMPLETE'),
    ('layer=5', 'layer=6'),
    ('y=384', 'y=385'),
    ('spec=0 demand=', 'spec=1 demand='),
    ('bank=3', 'bank=4'),
])
def test_invalid_evidence(validator: ModuleType, complete_log: tuple[Path, dict[str, int]],
                          old: str, new: str) -> None:
    """Never accept a small model or a broken checker as full-model evidence."""
    log, geometry = complete_log
    log.write_text(log.read_text().replace(old, new), encoding='utf-8')
    with pytest.raises(ValueError):
        validator.parse_result(log, geometry, 1, 0)
