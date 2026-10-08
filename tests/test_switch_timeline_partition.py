"""Verification-only event-span conservation and causality tests."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


@pytest.mark.parametrize('noncausal', [False, True])
def test_partition(tmp_path: Path, noncausal: bool) -> None:
    """Observed intervals must conserve cycles and reject reversed stages."""
    source = Path(__file__).resolve().parents[1]/'hardware/m10_u200/scripts/analyze_switch_full_size_timelines.py'
    spec = importlib.util.spec_from_file_location('switch_timeline_partition', source)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    timeline = tmp_path/'timeline.csv'
    timeline.write_text('cycle,layer_id,tile_id,event\n'
                        '0,0,-1,LAYER_START\n10,0,0,GU_START\n'
                        f'{5 if noncausal else 20},0,0,DOWN_START\n'
                        '30,0,1,GU_START\n40,0,1,DOWN_START\n'
                        '50,0,-1,LAYER_DONE\n', encoding='utf-8')
    if noncausal:
        with pytest.raises(ValueError, match='noncausal'):
            module.partition(timeline, 1, 2)
    else:
        totals = module.partition(timeline, 1, 2)['totals']
        assert totals['total_cycles'] == 50
        assert totals['first_gu_delay_cycles'] == 10
        assert totals['gu_to_down_cycles'] == 20
        assert totals['down_to_next_gu_cycles'] == 10
        assert totals['last_down_to_result_cycles'] == 10


@pytest.mark.parametrize('layout', ['axi', 'ddr', 'ambiguous', 'missing'])
def test_timeline_location(tmp_path: Path, layout: str) -> None:
    """The NOT MIG backend description must not route AXI evidence to DDR paths."""
    source = Path(__file__).resolve().parents[1]/'hardware/m10_u200/scripts/analyze_switch_full_size_timelines.py'
    spec = importlib.util.spec_from_file_location('switch_timeline_location', source)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    paths = [tmp_path/'A_whole_demand/timeline.csv', tmp_path/'cases/A_whole_demand/xsim/timeline.csv']
    for index, path in enumerate(paths):
        if layout == 'ambiguous' or layout == ('axi' if index == 0 else 'ddr'):
            path.parent.mkdir(parents=True)
            path.write_text('', encoding='utf-8')
    if layout in ('ambiguous', 'missing'):
        with pytest.raises(ValueError, match='missing or ambiguous'):
            module.timeline_location(tmp_path, 'A_whole_demand')
    else:
        assert module.timeline_location(tmp_path, 'A_whole_demand') == paths[0 if layout == 'axi' else 1]
