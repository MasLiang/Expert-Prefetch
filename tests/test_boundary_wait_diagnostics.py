"""分段等待不能双计数、缺事件不能默默补零；矩阵输入确定且覆盖截止点。"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

import pytest

from test_unified_rtl_comparison import load_runner


def fixture() -> tuple[str, dict[str, Any]]:
    """可手算的单 tile 接口样本，所有时间为 cycle。"""
    lines=['LAYER_START layer=0 cycle=0','MAIN_TRUTH layer=0 cycle=10 expert=3 weight=32768']
    for event,cycle,matrix in (('ADMIT',1,0),('ADMIT',3,1),('READY',12,0),('SELECT',14,0),
        ('COMPUTE_ACCEPT',15,0),('RELEASE_GU',21,0),('SLOT_FREE',22,0),('RELEASE_DOWN',31,1),('SLOT_FREE',32,1)):
        lines.append(f'MAIN_CTL event={event} layer=0 cycle={cycle} expert=3 tile=0 matrix={matrix} slot={matrix} generation=1 epoch=1')
    for name,cycle in (('GU_START',16),('GU_DONE',20),('DOWN_START',25),('DOWN_DONE',30)):
        lines.append(f'MAIN_STAGE layer=0 cycle={cycle} expert=3 tile=0 stage={name}')
    trace=[dict(event_type=e,cycle=c,end_cycle=end,layer_id=0,expert_id=3,tile_id=0,matrix_type=m)
        for e,c,m,end in (('pair_admit',2,None,None),('dma_done',11,'gate_up',None),
                         ('tile_select',11,'gate_up',None),('aie_start',15,'gate_up',19),('aie_start',24,'down',29))]
    return '\n'.join(lines),dict(trace=trace,layer_results={0:dict(layer_start_cycle=0,router_resolve_cycle=10)})


def test_partition_and_unmodeled_handshake_are_explicit(tmp_path: Path) -> None:
    """aggregate 只加三个区间；RTL 独有握手不假装主模型已表示。"""
    log,result=fixture();rows=load_runner().analyze_waits(log,result,tmp_path)
    partition=[v for v in rows if v['partition_member']]
    assert sum(v['main_cycles'] for v in partition)==15
    assert sum(v['rtl_cycles'] for v in partition)==16
    assert next(v for v in rows if v['interval']=='select_to_accept')['main_cycles'] is None
    assert next(v for v in rows if v['interval']=='release_gu_to_free')['rtl_cycles']==1
    assert (tmp_path/'control_events.csv').is_file()


@pytest.mark.parametrize('change',['missing','duplicate','premature'])
def test_bad_observations_fail_loudly(tmp_path: Path, change: str) -> None:
    """丢失、重复或提前握手都不能生成貌似对齐的 CSV。"""
    log,result=fixture()
    line=next(v for v in log.splitlines() if 'event=COMPUTE_ACCEPT ' in v)
    if change=='missing':log=log.replace(line,'')
    elif change=='duplicate':log+='\n'+line
    else:log=log.replace('event=COMPUTE_ACCEPT layer=0 cycle=15','event=COMPUTE_ACCEPT layer=0 cycle=13')
    with pytest.raises(ValueError):load_runner().analyze_waits(log,result,tmp_path)


def test_matrix_has_adjacent_late_points_and_matched_resource_pairs() -> None:
    """30 个预定义点，不依赖任何 RTL 观测决定输入。"""
    load_runner()
    path=Path(__file__).resolve().parents[1]/'hardware/m10_u200/scripts/run_boundary_validation.py'
    spec=importlib.util.spec_from_file_location('boundary_driver_test',path)
    assert spec is not None and spec.loader is not None
    module=importlib.util.module_from_spec(spec)
    sys.path.insert(0,str(path.parent))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.remove(str(path.parent))
    cases=module.boundary_cases()
    assert len(cases)==30 and cases==module.boundary_cases()
    assert [v['predictor_delay_cycles'] for v in cases if v['name'].startswith('late_')]==list(range(434,451))
    for idx in range(5):
        none,hit=(next(v for v in cases if v['name']==f'p{idx}_{mode}') for mode in ('none','hit'))
        assert {k:v for k,v in none.items() if k not in ('name','future_mode')}=={k:v for k,v in hit.items() if k not in ('name','future_mode')}
