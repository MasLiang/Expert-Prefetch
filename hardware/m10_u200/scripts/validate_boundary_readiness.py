"""真实 XSim 定向验证 GU ready 相对最后 truth 的三个边界。"""
from __future__ import annotations

import argparse
import hashlib
import re
from pathlib import Path
from typing import Any

from compare_unified_alignment import PROJECT, ROOT, command_run, write_json


def run_readiness(output: Path) -> dict[str, Any]:
    """在独立 snapshot 上验证 -2/0/+2 cycle；不改主设计或使用 force。"""
    output.mkdir(parents=True,exist_ok=True); build=output/'build';build.mkdir(exist_ok=True)
    sources=[ROOT/'rtl/package.sv',ROOT/'rtl/tile_execution_scheduler.sv',ROOT/'tb/tb_boundary_readiness.sv']
    write_json(output/'summary.json',dict(status='running'))
    write_json(output/'manifest.json',dict(source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [*sources,Path(__file__)]}))
    command_run(['xvlog','--sv',*map(str,sources)],build,output/'compile.log')
    rows=[]
    for offset,name in ((-2,'before'),(0,'same'),(2,'after')):
        snapshot='boundary_ready_'+name;directory=output/name;directory.mkdir(exist_ok=True)
        command_run(['xelab','--debug','off','--snapshot',snapshot,'--generic_top',f'READY_OFFSET={offset}',
                     'work.tb_boundary_readiness'],build,directory/'elaborate.log')
        batch=directory/'run.tcl';batch.write_text('run 10 us\nquit\n',encoding='utf-8')
        command_run(['xsim',snapshot,'--tclbatch',str(batch.resolve()),'--onerror','quit','--onfinish','quit',
                     '--log',str((directory/'simulation.log').resolve())],build,directory/'console.log')
        log=(directory/'simulation.log').read_text()
        if log.count('PASS:')!=1 or '$finish called' not in log or re.search(r'Fatal:|FATAL:|ERROR:|Error:',log):
            raise ValueError(f'readiness 自检失败 {name}')
        match=re.search(r'offset=(-?\d+) truth=(\d+) ready=(\d+) accept=(\d+)',log)
        if match is None:raise ValueError('缺少 readiness 观测')
        o,t,r,a=map(int,match.groups());rows.append(dict(offset=o,truth_cycle=t,ready_cycle=r,accept_cycle=a,
                                                     ready_to_accept_cycles=a-max(t,r)))
    result=dict(status='validation_complete',backend='XSim',passed=True,cases=rows)
    write_json(output/'summary.json',result)
    return result


def main() -> None:
    """默认保存至同一个边界验证目录。"""
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=PROJECT/'results/m10_boundary_validation/readiness_cases')
    print(run_readiness(parser.parse_args().output.resolve()))


if __name__=='__main__':
    main()
