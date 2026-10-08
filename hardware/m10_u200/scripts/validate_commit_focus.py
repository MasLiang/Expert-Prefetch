"""对无预取/命中相同基础激励做独立 commit 被动观测。"""
from __future__ import annotations

import csv
import argparse
import hashlib
import re
from pathlib import Path

from compare_unified_alignment import PROJECT,ROOT,command_run,write_json


def run_focus(output: Path) -> dict:
    """两个 XSim 实验正常自检结束后输出已 resident 但未 committed 的等待。"""
    output.mkdir(parents=True,exist_ok=True)
    build=output/'build';build.mkdir(exist_ok=True)
    package=ROOT/'rtl/package.sv'
    sources=[package,*sorted(p for p in (ROOT/'rtl').glob('*.sv') if p!=package),
             ROOT/'tb/tb_u200_moe.sv',ROOT/'tb/tb_commit_observer.sv']
    write_json(output/'manifest.json',dict(source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [*sources,Path(__file__)]}))
    command_run(['xvlog','--sv',*map(str,sources)],build,output/'compile.log')
    cases=[]
    for mode,name in ((0,'none'),(1,'hit')):
        directory=output/name;directory.mkdir(exist_ok=True);snapshot='commit_focus_'+name
        command_run(['xelab','--debug','off','--snapshot',snapshot,'--generic_top',f'FUTURE_MODE={mode}',
                     'work.tb_commit_observer'],build,directory/'elaborate.log')
        batch=directory/'run.tcl';batch.write_text('run 1 ms\nquit\n',encoding='utf-8')
        command_run(['xsim',snapshot,'--testplusarg','ALIGN_MAIN','--tclbatch',str(batch),'--onerror','quit',
                     '--onfinish','quit','--log',str(directory/'simulation.log')],build,directory/'console.log')
        log=(directory/'simulation.log').read_text()
        if log.count('PASS:')!=1 or '$finish called' not in log or re.search(r'Fatal:|FATAL:|ERROR:|Error:',log):
            raise ValueError(f'commit focus 功能检查失败 {name}')
        rows=[tuple(map(int,v)) for v in re.findall(r'FOCUS_GU cycle=(\d+) layer=(\d+) phase=(\d+) ready=(\d+) state=(\d+) payload=(\d+)',log)]
        with (directory/'commit_wait.csv').open('w',newline='',encoding='utf-8') as stream:
            writer=csv.writer(stream);writer.writerow(['cycle','layer','phase','ready','metadata_state','payload_complete']);writer.writerows(rows)
        blocked=[r for r in rows if r[2]==1 and r[3]==0 and r[4]==5 and r[5]==1]
        if mode==1 and not blocked:raise ValueError('没有捕获命中 payload ready、commit 未完成的实际等待')
        cases.append(dict(case=name,functional_passed=True,resident_uncommitted_wait_cycles=len(blocked),events=blocked))
    result=dict(status='validation_complete',cases=cases)
    write_json(output/'summary.json',result)
    return result


def main() -> None:
    """在独立目录保存 commit 等待观测。"""
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=PROJECT/'results/m10_boundary_validation/commit_focus')
    print(run_focus(parser.parse_args().output.resolve()))


if __name__=='__main__':
    main()
