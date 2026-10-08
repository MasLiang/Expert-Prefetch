"""小型控制边界矩阵：真实 XSim + 主模型，不做 paper-scale DSE。"""
from __future__ import annotations

import argparse
import csv
import json
import hashlib
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

from compare_unified_alignment import PROJECT, write_json
from validate_boundary_readiness import run_readiness
from validate_commit_focus import run_focus


def boundary_cases() -> list[dict[str, Any]]:
    """五组资源条件成对 no-prefetch/hit，17 个逐拍晚预测点及三个回归点。"""
    cases=[]
    for idx,(slots,entries,depth) in enumerate(((2,2,2),(2,4,2),(2,8,4),(4,4,2),(4,8,4))):
        for name,mode in (('none',0),('hit',1)):
            cases.append(dict(name=f'p{idx}_{name}',layers=3,future_mode=mode,slots_per_bank=slots,
                descriptor_entries=entries,fifo_depth=depth,expect_early_overlap=int(entries>2)))
    cases.extend(dict(name=f'late_{delay}',layers=3,future_mode=1,slots_per_bank=2,
                      predictor_delay_cycles=delay) for delay in range(434,451))
    cases.extend((dict(name='wrong_three',layers=3,future_mode=2,slots_per_bank=2),
        dict(name='hit_eight',layers=8,future_mode=1,slots_per_bank=2),
        dict(name='ddr_gaps',layers=3,future_mode=1,slots_per_bank=2,ddr_gaps=1,bandwidth_efficiency=0.5)))
    return cases


def summarize(output: Path) -> dict[str, Any]:
    """聚合收益方向与 admission 截止边界；不将小残差自动标为可忽略。"""
    run=json.loads((output/'summary.json').read_text())
    by_name={row['case']:row for row in run['cases']}
    gains=[]
    for idx in range(5):
        baseline,hit=by_name[f'p{idx}_none'],by_name[f'p{idx}_hit']
        m=baseline['main_cycles']-hit['main_cycles'];r=baseline['rtl_cycles']-hit['rtl_cycles']
        gains.append(dict(condition=idx,main_saved_cycles=m,rtl_saved_cycles=r,
            gain_sign_equal=(m>0)-(m<0)==(r>0)-(r<0),saved_delta_cycles=m-r))
    late=[]
    for delay in range(434,451):
        directory=output/f'late_{delay}';row=by_name[directory.name]
        resources=json.loads((directory/'resource_comparison.json').read_text())
        late.append(dict(delay_cycles=delay,main_cycles=row['main_cycles'],rtl_cycles=row['rtl_cycles'],
            main_speculative_bytes=resources['main_speculative_bytes'][1],rtl_speculative_bytes=resources['rtl_speculative_bytes'][1],
            bytes_equal=row['speculative_bytes_equal'],rule_equal=row['execution_rule_alignment_passed']))
    for filename,values in (('gain_comparison.csv',gains),('late_boundary.csv',late)):
        with (output/filename).open('w',newline='',encoding='utf-8') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(values[0]));writer.writeheader();writer.writerows(values)
    segments=[r for directory in by_name for r in csv.DictReader((output/directory/'wait_segments.csv').open())]
    outcome=dict(status='validation_complete',case_count=len(by_name),functional_passed=run['rtl_functional_checks_passed'],
        all_execution_rules_equal=run['execution_rule_alignment_passed'],
        all_fifo_occupancy_peaks_equal=all(v.get('fifo_occupancy_peaks_equal',False) for v in by_name.values()),
        admission_boundary_equal=all(v['bytes_equal'] for v in late),
        fixed_plan_gain_alignment_passed=all(v['saved_delta_cycles']==0 for v in gains),
        all_cycles_equal=all(v['delta_cycles']==0 for v in by_name.values()),
        max_abs_delta_cycles=max(abs(v['delta_cycles']) for v in by_name.values()),
        cycle_mismatch_cases=[dict(case=name,delta_cycles=row['delta_cycles'])
                              for name,row in by_name.items() if row['delta_cycles']!=0],
        resource_gain_sign_mismatches=sum(not v['gain_sign_equal'] for v in gains),
        late_rule_mismatches=[v['delay_cycles'] for v in late if not v['rule_equal']],
        main_first_prefetch_loss_delay=min(v['delay_cycles'] for v in late if v['main_speculative_bytes']<late[0]['main_speculative_bytes']),
        rtl_first_prefetch_loss_delay=min(v['delay_cycles'] for v in late if v['rtl_speculative_bytes']<late[0]['rtl_speculative_bytes']),
        rtl_readiness_classes=sorted({v['ready_class'] for v in segments}),
        selected_to_accept_cycles=sorted({int(v['rtl_cycles']) for v in segments if v['interval']=='select_to_accept'}),
        accept_to_gu_cycles=sorted({int(v['rtl_cycles']) for v in segments if v['interval']=='accept_to_gu'}),
        exact_alignment=False,can_ignore_boundary_error=False,
        limitations=['只测当前固定夹具与no-prefetch/full-expert计划，未比较所有部署Scheduler排名',
                     'descriptor/FIFO credit snapshots是全局观测；slot FREE另有generation级观测',
                     '同拍GU/truth由独立定向scheduler bench验证，矩阵未保证出现同拍'],
        gains=gains)
    write_json(output/'boundary_summary.json',outcome)
    scripts=Path(__file__).parent
    write_json(output/'analysis_manifest.json',dict(mode='postprocess_existing_run',
        source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in
            [Path(__file__),scripts/'analyze_boundary_waits.py',scripts/'validate_boundary_readiness.py',scripts/'validate_commit_focus.py']},
        matrix_manifest_sha256=hashlib.sha256((output/'manifest.json').read_bytes()).hexdigest()))
    return outcome


def main() -> None:
    """重建矩阵；analyze-only 仅分析已有完整运行，不宣称重新仿真。"""
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=PROJECT/'results/m10_boundary_validation')
    parser.add_argument('--analyze-only',action='store_true')
    parser.add_argument('--config',type=Path,default=PROJECT/'configs/hardware/u200_interface_alignment.yaml',
                        help='显式选择结构配置；旧默认保留以便复现历史结果')
    args=parser.parse_args();output=args.output.resolve();output.mkdir(parents=True,exist_ok=True)
    if not args.analyze_only:
        run_readiness(output/'readiness_cases')
        run_focus(output/'commit_focus')
        cfg=yaml.safe_load(args.config.read_text())
        cfg['cases']=boundary_cases();cfg['output_directory']=str(output)
        path=output/'experiment.yaml';path.write_text(yaml.safe_dump(cfg,sort_keys=False),encoding='utf-8')
        write_json(output/'boundary_summary.json',dict(status='running'))
        command=[sys.executable,str(Path(__file__).with_name('compare_unified_alignment.py')),'--config',str(path)]
        with (output/'matrix_console.log').open('w',encoding='utf-8') as stream:
            subprocess.run(command,check=True,stdout=stream,stderr=subprocess.STDOUT)
    print(json.dumps(summarize(output),ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
