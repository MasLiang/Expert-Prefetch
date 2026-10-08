"""运行相同 Switch 四策略的 MIG/CDC/RDIMM 数字时序仿真，不冒充板级测试。"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import time
from typing import Any

from run_switch_full_size import CASES, PROJECT, ROOT, parse_result


def command_run(command: list[str], directory: Path, log: Path, timeout: int) -> float:
    """限制 wall-time；超时退休本次独立进程组，不能留下后台 XSim 子进程。"""
    started = time.monotonic()
    with log.open('w', encoding='utf-8') as stream:
        process = subprocess.Popen(command, cwd=directory, stdout=stream,
            stderr=subprocess.STDOUT, start_new_session=True)
        try:
            code = process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            # PID is the leader of the newly created, task-owned process group.
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            raise RuntimeError(f'wall-time timeout ({timeout}s): {log}') from None
    if code or re.search(r'(?im)^\s*(ERROR:|Fatal:|FATAL:)', log.read_text(encoding='utf-8')):
        raise RuntimeError(f'command failed ({code}): {log}')
    return time.monotonic()-started


def ddr_evidence(directory: Path, result: dict[str, Any]) -> dict[str, Any]:
    """核对四路初始化、实际 AXI 服务样本和 DDR 命令；不假定固定延迟。"""
    log = (directory/'simulate.log').read_text(encoding='utf-8')
    calibrated = re.findall(r'DDR_CALIBRATED bank=(\d+) cycle=(\d+)', log)
    initialized = re.findall(r'DDR_INITIALIZED bank=(\d+) cycle=(\d+)', log)
    for markers in (calibrated, initialized):
        if len(markers) != 4 or {int(b) for b, _ in markers} != set(range(4)):
            raise ValueError('all four MIG/RDIMM paths must initialize')
    marker = re.search(r'FULL_BACKEND MIG_RDIMM initialized_cycle=(\d+)', log)
    if marker is None:
        raise ValueError('not a MIG-backed Switch run')
    starts = re.findall(r'FULL_LAYER_START layer=(\d+) cycle=(\d+)', log)
    if not starts or int(starts[0][1]) <= max(int(c) for _, c in initialized):
        raise ValueError('DDR initialization leaked into measurement')
    evidence = []
    for bank in range(4):
        previous = -1
        count = size = latency_sum = service_sum = 0
        low: int | None = None
        high = 0
        with (directory/f'ddr_bank_{bank}.csv').open(encoding='utf-8') as stream:
            for row in csv.DictReader(stream):
                item = {k: int(v) for k, v in row.items()}
                a, first, last = item['ar_cycle'], item['first_r_cycle'], item['last_r_cycle']
                if item['bank'] != bank or not previous < a <= first <= last:
                    raise ValueError('invalid DDR service interval')
                if a <= max(int(c) for _, c in initialized):
                    raise ValueError('preload burst in measurement')
                payload = item['beats']*16
                if payload > 16*(last-a+1):
                    raise ValueError('bank AXI bandwidth violation')
                previous = last
                count += 1
                size += payload
                latency = first-a
                low = latency if low is None else min(low, latency)
                high = max(high, latency)
                latency_sum += latency
                service_sum += last-a+1
        expected = next(b for b in result['banks'] if b['bank'] == bank)
        if count != expected['bursts'] or size != expected['bytes'] or count == 0:
            raise ValueError('DDR service/physical metrics mismatch')
        evidence.append(dict(bank=bank, bursts=count, bytes=size,
            first_latency_cycles_min=low, first_latency_cycles_max=high,
            first_latency_cycles_mean=latency_sum/count, service_cycles_mean=service_sum/count))
    commands = re.findall(r'DDR_DRAM_SUMMARY bank=(\d+) reads=(\d+) activates=(\d+) refreshes=(\d+) precharges=(\d+)', log)
    if len(commands) != 4 or {int(row[0]) for row in commands} != set(range(4)) or any(int(row[1]) == 0 for row in commands):
        raise ValueError('missing actual DDR read command observations')
    return dict(initialization_done_cycles=dict(initialized), service=evidence,
        dram_commands=[dict(zip(('bank', 'reads', 'activates', 'refreshes', 'precharges'), map(int, row))) for row in commands])


def run_case(output: Path, name: str, geometry: dict[str, int], timeout: int) -> dict[str, Any]:
    """执行 Vivado 导出的脚本，检查完整日志与所有现有 full-size 功能门。"""
    directory = output/'cases'/name/'xsim'
    wall = command_run(['bash', 'tb_switch_full_size.sh'], directory, directory.parent/'console.log', timeout)
    for filename in ('compile.log', 'elaborate.log', 'simulate.log'):
        content = (directory/filename).read_text(encoding='utf-8')
        if re.search(r'(?im)^\s*(ERROR:|Fatal:|FATAL:)|timing violation', content):
            raise ValueError(f'DDR simulation error: {directory/filename}')
    result = dict(case=name, wall_seconds=wall,
                  **parse_result(directory/'simulate.log', geometry, *CASES[name]))
    result['ddr'] = ddr_evidence(directory, result)
    elaboration = (directory/'elaborate.log').read_text(encoding='utf-8')
    result['vendor_unsupported_assertion_count'] = elaboration.count('XSIM 43-4398')
    result['vendor_elaboration_warning_count'] = elaboration.count('WARNING:')
    (directory.parent/'summary.json').write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
    print(f'{name}: PASS cycles={result["metrics"]["cycles"]} wall={wall:.1f}s', flush=True)
    return result


def main() -> None:
    """与行为响应器分目录；预载不计入性能、错误或超时不能标成 PASS。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--example-project', type=Path, default=PROJECT/'results/m10_ddr4_controller/example/ddr4_0_ex/ddr4_0_ex.xpr')
    parser.add_argument('--mini', action='store_true')
    parser.add_argument('--layers', type=int, choices=range(2, 9), default=6)
    parser.add_argument('--jobs', type=int, choices=(1, 2, 3, 4), default=1)
    parser.add_argument('--timeout', type=int, default=14400, help='每组 wall-time 秒上限')
    parser.add_argument('--cases', nargs='+', choices=tuple(CASES), default=list(CASES))
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    if args.timeout <= 0:
        parser.error('timeout must be positive')
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    geometry = dict(H=16 if args.mini else 768, F=32 if args.mini else 3072,
                    FT=16 if args.mini else 128, EXPERTS=8, LAYERS=args.layers,
                    SLOT_BYTES=512 if args.mini else 131072, SLOTS_PER_BANK=32)
    sources = [*sorted((ROOT/'rtl').glob('*.sv')), ROOT/'tb/ddr4_bank_backend.sv',
               ROOT/'tb/tb_switch_full_size.sv', ROOT/'scripts/prepare_switch_ddr4.tcl',
               ROOT/'scripts/run_switch_full_size.py', Path(__file__).resolve()]
    hashes = {str(p.relative_to(PROJECT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    summary: dict[str, Any] = dict(status='running', geometry=geometry,
        full_switch_dimensions=not args.mini, trained_checkpoint=False, board_measured=False,
        backend='Vivado XSim 2023.2 / four MIG + async AXI CDC + RDIMM model / PHY BFM',
        clock_hz=200000000, mac_lanes=16, predictor_pattern=0,
        source_sha256=hashes, results=[], requested_cases=args.cases,
        measurement='first begin to final result; excludes DDR calibration/preload and final drain',
        limitations=['digital timing with PHY BFM; not physical-board timing',
                     'unsupported vendor assertions are reported, not treated as coverage',
                     '128-bit bank AXI; not a bandwidth-optimal U200 shell configuration'])
    path = output/'summary.json'
    path.write_text(json.dumps(summary, indent=2)+'\n', encoding='utf-8')
    try:
        command_run(['vivado', '-mode', 'batch', '-source', str(ROOT/'scripts/prepare_switch_ddr4.tcl'),
                     '-tclargs', str(output), str(args.example_project.resolve()),
                     *[str(geometry[k]) for k in ('H', 'F', 'FT', 'EXPERTS', 'LAYERS')]],
                    output, output/'prepare.log', 600)
        if 'SWITCH_DDR_PREPARE_PASS' not in (output/'prepare.log').read_text():
            raise ValueError('preparation did not complete')
        if args.prepare_only:
            summary['status'] = 'PREPARED_NOT_SIMULATED'
            path.write_text(json.dumps(summary, indent=2)+'\n', encoding='utf-8')
            return
        with ThreadPoolExecutor(max_workers=args.jobs) as pool:
            futures = [pool.submit(run_case, output, name, geometry, args.timeout) for name in args.cases]
            for future in futures:
                summary['results'].append(future.result())
                path.write_text(json.dumps(summary, indent=2)+'\n', encoding='utf-8')
        for source, digest in hashes.items():
            if hashlib.sha256((PROJECT/source).read_bytes()).hexdigest() != digest:
                raise ValueError(f'source changed during simulation: {source}')
        rows = []
        baseline = next((r['metrics']['cycles'] for r in summary['results'] if r['case'] == 'A_whole_demand'), None)
        stream = next((r['metrics']['cycles'] for r in summary['results'] if r['case'] == 'B_stream_demand'), None)
        for r in summary['results']:
            m = r['metrics']
            rows.append(dict(case=r['case'], **m, milliseconds=m['cycles']/200000,
                speedup_vs_A=None if baseline is None else baseline/m['cycles'],
                speedup_vs_B=None if stream is None else stream/m['cycles']))
        with (output/'comparison.csv').open('w', newline='', encoding='utf-8') as stream_file:
            writer = csv.DictWriter(stream_file, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        summary['status'] = 'PASS'
    except Exception as error:
        summary['status'] = 'FAIL'
        summary['error'] = str(error)
        raise
    finally:
        path.write_text(json.dumps(summary, indent=2)+'\n', encoding='utf-8')


if __name__ == '__main__':
    main()
