"""完整 Switch MoE 尺寸的 XSim 四策略比较；明确使用 AXI responder。"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
import hashlib
import json
from pathlib import Path
import re
import subprocess
import time
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parents[1]
CASES = {'A_whole_demand': (1, 0), 'B_stream_demand': (0, 0),
         'C_whole_full_prefetch': (1, 1), 'D_stream_tile_prefetch': (0, 1)}


def command_run(command: list[str], directory: Path, log: Path, timeout: int = 1800) -> float:
    """保存完整日志并同时限制 wall time；不能以工具 exit code 代替 PASS。"""
    start = time.monotonic()
    with log.open('w', encoding='utf-8') as stream:
        result = subprocess.run(command, cwd=directory, stdout=stream,
                                stderr=subprocess.STDOUT, timeout=timeout, check=False)
    if result.returncode:
        raise RuntimeError(f'command failed ({result.returncode}): {log}')
    text = log.read_text(encoding='utf-8')
    if re.search(r'(?im)^\s*(ERROR:|Fatal:|FATAL:)', text):
        raise RuntimeError(f'command error despite return code zero: {log}')
    return time.monotonic()-start


def parse_result(log: Path, geometry: dict[str, int], whole: int,
                 prefetch: int) -> dict[str, Any]:
    """验收完整尺寸 PASS、完整层数、golden 结果与物理/telemetry 守恒。"""
    text = log.read_text(encoding='utf-8')
    if text.count('TEST_PASS full Switch MoE') != 1 or '$finish called' not in text:
        raise ValueError(f'missing terminal PASS: {log}')
    if re.search(r'(?im)^\s*(ERROR:|Fatal:|FATAL:)', text):
        raise ValueError(f'simulation errors: {log}')
    marker = re.search(r'TEST_PASS full Switch MoE H=(\d+) F=(\d+) E=(\d+) layers=(\d+) whole=(\d+) prefetch=(\d+)', text)
    expected = (geometry['H'], geometry['F'], geometry['EXPERTS'], geometry['LAYERS'], whole, prefetch)
    if marker is None or tuple(map(int, marker.groups())) != expected:
        raise ValueError('snapshot geometry does not match requested full model')
    layers = [tuple(map(int, match)) for match in re.findall(
        r'FULL_LAYER_RESULT layer=(\d+) cycle=(\d+) y=(\d+) first_gu=(\d+)', text)]
    if [layer[0] for layer in layers] != list(range(geometry['LAYERS'])):
        raise ValueError('incomplete layer execution')
    activation = 16
    for layer, _, observed, _ in layers:
        expected_y = geometry['F']//geometry['FT'] * activation * (1 + (layer%geometry['EXPERTS'])%2)
        if observed != expected_y:
            raise ValueError('independent layer golden mismatch')
        activation = min(127, expected_y>>((geometry['F']//geometry['FT']-1).bit_length()))
    match = re.search(r'FULL_METRICS (.+)', text)
    if match is None:
        raise ValueError('missing measured metrics')
    metrics = {key: int(value) for key, value in re.findall(r'(\w+)=(\d+)', match.group(1))}
    banks = [dict(bank=int(b), bytes=int(size), bursts=int(n)) for b, size, n in
             re.findall(r'FULL_BANK bank=(\d+) bytes=(\d+) bursts=(\d+)', text)]
    if {b['bank'] for b in banks} != set(range(4)) or any(b['bytes'] <= 0 for b in banks):
        raise ValueError('four-bank transfers missing')
    if sum(b['bytes'] for b in banks) != metrics['spec']+metrics['demand']:
        raise ValueError('byte conservation failed')
    if metrics['spec'] != metrics['useful']+metrics['wasted']:
        raise ValueError('speculative accounting failed')
    if not prefetch and metrics['demand'] != 2*geometry['H']*geometry['F']*geometry['LAYERS']:
        raise ValueError('baseline did not transfer complete true Expert matrices')
    return dict(functional_pass=True, metrics=metrics, banks=banks, layers=layers)


def main() -> None:
    """重建当前 RTL snapshot，运行四 case，输出可复现 CSV/JSON 和 cycle timeline。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=PROJECT/'results/m10_switch_full_size')
    parser.add_argument('--experts', type=int, choices=(8, 16), default=8)
    parser.add_argument('--layers', type=int, choices=range(2, 9), default=6)
    parser.add_argument('--beat-gap', type=int, default=0)
    parser.add_argument('--startup', type=int, default=0)
    parser.add_argument('--pattern', type=int, choices=(0, 1), default=0)
    parser.add_argument('--cases', nargs='+', choices=tuple(CASES), default=list(CASES))
    parser.add_argument('--mini', action='store_true', help='仅验证新 bench；明确不作为完整尺寸结果')
    parser.add_argument('--jobs', type=int, choices=(1, 2, 3, 4), default=2)
    parser.add_argument('--incremental', action='store_true')
    parser.add_argument('--build-only', action='store_true')
    parser.add_argument('--alignment-debug', action='store_true', help='输出请求级观测；不改变 DUT 激励')
    args = parser.parse_args()
    if args.beat_gap < 0 or args.startup < 0:
        parser.error('DDR responder gaps/startup must be non-negative cycles')
    output = args.output.resolve()
    build = output/'build'
    build.mkdir(parents=True, exist_ok=True)
    geometry = dict(H=16 if args.mini else 768, F=32 if args.mini else 3072,
                    FT=16 if args.mini else 128, EXPERTS=args.experts, LAYERS=args.layers,
                    SLOT_BYTES=512 if args.mini else 131072,
                    SLOTS_PER_BANK=32)
    package = ROOT/'rtl/package.sv'
    sources = [package, *sorted(p for p in (ROOT/'rtl').glob('*.sv') if p != package),
               ROOT/'tb/tb_switch_full_size.sv']
    hashes = {str(p.relative_to(PROJECT)): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in [*sources, Path(__file__), ROOT/'scripts/run_switch_full_size_snapshots.py']}
    summary: dict[str, Any] = dict(status='running', geometry=geometry,
        full_switch_dimensions=not args.mini, trained_checkpoint=False,
        backend='Vivado XSim 2023.2 / four-bank behavioral AXI responder; NOT MIG/RDIMM',
        clock_hz=200_000_000, mac_lanes=16, ops_per_mac=2,
        responder=dict(startup_cycles=args.startup, beat_gap_cycles=args.beat_gap),
        predictor_pattern=args.pattern, alignment_debug=args.alignment_debug, source_sha256=hashes, results=[])
    summary_path = output/'summary.json'
    summary_path.write_text(json.dumps(summary, indent=2)+'\n', encoding='utf-8')
    summary['compile_wall_seconds'] = command_run(['xvlog', '--sv', *map(str, sources)], build, build/'compile.log')
    snapshots = {}
    for whole in sorted({CASES[case][0] for case in args.cases}):
        name = f'full_switch_whole_{whole}'
        command = ['xelab', '--debug', 'off', '--mt', '4', '--snapshot', name]
        if args.incremental:
            command += ['--incr']
        for key, value in dict(geometry, WHOLE=whole).items():
            command += ['--generic_top', f'{key}={value}']
        command_run([*command, 'work.tb_switch_full_size'], build, build/f'{name}_elaborate.log')
        snapshots[whole] = name
    if args.build_only:
        summary['status']='BUILT_NOT_SIMULATED'
        summary_path.write_text(json.dumps(summary, indent=2)+'\n', encoding='utf-8')
        return
    # Independent runtime directories avoid concurrent xsim snapshot file writes.
    from run_switch_full_size_snapshots import run_case
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futures = [pool.submit(run_case, case, output, output, summary) for case in args.cases]
        for future in futures:
            try:
                summary['results'].append(future.result())
            except Exception as error:
                summary['status']='FAIL'
                summary['error']=str(error)
                summary_path.write_text(json.dumps(summary, indent=2)+'\n', encoding='utf-8')
                raise
            summary_path.write_text(json.dumps(summary, indent=2)+'\n', encoding='utf-8')
    for path, expected_hash in hashes.items():
        if hashlib.sha256((PROJECT/path).read_bytes()).hexdigest() != expected_hash:
            raise ValueError(f'source changed during run: {path}')
    by_case = {r['case']: r for r in summary['results']}
    rows = []
    for result in summary['results']:
        cycles = result['metrics']['cycles']
        rows.append(dict(case=result['case'], **result['metrics'],
            milliseconds=cycles/200_000,
            speedup_vs_A=by_case['A_whole_demand']['metrics']['cycles']/cycles if 'A_whole_demand' in by_case else None,
            speedup_vs_B=by_case['B_stream_demand']['metrics']['cycles']/cycles if 'B_stream_demand' in by_case else None))
    with (output/'comparison.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    summary['status']='PASS'
    summary_path.write_text(json.dumps(summary, indent=2)+'\n', encoding='utf-8')
    print(f'Completed {len(rows)} cases; full dimensions={not args.mini}', flush=True)


if __name__ == '__main__':
    main()
