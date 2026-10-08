"""用当前主模拟器完整重跑五个已测模型，并验证共同输入与旧结果逐点比较。"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SUITES = (
    ('switch_four_cases', 'switch_real_model_four_cases', 'flame_condition_comparison.py',
     'configs/experiments/switch_real_model_four_cases.yaml', 2400,
     ('model', 'mac_groups', 'bytes_per_cycle_per_channel', 'ocm_bytes', 'case', 'barrier', 'seed', 'sample')),
    ('switch_wide', 'switch_wide_tile_prefetch', 'switch_wide_tile_prefetch.py',
     'configs/experiments/switch_wide_tile_prefetch.yaml', 2400,
     ('model', 'bw_bytes_per_cycle_per_channel', 'policy', 'sample')),
    ('olmoe_mixtral', 'olmoe_mixtral_cases', 'olmoe_mixtral_cases.py',
     'configs/experiments/olmoe_mixtral_cases.yaml', 4500,
     ('model', 'ocm_bytes', 'ops_per_cycle', 'policy', 'sample')),
)


def read_csv(path: Path) -> list[dict[str, str]]:
    """完整读取包含列名的 CSV，缺失文件不静默忽略。"""
    with path.open(newline='', encoding='utf-8') as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """保存确定顺序的非空 CSV。"""
    if not rows:
        raise ValueError(f'empty results: {path}')
    with path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def paired_deltas(old: list[dict[str, str]], new: list[dict[str, str]],
                  keys: tuple[str, ...], suite: str) -> list[dict[str, Any]]:
    """按完整实验身份配对，不以不匹配的 baseline 补值。"""
    def index(rows: list[dict[str, str]]) -> dict[tuple[str, ...], dict[str, str]]:
        result = {tuple(row[k] for k in keys): row for row in rows}
        if len(result) != len(rows):
            raise ValueError(f'duplicate sample keys in {suite}')
        return result
    left, right = index(old), index(new)
    if left.keys() != right.keys():
        raise ValueError(f'experiment identities changed: {suite}')
    result = []
    for key, row in right.items():
        previous = left[key]
        a, b = int(previous['total_cycles']), int(row['total_cycles'])
        result.append(dict(suite=suite, model=row['model'], identity=json.dumps(dict(zip(keys, key)), sort_keys=True),
            old_cycles=a, current_cycles=b, delta_cycles=b-a, delta_percent=100*(b/a-1),
            old_speculative_bytes=int(previous['speculative_bytes']),
            current_speculative_bytes=int(row['speculative_bytes']),
            delta_speculative_bytes=int(row['speculative_bytes'])-int(previous['speculative_bytes'])))
    return result


def run_suite(suite: tuple[Any, ...], output: Path, workers: int) -> None:
    """执行原实验驱动，不复用旧 checkpoint，也不改变配置参数。"""
    name, _, script, config, _, _ = suite
    env = dict(os.environ)
    env['PYTHONPATH'] = str(ROOT/'src')+os.pathsep+env.get('PYTHONPATH', '')
    with (output/f'{name}.log').open('w', encoding='utf-8') as stream:
        subprocess.run([sys.executable, '-u', str(ROOT/'examples'/script), '--config', str(ROOT/config),
                        '--workers', str(workers), '--output', str(output/name)],
                       cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT, check=True)


def collect(output: Path) -> dict[str, Any]:
    """要求全部完整、源 hashes 一致、输入完全相同，并生成统一对照报告。"""
    deltas, headlines, summaries = [], [], []
    for suite in SUITES:
        name, historical, _, config, expected_count, keys = suite
        directory, old_dir = output/name, ROOT/'results'/historical
        summary = json.loads((directory/'summary.json').read_text())
        old_summary = json.loads((old_dir/'summary.json').read_text())
        if summary['status'] != 'complete' or old_summary['status'] != 'complete':
            raise ValueError(f'incomplete suite: {name}')
        if summary['simulator'] != 'UnifiedSequenceSimulator':
            raise ValueError('deprecated/non-main simulator result')
        configuration_fields = {
            'switch_four_cases': ('geometry', 'base'),
            'switch_wide': ('settings',),
            'olmoe_mixtral': ('experiment', 'base_settings'),
        }[name]
        for field in configuration_fields:
            if summary[field] != old_summary[field]:
                raise ValueError(f'hardware/model/experiment configuration changed: {name}: {field}')
        for path, digest in summary['source_sha256'].items():
            if hashlib.sha256((ROOT/path).read_bytes()).hexdigest() != digest:
                raise ValueError(f'source changed during suite {name}: {path}')
        inputs = sorted(directory.glob('*_inputs.json'))
        if not inputs or {p.name for p in inputs} != {p.name for p in old_dir.glob('*_inputs.json')}:
            raise ValueError(f'missing or changed trace files: {name}')
        for path in inputs:
            if json.loads(path.read_text()) != json.loads((old_dir/path.name).read_text()):
                raise ValueError(f'input truth/candidates/ranking changed: {path}')
        before, current = read_csv(old_dir/'per_sample.csv'), read_csv(directory/'per_sample.csv')
        if len(current) != expected_count:
            raise ValueError(f'wrong simulation count: {name}: {len(current)}')
        points = paired_deltas(before, current, keys, name)
        deltas.extend(points)
        comparisons = read_csv(directory/'comparison.csv')
        for row in comparisons:
            bw = row.get('bytes_per_cycle_per_channel', row.get('bw_bytes_per_cycle_per_channel', '16'))
            rate = row.get('ops_per_cycle', '32')
            capacity = row.get('ocm_bytes', '16777216')
            barrier = row.get('barrier', 'False')
            if bw != '8' and name != 'olmoe_mixtral':
                continue
            if capacity != '16777216' or barrier != 'False':
                continue
            cycles = float(row['cycles_per_sequence']) if 'cycles_per_sequence' in row else int(row['total_cycles'])/int(row['samples'])
            headlines.append(dict(suite=name, model=row['model'], policy=row.get('case', row.get('policy')),
                nominal_bytes_per_cycle_per_channel=bw, ops_per_cycle=rate, ocm_bytes=capacity,
                cycles_per_sequence=cycles,
                speedup_vs_whole_demand=row.get('speedup_vs_A', row.get('speedup_vs_whole_demand', '')),
                speedup_vs_stream_demand=row.get('speedup_vs_B', row.get('speedup_vs_stream_demand', row.get('speedup_vs_streaming', '')))))
        figures = sorted(p.name for p in directory.glob('*.png'))
        if not figures or any(not (directory/Path(p).with_suffix('.pdf')).is_file() for p in figures):
            raise ValueError(f'missing PNG/PDF figures: {name}')
        summaries.append(dict(suite=name, config=config, simulations=len(current), configurations=len(comparisons),
            exact_input_match=True, exact_configuration_match=True,
            changed_samples=sum(v['delta_cycles'] != 0 for v in points),
            changed_speculative_byte_samples=sum(v['delta_speculative_bytes'] != 0 for v in points),
            maximum_absolute_delta_cycles=max(abs(v['delta_cycles']) for v in points),
            maximum_absolute_delta_percent=max(abs(v['delta_percent']) for v in points), figures=figures))
    write_csv(output/'sample_deltas.csv', deltas)
    write_csv(output/'headline_comparison.csv', headlines)
    result = dict(status='COMPLETE', simulator='UnifiedSequenceSimulator', simulation_count=len(deltas),
        configuration_count=sum(s['configurations'] for s in summaries), suites=summaries,
        headline_rows=headlines,
        limits=['原矩阵的硬件与输入保持不变，不把它们改写为六层 RTL 全命中夹具',
            'byte OCM/expert_locked；Switch burst rules 与 OLMoE/Mixtral matrix-fragment DMA 分开解读',
            '没有强制启用 fixed-slot/pair RTL 规则；它们是硬件 profile 选择，不是统一替代原研究模型',
            '官方结构+人工 trace+INT8 部署假设，非完整 Transformer 推理或板级测量',
            'Mixtral whole-residency case 不可行，仅运行 streaming；all-tile target 不等于整 Expert 同时驻留',
            '没有重跑所有历史 DSE/质量/PE 扫描，只重跑最近三套五模型对照矩阵'])
    expected_models = {'switch_base_8_decoder', 'switch_base_16_decoder', 'olmoe_1b_7b', 'mixtral_8x7b', 'mixtral_8x22b'}
    if {row['model'] for row in deltas} != expected_models:
        raise ValueError('missing or unexpected models')
    result['driver_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    (output/'summary.json').write_text(json.dumps(result, indent=2, ensure_ascii=False)+'\n')
    lines = ['# 当前主 Simulator 五模型复测', '',
        f'实际完成 {len(deltas):,} 次模拟、{result["configuration_count"]} 个配置点。原始激励完全相同，旧结果没有覆盖。', '',
        '| 实验 | 模拟次数 | cycles 改变样本 | 最大周期差 | 最大相对差 |', '|---|---:|---:|---:|---:|']
    for suite in summaries:
        lines.append(f'| {suite["suite"]} | {suite["simulations"]} | {suite["changed_samples"]} | {suite["maximum_absolute_delta_cycles"]} | {suite["maximum_absolute_delta_percent"]:.6f}% |')
    lines += ['', '## 代表点：16 MiB OCM', '',
        'Switch 为 16-lane/32 ops/cycle、每通道名义 8 byte/cycle，payload multiplier=2；',
        'OLMoE/Mixtral 为每通道名义 16 byte/cycle，32/512 ops/cycle。不同 backend 配置不能直接横比。', '',
        '| 实验 | 模型 | 策略 | ops/cycle | 平均 cycles | 相对整 Expert 无预取 | 相对 streaming 无预取 |',
        '|---|---|---|---:|---:|---:|---:|']
    for row in headlines:
        a = f'{float(row["speedup_vs_whole_demand"]):.6f}×' if row['speedup_vs_whole_demand'] else '不可行/无分母'
        b = f'{float(row["speedup_vs_stream_demand"]):.6f}×'
        lines.append(f'| {row["suite"]} | {row["model"]} | {row["policy"]} | {row["ops_per_cycle"]} | {row["cycles_per_sequence"]:.2f} | {a} | {b} |')
    lines += ['', '## 配对验证', '',
        '全部样本按模型、硬件参数、策略与样本身份配对；保存的 truth、候选集合、排序和分数逐字段一致。',
        '每套结果的源文件 SHA-256 已与当前工作区核对。周期差定义为 current − old，负值表示节省；',
        '投机传输 byte 差单独列出，不与周期差混用。图中的 speedup 分母保持原实验定义。', '',
        '## 限制', '', *['- '+v for v in result['limits']], '', '## 复现', '',
        '```bash', 'PYTHONPATH=src python examples/retest_real_models.py --workers 4 --parallel-experiments 3', '```', '',
        '完整逐样本差值见 `sample_deltas.csv`，每套实验目录包含输入、CSV、summary 与 PNG/PDF。', '']
    (output/'REPORT.md').write_text('\n'.join(lines), encoding='utf-8')
    return result


def main() -> None:
    """一条命令复跑与汇总；collect-only 仍严格验证源代码和所有结果。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT/'results/current_model_retest_20261007')
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--parallel-experiments', type=int, default=3)
    parser.add_argument('--collect-only', action='store_true')
    args = parser.parse_args()
    if args.workers < 1 or not 1 <= args.parallel_experiments <= len(SUITES):
        parser.error('workers must be positive; parallel-experiments must be 1..3')
    args.output.mkdir(parents=True, exist_ok=True)
    if not args.collect_only:
        (args.output/'summary.json').write_text(json.dumps(dict(status='RUNNING', expected_simulations=9300))+'\n')
        with ThreadPoolExecutor(max_workers=args.parallel_experiments) as pool:
            futures = [pool.submit(run_suite, suite, args.output.resolve(), args.workers) for suite in SUITES]
            for future in futures:
                future.result()
    result = collect(args.output)
    print(json.dumps({k:v for k,v in result.items() if k in ('status','simulation_count','configuration_count','suites')}, indent=2))


if __name__ == '__main__':
    main()
