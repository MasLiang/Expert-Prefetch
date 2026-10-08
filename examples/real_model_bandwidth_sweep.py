"""五模型共享历史激励的带宽单因素扫描；不修改模拟器或调度算法。"""
from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
import multiprocessing as mp
from pathlib import Path
from typing import Any

import yaml

from switch_wide_tile_prefetch import POLICIES as SWITCH_POLICIES, run_sample as run_switch
from switch_prefetch_comparison import run_sample as run_moe, write_csv
from olmoe_mixtral_cases import STREAMING, WHOLE
from retest_real_models import ROOT, read_csv
from edge_moe_sim.config import load_model_config
from edge_moe_sim.real_model_profile import load_real_moe_profiles


def validate_settings(settings: dict[str, Any]) -> None:
    """拒绝无效或重复带宽、非整数样本数和非法容量。"""
    for key in ('samples_per_model', 'ocm_bytes'):
        if type(settings[key]) is not int or settings[key] <= 0:
            raise ValueError(f'{key} must be a positive integer')
    bandwidths = settings['bytes_per_cycle_per_channel']
    if not bandwidths or any(type(b) is not int or b <= 0 for b in bandwidths):
        raise ValueError('bandwidths must be positive integer byte/cycle values')
    if bandwidths != sorted(set(bandwidths)):
        raise ValueError('bandwidths must be unique and increasing')


def worker(job: tuple[Any, ...]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """调用原驱动，硬件仅覆盖 bandwidth；统一输出身份、周期和 byte 指标。"""
    family, model, settings, base, common, bw, rate, policy, record = job
    if family == 'switch':
        raw, timing = run_switch((model, settings, base, common, bw, policy, record))
        hw = base['hardware']
    else:
        hw = {**settings['hardware'],
              'bandwidth_bytes_per_sec_per_channel': bw * settings['hardware']['clock_hz']}
        changed = {**settings, 'hardware': hw}
        timing = []
        raw = run_moe(model, changed, record, settings['ocm_bytes'], rate, policy, timing)
    identity = dict(model=model.name, family=family, ops_per_cycle=rate,
        bytes_per_cycle_per_channel=bw, ocm_bytes=settings['ocm_bytes'], policy=policy,
        sample=record['sample'])
    result = {**identity, 'clock_hz': hw['clock_hz'],
        'num_memory_channels': hw['num_memory_channels'],
        'nominal_total_bandwidth_gbps': bw * hw['clock_hz'] * hw['num_memory_channels'] / 1e9,
        'bandwidth_efficiency': hw['bandwidth_efficiency']}
    for key in ('total_cycles', 'memory_wait_cycles', 'peak_ocm_bytes', 'demand_bytes',
                'speculative_bytes', 'useful_speculative_bytes', 'wasted_speculative_bytes'):
        result[key] = raw[key]
    if result['speculative_bytes'] != result['useful_speculative_bytes'] + result['wasted_speculative_bytes']:
        raise AssertionError('speculative byte conservation')
    if result['peak_ocm_bytes'] > result['ocm_bytes']:
        raise AssertionError('OCM capacity exceeded')
    # 两个历史驱动的 prediction 字段不完全相同，嵌入 JSON 以避免 CSV 字段静默丢失。
    details = [{**identity, 'prediction_json': json.dumps(t, sort_keys=True)} for t in timing]
    return result, details


def aggregate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """按同输入 baseline 的总周期求速度比；不平均逐样本 speedup。"""
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        key = tuple(row[k] for k in ('model', 'ops_per_cycle', 'bytes_per_cycle_per_channel', 'policy'))
        groups.setdefault(key, []).append(row)
    totals = {k: sum(r['total_cycles'] for r in v) for k, v in groups.items()}
    output = []
    for (model, rate, bw, policy), values in sorted(groups.items()):
        row = {k: values[0][k] for k in ('model', 'family', 'ops_per_cycle', 'bytes_per_cycle_per_channel',
            'ocm_bytes', 'policy', 'clock_hz', 'num_memory_channels', 'nominal_total_bandwidth_gbps',
            'bandwidth_efficiency')}
        n = len(values)
        total = totals[(model, rate, bw, policy)]
        baseline = 'stream_demand' if row['family'] == 'switch' else 'no_prediction'
        whole = 'whole_demand' if row['family'] == 'switch' else 'demand_full_expert'
        target_ids = {v['sample'] for v in values}
        if len(target_ids) != n:
            raise ValueError('duplicate samples')
        for baseline_name in (baseline, whole):
            reference = groups.get((model, rate, bw, baseline_name))
            if reference is not None and {v['sample'] for v in reference} != target_ids:
                raise ValueError('unpaired baseline samples')
        row.update(samples=n, cycles_per_sequence=total/n,
            latency_ms=total/n/row['clock_hz']*1000,
            speedup_vs_stream_demand=totals[(model, rate, bw, baseline)]/total,
            speedup_vs_whole_demand=totals[(model, rate, bw, whole)]/total
                if (model, rate, bw, whole) in totals else None,
            saved_cycles_vs_stream_demand=(totals[(model, rate, bw, baseline)]-total)/n,
            peak_ocm_bytes=max(v['peak_ocm_bytes'] for v in values))
        low_bw = min(k[2] for k in totals if k[0] == model and k[1] == rate and k[3] == policy)
        row['absolute_speedup_vs_lowest_bandwidth_same_policy'] = totals[(model, rate, low_bw, policy)]/total
        for key in ('memory_wait_cycles', 'demand_bytes', 'speculative_bytes',
                    'useful_speculative_bytes', 'wasted_speculative_bytes'):
            row[key+'_per_sequence'] = sum(v[key] for v in values)/n
        output.append(row)
    return output


def crossover(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """报告离散测试点阈值和最优带宽，不假设单调性或插值连续阈值。"""
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        if row['policy'] in {'stream_demand', 'no_prediction', 'whole_demand', 'demand_full_expert', 'demand_tile_serial'}:
            continue
        groups.setdefault((row['model'], row['ops_per_cycle'], row['policy']), []).append(row)
    result = []
    for (model, rate, policy), points in sorted(groups.items()):
        points.sort(key=lambda p: p['bytes_per_cycle_per_channel'])
        best = max(points, key=lambda p: p['speedup_vs_stream_demand'])
        result.append(dict(model=model, ops_per_cycle=rate, policy=policy,
            first_tested_bw_positive=next((p['bytes_per_cycle_per_channel'] for p in points
                if p['speedup_vs_stream_demand'] > 1), None),
            first_tested_bw_above_1percent=next((p['bytes_per_cycle_per_channel'] for p in points
                if p['speedup_vs_stream_demand'] > 1.01), None),
            best_tested_bw=best['bytes_per_cycle_per_channel'],
            best_speedup_vs_stream_demand=best['speedup_vs_stream_demand'],
            highest_bw_speedup_vs_stream_demand=points[-1]['speedup_vs_stream_demand']))
    return result


def plot(output: Path, rows: list[dict[str, Any]]) -> None:
    """保存绝对延迟、额外收益、aggregate starvation 三组科学曲线 PNG/PDF。"""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    panels = sorted({(r['model'], r['ops_per_cycle']) for r in rows})
    metrics = (('latency_ms', 'Six-layer latency (ms)'),
        ('speedup_vs_stream_demand', 'Speedup vs streaming demand at SAME bandwidth'),
        ('memory_wait_cycles_per_sequence', 'Aggregate expert starvation cycles (not global idle)'))
    for metric, label in metrics:
        fig, axes = plt.subplots(4, 2, figsize=(14, 18), squeeze=False)
        for ax, (model, rate) in zip(axes.flat, panels):
            policies = sorted({r['policy'] for r in rows if r['model'] == model and r['ops_per_cycle'] == rate})
            for policy in policies:
                points = sorted((r for r in rows if r['model'] == model and r['ops_per_cycle'] == rate
                    and r['policy'] == policy), key=lambda r: r['bytes_per_cycle_per_channel'])
                ax.plot([p['nominal_total_bandwidth_gbps'] for p in points], [p[metric] for p in points],
                    marker='o', markersize=3, label=policy)
            ax.set_xscale('log', base=2)
            ax.set_xlabel('Configured nominal aggregate bandwidth (GB/s); NOT board measured')
            ax.set_ylabel(label)
            ax.set_title(f'{model}; {rate} ops/cycle; OCM=16 MiB')
            ax.grid(alpha=.2)
            if metric == 'speedup_vs_stream_demand':
                ax.axhline(1, color='gray', linestyle='--')
            ax.legend(fontsize=7)
        fig.suptitle('Bandwidth-only sweep / fixed original compute and DMA backend / synthetic saved trace')
        fig.tight_layout()
        for extension in ('png', 'pdf'):
            fig.savefig(output/f'{metric}.{extension}')
        plt.close(fig)


def make_jobs(settings: dict[str, Any], output: Path) -> tuple[list[tuple[Any, ...]], list[dict[str, Any]], dict[str, Any]]:
    """读取已有 JSON trace 的固定前缀，保存完整配置快照；不重新抽样。"""
    root = ROOT/settings['reference_root']
    switch = yaml.safe_load((ROOT/settings['switch_config']).read_text())
    moe = yaml.safe_load((ROOT/settings['moe_config']).read_text())
    base_switch = yaml.safe_load((ROOT/switch['base_config']).read_text())
    common = yaml.safe_load((ROOT/switch['common_config']).read_text())
    base_moe = yaml.safe_load((ROOT/moe['base_config']).read_text())
    profiles = load_real_moe_profiles(ROOT/moe['catalogue'], weight_bits=moe['weight_bits'],
        activation_bits=moe['activation_bits'])
    jobs, declarations = [], []
    for family, models in (('switch', [load_model_config(ROOT/p) for p in switch['models']]),
                           ('moe', [replace(p.model, moe_layer_indices=tuple(range(moe['simulated_layers'])))
                                    for p in profiles if p.model.name in moe['models']])):
        for model in models:
            suite = 'switch_wide' if family == 'switch' else 'olmoe_mixtral'
            path = root/suite/f'{model.name}_inputs.json'
            saved = json.loads(path.read_text())
            records = saved[:settings['samples_per_model']]
            if len(records) != settings['samples_per_model']:
                raise ValueError(f'insufficient saved samples for {model.name}')
            (output/f'{model.name}_inputs.json').write_text(json.dumps(records, indent=2))
            observed_hits = sum((p['true_rank'] < 2) if family == 'switch' else sum(r >= 0 for r in p['true_ranks'])
                for rec in records for p in rec['predictions'])
            decisions = len(records)*(len(model.moe_layer_indices)-1)*model.top_k
            declarations.append(dict(model=model.name, family=family, samples=len(records),
                source_input=str(path.relative_to(ROOT)), source_input_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                exact_saved_prefix=True, observed_top2_or_candidate_recall=observed_hits/decisions,
                hidden_size=model.hidden_size, intermediate_size=model.intermediate_size,
                experts=model.num_experts, top_k=model.top_k, simulated_layers=len(model.moe_layer_indices)))
            cfg = {**switch, 'ocm_bytes': settings['ocm_bytes']} if family == 'switch' else {
                **base_moe, 'ocm_bytes': settings['ocm_bytes'],
                'candidate_count': min(model.num_experts, model.top_k*moe['candidate_multiplier'])}
            rates = [switch['ops_per_cycle']] if family == 'switch' else base_moe['ops_per_cycle']
            policies = SWITCH_POLICIES if family == 'switch' else STREAMING + (WHOLE if model.name.startswith('olmoe') else ())
            for rate in rates:
                for bw in settings['bytes_per_cycle_per_channel']:
                    for policy in policies:
                        for record in records:
                            jobs.append((family, model, cfg, base_switch, common, bw, rate, policy, record))
    return jobs, declarations, dict(switch=switch, switch_base=base_switch, switch_common=common,
        moe=moe, moe_base=base_moe)


def check_reference(rows: list[dict[str, Any]], root: Path) -> int:
    """重叠带宽点逐指标与前次复测一致，防止新增驱动改变执行规则。"""
    references = {}
    for suite in ('switch_wide', 'olmoe_mixtral'):
        for row in read_csv(root/suite/'per_sample.csv'):
            bw = row.get('bw_bytes_per_cycle_per_channel', '16')
            rate = row.get('ops_per_cycle', '32')
            capacity = row.get('ocm_bytes', '16777216')
            key = (row['model'], int(rate), int(bw), int(capacity), row['policy'], int(row['sample']))
            references[key] = row
    matches = 0
    for row in rows:
        key = tuple(row[k] for k in ('model', 'ops_per_cycle', 'bytes_per_cycle_per_channel', 'ocm_bytes', 'policy', 'sample'))
        previous = references.get(key)
        if previous is not None:
            for metric in ('total_cycles', 'memory_wait_cycles', 'peak_ocm_bytes', 'demand_bytes',
                           'speculative_bytes', 'useful_speculative_bytes', 'wasted_speculative_bytes'):
                if row[metric] != int(previous[metric]):
                    raise AssertionError(f'reference changed: {key}: {metric}')
            matches += 1
    if not matches:
        raise ValueError('no overlap with reference bandwidths')
    return matches


def main() -> None:
    """运行完整扫描，定期 checkpoint；保存显式运行/失败/完成状态和中文报告。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT/'configs/experiments/real_model_bandwidth_sweep.yaml')
    parser.add_argument('--output', type=Path, default=ROOT/'results/real_model_bandwidth_sweep_20261007')
    parser.add_argument('--workers', type=int, default=12)
    parser.add_argument('--samples', type=int, help='覆盖每模型样本数；必须不超过原始 trace')
    args = parser.parse_args()
    settings = yaml.safe_load(args.config.read_text())
    if args.samples is not None:
        settings['samples_per_model'] = args.samples
    validate_settings(settings)
    if args.workers < 1:
        parser.error('workers must be positive')
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output/'summary.json').exists():
        raise ValueError('output already has summary; select a new output directory')
    status_path = args.output/'summary.json'
    status_path.write_text(json.dumps(dict(status='RUNNING', settings=settings), indent=2))
    try:
        jobs, models, snapshots = make_jobs(settings, args.output)
        expected = len(jobs)
        raw, timing = [], []
        sources = [args.config.resolve(), Path(__file__), ROOT/settings['switch_config'], ROOT/settings['moe_config'],
            *sorted((ROOT/'src/edge_moe_sim').rglob('*.py')),
            *[ROOT/'examples'/p for p in ('switch_wide_tile_prefetch.py', 'switch_prefetch_comparison.py',
                'olmoe_mixtral_cases.py', 'mac_four_cases.py')]]
        hashes = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
        with mp.get_context('spawn').Pool(args.workers) as pool:
            for row, details in pool.imap(worker, jobs, chunksize=1):
                raw.append(row)
                timing.extend(details)
                if len(raw) % settings['samples_per_model'] == 0:
                    write_csv(args.output/'per_sample.csv', raw)
                    status_path.write_text(json.dumps(dict(status='RUNNING', simulations=len(raw), expected_simulations=expected), indent=2))
                    print(f"completed {len(raw)}/{expected}: {row['model']} ops={row['ops_per_cycle']} BW={row['bytes_per_cycle_per_channel']} {row['policy']}", flush=True)
        overlap = check_reference(raw, ROOT/settings['reference_root'])
        rows = aggregate(raw)
        thresholds = crossover(rows)
        write_csv(args.output/'comparison.csv', rows)
        write_csv(args.output/'crossover.csv', thresholds)
        write_csv(args.output/'per_prediction.csv', timing)
        plot(args.output, rows)
        for path, digest in hashes.items():
            if hashlib.sha256((ROOT/path).read_bytes()).hexdigest() != digest:
                raise ValueError(f'source changed during sweep: {path}')
        summary = dict(status='COMPLETE', simulator='UnifiedSequenceSimulator', settings=settings,
            simulation_count=len(raw), configuration_count=len(rows), reference_matched_samples=overlap,
            models=models, original_config_snapshots=snapshots, source_sha256=hashes,
            limits=['仅带宽变化；四通道/原 DMA backend 固定，不是 U280/V80 实际接口或测量',
                'HBM 级聚合带宽是抽象敏感性参数，不代表现有四通道 RTL 能提供该带宽',
                'Switch physical payload multiplier=2；各 backend 的 byte 与有效带宽需分开解读',
                '16 MiB 实际 byte OCM，不把 HBM 容量当 OCM；Mixtral whole-residency 不可行',
                '原始 trace 固定前缀，样本覆盖率单独报告，不声称子集仍精确等于 84.4%',
                '六 MoE 层/单 token/INT8 假设/人工 predictor，不是完整模型推理',
                'memory_wait 是 aggregate expert starvation，不作为全局 idle 或可消除周期直接相减',
                '阈值为离散测试点，没有连续插值或统计显著性保证'])
        status_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False)+'\n')
        lines = ['# 五模型带宽单因素扫描', '',
            f'完成 {len(raw):,} 次模拟、{len(rows)} 个配置点；{overlap} 个重叠样本全部指标与前次复测一致。', '',
            '## 最优离散测试点（相对同带宽 streaming 无预取）', '',
            '| 模型 | ops/cycle | 策略 | 最优每通道 byte/cycle | 最大 speedup | 最高带宽 speedup |',
            '|---|---:|---|---:|---:|---:|']
        for row in thresholds:
            lines.append(f"| {row['model']} | {row['ops_per_cycle']} | {row['policy']} | {row['best_tested_bw']} | {row['best_speedup_vs_stream_demand']:.6f}× | {row['highest_bw_speedup_vs_stream_demand']:.6f}× |")
        lines += ['', '## 限制', '', *['- '+p for p in summary['limits']], '',
            '## 复现', '', '```bash', 'PYTHONPATH=src python examples/real_model_bandwidth_sweep.py --workers 12', '```', '',
            '原始输入、每样本/每预测/聚合 CSV、crossover CSV、配置快照和源 SHA-256 均在本目录。',
            'PNG/PDF 包含绝对延迟、同带宽预取收益和 aggregate starvation 三组曲线。', '']
        (args.output/'REPORT.md').write_text('\n'.join(lines), encoding='utf-8')
        print(f'COMPLETE: {len(raw)} simulations; {len(rows)} configurations; reference matches={overlap}', flush=True)
    except Exception as error:
        status_path.write_text(json.dumps(dict(status='FAILED', error=str(error)), indent=2))
        raise


if __name__ == '__main__':
    main()
