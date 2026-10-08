"""固定 U200-like 内存参数，扫描 DSP MAC lanes；不修改 RTL 或主模拟器语义。"""
from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
from pathlib import Path
from typing import Any

import yaml

import real_model_bandwidth_sweep as bandwidth
from retest_real_models import ROOT, read_csv
from switch_prefetch_comparison import write_csv
from edge_moe_sim.work_model import ceil_div, down_ops, gate_up_ops, router_ops


def dsp_estimate(lanes: int, settings: dict[str, Any]) -> dict[str, Any]:
    """返回研究预算估计，不将一 lane 一 DSP 假设冒充综合结果。"""
    mac_dsp = lanes*settings['dsp_per_mac_lane']
    total = mac_dsp+settings['controller_dsp_reference']
    budget = int(settings['device_dsp_count']*settings['dsp_budget_fraction'])
    return dict(mac_lanes=lanes, ops_per_cycle=lanes*settings['operations_per_mac'],
        estimated_mac_dsp=mac_dsp, controller_dsp_reference=settings['controller_dsp_reference'],
        estimated_total_dsp=total, device_dsp_count=settings['device_dsp_count'],
        estimated_device_dsp_percent=100*total/settings['device_dsp_count'],
        study_dsp_budget=budget, fits_study_dsp_budget=total <= budget)


def validate_settings(settings: dict[str, Any]) -> None:
    """只允许整数 lane/operation/DSP 配置；拒绝超预算或重复档位。"""
    bandwidth.validate_settings(settings)
    if len(settings['bytes_per_cycle_per_channel']) != 1:
        raise ValueError('DSP-only sweep requires exactly one fixed bandwidth')
    for key in ('operations_per_mac', 'dsp_per_mac_lane', 'device_dsp_count'):
        if type(settings[key]) is not int or settings[key] <= 0:
            raise ValueError(f'{key} must be a positive integer')
    if settings['operations_per_mac'] != 2:
        raise ValueError('INT8 MAC counts multiply and add as two operations')
    if type(settings['controller_dsp_reference']) is not int or settings['controller_dsp_reference'] < 0:
        raise ValueError('controller DSP reference must be nonnegative integer')
    fraction = settings['dsp_budget_fraction']
    if type(fraction) not in (int, float) or not 0 < fraction <= 1:
        raise ValueError('DSP budget fraction must be in (0,1]')
    lanes = settings['mac_lanes']
    if not lanes or any(type(n) is not int or n <= 0 or n % 16 for n in lanes):
        raise ValueError('lanes must be positive integer multiples of 16')
    if lanes != sorted(set(lanes)) or lanes[0] != 16:
        raise ValueError('lanes must be increasing, unique, and include 16-lane reference')
    if any(not dsp_estimate(n, settings)['fits_study_dsp_budget'] for n in lanes):
        raise ValueError('MAC configuration exceeds study DSP budget')


def scale_job(job: tuple[Any, ...], lanes: int, settings: dict[str, Any]) -> tuple[Any, ...]:
    """同时修改标签与实际性能参数，避免 Switch 被标为宽 MAC 却仍以 32 ops/cycle 运行。"""
    family, model, config, base, common, bw, _, policy, record = job
    rate = lanes*settings['operations_per_mac']
    changed = {**config, 'ops_per_cycle': rate}
    return family, model, changed, base, common, bw, rate, policy, record


def worker(task: tuple[Any, ...]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """调用既有真实事件队列，计算纯工作量占比下界而非伪造硬件利用率。"""
    job, lanes, settings = task
    scaled = scale_job(job, lanes, settings)
    row, details = bandwidth.worker(scaled)
    model, record = scaled[1], scaled[-1]
    payload_ops = len(record['truth'])*router_ops(model, 1)+sum(len(t) for t in record['truth'])*(
        gate_up_ops(model, 1)+down_ops(model, 1))
    ideal_cycles = ceil_div(payload_ops, row['ops_per_cycle'])
    if ideal_cycles > row['total_cycles']:
        raise AssertionError('arithmetic lower bound exceeds simulated makespan')
    row.update(dsp_estimate(lanes, settings))
    row.update(payload_ops=payload_ops, ideal_payload_compute_lower_bound_cycles=ideal_cycles,
        payload_fraction_lower_bound=ideal_cycles/row['total_cycles'])
    for detail in details:
        detail['mac_lanes'] = lanes
    return row, details


def aggregate(raw: list[dict[str, Any]], settings: dict[str, Any]) -> list[dict[str, Any]]:
    """分别计算相对同 lanes baseline 与相对 16 lanes 的真实总周期 speedup。"""
    rows = bandwidth.aggregate(raw)
    totals: dict[tuple[Any, ...], int] = {}
    ops: dict[tuple[Any, ...], int] = {}
    bounds: dict[tuple[Any, ...], int] = {}
    sample_ids: dict[tuple[Any, ...], set[int]] = {}
    for r in raw:
        key = r['model'], r['ops_per_cycle'], r['bytes_per_cycle_per_channel'], r['policy']
        totals[key] = totals.get(key, 0)+r['total_cycles']
        ops[key] = ops.get(key, 0)+r['payload_ops']
        bounds[key] = bounds.get(key, 0)+r['ideal_payload_compute_lower_bound_cycles']
        sample_ids.setdefault(key, set()).add(r['sample'])
    for r in rows:
        # 仅一个 BW，不把恒为 1 的带宽比混作算力扩宽 speedup。
        r.pop('absolute_speedup_vs_lowest_bandwidth_same_policy')
        lanes = r['ops_per_cycle']//settings['operations_per_mac']
        r.update(dsp_estimate(lanes, settings))
        key = r['model'], r['ops_per_cycle'], r['bytes_per_cycle_per_channel'], r['policy']
        reference = key[0], 16*settings['operations_per_mac'], key[2], key[3]
        if sample_ids[key] != sample_ids[reference]:
            raise ValueError('16-lane baseline uses different samples')
        previous = max(n for n in settings['mac_lanes'] if n <= lanes and (n < lanes or lanes == 16))
        previous_key = key[0], previous*settings['operations_per_mac'], key[2], key[3]
        r.update(speedup_vs_16_lanes_same_policy=totals[reference]/totals[key],
            speedup_vs_previous_lane_point=totals[previous_key]/totals[key],
            ideal_payload_compute_lower_bound_cycles_per_sequence=bounds[key]/r['samples'],
            payload_ops_per_sequence=ops[key]/r['samples'], payload_fraction_lower_bound=bounds[key]/totals[key])
    return rows


def knees(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """找已测最短延迟与 1% 邻域；不把非单调曲线当连续或普适拐点。"""
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault((row['model'], row['policy']), []).append(row)
    result = []
    for (model, policy), points in sorted(groups.items()):
        points.sort(key=lambda r:r['mac_lanes'])
        fastest = min(points, key=lambda r:r['cycles_per_sequence'])
        within = next(p for p in points if p['cycles_per_sequence'] <= fastest['cycles_per_sequence']*1.01)
        prefetch_best = max(points, key=lambda r:r['speedup_vs_stream_demand'])
        result.append(dict(model=model, policy=policy, fastest_tested_lanes=fastest['mac_lanes'],
            minimum_cycles_per_sequence=fastest['cycles_per_sequence'],
            smallest_tested_lanes_within_1percent_fastest=within['mac_lanes'],
            largest_lanes_speedup_vs_16=points[-1]['speedup_vs_16_lanes_same_policy'],
            largest_lanes_payload_fraction_lower_bound=points[-1]['payload_fraction_lower_bound'],
            best_prefetch_speedup_vs_stream_demand=prefetch_best['speedup_vs_stream_demand'],
            lanes_at_best_prefetch_speedup=prefetch_best['mac_lanes']))
    return result


def check_reference(raw: list[dict[str, Any]], reference: Path) -> int:
    """与前次带宽扫描的重叠点逐指标配对，未重跑的数据不计入本次模拟。"""
    keys = ('model','ops_per_cycle','bytes_per_cycle_per_channel','policy','sample')
    previous = {tuple(row[k] for k in keys):row for row in read_csv(reference/'per_sample.csv')}
    matches = 0
    for row in raw:
        saved = previous.get(tuple(str(row[k]) for k in keys))
        if saved is not None:
            if row['ocm_bytes'] != int(saved['ocm_bytes']):
                raise ValueError('reference OCM differs')
            for metric in ('total_cycles','memory_wait_cycles','peak_ocm_bytes','demand_bytes',
                           'speculative_bytes','useful_speculative_bytes','wasted_speculative_bytes'):
                if row[metric] != int(saved[metric]):
                    raise AssertionError(f'reference mismatch: {row["model"]}: {metric}')
            matches += 1
    if not matches:
        raise ValueError('no matched reference samples')
    return matches


def plot(output: Path, rows: list[dict[str, Any]]) -> None:
    """绘制延迟、扩宽收益、预取收益与理想计算占比下界四组 PNG/PDF。"""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    metrics = (('latency_ms','Six-layer latency (ms)'),
        ('speedup_vs_16_lanes_same_policy','Speedup vs 16 lanes, SAME policy'),
        ('speedup_vs_stream_demand','Prefetch speedup vs streaming at SAME lanes'),
        ('payload_fraction_lower_bound','Arithmetic-work fraction LOWER BOUND, NOT measured utilization'))
    models = sorted({r['model'] for r in rows})
    lanes = sorted({r['mac_lanes'] for r in rows})
    for metric, label in metrics:
        fig, axes = plt.subplots(len(models), 1, figsize=(12, 4*len(models)), squeeze=False)
        for ax, model in zip(axes.flat, models):
            for policy in sorted({r['policy'] for r in rows if r['model']==model}):
                points = sorted((r for r in rows if r['model']==model and r['policy']==policy),key=lambda r:r['mac_lanes'])
                ax.plot([p['mac_lanes'] for p in points],[p[metric] for p in points], marker='o',label=policy)
            ax.set_xscale('log',base=2)
            ax.set_xticks(lanes,[str(n) for n in lanes])
            ax.set_xlabel('Hypothetical DSP MAC lanes; widened OCM feed assumed')
            ax.set_ylabel(label)
            ax.set_title(f'{model}; nominal 12.8 GB/s; OCM=16 MiB; configured, NOT measured')
            ax.grid(alpha=.2)
            ax.legend(fontsize=7)
            if metric.startswith('speedup'):
                ax.axhline(1,color='gray',linestyle='--')
        fig.tight_layout()
        for ext in ('png','pdf'):
            fig.savefig(output/f'{metric}.{ext}')
        plt.close(fig)


def main() -> None:
    """运行共享激励的五模型 DSP 扫描，保存配置、资源预算、checkpoint 和完整报告。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=ROOT/'configs/experiments/u200_dsp_lane_sweep.yaml')
    parser.add_argument('--output',type=Path,default=ROOT/'results/u200_dsp_lane_sweep_20261007')
    parser.add_argument('--workers',type=int,default=12)
    parser.add_argument('--samples',type=int)
    args = parser.parse_args()
    settings = yaml.safe_load(args.config.read_text())
    if args.samples is not None:
        settings['samples_per_model'] = args.samples
    validate_settings(settings)
    if args.workers < 1:
        parser.error('workers must be positive')
    args.output.mkdir(parents=True,exist_ok=True)
    status = args.output/'summary.json'
    if status.exists():
        raise ValueError('existing output; choose a new --output directory')
    status.write_text(json.dumps(dict(status='RUNNING',settings=settings),indent=2))
    try:
        prototypes, models, snapshots = bandwidth.make_jobs(settings,args.output)
        prototypes = [job for job in prototypes if job[6]==32]
        tasks = [(job,lanes,settings) for lanes in settings['mac_lanes'] for job in prototypes]
        budgets = [dsp_estimate(n,settings) for n in settings['mac_lanes']]
        write_csv(args.output/'dsp_budget.csv',budgets)
        sources = [args.config.resolve(),Path(__file__),ROOT/settings['switch_config'],ROOT/settings['moe_config'],
            ROOT/snapshots['switch']['base_config'],ROOT/snapshots['switch']['common_config'],
            ROOT/snapshots['moe']['catalogue'], *[ROOT/p for p in snapshots['switch']['models']],
            *[ROOT/'examples'/name for name in ('real_model_bandwidth_sweep.py','switch_wide_tile_prefetch.py',
                'switch_prefetch_comparison.py','olmoe_mixtral_cases.py','mac_four_cases.py')],
            *sorted((ROOT/'src/edge_moe_sim').rglob('*.py'))]
        hashes = {str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
        raw, timing = [], []
        with mp.get_context('spawn').Pool(args.workers) as pool:
            for row, details in pool.imap(worker,tasks,chunksize=1):
                raw.append(row)
                timing.extend(details)
                if len(raw)%settings['samples_per_model']==0:
                    write_csv(args.output/'per_sample.csv',raw)
                    status.write_text(json.dumps(dict(status='RUNNING',simulations=len(raw),expected_simulations=len(tasks))))
                    print(f"completed {len(raw)}/{len(tasks)}: {row['mac_lanes']} lanes {row['model']} {row['policy']}",flush=True)
        matched = check_reference(raw,ROOT/settings['bandwidth_reference'])
        rows = aggregate(raw,settings)
        points = knees(rows)
        write_csv(args.output/'comparison.csv',rows)
        write_csv(args.output/'knees.csv',points)
        write_csv(args.output/'per_prediction.csv',timing)
        plot(args.output,rows)
        for path,digest in hashes.items():
            if hashlib.sha256((ROOT/path).read_bytes()).hexdigest()!=digest:
                raise ValueError(f'source changed during sweep: {path}')
        limits = ['一 lane 一 DSP、每拍一 MAC 为研究假设，未新增 RTL/综合/PnR；60 个控制 DSP 仅来自小参数 OOC 参考',
            '假定 OCM/激活广播/归约供数随 lanes 扩宽；当前 128-bit 读口不能自动支持更多 lanes',
            '48/96/192/384 lanes 的行尾、排空、频率与新增资源代价尚未按宽阵列微架构建模',
            '固定四通道名义 12.8 GB/s、16 MiB byte OCM；不是 U200 实测吞吐',
            'Switch 保留 burst 与 payload multiplier=2；OLMoE/Mixtral 保留 matrix-fragment DMA，不横比 backend',
            '一个逻辑 Engine；只扩宽吞吐，不增加 Expert 并行数；Router 与 Expert 的吞吐一起变化',
            '每模型前 20 个保存样本，六 MoE 层/单 token/INT8/人工预测，不是完整 checkpoint 推理',
            'payload fraction 是总工作量/配置吞吐得到的算术下界，不是测得的 DSP 或 AIE busy utilization',
            '拐点是离散测试中距最快点 1% 内的最小 lanes，不保证全局最优或统计显著性',
            'Mixtral whole-residency 超过 OCM，不伪造此 baseline；full prefetch 指 streaming 全 tile 目标']
        summary = dict(status='COMPLETE',simulator='UnifiedSequenceSimulator',settings=settings,
            simulation_count=len(raw),configuration_count=len(rows),reference_matched_samples=matched,
            models=models,original_config_snapshots=snapshots,source_sha256=hashes,limits=limits)
        lines = ['# U200-like 固定带宽 DSP MAC 扩宽扫描','',
            f'完成 {len(raw):,} 次模拟、{len(rows)} 个配置点；{matched} 个重叠样本指标与前次带宽扫描一致。','',
            '## Streaming 无预取的离散饱和点','',
            '| 模型 | 距最快 1% 内最小 lanes | 最快点 lanes | 512 相对 16 lanes 提速 | 512 lanes 算术工作占比下界 |',
            '|---|---:|---:|---:|---:|']
        for p in points:
            if p['policy'] in ('stream_demand','no_prediction'):
                lines.append(f"| {p['model']} | {p['smallest_tested_lanes_within_1percent_fastest']} | {p['fastest_tested_lanes']} | {p['largest_lanes_speedup_vs_16']:.6f}× | {100*p['largest_lanes_payload_fraction_lower_bound']:.2f}% |")
        lines += ['','## 限制','',*['- '+p for p in limits],'','## 复现','',
            '```bash','PYTHONPATH=src python examples/u200_dsp_lane_sweep.py --workers 12 --output results/u200_dsp_lane_sweep_new','```','',
            'DSP 预算、每样本/每预测/聚合 CSV、knees CSV、配置快照和四组 PNG/PDF 均保存在本目录。','']
        (args.output/'REPORT.md').write_text('\n'.join(lines),encoding='utf-8')
        status.write_text(json.dumps(summary,indent=2,ensure_ascii=False)+'\n')
        print(f'COMPLETE: {len(raw)} simulations; {len(rows)} points; matched={matched}',flush=True)
    except Exception as error:
        status.write_text(json.dumps(dict(status='FAILED',error=str(error)),indent=2))
        raise


if __name__ == '__main__':
    main()
