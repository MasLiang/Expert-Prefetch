"""主模型宽候选实验：相同容量/预算，对照完整、Uniform 和 Ranked 分配。"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import multiprocessing as mp
from pathlib import Path
import random
from typing import Any

import yaml

from flame_condition_comparison import make_traces
from mac_four_cases import ROOT, common_rules, write_csv
from edge_moe_sim.advanced_schema import AdvancedExecutionConfig
from edge_moe_sim.batch_predictor import BatchPredictorOutput, PrecomputedBatchPredictor, TokenPrediction
from edge_moe_sim.batch_routing import BatchRoutingEvent
from edge_moe_sim.config import load_model_config
from edge_moe_sim.hardware_profile import HardwareProfile
from edge_moe_sim.schema import CandidatePrefetch, ModelConfig
from edge_moe_sim.tile_scheduler import (
    FullExpertPrefetchScheduler, RankedTilePrefetchScheduler, UniformTilePrefetchScheduler,
)
from edge_moe_sim.unified_schema import UnifiedLayerSequence, UnifiedPredictionSpec
from edge_moe_sim.unified_sequence_simulator import UnifiedSequenceSimulator
from edge_moe_sim.work_model import expert_weight_bytes

POLICIES = ('whole_demand', 'stream_demand', 'whole_full_prefetch',
            'stream_full_prefetch', 'stream_uniform', 'stream_ranked')


def make_wide_traces(model: ModelConfig, settings: dict[str, Any]) -> list[dict[str, Any]]:
    """保留 Top-1/2 命中，随机排列所有剩余 Expert；尾部不是 CEPR 实测。"""
    records = make_traces(model, {**settings, 'candidate_count': 2})
    rng = random.Random(settings['seed'] + 1)
    scores = [settings['top1_coverage'], settings['coverage'] - settings['top1_coverage']]
    scores += [(1 - settings['coverage']) / (model.num_experts - 2)] * (model.num_experts - 2)
    for record in records:
        for prediction in record['predictions']:
            tail = [e for e in range(model.num_experts) if e not in prediction['candidate_ids']]
            rng.shuffle(tail)
            prediction['candidate_ids'] += tail
            prediction['scores'] = scores.copy()
            truth = record['truth'][model.moe_layer_indices.index(prediction['target'])][0]
            prediction['true_rank'] = prediction['candidate_ids'].index(truth)
    return records


def run_sample(job: tuple[Any, ...]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """运行真实事件队列；输出计划、入队和 Router-time 完成覆盖，不以计划冒充就绪。"""
    model, settings, base, common, bw, policy, record = job
    whole = policy in {'whole_demand', 'whole_full_prefetch'}
    predict = 'prefetch' in policy or policy in {'stream_uniform', 'stream_ranked'}
    schedulers = {'whole_full_prefetch': FullExpertPrefetchScheduler(),
                  'stream_full_prefetch': FullExpertPrefetchScheduler(),
                  'stream_uniform': UniformTilePrefetchScheduler(),
                  'stream_ranked': RankedTilePrefetchScheduler(settings['ranked_depth_decay'])}
    fields = dict(base['hardware'])
    fields.update(name='wide_candidate_configured_not_measured', total_aie_units=1,
                  burst_bytes=settings['burst_bytes'],
                  bandwidth_bytes_per_sec_per_channel=bw * fields['clock_hz'])
    hardware = HardwareProfile(**fields, ocm_bytes=settings['ocm_bytes'],
                               ops_per_cycle_per_aie=settings['ops_per_cycle'])
    sequence = UnifiedLayerSequence(tuple(
        BatchRoutingEvent(record['sample'], 0, layer, (tuple(truth),))
        for layer, truth in zip(model.moe_layer_indices, record['truth'])))
    specs, predictors = [], {}
    if predict:
        for prediction in record['predictions']:
            source, target = prediction['source'], prediction['target']
            output = BatchPredictorOutput(record['sample'], 0, target, 0, (TokenPrediction(0, tuple(
                CandidatePrefetch(e, score) for e, score in
                zip(prediction['candidate_ids'], prediction['scores']))),))
            predictors[(source, target)] = PrecomputedBatchPredictor({(record['sample'], 0, target): output})
            specs.append(UnifiedPredictionSpec(source, target, 'source_router_done',
                base['predictor_latency_cycles'], model.num_experts,
                'full_expert_prefetch' if 'full_prefetch' in policy else 'uniform_prefix'))
    budget = settings['prefetch_budget_experts'] * expert_weight_bytes(model)
    config = AdvancedExecutionConfig(settings['f_tile_size'], placement_policy='striped_tile',
        candidate_count=model.num_experts, speculative_byte_budget=budget,
        nonlinear_elements_per_cycle=base['nonlinear_elements_per_cycle'],
        execution_rules=common_rules(model, settings, common),
        demand_loading='full_expert_load' if whole else 'tile_streaming')
    result = UnifiedSequenceSimulator(model, hardware).simulate(sequence, config,
        predictions=tuple(specs), predictors=predictors,
        tile_prefetch_scheduler=schedulers.get(policy))
    if len(result.layer_results) != len(sequence.layers) or result.peak_ocm_bytes > hardware.ocm_bytes:
        raise AssertionError('incomplete execution or OCM overflow')
    memory = result.memory_metrics
    if memory['speculative_bytes'] != memory['useful_speculative_bytes'] + memory['wasted_speculative_bytes']:
        raise AssertionError('speculative byte conservation violated')
    identity = dict(model=model.name, bw_bytes_per_cycle_per_channel=bw, policy=policy, sample=record['sample'])
    details = []
    for prediction in result.predictions:
        target = result.layer_results[prediction.target_layer_id]
        requests = [r for r in result.speculative_request_records if r['target_layer'] == prediction.target_layer_id]
        queued = {r['expert_id'] for r in requests}
        started = {r['expert_id'] for r in requests if r['issue_cycle'] is not None}
        first_completed = {r['expert_id'] for r in requests if r['matrix_type'] == 'gate_up'
            and r['tile_id'] == 0 and r['completion_cycle'] is not None
            and r['completion_cycle'] <= target.router_resolve_cycle}
        true_rank = prediction.candidate_ids.index(target.true_expert_ids[0])
        details.append({**identity, **asdict(prediction), 'true_rank': true_rank,
            'top1_hit': true_rank == 0, 'top2_hit': true_rank < 2,
            'planned_experts': sum(d > 0 for d in prediction.planned_candidate_depths),
            'queued_experts': len(queued), 'started_experts': len(started),
            'first_up_completed_experts_before_router': len(first_completed),
            'true_startable_at_router': target.startable_recall,
            'true_first_up_completed_before_router': target.true_expert_ids[0] in first_completed})
    row = {**identity, 'total_cycles': result.inference_makespan_cycles,
        'memory_wait_cycles': sum(r.aie_starvation_cycles for r in result.layer_results.values()),
        'peak_ocm_bytes': result.peak_ocm_bytes,
        'demand_bytes': sum(r.demand_bytes for r in result.layer_results.values()),
        **{key: memory[key] for key in ('speculative_bytes', 'useful_speculative_bytes', 'wasted_speculative_bytes')},
        'prediction_targets': len(details)}
    for key in ('planned_experts', 'queued_experts', 'started_experts',
                'first_up_completed_experts_before_router', 'true_startable_at_router',
                'true_first_up_completed_before_router'):
        row[key] = sum(d[key] for d in details)
    return row, details


def aggregate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """计算配对总体 makespan 速度比，不平均各样本速度比。"""
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault((row['model'], row['bw_bytes_per_cycle_per_channel'], row['policy']), []).append(row)
    totals = {key: sum(r['total_cycles'] for r in values) for key, values in grouped.items()}
    output = []
    for (model, bw, policy), values in sorted(grouped.items()):
        n = len(values)
        total = totals[(model, bw, policy)]
        targets = sum(r['prediction_targets'] for r in values)
        row = dict(model=model, bw_bytes_per_cycle_per_channel=bw, policy=policy, samples=n,
                   cycles_per_sequence=total/n, peak_ocm_bytes=max(r['peak_ocm_bytes'] for r in values))
        for baseline in ('whole_demand', 'stream_demand', 'stream_full_prefetch'):
            row['speedup_vs_' + baseline] = totals[(model, bw, baseline)] / total
        for key in ('memory_wait_cycles', 'demand_bytes', 'speculative_bytes',
                    'useful_speculative_bytes', 'wasted_speculative_bytes'):
            row[key + '_per_sequence'] = sum(r[key] for r in values) / n
        for key in ('planned_experts', 'queued_experts', 'started_experts',
                    'first_up_completed_experts_before_router', 'true_startable_at_router',
                    'true_first_up_completed_before_router'):
            row[key + '_per_target'] = sum(r[key] for r in values)/targets if targets else None
        output.append(row)
    return output


def plot(directory: Path, rows: list[dict[str, Any]]) -> None:
    """输出速度比和计划/完成覆盖两组 PNG/PDF。"""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    panels = sorted({(r['model'], r['bw_bytes_per_cycle_per_channel']) for r in rows})
    for kind in ('speedup', 'coverage'):
        fig, axes = plt.subplots(1, len(panels), figsize=(4*len(panels), 4), squeeze=False)
        for ax, (model, bw) in zip(axes[0], panels):
            policies = POLICIES if kind == 'speedup' else POLICIES[2:]
            points = [next(r for r in rows if (r['model'], r['bw_bytes_per_cycle_per_channel'], r['policy'])
                           == (model, bw, p)) for p in policies]
            if kind == 'speedup':
                ax.bar(range(len(points)), [r['speedup_vs_stream_demand'] for r in points])
                ax.axhline(1, color='gray', linestyle='--')
                ax.set_ylabel('Speedup vs streaming without prefetch')
            else:
                ax.bar([i-.18 for i in range(len(points))], [r['planned_experts_per_target'] for r in points],
                       width=.36, label='Planned')
                ax.bar([i+.18 for i in range(len(points))], [r['first_up_completed_experts_before_router_per_target']
                       for r in points], width=.36, label='First Up completed by Router')
                ax.set_ylabel('Experts per predicted layer')
                ax.legend(fontsize=7)
            ax.set_xticks(range(len(points)), policies, rotation=60, ha='right', fontsize=7)
            ax.set_title(f'{model}\nnominal BW={bw} bytes/cycle/channel', fontsize=8)
        fig.tight_layout()
        for ext in ('png', 'pdf'):
            fig.savefig(directory/f'{kind}.{ext}')
        plt.close(fig)


def main() -> None:
    """保存同源宽候选 JSON、逐样本/逐预测 CSV、配置摘要和图。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT/'configs/experiments/switch_wide_tile_prefetch.yaml')
    parser.add_argument('--output', type=Path, default=ROOT/'results/switch_wide_tile_prefetch')
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--samples', type=int, help='覆盖配置样本数，仅用于快速检查')
    args = parser.parse_args()
    settings = yaml.safe_load(args.config.read_text())
    if args.samples is not None:
        settings['samples'] = args.samples
    if args.workers < 1 or settings['samples'] < 1:
        raise ValueError('workers and samples must be positive')
    base = yaml.safe_load((ROOT/settings['base_config']).read_text())
    common = yaml.safe_load((ROOT/settings['common_config']).read_text())
    args.output.mkdir(parents=True, exist_ok=True)
    raw, details, declarations = [], [], []
    with mp.get_context('spawn').Pool(args.workers) as pool:
        for path in settings['models']:
            model = load_model_config(ROOT/path)
            records = make_wide_traces(model, settings)
            (args.output/f'{model.name}_inputs.json').write_text(json.dumps(records, indent=2))
            declarations.append(asdict(model))
            for bw in settings['bytes_per_cycle_per_channel']:
                for policy in POLICIES:
                    results = pool.map(run_sample, [(model, settings, base, common, bw, policy, r) for r in records])
                    raw.extend(r for r, _ in results)
                    details.extend(d for _, items in results for d in items)
                    write_csv(args.output/'checkpoint.csv', raw)
                    print(f'completed {model.name} BW={bw} {policy}', flush=True)
    comparison = aggregate(raw)
    write_csv(args.output/'per_sample.csv', raw)
    write_csv(args.output/'per_prediction.csv', details)
    write_csv(args.output/'comparison.csv', comparison)
    plot(args.output, comparison)
    sources = [args.config.resolve(), Path(__file__), ROOT/'examples/flame_condition_comparison.py',
               ROOT/'examples/switch_prefetch_comparison.py', ROOT/'examples/mac_four_cases.py',
               ROOT/settings['base_config'], ROOT/settings['common_config'],
               *[ROOT/p for p in settings['models']], *sorted((ROOT/'src/edge_moe_sim').rglob('*.py'))]
    summary = dict(status='complete', simulator='UnifiedSequenceSimulator', settings=settings,
        base_hardware=base['hardware'], execution_rules=common['execution_rules'], models=declarations,
        policies=POLICIES, sample_results=len(raw), prediction_records=len(details),
        source_sha256={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
        limits=['官方尺寸，INT8 假设；六个 decoder MoE 层，无 attention/encoder 或跨 token 复用',
            'Top-1/2 maxima 构造为受控分布；其余 ranking 随机、confidence 平分尾部，非 CEPR 实测',
            '所有候选列表包含全部 Expert，candidate recall=1 不代表 predictor accuracy=1',
            'OCM=16MiB，预取上限=两个完整 Expert；按预测释放时实际空闲容量分配',
            '逻辑 tile 含 Up+Down，两矩阵均预取；先按 allocator 顺序发 Up，再发 Down',
            '保持真实 DMA 优先级/非抢占/OCM；不使用未来 Router truth 或预测完成时间作规划',
            'byte OCM/unlimited credits/expert_locked，非 RTL 仿真或板卡实测',
            '全部策略相同固定 allocator 控制成本；不建模不同算法的 RTL 面积或周期差异'])
    (args.output/'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
