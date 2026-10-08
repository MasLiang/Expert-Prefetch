"""小规模 batched-decode 机会分析；只调用主模型，不修改调度语义。"""
from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
from pathlib import Path
from random import Random
from typing import Any

import yaml

from prefetch_phase_sweep import (
    ROOT, AdvancedExecutionConfig, BatchPredictorOutput, BatchRoutingEvent,
    CandidatePrefetch, HardwareProfile, ModelConfig, PrecomputedBatchPredictor,
    TokenPrediction, UnifiedLayerSequence, UnifiedPredictionSpec,
    UnifiedSequenceSimulator, write_csv,
)


def validate(config: dict[str, Any]) -> None:
    """检查受控实验范围，拒绝无法比较的输入。"""
    if config['top_k'] != 1 or config['layers'] < 2:
        raise ValueError('初版仅支持 Top-1 和至少两个 layer')
    if not 0 <= config['prediction_accuracy'] <= 1:
        raise ValueError('prediction_accuracy 必须位于 [0,1]')
    if any(b < 1 or b > config['num_experts'] for b in config['batch_sizes']):
        raise ValueError('dispersed 要求 batch 不超过 expert 数')
    if any(t < 1 or config['intermediate_size'] % t for t in config['f_tile_sizes']):
        raise ValueError('tile 必须整除 intermediate_size')
    if any(r <= 0 for r in config['ratios']):
        raise ValueError('ratio 必须为正')
    if set(config['routing_patterns']) - {'dispersed', 'balanced', 'hot'}:
        raise ValueError('未知 routing pattern')
    if set(config['dma_orders']) - {'native', 'hot_first'}:
        raise ValueError('未知 DMA 顺序')
    if any(d < 1 for d in config['prefetch_depths']):
        raise ValueError('prefetch depth 必须为正')
    for name in ('batch_sizes', 'ratios', 'f_tile_sizes', 'dma_orders', 'seeds',
                 'routing_patterns', 'prefetch_depths'):
        if not config[name] or len(set(config[name])) != len(config[name]):
            raise ValueError(f'{name} 必须非空且无重复')


def make_record(config: dict[str, Any], batch: int, pattern: str, seed: int) -> dict[str, Any]:
    """构造层间 barrier 的独立请求 batch 和固定 per-token 合成预测。"""
    rng = Random(seed + batch * 101 + ['dispersed', 'balanced', 'hot'].index(pattern) * 10007)
    truth, predicted = [], []
    for _ in range(config['layers']):
        experts = rng.sample(range(config['num_experts']), config['num_experts'])
        if pattern == 'dispersed':
            ids = experts[:batch]
        elif pattern == 'balanced':
            ids = [experts[i % min(4, batch)] for i in range(batch)]
        else:
            hot = max(1, (3 * batch + 3) // 4)
            ids = [experts[0]] * hot + [experts[1 + i % 3] for i in range(batch - hot)]
        rng.shuffle(ids)
        predictions = [e if rng.random() < config['prediction_accuracy'] else
                       rng.choice([x for x in range(config['num_experts']) if x != e]) for e in ids]
        truth.append(ids)
        predicted.append(predictions)
    return dict(batch_size=batch, pattern=pattern, seed=seed, truth=truth, predicted=predicted)


def ordered_routes(record: dict[str, Any], order: str, num_experts: int) -> tuple[list[list[int]], list[list[int]]]:
    """仅对 striped placement 重命名 ID，改变同周期 demand 的 ID 顺序。

    同时重命名 truth/prediction；不改变 token mapping、权重尺寸或通道归属。
    这是有限 DMA 顺序探针，不是新 scheduler 或全局最优求解器。
    """
    if order not in {'native', 'hot_first'}:
        raise ValueError('未知 DMA 顺序')
    truth, predicted = [], []
    for ids, guesses in zip(record['truth'], record['predicted']):
        counts = {e: ids.count(e) for e in set(ids)}
        ranked = sorted(range(num_experts), key=lambda e: (-counts.get(e, 0), e))
        mapping = {e: rank for rank, e in enumerate(ranked)} if order == 'hot_first' else {e: e for e in range(num_experts)}
        truth.append([mapping[e] for e in ids])
        predicted.append([mapping[e] for e in guesses])
    return truth, predicted


def simulate(config: dict[str, Any], record: dict[str, Any], ratio: float,
             tile: int, order: str, policy: str, depth: int = 0) -> dict[str, Any]:
    """在同一 OCM/单 Engine/四通道硬件上运行一条完整 batch 序列。"""
    if policy not in {'OFF', 'ORACLE', 'PREDICTED'} or (policy != 'OFF' and depth < 1):
        raise ValueError('未知 policy 或无效 prefetch depth')
    h, f, batch = config['hidden_size'], config['intermediate_size'], record['batch_size']
    bandwidth = ratio * config['ops_per_cycle'] / (2 * config['channels'])
    model = ModelConfig(name='synthetic_multi_token', hidden_size=h, intermediate_size=f,
        num_experts=config['num_experts'], top_k=1, weight_bits=8, activation_bits=8,
        moe_layer_indices=tuple(range(config['layers'])))
    hardware = HardwareProfile(name='synthetic_not_board', clock_hz=config['clock_hz'],
        num_memory_channels=config['channels'],
        bandwidth_bytes_per_sec_per_channel=bandwidth * config['clock_hz'],
        bandwidth_efficiency=1, dma_setup_cycles=config['dma_setup_cycles'],
        max_inflight_per_channel=1, burst_bytes=16384, supports_channel_striping=True,
        total_aie_units=1, ops_per_cycle_per_aie=config['ops_per_cycle'], aie_efficiency=1,
        aie_task_startup_cycles=config['compute_startup_cycles'], ocm_bytes=config['ocm_bytes'])
    truth, guesses = ordered_routes(record, order, config['num_experts'])
    sequence = UnifiedLayerSequence(tuple(BatchRoutingEvent(record['seed'], 0, layer,
        tuple((e,) for e in ids)) for layer, ids in enumerate(truth)))
    specs, predictors = [], {}
    if policy != 'OFF':
        for layer in range(1, config['layers']):
            ids = truth[layer] if policy == 'ORACLE' else guesses[layer]
            output = BatchPredictorOutput(record['seed'], 0, layer, 0,
                tuple(TokenPrediction(i, (CandidatePrefetch(e, 1.),)) for i, e in enumerate(ids)), 'sum')
            predictors[layer - 1, layer] = PrecomputedBatchPredictor({(record['seed'], 0, layer): output})
            specs.append(UnifiedPredictionSpec(layer - 1, layer, 'source_router_done',
                config['predictor_latency_cycles'], min(batch, config['num_experts']),
                'uniform_prefix', min(depth, f // tile), wait_for_previous_layer_weights=False))
    execution = AdvancedExecutionConfig(tile, placement_policy='striped_tile',
        max_parallel_experts=1, candidate_count=min(batch, config['num_experts']),
        speculative_byte_budget=config['ocm_bytes'],
        nonlinear_elements_per_cycle=config['nonlinear_elements_per_cycle'])
    result = UnifiedSequenceSimulator(model, hardware).simulate(sequence, execution,
        predictions=tuple(specs), predictors=predictors, trace_enabled=False)
    metrics = result.memory_metrics
    required = sum(len(set(ids)) for ids in truth) * 3 * h * f
    transferred = int(metrics['transferred_bytes'])
    if result.peak_ocm_bytes > config['ocm_bytes'] or transferred < required:
        raise AssertionError('OCM/必要 bytes 违反约束')
    if metrics['speculative_bytes'] != metrics['useful_speculative_bytes'] + metrics['wasted_speculative_bytes']:
        raise AssertionError('speculative bytes 不守恒')
    if policy == 'OFF' and transferred != required:
        raise AssertionError('Grouped baseline 必须每个实际 Expert 加载一次')
    if policy == 'ORACLE' and metrics['wasted_speculative_bytes']:
        raise AssertionError('Oracle 不应加载错误 Expert')
    unique = sum(len(set(ids)) for ids in truth)
    predictions = [p for ids in guesses[1:] for p in ids]
    target_truth = [e for ids in truth[1:] for e in ids]
    recall = sum(len(set(ids) & set(ps)) for ids, ps in zip(truth[1:], guesses[1:])) / sum(len(set(ids)) for ids in truth[1:])
    cycles = result.inference_makespan_cycles
    router_payload = config['layers'] * 2 * batch * h * config['num_experts'] / config['ops_per_cycle']
    compute_ops = config['layers'] * batch * (2*h*config['num_experts'] + 6*h*f)
    payload_lower_bound = max(compute_ops/config['ops_per_cycle'],
                              required/(config['channels']*bandwidth))
    return dict(batch_size=batch, reference_for_batch_size=batch,
        pattern=record['pattern'], seed=record['seed'], ratio=ratio,
        tile=tile, dma_order=order, policy=policy, depth=depth, total_cycles=cycles,
        drain_cycles=result.final_resource_idle_cycle, required_weight_bytes=required,
        transferred_bytes=transferred, extra_bytes=transferred-required,
        mean_unique_experts=unique/config['layers'], reuse_factor=batch*config['layers']/unique,
        max_tokens_per_expert=max(ids.count(e) for ids in truth for e in set(ids)),
        token_prediction_accuracy=1. if policy == 'ORACLE' else
            sum(a == b for a, b in zip(predictions, target_truth))/len(target_truth),
        expert_candidate_recall=1. if policy == 'ORACLE' else recall,
        peak_ocm_bytes=result.peak_ocm_bytes, speculative_bytes=metrics['speculative_bytes'],
        wasted_speculative_bytes=metrics['wasted_speculative_bytes'],
        memory_payload_utilization=transferred/(config['channels']*bandwidth*result.final_resource_idle_cycle),
        memory_channel_occupancy=metrics['memory_busy_cycles']/(config['channels']*result.final_resource_idle_cycle),
        compute_payload_utilization=result.aie_metrics['useful_aie_cycles']/cycles,
        router_payload_fraction=router_payload/cycles,
        required_compute_ops=compute_ops, payload_only_lower_bound_cycles=payload_lower_bound,
        speedup_ceiling_payload_only=cycles/payload_lower_bound,
        future_blocking_cycles=result.critical_path_future_speculation_blocking_cycles,
        tokens_per_cycle=batch/cycles, amortized_cycles_per_token=cycles/batch)


def run_cell(job: tuple[dict[str, Any], dict[str, Any], float]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """比较独立冷执行与有限配置优化后的共享；C/D 使用 B-opt 相同配置。"""
    config, record, ratio = job
    raw = [simulate(config, record, ratio, tile, order, 'OFF')
           for tile in config['f_tile_sizes'] for order in config['dma_orders']]
    native = min((r for r in raw if r['dma_order'] == 'native'), key=lambda r: r['total_cycles'])
    best = min(raw, key=lambda r: (r['total_cycles'], r['tile'], r['dma_order']))
    single = {**record, 'batch_size': 1, 'truth': [[ids[0]] for ids in record['truth']],
              'predicted': [[ids[0]] for ids in record['truth']]}
    independent = [simulate(config, single, ratio, tile, 'native', 'OFF') for tile in config['f_tile_sizes']]
    a = min(independent, key=lambda r: r['total_cycles'])
    for r in independent:
        raw.append({**r, 'policy': 'A_SINGLE_COLD_REFERENCE', 'reference_for_batch_size': record['batch_size']})
    predictions = [simulate(config, record, ratio, best['tile'], best['dma_order'], policy, depth)
                   for policy in ('ORACLE', 'PREDICTED') for depth in config['prefetch_depths']]
    raw.extend(predictions)
    c = min([best] + [r for r in predictions if r['policy'] == 'ORACLE'], key=lambda r: r['total_cycles'])
    d = min([best] + [r for r in predictions if r['policy'] == 'PREDICTED'], key=lambda r: r['total_cycles'])
    summary = {**best, 'policy': 'B_OPT_TESTED', 'A_cycles': a['total_cycles']*record['batch_size'],
        'B_cycles': native['total_cycles'], 'B_opt_cycles': best['total_cycles'],
        'C_best_tested_with_OFF_cycles': c['total_cycles'], 'D_best_tested_with_OFF_cycles': d['total_cycles'],
        'sharing_speedup': a['total_cycles']*record['batch_size']/best['total_cycles'],
        'prediction_oracle_speedup': best['total_cycles']/c['total_cycles'],
        'prediction_synthetic_speedup': best['total_cycles']/d['total_cycles'],
        'oracle_selected_depth': c['depth'], 'synthetic_selected_depth': d['depth'],
        'A_weight_bytes': a['transferred_bytes']*record['batch_size'],
        'weight_reduction_factor': a['transferred_bytes']*record['batch_size']/best['transferred_bytes'],
        'independent_reference_tile': a['tile'], 'grouped_order_speedup': native['total_cycles']/best['total_cycles']}
    return raw, summary


def plot(directory: Path, rows: list[dict[str, Any]]) -> None:
    """绘制共享收益和预测增量；曲线是匹配 seed 的 pooled 比值。"""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for filename, numerator, denominator, label in (
        ('sharing_gain', 'A_cycles', 'B_opt_cycles', 'Independent / grouped streaming'),
        ('prediction_headroom', 'B_opt_cycles', 'C_best_tested_with_OFF_cycles', 'Grouped / causal oracle (tested + OFF)'),
        ('synthetic_prediction_gain', 'B_opt_cycles', 'D_best_tested_with_OFF_cycles', 'Grouped / synthetic prediction (tested + OFF)')):
        ratios = sorted({r['ratio'] for r in rows})
        fig, axes = plt.subplots(1, len(ratios), figsize=(13, 4), squeeze=False)
        for ax, ratio in zip(axes[0], ratios):
            for pattern in ('dispersed', 'balanced', 'hot'):
                selected = [r for r in rows if r['ratio'] == ratio and r['pattern'] == pattern]
                batches = sorted({r['batch_size'] for r in selected})
                gains = [sum(r[numerator] for r in selected if r['batch_size'] == b)/
                         sum(r[denominator] for r in selected if r['batch_size'] == b) for b in batches]
                ax.plot(batches, gains, marker='o', label=pattern)
            ax.set_xscale('log', base=2); ax.set_title(f'single-token rho={ratio:g}')
            ax.set_xlabel('Batch size'); ax.set_ylabel(label); ax.axhline(1, color='gray', linestyle='--')
            ax.grid(alpha=.2); ax.legend()
        fig.tight_layout()
        for ext in ('png', 'pdf'):
            fig.savefig(directory/f'{filename}.{ext}')
        plt.close(fig)


def main() -> None:
    """生成可复现输入、实际运行数据和三组图；禁止覆盖已有产物。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT/'configs/experiments/multi_token_opportunity.yaml')
    parser.add_argument('--output', type=Path, default=ROOT/'results/multi_token_opportunity_20261008')
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    validate(config)
    if args.workers < 1 or args.output.exists():
        raise ValueError('workers 必须为正，output 必须尚不存在')
    sources = [args.config.resolve(), Path(__file__), ROOT/'examples/prefetch_phase_sweep.py'] + sorted((ROOT/'src/edge_moe_sim').rglob('*.py'))
    manifest = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    records = [make_record(config, b, p, s) for b in config['batch_sizes']
               for p in config['routing_patterns'] for s in config['seeds']]
    args.output.mkdir(parents=True)
    (args.output/'inputs.json').write_text(json.dumps(records, indent=2))
    jobs = [(config, record, ratio) for record in records for ratio in config['ratios']]
    raw, comparisons = [], []
    with mp.Pool(args.workers) as pool:
        for i, (rows, result) in enumerate(pool.imap(run_cell, jobs), 1):
            raw.extend(rows); comparisons.append(result)
            if i % 12 == 0:
                print(f'{i}/{len(jobs)} cells complete', flush=True)
    write_csv(args.output/'per_run.csv', raw)
    write_csv(args.output/'comparison.csv', comparisons)
    plot(args.output, comparisons)
    if any(hashlib.sha256((ROOT/path).read_bytes()).hexdigest() != digest for path, digest in manifest.items()):
        raise RuntimeError('运行期间 source/config 改变；不能标记 COMPLETE')
    summary = dict(status='COMPLETE', configuration=config, cells=len(jobs), simulations=len(raw),
        prediction='one previous-layer Router-done + 20 cycles; all candidates aggregated by sum',
        baseline='B-opt is best of tested tile / demand-ID orders, not global optimal',
        independent='A = batch * cold single-token sequence, one sequential engine, no persistent cache',
        prediction_envelope='posthoc minimum of tested depths and OFF; not a deployed adaptive policy or rigorous upper bound',
        source_sha256=manifest)
    (args.output/'summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: summary[k] for k in ('status', 'cells', 'simulations')}), flush=True)


if __name__ == '__main__':
    main()
