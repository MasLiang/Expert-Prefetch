"""主模型控制开销敏感性扫描；不是新增 RTL 实测或板级 DSE。"""
from __future__ import annotations

import csv
import hashlib
import itertools
import tempfile
from copy import deepcopy
from pathlib import Path

import yaml

from compare_unified_alignment import PROJECT, simulate_fixture, write_json


def main() -> None:
    """扫描 OCM、带宽、预测到达与控制开销；不使用 RTL 周期作输入。"""
    path = PROJECT / 'configs/hardware/u200_control_alignment.yaml'
    base = yaml.safe_load(path.read_text(encoding='utf-8'))
    output = PROJECT / 'results/m10_control_sensitivity'
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / 'summary.json', dict(status='running', simulator='UnifiedSequenceSimulator', rtl_run=False))
    control_fields = ('dma_selection_cycles', 'dma_burst_gap_cycles', 'router_control_cycles',
        'tile_dispatch_cycles', 'nonlinear_handoff_cycles', 'layer_combine_cycles',
        'layer_retirement_cycles', 'demand_dispatch_cycles', 'route_scan_cycles_per_entry', 'descriptor_retirement_cycles')
    rows = []
    groups: dict[tuple[int, float, int], list[int]] = {}
    with tempfile.TemporaryDirectory(prefix='model_sweep_', dir=output) as temporary:
        for slots, efficiency, delay, scale in itertools.product((2, 4), (0.5, 1.0), (0, 64, 448), (0, 1, 2)):
            cfg = deepcopy(base)
            for field in control_fields:
                cfg['execution_rules'][field] *= scale
            reference = 0
            for name, mode in (('no_prefetch', 0), ('hit', 1), ('zero_hit', 2)):
                case = dict(name=name, layers=3, future_mode=mode, slots_per_bank=slots,
                            bandwidth_efficiency=efficiency, predictor_delay_cycles=delay)
                directory = Path(temporary) / str(len(rows))
                directory.mkdir()
                result = simulate_fixture(cfg, case, directory)
                cycles = result['makespan_cycles']
                if mode == 0:
                    reference = cycles
                saved = reference - cycles
                rows.append(dict(scenario=name, layers=3, slots_per_bank=slots,
                    bandwidth_efficiency=efficiency, predictor_delay_cycles=delay, control_scale=scale,
                    cycles=cycles, no_prefetch_cycles=reference, saved_cycles=saved, speedup=reference / cycles,
                    peak_ocm_allocated_bytes=result['peak_ocm_bytes'],
                    descriptor_peak=result['memory_metrics']['descriptor_peak'],
                    credit_backpressure_count=result['memory_metrics']['credit_backpressure_count']))
                if mode == 1:
                    groups.setdefault((slots, efficiency, delay), []).append(saved)
    with (output / 'sensitivity.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    write_json(output / 'manifest.json', dict(config=base, control_fields=control_fields,
        scales=[0, 1, 2], ocm_slots_per_bank=[2, 4], bandwidth_efficiencies=[0.5, 1.0],
        predictor_delay_cycles=[0, 64, 448], source_sha256={str(p.relative_to(PROJECT)):
            hashlib.sha256(p.read_bytes()).hexdigest() for p in [path, Path(__file__),
            *sorted((PROJECT / 'src/edge_moe_sim').rglob('*.py'))]}))
    write_json(output / 'summary.json', dict(status='analysis_complete', simulator='UnifiedSequenceSimulator',
        rtl_run=False, point_count=len(rows), condition_count=len(groups),
        hit_negative_gain_points=sum(r['scenario']=='hit' and r['saved_cycles']<0 for r in rows),
        hit_positive_gain_points=sum(r['scenario']=='hit' and r['saved_cycles']>0 for r in rows),
        hit_gain_sign_changes_across_control_scales=sum(min(v)<0<max(v) for v in groups.values()),
        max_hit_saved_cycle_spread_across_control_scales=max(max(v)-min(v) for v in groups.values()),
        zero_hit_negative_gain_points=sum(r['scenario']=='zero_hit' and r['saved_cycles']<0 for r in rows),
        limitations=['三层小型 synthetic 夹具', '主模型分析，未逐点运行 RTL',
                     '0/1/2倍控制开销是敏感性条件，不代表三种实测硬件', '不是误差已收敛或真实模型性能结论']))
    print(f'{len(rows)} model sensitivity points complete: {output}', flush=True)


if __name__ == '__main__':
    main()
