"""完整尺寸 AXI/主模型请求级对照：不接受仅有相同总周期的假对齐。"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import re
from typing import Any


def read_observations(log: str) -> dict[str, list[dict[str, Any]]]:
    """解析只读 handshake 观测，cycle 为原始 RTL 绝对周期。"""
    patterns = {
        'pair': r'ALIGN_PAIR cycle=(\d+) layer=(\d+) expert=(\d+) tile=(\d+) class=(\d+) slots=(\d+)',
        'claim': r'ALIGN_CLAIM cycle=(\d+) layer=(\d+) expert=(\d+) tile=(\d+) matrix=(\d+) class=(\d+)',
        'done': r'ALIGN_DONE cycle=(\d+) layer=(\d+) expert=(\d+) tile=(\d+) matrix=(\d+)',
        'route': r'ALIGN_ROUTE cycle=(\d+) layer=(\d+) expert=(\d+) tile=(\d+) matrix=(\d+) truth=(\d+)',
    }
    result = {}
    for kind, pattern in patterns.items():
        rows = []
        for match in re.findall(pattern, log):
            values = list(map(int, match))
            row = dict(cycle=values[0], layer_id=values[1], expert_id=values[2], tile_id=values[3])
            if kind == 'pair':
                row.update(request_state='demand' if values[4] else 'speculative', slots=values[5])
            else:
                row['matrix_type'] = 'gate_up' if values[4] == 0 else 'down'
                if kind == 'claim':
                    row['request_state'] = 'demand' if values[5] else 'speculative'
                if kind == 'route':
                    row['truth'] = bool(values[5])
            rows.append(row)
        result[kind] = rows
    if not result['pair'] or not result['claim'] or not result['route']:
        raise ValueError('缺少请求级日志；必须运行 --alignment-debug 的新 snapshot')
    return result


def identity(row: dict[str, Any]) -> tuple[Any, ...]:
    """共享逻辑请求键；不依赖两侧独立 request_id 分配。"""
    return row['layer_id'], row['expert_id'], row['tile_id'], row.get('matrix_type')


def first_claims(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """DMA 中断后会重新 claim；仅首个 claim 对应主模型 dma_start。"""
    result, seen = [], set()
    for row in rows:
        key = identity(row)
        if key not in seen:
            seen.add(key)
            result.append(row)
    return result


def compare(main: dict[str, Any], rtl_summary: dict[str, Any], log: str,
            timeline: list[dict[str, str]]) -> dict[str, Any]:
    """检查完整功能通过、首次 DMA 顺序、compute 顺序、真实 byte 和峰值 slot。"""
    if not rtl_summary['functional_pass'] or log.count('TEST_PASS full Switch MoE') != 1 or '$finish called' not in log:
        raise ValueError('不接受未完成 RTL 仿真')
    if re.search(r'(?im)^\s*(ERROR:|Fatal:|FATAL:)', log):
        raise ValueError('RTL transcript 存在错误')
    observed = read_observations(log)
    rtl_starts = first_claims(observed['claim'])
    main_starts = [v for v in main['trace'] if v['event_type'] == 'dma_start']
    origin = int(next(v['cycle'] for v in timeline if v['event'] == 'LAYER_START'))
    first = next((dict(index=i, rtl=a, main=b) for i, (a, b) in enumerate(zip(rtl_starts, main_starts))
                  if identity(a) != identity(b)), None)
    main_compute = [identity(v) for v in main['trace'] if v['event_type'] == 'aie_start']
    rtl_compute = [(int(v['layer_id']), int(v['expert_id']), int(v['tile_id']),
                    'gate_up' if v['event'] == 'GU_START' else 'down')
                   for v in timeline if v['event'] in ('GU_START', 'DOWN_START')]
    m, r = main['memory_metrics'], rtl_summary['metrics']
    gates = dict(dma_first_start_order_equal=[identity(v) for v in rtl_starts] == [identity(v) for v in main_starts],
        compute_order_equal=main_compute == rtl_compute,
        demand_bytes_equal=m['demand_bytes'] == r['demand'],
        speculative_bytes_equal=m['speculative_bytes'] == r['spec'],
        useful_bytes_equal=m['useful_speculative_bytes'] == r['useful'],
        wasted_bytes_equal=m['wasted_speculative_bytes'] == r['wasted'],
        canceled_bytes_equal=m['canceled_speculative_bytes'] == r['cancelled'],
        peak_slots_equal=main['peak_ocm_bytes']//131072 == r['peak_slots'])
    rc, mc = r['cycles'], main['makespan_cycles']
    matched = [dict(key=identity(a), rtl_cycle_relative=a['cycle']-origin,
                    main_cycle=b['cycle'], main_minus_rtl_cycles=b['cycle']-(a['cycle']-origin))
               for a, b in zip(rtl_starts, main_starts) if identity(a) == identity(b)]
    layer_timing = []
    for layer, value in main.get('layer_results', {}).items():
        observed_cycles = {v['event']: int(v['cycle']) for v in timeline
                           if int(v['layer_id']) == int(layer) and v['event'] in ('LAYER_START', 'LAYER_DONE')}
        main_duration = value['layer_done_cycle']-value['layer_start_cycle']
        rtl_duration = observed_cycles['LAYER_DONE']-observed_cycles['LAYER_START']
        layer_timing.append(dict(layer_id=int(layer), main_cycles=main_duration, rtl_cycles=rtl_duration,
                                 delta_cycles=main_duration-rtl_duration))
    return dict(status='REQUEST_ORDER_AND_BYTES_PASS' if all(gates.values()) else 'MISMATCH',
        gates=gates, first_dma_order_mismatch=first,
        rtl_start_count=len(rtl_starts), main_start_count=len(main_starts),
        rtl_cycles=rc, main_cycles=mc, cycle_delta=mc-rc, cycle_error_percent=100*(mc/rc-1),
        cycle_exact=mc == rc, matched_first_starts=matched, layer_timing=layer_timing,
        note='请求首发顺序/bytes/峰值对齐不等于所有 burst、取消事件和控制路径逐拍一致')


def main() -> None:
    """保存请求级观测与对照 JSON；输入不完整或门限失败返回非零。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--main', type=Path, required=True, help='主模型 main_result.json')
    parser.add_argument('--rtl-case', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    log = (args.rtl_case/'simulation.log').read_text()
    with (args.rtl_case/'timeline.csv').open() as stream:
        timeline = list(csv.DictReader(stream))
    result = compare(json.loads(args.main.read_text()), json.loads((args.rtl_case/'summary.json').read_text()), log, timeline)
    args.output.mkdir(parents=True, exist_ok=True)
    for filename, value in (('request_comparison.json', result), ('rtl_observations.json', read_observations(log))):
        (args.output/filename).write_text(json.dumps(value, indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k != 'matched_first_starts'}, indent=2))
    if result['status'] != 'REQUEST_ORDER_AND_BYTES_PASS':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
