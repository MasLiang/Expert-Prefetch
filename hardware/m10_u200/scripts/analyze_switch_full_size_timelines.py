"""将完整尺寸 layer timeline 精确拆为事件间隔，不拟合额外周期。"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def partition(path: Path, layers: int, tiles: int) -> dict[str, Any]:
    """拆分 makespan；间隔是观测量，不能都称为 memory stall。"""
    if layers <= 0 or tiles <= 0:
        raise ValueError('layers and tiles must be positive')
    with path.open(encoding='utf-8') as stream:
        rows = list(csv.DictReader(stream))
    stages: dict[tuple[int, int, str], int] = {}
    for row in rows:
        key = (int(row['layer_id']), int(row['tile_id']), row['event'])
        if key in stages:
            raise ValueError(f'duplicate timeline key: {key}')
        stages[key] = int(row['cycle'])
    records = []
    for layer in range(layers):
        start = stages[layer, -1, 'LAYER_START']
        done = stages[layer, -1, 'LAYER_DONE']
        gu = [stages[layer, tile, 'GU_START'] for tile in range(tiles)]
        down = [stages[layer, tile, 'DOWN_START'] for tile in range(tiles)]
        ordered = [start]
        for g, d in zip(gu, down):
            ordered.extend((g, d))
        ordered.append(done)
        if ordered != sorted(ordered):
            raise ValueError('noncausal layer timeline')
        record = dict(layer_id=layer, layer_cycles=done-start,
            first_gu_delay_cycles=gu[0]-start,
            gu_to_down_cycles=sum(d-g for g, d in zip(gu, down)),
            down_to_next_gu_cycles=sum(gu[t+1]-down[t] for t in range(tiles-1)),
            last_down_to_result_cycles=done-down[-1])
        if sum(record[k] for k in ('first_gu_delay_cycles', 'gu_to_down_cycles',
                                   'down_to_next_gu_cycles', 'last_down_to_result_cycles')) != done-start:
            raise ValueError('partition not conserving layer duration')
        records.append(record)
    totals = {key: sum(record[key] for record in records) for key in records[0] if key != 'layer_id'}
    totals['inter_layer_cycles'] = sum(stages[l+1, -1, 'LAYER_START']-stages[l, -1, 'LAYER_DONE'] for l in range(layers-1))
    if any(stages[l+1, -1, 'LAYER_START'] < stages[l, -1, 'LAYER_DONE'] for l in range(layers-1)):
        raise ValueError('overlapping sequential layers')
    totals['total_cycles'] = stages[layers-1, -1, 'LAYER_DONE']-stages[0, -1, 'LAYER_START']
    if totals['layer_cycles']+totals['inter_layer_cycles'] != totals['total_cycles']:
        raise ValueError('partition not conserving sequence duration')
    return dict(totals=totals, layers=records)


def timeline_location(root: Path, case: str) -> Path:
    """识别两种真实输出布局，不能因 backend 文本中的 NOT MIG 误判。"""
    candidates = [root/case/'timeline.csv', root/'cases'/case/'xsim/timeline.csv']
    present = [path for path in candidates if path.is_file()]
    if len(present) != 1:
        raise ValueError(f'missing or ambiguous timeline for {case}: {present}')
    return present[0]


def main() -> None:
    """与 B 逐项比较真实事件间隔，输出可追溯分解 JSON。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output_directory', type=Path)
    args = parser.parse_args()
    root = args.output_directory
    summary = json.loads((root/'summary.json').read_text(encoding='utf-8'))
    if summary['status'] != 'PASS':
        raise ValueError('only complete passing experiments can be analyzed')
    geometry = summary['geometry']
    cases = {r['case']: partition(timeline_location(root, r['case']), geometry['LAYERS'],
                 geometry['F']//geometry['FT']) for r in summary['results']}
    for result in summary['results']:
        if cases[result['case']]['totals']['total_cycles'] != result['metrics']['cycles']:
            raise ValueError('timeline and metrics disagree')
    baseline = cases['B_stream_demand']['totals']
    for record in cases.values():
        record['difference_vs_B'] = {key: value-baseline[key] for key, value in record['totals'].items()}
    (root/'timeline_partition.json').write_text(json.dumps(cases, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({key: value['difference_vs_B'] for key, value in cases.items()}, indent=2))


if __name__ == '__main__':
    main()
