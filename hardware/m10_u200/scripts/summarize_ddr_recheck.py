"""验收当前小尺寸 DDR 自检，不将结果标成完整 Switch 模型性能。"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import re
from typing import Any


def summarize(directory: Path) -> dict[str, Any]:
    """读取完整 runtime/编译日志，验证四 bank、数值与 payload 守恒。"""
    sim = directory / 'xsim'
    runtime = (sim / 'recheck_runtime.log').read_text(encoding='utf-8')
    if runtime.count('PASS: real 3-layer MoE chain') != 1 or '$finish called' not in runtime:
        raise ValueError('缺少明确完整 PASS/$finish；不能报告性能')
    for path in (sim / 'compile.log', sim / 'elaborate.log', sim / 'recheck_runtime.log'):
        text = path.read_text(encoding='utf-8')
        if re.search(r'(?im)^\s*(ERROR:|Fatal:|FATAL:)|timing violation', text):
            raise ValueError(f'仿真错误：{path}')
    calibrated = re.findall(r'DDR_CALIBRATED bank=(\d+) cycle=(\d+)', runtime)
    initialized = re.findall(r'DDR_INITIALIZED bank=(\d+) cycle=(\d+)', runtime)
    for label, values in [('calibrated', calibrated), ('initialized', initialized)]:
        if len(values) != 4 or {int(b) for b, _ in values} != set(range(4)):
            raise ValueError(f'四 bank 未全部完成 {label}')
    starts = [(int(l), int(c)) for l, c in re.findall(r'LAYER_START layer=(\d+) cycle=(\d+)', runtime)]
    results = [(int(l), int(c), int(y)) for l, c, y in re.findall(
        r'LAYER_RESULT layer=(\d+) cycle=(\d+) y=(-?\d+)', runtime)]
    if [l for l, _ in starts] != [0, 1, 2] or [l for l, _, _ in results] != [0, 1, 2]:
        raise ValueError('层执行不完整')
    if [y for _, _, y in results] != [24, 60, 254]:
        raise ValueError('数值与参考不一致')
    samples = []
    for bank in range(4):
        with (sim / f'ddr_bank_{bank}.csv').open(encoding='utf-8') as stream:
            rows = list(csv.DictReader(stream))
        if not rows:
            raise ValueError('存在未使用的 DDR bank')
        previous_done = -1
        for row in rows:
            item = {k: int(v) for k, v in row.items()}
            if not previous_done < item['ar_cycle'] <= item['first_r_cycle'] <= item['last_r_cycle']:
                raise ValueError('DDR burst 次序非法')
            if item['first_r_cycle'] <= max(int(c) for _, c in initialized):
                raise ValueError('初始化混入测量区间')
            item['payload_bytes'] = item['beats'] * 16
            item['first_latency_cycles'] = item['first_r_cycle'] - item['ar_cycle']
            item['service_cycles'] = item['last_r_cycle'] - item['ar_cycle'] + 1
            if item['payload_bytes'] > 16 * item['service_cycles']:
                raise ValueError('带宽守恒失败')
            samples.append(item)
            previous_done = item['last_r_cycle']
    if sum(s['payload_bytes'] for s in samples) != 9216:
        raise ValueError('真实三层权重 payload 总量不符')
    project = Path(__file__).resolve().parents[3]
    root = Path(__file__).resolve().parents[1]
    paths = list((root / 'rtl').glob('*.sv')) + [
        root / 'tb/tb_u200_moe.sv', root / 'tb/ddr4_bank_backend.sv',
        root / 'configs/ddr4_0.tcl', root / 'configs/ddr_axi_cdc.tcl']
    return dict(status='PASS', backend='Vivado XSim 2023.2 / MIG + PHY BFM + RDIMM',
        full_switch_model_verified=False,
        workload=dict(hidden_size=16, intermediate_size=32, num_experts=6,
                      top_k=2, layers=3, expert_type='swiglu', prediction='disabled'),
        total_cycles=results[-1][1]-starts[0][1], clock_hz=200_000_000,
        measurement='first layer begin to final result; excludes DDR calibration/preload/cleanup',
        layer_starts=starts, layer_results=results, transferred_bytes=9216,
        burst_count=len(samples), service_samples=samples,
        unsupported_vendor_assertions=(sim / 'elaborate.log').read_text().count('XSIM 43-4398'),
        source_sha256={str(p.relative_to(project)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})


def main() -> None:
    """保存独立验收摘要，不覆盖先前 DDR 运行产物。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('case_directory', type=Path)
    args = parser.parse_args()
    result = summarize(args.case_directory)
    (args.case_directory / 'summary.json').write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({k: result[k] for k in ('status', 'total_cycles', 'transferred_bytes', 'burst_count')}))


if __name__ == '__main__':
    main()
