"""验证现有 RTL 的宽候选 Greedy 与 Switch 四 case；不冒充 Uniform/Ranked 验证。"""
from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import re
from typing import Any

import yaml

from compare_four_cases import CASES, simulate, verify_rtl
from compare_u200_default_alignment import PROJECT, ROOT, command_run
from compare_u200_alignment import write_json
from edge_moe_sim.hardware_greedy_v1 import (
    HardwareGreedyV1Candidate, HardwareGreedyV1Config, hardware_greedy_v1,
)


def golden(candidates: int) -> tuple[list[int], tuple[Any, ...]]:
    """使用与宽候选实验相同 confidence 分布，显式量化为 RTL Q12。"""
    scores = [.758, .086] + [.156/(candidates-2)]*(candidates-2)
    confidences = [max(1, min(4095, round(s*4095))) for s in scores]
    plan = hardware_greedy_v1(tuple(HardwareGreedyV1Candidate(e, e, c, (196608,)*24)
                                  for e, c in enumerate(confidences)),
                             HardwareGreedyV1Config(1, 9437184, (9437184,)))
    return confidences, plan


def run_xsim(build: Path, directory: Path, snapshot: str, top: str,
             generics: dict[str, int], plusargs: list[str]) -> str:
    """编译新 snapshot，运行最多 1ms，并检查完整 transcript 与终止标记。"""
    directory.mkdir(parents=True, exist_ok=True)
    command = ['xelab', '--debug', 'off', '--mt', '4', '--snapshot', snapshot]
    for key, value in generics.items():
        command += ['--generic_top', f'{key}={value}']
    command_run([*command, f'work.{top}'], build, directory/'elaborate.log')
    batch = directory/'run.tcl'
    batch.write_text('run 1 ms\nquit\n')
    command = ['xsim', snapshot, '--tclbatch', str(batch), '--onerror', 'quit',
               '--onfinish', 'quit', '--log', str(directory/'simulation.log')]
    for arg in plusargs:
        command += ['--testplusarg', arg]
    command_run(command, build, directory/'console.log')
    text = (directory/'simulation.log').read_text()
    if text.count('PASS:') != 1 or '$finish called' not in text or re.search(r'Fatal:|FATAL:|ERROR:|Error:', text):
        raise ValueError(f'RTL failed acceptance: {snapshot}')
    return text


def main() -> None:
    """保存 bit-exact allocation、数值通路结果及明确的未验收范围。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=PROJECT/'results/m10_wide_prefetch_validation')
    args = parser.parse_args()
    output = args.output.resolve()
    build = output/'rtl_build'
    build.mkdir(parents=True, exist_ok=True)
    config_path = PROJECT/'configs/hardware/u200_switch_four_case_validation.yaml'
    config = yaml.safe_load(config_path.read_text())
    package = ROOT/'rtl/package.sv'
    sources = [package, *sorted(p for p in (ROOT/'rtl').glob('*.sv') if p != package),
               ROOT/'tb/tb_wide_candidate_allocation.sv', ROOT/'tb/tb_u200_moe.sv',
               ROOT/'tb/tb_switch_relu_backend.sv']
    initial_hashes = {str(p.relative_to(PROJECT)): hashlib.sha256(p.read_bytes()).hexdigest()
                      for p in [*sources, Path(__file__), config_path, PROJECT/config['alignment_base']]}
    write_json(output/'summary.json', dict(status='running', strict_uniform_ranked_validation=False))
    command_run(['xvlog', '--sv', *map(str, sources)], build, build/'compile.log')
    allocations = []
    for n in (8, 16):
        confidences, plan = golden(n)
        confidence_path = output/f'confidence_{n}.mem'
        expected_path = output/f'expected_{n}.mem'
        confidence_path.write_text(''.join(f'{c:08x}\n' for c in confidences))
        expected_path.write_text(''.join(f'{(p.expert_id << 8) | p.tile_id:08x}\n' for p in plan))
        write_json(output/f'golden_{n}.json', dict(confidences_q12=confidences, allocations=[asdict(p) for p in plan]))
        for stall in (0, 1):
            name = f'wide_{n}_stall_{stall}'
            text = run_xsim(build, output/name, name, 'tb_wide_candidate_allocation',
                dict(CANDIDATES=n, STALL=stall),
                [f'CONFIDENCE={confidence_path}', f'EXPECTED={expected_path}'])
            observed = [(int(e), int(t), int(b)) for _, _, e, t, b in re.findall(
                r'WIDE_ALLOC index=(\d+) cycle=(\d+) expert=(\d+) tile=(\d+) bytes=(\d+)', text)]
            if observed != [(p.expert_id, p.tile_id, p.logical_bytes) for p in plan]:
                raise ValueError('RTL allocation sequence differs from independent integer golden')
            depths = {int(e): int(d) for e, d in re.findall(r'WIDE_DEPTH expert=(\d+) depth=(\d+)', text)}
            allocations.append(dict(candidates=n, stall=stall, allocations=len(observed),
                covered_experts=sum(d > 0 for d in depths.values()), depths=depths,
                allocated_bytes=sum(b for _, _, b in observed)))
            print(f'{name}: PASS', flush=True)
    run_xsim(build, output/'switch_dense', 'switch_dense', 'tb_switch_relu_backend', {}, [])
    print('switch_dense: PASS', flush=True)
    rows = []
    for point in config['bandwidth_points'][:2]:
        for pattern in config['prediction_patterns']:
            for case in CASES:
                name = f"{point['name']}_{pattern}_{case}"
                directory = output/name
                directory.mkdir(exist_ok=True)
                main_result = simulate(config, point, pattern, case, directory)
                whole = case in CASES[:1] or case == CASES[2]
                prefetch = case in CASES[2:]
                generics = dict(LAYERS=3, ROUTE_K=1, SWITCH_RELU=1, BUDGET=0, DDR_GAPS=0,
                    DDR_BEAT_GAP=point['ddr_beat_gap'], WAIT_FULL_EXPERT=int(whole),
                    FUTURE_MODE=(1 if pattern == 'hit' else 2) if prefetch else 0,
                    FUTURE_BUDGET_BYTES=1024 if whole else 512,
                    OCM_SLOTS_PER_BANK=config['slots_per_bank'],
                    DESCRIPTOR_ENTRIES=config['descriptor_entries'], FIFO_DEPTH=config['fifo_depth'],
                    EXPECT_EARLY_OVERLAP=0)
                text = run_xsim(build, directory, name, 'tb_u200_moe', generics, ['ALIGN_MAIN'])
                cycles, layers = verify_rtl(text, 3, whole, switch=True, prefetch=prefetch)
                write_json(directory/'rtl_result.json', dict(cycles=cycles, per_layer=layers, generics=generics))
                rows.append(dict(bandwidth=point['name'], prediction=pattern, case=case,
                                 rtl_cycles=cycles, main_cycles=main_result['makespan_cycles']))
                print(f'{name}: PASS RTL={cycles}', flush=True)
    lookup = {(r['bandwidth'], r['prediction'], r['case']): r for r in rows}
    for r in rows:
        for backend in ('rtl', 'main'):
            r[f'{backend}_speedup_vs_B'] = lookup[(r['bandwidth'], r['prediction'], CASES[1])][backend+'_cycles']/r[backend+'_cycles']
    with (output/'comparison.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    if any(hashlib.sha256((PROJECT/p).read_bytes()).hexdigest() != h for p, h in initial_hashes.items()):
        raise RuntimeError('RTL/test source changed during validation')
    write_json(output/'summary.json', dict(status='complete', tested_contracts_passed=True,
        backend='Vivado XSim 2023.2 behavioral', allocation_tests=allocations,
        numerical_cases=17, performance_rows=rows, source_sha256=initial_hashes,
        strict_uniform_ranked_validation=False,
        limits=['宽候选测试仅验证 HardwareGreedy-v1 allocation/credit，不包含 DMA/OCM/计算周期',
                '四 case 是 H16/F32/E6/三层单候选数值夹具，非 H768/F3072 宽候选实验',
                'RTL 没有 Uniform/Ranked 策略开关，不能验证其等待消除比例或速度比',
                '真实 Switch byte 几何只用于 allocator 单元测试；不等于 OCM payload 实际容纳这些 bytes',
                'AXI behavioral responder 非 DDR4 PHY 或板卡实测；不宣称时序收敛']))


if __name__ == '__main__':
    main()
