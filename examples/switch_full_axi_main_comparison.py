"""以完整六层 AXI RTL 的声明式夹具运行主模型，不拟合 RTL 观测周期。"""
from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

from edge_moe_sim.advanced_schema import AdvancedExecutionConfig
from edge_moe_sim.batch_predictor import BatchPredictorOutput, PrecomputedBatchPredictor, TokenPrediction
from edge_moe_sim.batch_routing import BatchRoutingEvent
from edge_moe_sim.execution_rules import ExecutionRules
from edge_moe_sim.hardware_profile import HardwareProfile
from edge_moe_sim.prefetch_config import PrefetchConfig
from edge_moe_sim.schema import CandidatePrefetch, ModelConfig
from edge_moe_sim.unified_schema import UnifiedLayerSequence, UnifiedPredictionSpec
from edge_moe_sim.unified_sequence_simulator import UnifiedSequenceSimulator

ROOT = Path(__file__).resolve().parents[1]
CASES = ('A_whole_demand', 'B_stream_demand', 'C_whole_full_prefetch', 'D_stream_tile_prefetch')


def fixture(case: str) -> tuple[Any, ...]:
    """复现 H768/F3072/E8、六层 truth、Q12 候选与物理 slot/credit。

    prediction ready = begin/candidate 控制交接四拍 + 每候选一拍。
    层间 Router reload = 384 个 16-byte word + 五拍主机/chain 交接。
    不读取 RTL timeline，不使用总周期残差作为参数。
    """
    if case not in CASES:
        raise ValueError(f'unknown case: {case}')
    whole, predict = case in (CASES[0], CASES[2]), case in CASES[2:]
    model = ModelConfig('switch_full_axi_fixture', 768, 3072, 8, 1, 8, 8,
                        fuse_gate_up=False, expert_architecture='relu_ffn')
    base = yaml.safe_load((ROOT/'configs/hardware/u200_ingress_alignment.yaml').read_text())
    rules = replace(ExecutionRules(**base['execution_rules']),
        ocm_slot_bytes=131072, ocm_banks=4, descriptor_entries=128,
        reserved_demand_descriptors=2, dma_fifo_depth=64,
        reserved_demand_slots=2*(3072//128) if whole else 2, wait_full_expert=whole,
        router_control_cycles=8+2+3, prefetch_allocator_cycles=8+5+3,
        prefetch_pair_interval_cycles=6, demand_ingress_blocks_prefetch=True)
    hardware = HardwareProfile(**{**base['hardware'],
        'name': 'full_switch_four_bank_AXI_structural_estimate',
        'hardware_tag': 'AXI_BEHAVIORAL_NOT_BOARD_MEASURED'}, ocm_bytes=16*1024*1024)
    routes = tuple(BatchRoutingEvent(0, 0, layer, ((layer % 8,),), ((1.0,),)) for layer in range(6))
    sequence = UnifiedLayerSequence(routes, inter_layer_cycles={(l, l+1): 768*8//16+5 for l in range(5)})
    count = 2 if whole else 8
    outputs, specs, predictors = [], [], {}
    if predict:
        for target in range(1, 6):
            candidates = tuple(CandidatePrefetch((target+rank)%8,
                3104 if rank == 0 else 352 if rank == 1 else 639//6) for rank in range(count))
            output = BatchPredictorOutput(0, 0, target, 0, (TokenPrediction(0, candidates),))
            outputs.append(output)
            predictors[(target-1, target)] = PrecomputedBatchPredictor({(0, 0, target): output})
            specs.append(UnifiedPredictionSpec(target-1, target, 'source_layer_start',
                predictor_latency_cycles=4+count, candidate_count=count,
                prefetch_policy='full_expert_prefetch' if whole else 'uniform_prefix',
                wait_for_previous_layer_weights=False))
    execution = AdvancedExecutionConfig(128, placement_policy='striped_tile', candidate_count=count,
        speculative_byte_budget=2*2*768*3072, nonlinear_elements_per_cycle=1,
        execution_rules=rules, prefetch=PrefetchConfig(allocator='hardware_greedy_v1',
            dma_order='pair_first', confidence_encoding='q12', channel_credit_bytes=(2*2*768*3072,)))
    return model, hardware, execution, sequence, tuple(specs), predictors, outputs


def write_json(path: Path, value: Any) -> None:
    """保存可重放的 UTF-8 JSON。"""
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False)+'\n', encoding='utf-8')


def main() -> None:
    """运行主模型四 case，并与已完成的 RTL CSV 比较；不启动 RTL。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--rtl', type=Path, default=ROOT/'results/m10_switch_axi_six_layers_20261007')
    parser.add_argument('--output', type=Path, default=ROOT/'results/m10_switch_axi_main_comparison')
    args = parser.parse_args()
    rtl_summary = json.loads((args.rtl/'summary.json').read_text())
    expected = dict(H=768, F=3072, FT=128, EXPERTS=8, LAYERS=6, SLOT_BYTES=131072, SLOTS_PER_BANK=32)
    if (rtl_summary['status'] != 'PASS' or rtl_summary['geometry'] != expected
            or rtl_summary['predictor_pattern'] != 0
            or rtl_summary['responder'] != dict(startup_cycles=0, beat_gap_cycles=0)
            or 'behavioral AXI' not in rtl_summary['backend']):
        raise ValueError('RTL 结果不是本脚本支持的完整六层四通道 AXI/Top1 全命中夹具')
    with (args.rtl/'comparison.csv').open() as stream:
        rtl = {r['case']: r for r in csv.DictReader(stream)}
    args.output.mkdir(parents=True, exist_ok=True)
    rows = []
    for case in CASES:
        model, hardware, execution, sequence, specs, predictors, outputs = fixture(case)
        directory = args.output/case
        directory.mkdir(exist_ok=True)
        write_json(directory/'shared_input.json', dict(model=asdict(model), hardware=asdict(hardware),
            execution=asdict(execution), sequence=asdict(replace(sequence, inter_layer_cycles={})),
            inter_layer_cycles=[dict(source=s, target=t, cycles=c) for (s,t),c in sequence.inter_layer_cycles.items()],
            prediction_specs=[asdict(s) for s in specs]))
        for filename, records in (('routing.jsonl', sequence.layers), ('predictor.jsonl', outputs)):
            (directory/filename).write_text(''.join(json.dumps(asdict(r))+'\n' for r in records))
        result = UnifiedSequenceSimulator(model, hardware).simulate(sequence, execution,
            predictions=specs, predictors=predictors, trace_enabled=True)
        if len(result.layer_results) != 6 or result.peak_ocm_bytes > hardware.ocm_bytes:
            raise AssertionError('incomplete sequence or capacity overflow')
        memory = result.memory_metrics
        if memory['speculative_bytes'] != memory['useful_speculative_bytes']+memory['wasted_speculative_bytes']:
            raise AssertionError('speculative byte conservation')
        write_json(directory/'main_result.json', asdict(result))
        rc = int(rtl[case]['cycles'])
        rows.append(dict(case=case, main_cycles=result.inference_makespan_cycles, rtl_cycles=rc,
            main_minus_rtl_cycles=result.inference_makespan_cycles-rc,
            main_error_percent=100*(result.inference_makespan_cycles/rc-1),
            main_memory_wait_cycles=sum(l.aie_starvation_cycles for l in result.layer_results.values()),
            main_peak_ocm_bytes=result.peak_ocm_bytes,
            main_demand_bytes=memory['demand_bytes'], rtl_demand_bytes=int(rtl[case]['demand']),
            main_speculative_bytes=memory['speculative_bytes'], rtl_speculative_bytes=int(rtl[case]['spec']),
            main_useful_bytes=memory['useful_speculative_bytes'], rtl_useful_bytes=int(rtl[case]['useful']),
            main_wasted_bytes=memory['wasted_speculative_bytes'], rtl_wasted_bytes=int(rtl[case]['wasted'])))
        print(json.dumps(rows[-1]), flush=True)
    for row in rows:
        for index, label in ((0, 'A'), (1, 'B')):
            row['main_speedup_vs_'+label] = rows[index]['main_cycles']/row['main_cycles']
            row['rtl_speedup_vs_'+label] = rows[index]['rtl_cycles']/row['rtl_cycles']
    with (args.output/'comparison.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    sources = [Path(__file__), ROOT/'configs/hardware/u200_ingress_alignment.yaml',
               *sorted((ROOT/'src/edge_moe_sim').rglob('*.py'))]
    write_json(args.output/'summary.json', dict(status='COMPLETED_NOT_CYCLE_EXACT_CERTIFIED',
        simulator='UnifiedSequenceSimulator', rtl_directory=str(args.rtl.resolve()), cases=rows,
        source_sha256={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
        limitations=['性能模型不计算实际 tensor 数值；truth/gating/候选来自同一夹具规则',
            '复用小尺寸验证的控制成本；完整尺寸 Router/combine/payload 数据通路非逐拍模型',
            '静态计划与 RTL 动态逐次分配/credit 刷新可能不同；不宣称 byte/order/cycle 完全对齐',
            '不是真实 DDR 或板级性能测量；无从 RTL 总周期拟合的补偿项']))


if __name__ == '__main__':
    main()
