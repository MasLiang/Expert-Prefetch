"""共同夹具驱动主模拟器及真实多层 RTL；诊断差异，不伪称周期对齐。"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import yaml
from compare_u200_default_alignment import PROJECT, ROOT, command_run
from compare_u200_alignment import write_json
from validate_multilayer import export_events
from analyze_boundary_waits import analyze_waits
from edge_moe_sim.advanced_schema import AdvancedExecutionConfig
from edge_moe_sim.execution_rules import ExecutionRules
from edge_moe_sim.batch_predictor import BatchPredictorOutput, PrecomputedBatchPredictor, TokenPrediction
from edge_moe_sim.batch_routing import BatchRoutingEvent
from edge_moe_sim.hardware_profile import HardwareProfile
from edge_moe_sim.schema import CandidatePrefetch, ModelConfig
from edge_moe_sim.unified_schema import UnifiedLayerSequence, UnifiedPredictionSpec
from edge_moe_sim.unified_sequence_simulator import UnifiedSequenceSimulator


def simulate_fixture(config: dict[str, Any], case: dict[str, Any], directory: Path) -> dict[str, Any]:
    """从预先声明的夹具产生输入；不读取 RTL 观测周期或 truth。"""
    fixture = config['fixture']
    expected = dict(hidden_size=16, intermediate_size=32, num_experts=6, top_k=2,
                    weight_bits=8, activation_bits=8, f_tile_size=16, slot_bytes=512,
                    future_budget_bytes=1536)
    if fixture != expected or not 2 <= case['layers'] <= 8 or case['future_mode'] not in (0, 1, 2):
        raise ValueError('配置超出 tb_u200_moe 的共同夹具范围')
    required_rules = dict(ocm_slot_bytes=fixture['slot_bytes'], ocm_banks=2,
        reserved_demand_slots=2, epoch_base=1, atomic_tile_pair=True,
        compute_policy='tile_ready_first', retry_future_prefetch=True, request_order='pair_fifo')
    if any(config.get('execution_rules', {}).get(k) != v for k, v in required_rules.items()):
        raise ValueError('execution_rules 与当前 RTL 夹具的 bank/slot/admission/调度规则不一致')
    model = ModelConfig(name='rtl_common_fixture', **{k: fixture[k] for k in
        ('hidden_size', 'intermediate_size', 'num_experts', 'top_k', 'weight_bits', 'activation_bits')})
    routes = tuple(BatchRoutingEvent(0, 0, layer, ((layer % 4 + 1, layer % 4 + 2),),
                                    ((0.5, 0.5),)) for layer in range(case['layers']))
    predictions = []
    outputs = []
    predictors = {}
    for target in range(1, case['layers']):
        if not case['future_mode']:
            continue
        expert = target % 4 + 1 if case['future_mode'] == 1 else 5
        # begin/candidate 接受两拍，后续事务含 teardown + begin + candidate 三拍。
        ready = 2 + 3 * (target - 1) + case.get('predictor_delay_cycles', 0)
        output = BatchPredictorOutput(0, 0, target, ready,
            (TokenPrediction(0, (CandidatePrefetch(expert, 3000 / 4096),)),))
        outputs.append(output)
        predictors[(0, target)] = PrecomputedBatchPredictor({(0, 0, target): output})
        predictions.append(UnifiedPredictionSpec(0, target, 'source_layer_start',
            predictor_latency_cycles=ready, prefetch_policy='full_expert_prefetch',
            wait_for_previous_layer_weights=False))
    hardware = {**config['hardware'], 'bandwidth_efficiency': case.get('bandwidth_efficiency', config['hardware']['bandwidth_efficiency'])}
    profile = HardwareProfile(**hardware,
        ocm_bytes=2 * case['slots_per_bank'] * fixture['slot_bytes'])
    execution = AdvancedExecutionConfig(f_tile_size=fixture['f_tile_size'],
        placement_policy='striped_tile', max_parallel_experts=1,
        speculative_byte_budget=fixture['future_budget_bytes'], nonlinear_elements_per_cycle=1,
        execution_rules=replace(ExecutionRules(**config.get('execution_rules', {})),
            reserved_demand_descriptors=2 if case.get('descriptor_entries', config['execution_rules'].get('descriptor_entries', 0)) else 0,
            descriptor_entries=case.get('descriptor_entries', config['execution_rules'].get('descriptor_entries', 0)),
            dma_fifo_depth=case.get('fifo_depth', config['execution_rules'].get('dma_fifo_depth', 0))))
    for filename, values in (('routing.jsonl', routes), ('predictor.jsonl', outputs)):
        (directory / filename).write_text(''.join(json.dumps(asdict(v)) + '\n' for v in values), encoding='utf-8')
    write_json(directory / 'shared_input.json', dict(model=asdict(model), hardware=asdict(profile),
        execution=asdict(execution), prediction_specs=[asdict(v) for v in predictions],
        rtl_generics=rtl_generics(case)))
    result = asdict(UnifiedSequenceSimulator(model, profile).simulate(
        UnifiedLayerSequence(routes), execution, predictions=tuple(predictions),
        predictors=predictors, trace_enabled=True))
    write_json(directory / 'main_result.json', result)
    (directory / 'main_trace.jsonl').write_text(''.join(json.dumps(v) + '\n' for v in result['trace']), encoding='utf-8')
    return result


def rtl_generics(case: dict[str, Any]) -> dict[str, int]:
    """共同输入参数直接驱动真实 bench，不读取观测结果。"""
    return dict(LAYERS=case['layers'], FUTURE_MODE=case['future_mode'],
        OCM_SLOTS_PER_BANK=case['slots_per_bank'], BUDGET=0, DDR_GAPS=case.get('ddr_gaps', 0),
        DESCRIPTOR_ENTRIES=case.get('descriptor_entries', 8), FIFO_DEPTH=case.get('fifo_depth', 4),
        PREDICTOR_DELAY_CYCLES=case.get('predictor_delay_cycles', 0),
        EXPECT_EARLY_OVERLAP=case.get('expect_early_overlap', 1))


def main() -> None:
    """重建 XSim 并生成明确分离功能验证与完整对齐的结果。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=PROJECT / 'configs/hardware/u200_unified_alignment.yaml')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--require-alignment', action='store_true')
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding='utf-8'))
    output = (args.output or PROJECT / config.get('output_directory', 'results/m10_unified_rules_alignment')).resolve()
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / 'summary.json', dict(status='running', alignment_passed=False))
    build = output / 'rtl_build'
    build.mkdir(exist_ok=True)
    package = ROOT / 'rtl/package.sv'
    sources = [package, *sorted(p for p in (ROOT / 'rtl').glob('*.sv') if p != package), ROOT / 'tb/tb_u200_moe.sv']
    write_json(output / 'manifest.json', dict(config=config, simulator='UnifiedSequenceSimulator',
        model_source_sha256={str(p.relative_to(PROJECT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted((PROJECT / 'src/edge_moe_sim').rglob('*.py'))},
        source_sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in
            [*sources, args.config, Path(__file__), Path(__file__).with_name('analyze_boundary_waits.py'), PROJECT / 'src/edge_moe_sim/unified_sequence_simulator.py',
             PROJECT / 'src/edge_moe_sim/execution_rules.py', PROJECT / 'src/edge_moe_sim/hardware/capacity_ocm.py',
             PROJECT / 'src/edge_moe_sim/hardware/multi_channel_memory.py']}))
    command_run(['xvlog', '--sv', *map(str, sources)], build, build / 'compile.log')
    rows = []
    for case in config['cases']:
        directory = output / case['name']
        directory.mkdir(exist_ok=True)
        result = simulate_fixture(config, case, directory)
        snapshot = 'unified_' + case['name']
        command = ['xelab', '--debug', 'off', '--mt', '4', '--snapshot', snapshot]
        for key, value in rtl_generics(case).items():
            command += ['--generic_top', f'{key}={value}']
        command_run([*command, 'work.tb_u200_moe'], build, directory / 'elaborate.log')
        batch = directory / 'run.tcl'
        batch.write_text('run 1 ms\nquit\n', encoding='utf-8')
        command_run(['xsim', snapshot, '--testplusarg', 'ALIGN_MAIN', '--tclbatch', str(batch), '--onerror', 'quit',
                     '--onfinish', 'quit', '--log', str(directory / 'simulation.log')],
                    build, directory / 'console.log')
        log = (directory / 'simulation.log').read_text(encoding='utf-8')
        if log.count('PASS:') != 1 or '$finish called' not in log or re.search(r'Fatal:|FATAL:|ERROR:|Error:', log):
            raise ValueError(f"RTL 功能验收失败：{case['name']}")
        export_events(log, directory)
        analyze_waits(log,result,directory)
        truths = re.findall(r'MAIN_TRUTH layer=(\d+) cycle=(\d+) expert=(\d+) weight=(\d+)', log)
        for layer in range(case['layers']):
            observed = [(int(e), int(w)) for l, _, e, w in truths if int(l) == layer]
            if sorted(observed) != [(layer % 4 + 1, 16384), (layer % 4 + 2, 16384)]:
                raise ValueError(f'共同 routing truth 不匹配：layer={layer}, observed={observed}')
        stages = re.findall(r'MAIN_STAGE layer=(\d+) cycle=(\d+) expert=(\d+) tile=(\d+) stage=(\w+)', log)
        with (directory / 'rtl_stages.csv').open('w', newline='', encoding='utf-8') as stream:
            writer = csv.writer(stream)
            writer.writerow(['layer', 'cycle_absolute', 'expert', 'tile', 'stage'])
            writer.writerows(stages)
        expected_keys = {(str(l), str(e), str(t)) for l in range(case['layers'])
                         for e in (l % 4 + 1, l % 4 + 2) for t in range(2)}
        for stage in ('GU_START', 'GU_DONE', 'DOWN_START', 'DOWN_DONE'):
            keys = [(l, e, t) for l, _, e, t, s in stages if s == stage]
            if len(keys) != len(expected_keys) or set(keys) != expected_keys:
                raise ValueError(f'RTL tile 阶段缺失或重复：{stage}')
        main_stages = [v for v in result['trace'] if v['event_type'] == 'aie_start']
        for matrix in ('gate_up', 'down'):
            keys = [(str(v['layer_id']), str(v['expert_id']), str(v['tile_id']))
                    for v in main_stages if v['matrix_type'] == matrix]
            if len(keys) != len(expected_keys) or set(keys) != expected_keys:
                raise ValueError(f'主模型 tile 阶段缺失或重复：{matrix}')
        stage_cycles = {(l, e, t, s): int(c) for l, c, e, t, s in stages}
        with (directory / 'stage_comparison.csv').open('w', newline='', encoding='utf-8') as stream:
            writer = csv.writer(stream)
            writer.writerow(['layer', 'expert', 'tile', 'matrix', 'main_start', 'main_done',
                             'rtl_start_absolute', 'rtl_done_absolute', 'duration_delta_cycles'])
            for v in main_stages:
                prefix = 'GU' if v['matrix_type'] == 'gate_up' else 'DOWN'
                key = (str(v['layer_id']), str(v['expert_id']), str(v['tile_id']))
                start = stage_cycles[(*key, prefix + '_START')]
                end = stage_cycles[(*key, prefix + '_DONE')]
                if end < start:
                    raise ValueError('RTL 阶段完成早于开始')
                writer.writerow([*key, v['matrix_type'], v['cycle'], v['end_cycle'], start, end,
                                 end - start - (v['end_cycle'] - v['cycle'])])
        starts = {int(l): int(c) for l, c in re.findall(r'LAYER_START layer=(\d+) cycle=(\d+)', log)}
        ends = {int(l): int(c) for l, c in re.findall(r'LAYER_RESULT layer=(\d+) cycle=(\d+)', log)}
        if len(starts) != case['layers'] or len(ends) != case['layers']:
            raise ValueError('RTL 缺少完整的分层开始/结束观测')
        # 按层自身起点比较关键路径事件，避免把上一层残差混入本层等待。
        with (directory / 'pipeline_comparison.csv').open('w', newline='', encoding='utf-8') as stream:
            writer = csv.writer(stream)
            writer.writerow(['layer', 'milestone', 'main_relative_cycle', 'rtl_relative_cycle', 'delta_cycles'])
            for layer in range(case['layers']):
                model_layer = result['layer_results'][layer]
                origin = model_layer['layer_start_cycle']
                first_gu = next(v['cycle'] for v in main_stages if v['layer_id'] == layer and v['matrix_type'] == 'gate_up')
                rtl_gu = next(int(c) for l, c, _, _, stage in stages if int(l) == layer and stage == 'GU_START')
                last_truth = max(int(c) for l, c, _, _ in truths if int(l) == layer)
                for milestone, main_cycle, rtl_cycle in (
                    ('truth_complete', model_layer['router_resolve_cycle'], last_truth),
                    ('first_gu_start', first_gu, rtl_gu),
                    ('layer_done', model_layer['layer_done_cycle'], ends[layer])):
                    m, r = main_cycle - origin, rtl_cycle - starts[layer]
                    writer.writerow([layer, milestone, m, r, r - m])
        with (directory / 'rtl_interface_timeline.csv').open('w', newline='', encoding='utf-8') as stream:
            writer = csv.writer(stream)
            writer.writerow(['event', 'cycle_absolute', 'bank', 'last'])
            writer.writerows(re.findall(r'MAIN_AXI event=(AR|R) cycle=(\d+) bank=(\d+) last=(\d+)', log))
            writer.writerows(('JOIN', cycle, '', last) for cycle, last in re.findall(r'MAIN_JOIN cycle=(\d+) last=(\d+)', log))
        rtl_cycles = ends[case['layers'] - 1] - starts[0]
        observed_dma = re.findall(r'MAIN_DMA layer=(\d+) cycle=(\d+) expert=(\d+) tile=(\d+) matrix=(\d+) bytes=(\d+) class=(\d+)', log)
        rtl_order = list(dict.fromkeys((int(l), int(e), int(t), 'gate_up' if m == '0' else 'down')
                                      for l, _, e, t, m, _, _ in observed_dma))
        observed_admission = re.findall(r'MAIN_ADMIT layer=(\d+) cycle=(\d+) expert=(\d+) tile=(\d+) matrix=(\d+) bytes=(\d+) class=(\d+)', log)
        rtl_admission_order = [(int(l), int(e), int(t), 'gate_up' if m == '0' else 'down')
                               for l, _, e, t, m, _, _ in observed_admission]
        main_admission_order = [(v['layer_id'], v['expert_id'], v['tile_id'], matrix)
            for v in result['trace'] if v['event_type'] == 'pair_admit' for matrix in ('gate_up', 'down')]
        main_order = [(v['layer_id'], v['expert_id'], v['tile_id'], v['matrix_type'])
                      for v in result['trace'] if v['event_type'] == 'dma_start']
        rtl_compute_order = [(int(l), int(e), int(t), 'gate_up' if s == 'GU_START' else 'down')
                             for l, _, e, t, s in stages if s in ('GU_START', 'DOWN_START')]
        main_compute_order = [(v['layer_id'], v['expert_id'], v['tile_id'], v['matrix_type']) for v in main_stages]
        with (directory / 'rtl_dma.csv').open('w', newline='', encoding='utf-8') as stream:
            writer = csv.writer(stream)
            writer.writerow(['layer', 'cycle_absolute', 'expert', 'tile', 'matrix', 'burst_bytes', 'original_class'])
            writer.writerows(observed_dma)
        write_json(directory / 'order_comparison.json', dict(rtl_dma_order=rtl_order, main_dma_order=main_order,
            rtl_admission_order=rtl_admission_order, main_admission_order=main_admission_order,
            rtl_compute_order=rtl_compute_order, main_compute_order=main_compute_order))
        byte_match = re.search(r'CROSS_BYTES allocated=(\d+) transferred=(\d+) useful=(\d+) wasted=(\d+) cancelled=(\d+)', log)
        rtl_bytes = list(map(int, byte_match.groups())) if byte_match else [0] * 5
        if case['future_mode'] and byte_match is None:
            raise ValueError('缺少 RTL speculative byte telemetry')
        main_bytes = [sum(v[field] for v in result['predictions']) for field in (
            'initial_allocated_speculative_bytes', 'speculative_bytes', 'useful_bytes', 'wasted_bytes', 'canceled_bytes')]
        peak_match = re.search(r'MAIN_OCM peak_slots=(\d+)', log)
        if peak_match is None:
            raise ValueError('缺少 RTL OCM peak 观测')
        rtl_peak_bytes = int(peak_match.group(1)) * config['fixture']['slot_bytes']
        gates = dict(dma_order_equal=rtl_order == main_order,
                     admitted_descriptor_order_equal=rtl_admission_order == main_admission_order,
                     compute_order_equal=rtl_compute_order == main_compute_order,
                     speculative_bytes_equal=rtl_bytes == main_bytes,
                     peak_ocm_allocated_bytes_equal=rtl_peak_bytes == result['peak_ocm_bytes'])
        predictor_observed = {int(l): int(c) - starts[0] for l, c, _ in re.findall(
            r'MAIN_PREDICTOR layer=(\d+) cycle=(\d+) expert=(\d+)', log)}
        predictor_expected = {v['target_layer_id']: v['predictor_ready_cycle'] for v in result['predictions']}
        gates['predictor_ready_cycles_equal'] = predictor_observed == predictor_expected
        if config['execution_rules'].get('burst_arbitration', False):
            rtl_bursts = [(int(l), int(e), int(t), 'gate_up' if m == '0' else 'down', int(size))
                          for l, _, e, t, m, size, _ in observed_dma]
            main_bursts = [(v['layer_id'], v['expert_id'], v['tile_id'], v['matrix_type'], v['burst_bytes'])
                           for v in result['trace'] if v['event_type'] == 'dma_burst_start']
            gates['burst_sequence_equal'] = rtl_bursts == main_bursts
        credit_match = re.search(r'MAIN_CREDITS descriptors=(\d+) demand=(\d+) promoted=(\d+) speculative=(\d+)', log)
        if credit_match is None:
            raise ValueError('缺少 RTL descriptor/FIFO 峰值观测')
        rtl_credit_peaks = list(map(int, credit_match.groups()))
        model_credit_peaks = [result['memory_metrics']['descriptor_peak'], *[
            result['memory_metrics']['fifo_occupancy_peaks'][k] for k in ('demand', 'promoted', 'speculative')]]
        resolved_input = json.loads((directory / 'shared_input.json').read_text())
        resolved_rules = resolved_input['execution']['execution_rules']
        generics = rtl_generics(case)
        bounds = [generics['DESCRIPTOR_ENTRIES'], *([generics['FIFO_DEPTH']] * 3)]
        if any(value > bound for value, bound in zip(rtl_credit_peaks, bounds)):
            raise ValueError('RTL 控制资源超出配置上限')
        for value, bound in zip(model_credit_peaks, [resolved_rules['descriptor_entries'], *([resolved_rules['dma_fifo_depth']] * 3)]):
            if bound and value > bound:
                raise ValueError('主模型控制资源超出配置上限')
        write_json(directory / 'resource_comparison.json', dict(gates=gates,
            byte_fields=['allocated', 'transferred', 'useful', 'wasted', 'canceled'],
            rtl_speculative_bytes=rtl_bytes, main_speculative_bytes=main_bytes,
            predictor_observed_cycles=predictor_observed, predictor_expected_cycles=predictor_expected,
            rtl_credit_peaks=rtl_credit_peaks, main_credit_peaks=model_credit_peaks,
            main_fifo_reserved_credit_peaks=result['memory_metrics']['fifo_reserved_credit_peaks'],
            main_fifo_occupancy_peaks=result['memory_metrics']['fifo_occupancy_peaks'],
            main_fifo_reserved_not_enqueued_peaks=result['memory_metrics']['fifo_reserved_not_enqueued_peaks'],
            fifo_comparison_semantics='physical_occupancy_not_reserved_credit',
            rtl_peak_ocm_allocated_bytes=rtl_peak_bytes, main_peak_ocm_allocated_bytes=result['peak_ocm_bytes']))
        fifo_rows=[]
        for snapshot in result['memory_metrics'].get('fifo_timeline', ()):
            for kind in ('demand','promoted','speculative'):
                fifo_rows.append(dict(cycle=snapshot['cycle'],queue=kind,
                    reserved_credits=snapshot['reserved_credits'][kind],
                    occupancy=snapshot['occupancy'][kind],
                    reserved_not_enqueued=snapshot['reserved_not_enqueued'][kind]))
        if fifo_rows:
            with (directory/'main_fifo_timeline.csv').open('w',newline='',encoding='utf-8') as stream:
                writer=csv.DictWriter(stream,fieldnames=list(fifo_rows[0]))
                writer.writeheader();writer.writerows(fifo_rows)
        retire_rows=re.findall(r'MAIN_RETIRE cycle=(\d+) layer=(\d+) state=(\d+) ingress=(\d+) ready=(\d+) allocator=(\d+) demand=(\d+) resolver=(\d+) compute=(\d+)',log)
        with (directory/'rtl_retirement.csv').open('w',newline='',encoding='utf-8') as stream:
            writer=csv.writer(stream)
            writer.writerow(['cycle_absolute','layer','state','ingress_idle','retire_ready','allocator_active','demand_active','resolver_ready','compute_busy'])
            writer.writerows(retire_rows)
        if not all(gates.values()) and config.get('strict_execution_rules', True):
            raise ValueError(f"执行规则观测验收失败：{case['name']}: {gates}")
        row = dict(case=case['name'], main_cycles=result['makespan_cycles'], rtl_cycles=rtl_cycles,
                   delta_cycles=rtl_cycles - result['makespan_cycles'],
                   **gates, execution_rule_alignment_passed=all(gates.values()),
                   main_descriptor_peak=result['memory_metrics']['descriptor_peak'],
                   main_credit_backpressure_count=result['memory_metrics']['credit_backpressure_count'],
                   control_credit_bounds_passed=True, credit_peaks_equal=rtl_credit_peaks == model_credit_peaks,
                   fifo_occupancy_peaks_equal=rtl_credit_peaks[1:] == model_credit_peaks[1:],
                   rtl_functional_checks_passed=True, resource_equivalence=False, alignment_passed=False)
        rows.append(row)
        with (directory / 'per_layer.csv').open('w', newline='', encoding='utf-8') as stream:
            writer = csv.writer(stream)
            writer.writerow(['layer', 'main_start', 'main_done', 'rtl_start_relative', 'rtl_done_relative'])
            for layer in range(case['layers']):
                model_layer = result['layer_results'][layer]
                writer.writerow([layer, model_layer['layer_start_cycle'], model_layer['layer_done_cycle'],
                                 starts[layer] - starts[0], ends[layer] - starts[0]])
        write_json(directory / 'comparison.json', row)
        print(row, flush=True)
    with (output / 'comparison.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    write_json(output / 'summary.json', dict(status='diagnostic_complete', simulator='UnifiedSequenceSimulator',
        rtl_functional_checks_passed=True, execution_rule_alignment_passed=all(v['execution_rule_alignment_passed'] for v in rows), alignment_passed=False, cases=rows,
        execution_rules=config.get('execution_rules', {}),
        unresolved=['FIFO token 退休和 descriptor admission 的接口流水仅作事件级抽象',
                    'AXI address/response 气泡与串行控制链的估计误差',
                    '误差尚不能外推到真实模型或板级性能'] ))
    if args.require_alignment:
        raise SystemExit('完整对齐尚未通过，参见 summary.json 的 unresolved')


if __name__ == '__main__':
    main()
