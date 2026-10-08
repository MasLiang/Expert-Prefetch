"""在真实数值 RTL 与主模拟器上比较整 Expert 启动和 tile 启动四 case。"""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import csv
import hashlib
import json
from pathlib import Path
import re
from typing import Any

import yaml

from compare_u200_default_alignment import PROJECT, ROOT, command_run
from compare_u200_alignment import write_json
from edge_moe_sim.advanced_schema import AdvancedExecutionConfig
from edge_moe_sim.batch_predictor import BatchPredictorOutput, PrecomputedBatchPredictor, TokenPrediction
from edge_moe_sim.batch_routing import BatchRoutingEvent
from edge_moe_sim.execution_rules import ExecutionRules
from edge_moe_sim.hardware_profile import HardwareProfile
from edge_moe_sim.schema import CandidatePrefetch, ModelConfig
from edge_moe_sim.unified_schema import UnifiedLayerSequence, UnifiedPredictionSpec
from edge_moe_sim.unified_sequence_simulator import UnifiedSequenceSimulator

CASES = ('A_whole_demand', 'B_stream_demand', 'C_whole_prefetch', 'D_stream_tile_prefetch')


def simulate(config: dict[str, Any], point: dict[str, Any], pattern: str,
             case: str, directory: Path) -> dict[str, Any]:
    """声明相同 slot/pair/预算/Engine 规则，不使用 RTL 总周期拟合模型。"""
    base = yaml.safe_load((PROJECT/config['alignment_base']).read_text())
    layers = config['layers']
    whole = case in (CASES[0], CASES[2])
    predict = case in (CASES[2], CASES[3])
    architecture = config.get('expert_architecture', 'swiglu')
    model = ModelConfig(f'u200_numeric_{architecture}_fixture', 16, 32, 6, 1, 8, 8,
                        expert_architecture=architecture)
    pair_bytes = (model.input_projection_count+1)*256
    routes = tuple(BatchRoutingEvent(0, 0, layer, ((layer % 4 + 1,),), ((1.0,),))
                   for layer in range(layers))
    rules = replace(ExecutionRules(**base['execution_rules']),
        descriptor_entries=config['descriptor_entries'], reserved_demand_descriptors=2,
        dma_fifo_depth=config['fifo_depth'], router_control_cycles=11,
        reserved_demand_slots=4 if whole else 2,
        wait_full_expert=whole)
    fields = dict(base['hardware'])
    fields['bandwidth_bytes_per_sec_per_channel'] = point['nominal_bytes_per_cycle_per_channel']*fields['clock_hz']
    hardware = HardwareProfile(**fields, ocm_bytes=2*config['slots_per_bank']*512)
    specs = []
    predictors = {}
    outputs = []
    for target in range(1, layers):
        expert = target % 4 + 1 if pattern == 'hit' else 5
        ready = 2+3*(target-1)
        output = BatchPredictorOutput(0, 0, target, ready,
            (TokenPrediction(0, (CandidatePrefetch(expert, 3000/4096),)),))
        outputs.append(output)
        if predict:
            specs.append(UnifiedPredictionSpec(0, target, 'source_layer_start',
                predictor_latency_cycles=ready, candidate_count=1,
                prefetch_policy='full_expert_prefetch' if whole else 'uniform_prefix',
                prefix_depth=1, wait_for_previous_layer_weights=False))
            predictors[(0,target)] = PrecomputedBatchPredictor({(0,0,target):output})
    execution = AdvancedExecutionConfig(16, placement_policy='striped_tile',
        max_parallel_experts=1, candidate_count=1, nonlinear_elements_per_cycle=1,
        speculative_byte_budget=2*pair_bytes if whole else pair_bytes, execution_rules=rules)
    result = asdict(UnifiedSequenceSimulator(model, hardware).simulate(
        UnifiedLayerSequence(routes), execution, predictions=tuple(specs),
        predictors=predictors, trace_enabled=True))
    for name, values in (('routing.jsonl', routes), ('predictor.jsonl', outputs)):
        (directory/name).write_text(''.join(json.dumps(asdict(v))+'\n' for v in values))
    write_json(directory/'shared_input.json', dict(model=asdict(model), hardware=asdict(hardware),
        execution=asdict(execution), prediction_enabled=predict))
    write_json(directory/'main_result.json', result)
    return result


def verify_rtl(text: str, layers: int, whole: bool,
               *, switch: bool = False, prefetch: bool = False) -> tuple[int, list[dict[str, int]]]:
    """验收 PASS、truth、GU/DOWN 操作集合及 whole 权重门槛，再提取绝对 cycle。"""
    if text.count('PASS:') != 1 or '$finish called' not in text or re.search(r'Fatal:|FATAL:|ERROR:|Error:', text):
        raise ValueError('RTL 数值/协议验收失败，不能生成性能 PASS')
    starts = {int(l):int(c) for l,c in re.findall(r'LAYER_START layer=(\d+) cycle=(\d+)',text)}
    ends = {int(l):int(c) for l,c in re.findall(r'LAYER_RESULT layer=(\d+) cycle=(\d+)',text)}
    truths = [(int(l),int(e),int(w)) for l,_,e,w in re.findall(
        r'MAIN_TRUTH layer=(\d+) cycle=(\d+) expert=(\d+) weight=(\d+)',text)]
    if truths != [(l,l%4+1,32768) for l in range(layers)] or len(starts)!=layers or len(ends)!=layers:
        raise ValueError('层数/真实 Router 结果不匹配')
    stages = re.findall(r'MAIN_STAGE layer=(\d+) cycle=(\d+) expert=(\d+) tile=(\d+) stage=(\w+)',text)
    observed = {(int(l),int(e),int(t),s) for l,_,e,t,s in stages}
    expected = {(l,l%4+1,t,s) for l in range(layers) for t in (0,1)
                for s in ('GU_START','GU_DONE','DOWN_START','DOWN_DONE')}
    if observed != expected or len(stages)!=len(expected):
        raise ValueError('compute stage 丢失或重复执行')
    ready = {int(l):int(c) for l,c in re.findall(r'WEIGHTS_READY layer=(\d+) cycle=(\d+)',text)}
    if whole and any(int(c)<ready.get(int(l),10**20) for l,c,_,t,s in stages if t=='0' and s=='GU_START'):
        raise ValueError('whole mode 在完整权重到达之前启动')
    admissions = re.findall(r'MAIN_ADMIT layer=\d+ cycle=\d+ expert=\d+ tile=\d+ matrix=(\d+) bytes=(\d+) class=(\d+)',text)
    if not admissions or any(int(b)!=(256 if switch or m=='1' else 512) for m,b,_ in admissions):
        raise ValueError('Expert mode/descriptor payload bytes 不一致或未观察到 admission')
    if prefetch and not any(kind=='0' for _,_,kind in admissions):
        raise ValueError('预取 case 没有实际 speculative admission，不能验收性能')
    return ends[layers-1]-starts[0], [dict(layer=l,start=starts[l],end=ends[l],cycles=ends[l]-starts[l]) for l in range(layers)]


def main() -> None:
    """运行有界 XSim；只报告观察到的趋势一致性，不强行要求数值一致。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=PROJECT/'configs/hardware/u200_four_case_validation.yaml')
    parser.add_argument('--output',type=Path,default=PROJECT/'results/m10_four_cases_promotion_fixed')
    args = parser.parse_args()
    args.config = args.config.resolve()
    config = yaml.safe_load(args.config.read_text())
    architecture = config.get('expert_architecture', 'swiglu')
    if architecture not in ('swiglu', 'relu_ffn'):
        raise ValueError('expert_architecture must be swiglu or relu_ffn')
    switch = architecture == 'relu_ffn'
    pair_bytes = 512 if switch else 768
    if config['route_k']!=1 or not 2<=config['layers']<=3 or config['slots_per_bank']<4:
        raise ValueError('此数值夹具支持 Top-1、2–3 层并要求 OCM 能容纳整 Expert')
    output = args.output.resolve()
    build = output/'rtl_build'
    build.mkdir(parents=True,exist_ok=True)
    write_json(output/'summary.json',dict(status='running',passed=False))
    package = ROOT/'rtl/package.sv'
    sources = [package,*sorted(p for p in (ROOT/'rtl').glob('*.sv') if p!=package),
        ROOT/'tb/tb_u200_moe.sv',ROOT/'tb/tb_active_dma_promotion.sv']
    if switch:
        sources.extend([ROOT/'tb/tb_switch_relu_backend.sv', ROOT/'tb/tb_mac_chunk_pipeline.sv'])
    write_json(output/'manifest.json',dict(config=config,backend='XSim 2023.2 behavioral',
        expert_architecture=architecture,
        limitations=['Small H16/F32 numeric fixture, not a full pretrained model or FLAME reproduction',
                    '4-bank AXI behavioral responder, not DDR4 PHY or board measurement',
                    'Pair prefetch includes input projection and DOWN; D depth=1, C depth=2'],
        source_sha256={str(p.relative_to(PROJECT)):hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [*sources,args.config,PROJECT/config['alignment_base'],Path(__file__),
                *sorted((PROJECT/'src/edge_moe_sim').rglob('*.py'))]}))
    command_run(['xvlog','--sv',*map(str,sources)],build,build/'compile.log')
    rows = []
    if switch:
        checks = [('switch_dense', 'tb_switch_relu_backend', {}),
                  ('switch_mac_48', 'tb_mac_chunk_pipeline', dict(SWITCH_RELU=1,H=48,F_TILE=48)),
                  ('switch_mac_16', 'tb_mac_chunk_pipeline', dict(SWITCH_RELU=1,H=16,F_TILE=32))]
        for snapshot, top, generics in checks:
            directory = output/snapshot
            directory.mkdir(exist_ok=True)
            cmd = ['xelab','--debug','off','--mt','4','--snapshot',snapshot]
            for key,value in generics.items():
                cmd += ['--generic_top',f'{key}={value}']
            command_run([*cmd,f'work.{top}'],build,directory/'elaborate.log')
            batch = directory/'run.tcl'
            batch.write_text('run 1 ms\nquit\n')
            command_run(['xsim',snapshot,'--tclbatch',str(batch),'--onerror','quit','--onfinish','quit',
                         '--log',str(directory/'simulation.log')],build,directory/'console.log')
            text = (directory/'simulation.log').read_text()
            if text.count('PASS:')!=1 or '$finish called' not in text or re.search(r'Fatal:|FATAL:|ERROR:',text):
                raise ValueError(f'Switch numerical/protocol test failed: {snapshot}')
            print(f'{snapshot}: PASS',flush=True)
    for phase in range(4):
        directory = output/f'promotion_phase_{phase}'
        directory.mkdir(exist_ok=True)
        snapshot = f'active_promotion_{phase}'
        command_run(['xelab','--debug','off','--mt','4','--generic_top',f'PHASE={phase}',
            '--snapshot',snapshot,'work.tb_active_dma_promotion'],build,directory/'elaborate.log')
        batch = directory/'run.tcl'
        batch.write_text('run 20 us\nquit\n')
        command_run(['xsim',snapshot,'--tclbatch',str(batch),'--onerror','quit','--onfinish','quit',
                     '--log',str(directory/'simulation.log')],build,directory/'console.log')
        text = (directory/'simulation.log').read_text()
        if text.count('PASS:')!=1 or '$finish called' not in text or re.search(r'Fatal:|FATAL:|ERROR:',text):
            raise ValueError(f'promotion boundary test failed phase={phase}')
    for point in config['bandwidth_points']:
        for pattern in config['prediction_patterns']:
            for case in CASES:
                name = f"{point['name']}_{pattern}_{case}"
                directory = output/name
                directory.mkdir(exist_ok=True)
                result = simulate(config,point,pattern,case,directory)
                whole = case in (CASES[0],CASES[2])
                generics = dict(LAYERS=config['layers'],ROUTE_K=1,BUDGET=0,DDR_GAPS=0,
                    DDR_BEAT_GAP=point['ddr_beat_gap'],WAIT_FULL_EXPERT=int(whole),
                    SWITCH_RELU=int(switch),
                    FUTURE_MODE=(1 if pattern=='hit' else 2) if case in CASES[2:] else 0,
                    FUTURE_BUDGET_BYTES=2*pair_bytes if whole else pair_bytes,
                    OCM_SLOTS_PER_BANK=config['slots_per_bank'],DESCRIPTOR_ENTRIES=config['descriptor_entries'],
                    FIFO_DEPTH=config['fifo_depth'],EXPECT_EARLY_OVERLAP=0)
                snapshot = name
                cmd = ['xelab','--debug','off','--mt','4','--snapshot',snapshot]
                for key,value in generics.items(): cmd += ['--generic_top',f'{key}={value}']
                command_run([*cmd,'work.tb_u200_moe'],build,directory/'elaborate.log')
                batch = directory/'run.tcl'
                batch.write_text('run 1 ms\nquit\n')
                command_run(['xsim',snapshot,'--testplusarg','ALIGN_MAIN','--tclbatch',str(batch),
                    '--onerror','quit','--onfinish','quit','--log',str(directory/'simulation.log')],
                    build,directory/'console.log')
                cycles,per_layer = verify_rtl((directory/'simulation.log').read_text(),config['layers'],whole,
                                             switch=switch,prefetch=case in CASES[2:])
                write_json(directory/'rtl_result.json',dict(cycles=cycles,per_layer=per_layer,generics=generics))
                rows.append(dict(bandwidth=point['name'],prediction=pattern,case=case,
                    main_cycles=result['makespan_cycles'],rtl_cycles=cycles,
                    main_speculative_bytes=result['memory_metrics']['speculative_bytes'],
                    main_wasted_bytes=result['memory_metrics']['wasted_speculative_bytes']))
                print(f'{name}: PASS main={result["makespan_cycles"]} RTL={cycles}',flush=True)
    lookup = {(r['bandwidth'],r['prediction'],r['case']):r for r in rows}
    for row in rows:
        for backend in ('main','rtl'):
            for label,base in (('A',CASES[0]),('B',CASES[1])):
                row[f'{backend}_speedup_vs_{label}'] = lookup[(row['bandwidth'],row['prediction'],base)][f'{backend}_cycles']/row[f'{backend}_cycles']
        row['trend_vs_A_matches'] = (row['main_speedup_vs_A']>1)==(row['rtl_speedup_vs_A']>1)
        row['trend_vs_B_matches'] = (row['main_speedup_vs_B']>1)==(row['rtl_speedup_vs_B']>1)
    with (output/'comparison.csv').open('w',newline='') as stream:
        writer = csv.DictWriter(stream,fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    write_json(output/'summary.json',dict(status='complete',passed=True,
        functional_passed=True,promotion_cases=4,performance_cases=len(rows),cases=rows,
        expert_architecture=architecture,switch_backend_cases=3 if switch else 0,
        all_trends_match=all(r['trend_vs_A_matches'] and r['trend_vs_B_matches'] for r in rows),
        limitations=['Trend comparison only; no fitted makespan compensation or board timing claim']))


if __name__=='__main__':
    main()
