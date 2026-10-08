"""Run allocator × DMA-order comparisons on the main causal simulator."""
from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from edge_moe_sim.advanced_schema import AdvancedExecutionConfig
from edge_moe_sim.batch_predictor import BatchPredictorOutput, PrecomputedBatchPredictor, TokenPrediction
from edge_moe_sim.batch_routing import BatchRoutingEvent
from edge_moe_sim.execution_rules import ExecutionRules
from edge_moe_sim.hardware_profile import HardwareProfile
from edge_moe_sim.prefetch_config import PrefetchConfig
from edge_moe_sim.schema import CandidatePrefetch, ModelConfig
from edge_moe_sim.unified_schema import UnifiedLayerSequence, UnifiedPredictionSpec
from edge_moe_sim.unified_sequence_simulator import UnifiedSequenceSimulator


def main() -> None:
    """Emit reproducible comparison CSV and detailed JSON; numbers are synthetic."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('results/prefetch_allocator_order'))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    model = ModelConfig('synthetic_not_a_real_model', 4, 16, 8, 1, 8, 8)
    hardware = HardwareProfile('synthetic_not_measured', 1000, 2, 8000, 1, 1, 1,
                               32, True, 1, 4, 1, 1, 1024)
    sequence = UnifiedLayerSequence((BatchRoutingEvent(1, 0, 0, ((0,),)),
                                     BatchRoutingEvent(1, 0, 1, ((7,),))))
    output = BatchPredictorOutput(1, 0, 1, 0, (TokenPrediction(0, tuple(
        CandidatePrefetch(e, 1-.1*e) for e in range(8))),))
    predictor = {(0, 1): PrecomputedBatchPredictor({(1, 0, 1): output})}
    simulator = UnifiedSequenceSimulator(model, hardware)
    baseline = simulator.simulate(sequence, AdvancedExecutionConfig(4))
    rows = []
    for finite in (False, True):
        for allocator in ('uniform', 'ranked', 'hardware_greedy_v1'):
            for order in (('pair_first',) if finite else ('up_first', 'pair_first')):
                rules = (ExecutionRules(ocm_slot_bytes=64, atomic_tile_pair=True,
                    reserved_demand_slots=2, descriptor_entries=8, dma_fifo_depth=4,
                    retry_future_prefetch=True, request_order='pair_fifo', burst_arbitration=True)
                    if finite else ExecutionRules())
                config = AdvancedExecutionConfig(4, candidate_count=8, speculative_byte_budget=384,
                    execution_rules=rules, prefetch=PrefetchConfig(allocator=allocator, dma_order=order))
                result = simulator.simulate(sequence, config,
                    predictions=(UnifiedPredictionSpec(0, 1, candidate_count=8),),
                    predictors=predictor, trace_enabled=True)
                mode = 'finite_pair' if finite else 'byte_research'
                name = f'{mode}_{allocator}_{order}'
                (args.output/f'{name}.json').write_text(json.dumps({
                    'model': asdict(model), 'hardware': asdict(hardware),
                    'config': asdict(config), 'result': asdict(result)}, indent=2), encoding='utf-8')
                # Finite mode uses its own no-prefetch baseline: never mix resource rules.
                reference = simulator.simulate(sequence, AdvancedExecutionConfig(4, execution_rules=rules)) if finite else baseline
                record = result.predictions[0]
                rows.append(dict(mode=mode, allocator=allocator, dma_order=order,
                    total_cycles=result.makespan_cycles, baseline_cycles=reference.makespan_cycles,
                    speedup=reference.makespan_cycles/result.makespan_cycles,
                    planned_bytes=record.planned_speculative_bytes,
                    admitted_bytes=record.initial_allocated_speculative_bytes,
                    candidate_depths=json.dumps(record.planned_candidate_depths),
                    peak_ocm_bytes=result.peak_ocm_bytes))
    with (args.output/'comparison.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f'Generated {len(rows)} configurations in {args.output}')


if __name__ == '__main__':
    main()
