"""Acceptance tests for the bounded M7.2b real-timing refinement."""
from __future__ import annotations
import csv,json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results/m72b_opportunity_refinement'
def rows(name:str)->list[dict[str,str]]:
    """Read one M7.2b CSV."""
    with (OUT/name).open() as stream:return list(csv.DictReader(stream))

def test_m72b_points_are_real_and_inside_old_gap()->None:
    """All three controls use genuine unified source-router timing."""
    points=rows('m72b_opportunity_points.csv');assert len(points)==3
    assert all(row['source_milestone']=='source_router_done' and row['synthetic_lead_time']=='False' for row in points)
    assert all(.381<float(row['opportunity_expert_equivalents'])<4.546 for row in points)

def test_m72b_workload_hardware_and_rankings_match_m72()->None:
    """Refinement changes predictor latency, not workload/hardware/ranking."""
    new=rows('m72b_policy_winners.csv');old=rows('../m72_policy_phase/raw/m72_policy_evaluations.csv')
    assert {row['workload_trace_id'] for row in new}=={'m72-controlled-3layer-k4-b1-high_reuse'}
    assert {row['hardware_mode'] for row in new}=={'controlled_synthetic_hardware'}
    for quality in ('perfect','high','medium','low','wrong'):
        assert {row['predictor_trace_id'] for row in new if row['predictor_quality']==quality}=={next(row['predictor_trace_id'] for row in old if row['predictor_quality']==quality)}

def test_m72b_policy_rows_are_annotated_and_no_synthetic_override()->None:
    """Every feasible row carries point timing and canonical winner semantics."""
    data=rows('m72b_policy_evaluations.csv');assert len(data)==540
    assert all(row['synthetic_lead_time']=='False' for row in data)
    assert all(row['raw_latency_winners'] and row['preferred_policy_under_tie'] and row['point_realized_lead_cycles'] for row in data)

def test_m72b_winner_is_minimum_canonical_makespan()->None:
    """Winner is selected from no/full/ranked/best-BDS by sequence makespan."""
    data=rows('m72b_policy_evaluations.csv')
    for winner in rows('m72b_policy_winners.csv'):
        group=[row for row in data if row['configuration_id']==winner['configuration_id'] and row['feasible']=='True'];bds=min((row for row in group if row['policy']=='breadth_depth_search'),key=lambda row:int(row['search_rank']));canonical=[row for row in group if row['policy']!='breadth_depth_search']+[bds]
        assert int(winner['preferred_makespan'])==min(int(row['sequence_makespan']) for row in canonical)
        assert winner['robust_latency_winner']!='True' or (winner['latency_tie']=='False' and float(winner['strict_winner_margin_percent'])>=1)

def test_m72b_figures_and_run_budget()->None:
    """Only the prescribed refinement matrix ran and both figures are non-empty."""
    summary=json.loads((OUT/'run_summary.json').read_text());assert summary['new_simulator_runs']==543 and not summary['full_m72_search_rerun']
    for stem in ('fig_m72b_quality_vs_opportunity_refined','fig_m72b_speedup_vs_opportunity'):
        for suffix in ('.png','.pdf','.csv'):assert (OUT/(stem+suffix)).stat().st_size>100
