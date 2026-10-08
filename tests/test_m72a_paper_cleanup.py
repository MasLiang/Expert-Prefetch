"""Acceptance tests for M7.2a presentation-only corrections."""
from __future__ import annotations
import csv,sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'examples'))
import m72a_paper_cleanup as m72a

OUT=ROOT/'results/m72a_paper_figure_cleanup';SOURCE=ROOT/'results/m72_policy_phase'
def rows(name:str)->list[dict[str,str]]:
    """Read one generated M7.2a data file."""
    with (OUT/'data'/name).open() as stream:return list(csv.DictReader(stream))

def test_fig04_contains_only_unique_overall_best_bds_plans()->None:
    """Both Fig. 4 datasets exclude every non-best search candidate."""
    for name in ('fig04a_best_breadth_map.csv','fig04b_best_depth_map.csv'):
        data=rows(name);assert all(row['search_rank']=='1' and row['policy']=='breadth_depth_search' for row in data)
        assert len(data)==len({(row['comparison_id'],row['policy_plan_signature']) for row in data})

def test_latency_tie_and_preferred_policy_are_separate()->None:
    """Cost preference never erases the raw exact-latency policy set."""
    data=rows('m72a_corrected_winners.csv');tied=next(row for row in data if row['latency_tie']=='True')
    assert '|' in tied['raw_latency_winners'];assert tied['preferred_policy_under_tie'] in tied['raw_latency_winners']
    assert tied['strict_winner_margin_percent']=='0.0';assert tied['robust_latency_winner']=='False'

def test_robust_winner_requires_unique_and_one_percent_margin()->None:
    """Robust markers require a strict canonical-policy winner and >=1%."""
    for row in rows('m72a_corrected_winners.csv'):
        robust=row['robust_latency_winner']=='True'
        assert not robust or (row['latency_tie']=='False' and float(row['strict_winner_margin_percent'])>=1)

def test_channel_axis_uses_fixed_reference_bandwidth()->None:
    """Corrected channel opportunity removes the swept channel multiplier."""
    for row in rows('fig03a_policy_phase_lead_vs_channels.csv'):
        assert abs(float(row['reference_bandwidth_opportunity_expert_equivalents'])-float(row['opportunity'])/int(row['channels']))<1e-12

def test_vek_matrix_is_complete_explicit_and_canonical()->None:
    """VEK table has exactly one explicit row for every quality-policy cell."""
    data=rows('m72a_vek280_policy_matrix.csv');expected={(q,p) for q in ('perfect','medium','wrong') for p in ('No Prediction','Full','Uniform Prefix','Ranked Prefix','Best BDS')}
    assert {(row['predictor_quality'],row['canonical_policy']) for row in data}==expected;assert len(data)==15
    assert all(row['feasible'] in ('True','False') and 'infeasible_reason' in row for row in data)
    assert all(row['bds_search_rank']=='1' for row in data if row['canonical_policy']=='Best BDS')

def test_regret_excludes_infeasible_instead_of_zero_filling()->None:
    """Feasibility denominators and feasible-only regret samples stay separate."""
    summary={row['heuristic']:row for row in rows('m72a_static_policy_regret_summary.csv')};points=rows('m72a_static_policy_regret_points.csv')
    assert float(summary['Always Full']['feasible_fraction'])<1;assert float(summary['Always Prefix']['feasible_fraction'])<1
    assert len([row for row in points if row['heuristic']=='Always Full'])==int(summary['Always Full']['feasible_count'])

def test_search_tie_efficiency_contains_equal_latency_points()->None:
    """Resource savings are compared only when BDS and fixed latency match."""
    for row in rows('m72a_search_tie_efficiency.csv'):
        assert row['latency_tie']=='True';assert row['best_fixed_makespan']==row['search_makespan']

def test_plotting_cells_are_deterministic_and_original_is_separate()->None:
    """Phase plot rows are unique and M7.2a never aliases the M7.2 directory."""
    assert OUT.resolve()!=SOURCE.resolve()
    for name,keys in (('fig01a_policy_phase_quality_vs_opportunity_confidence.csv',('configuration_id',)),('fig02a_policy_phase_quality_vs_ocm_confidence.csv',('configuration_id',)),('fig03a_policy_phase_lead_vs_channels.csv',('configuration_id',)),('fig10a_vek280_theoretical_policy_comparison.csv',('predictor_quality','canonical_policy'))):
        data=rows(name);assert len(data)==len({tuple(row[key] for key in keys) for row in data})
