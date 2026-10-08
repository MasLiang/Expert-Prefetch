"""Tests for the M7.2a zero-simulation paper-review figure collection."""
from __future__ import annotations
import csv,json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results/m72a_paper_figure_cleanup/paper_review_figures'
def read(name:str)->list[dict[str,str]]:
    """Read one exact plotting CSV."""
    with (OUT/'data'/name).open() as stream:return list(csv.DictReader(stream))

def test_review_generation_records_zero_simulations()->None:
    """The review pass is a CSV-only regeneration."""
    summary=json.loads((OUT/'generation_summary.json').read_text());assert summary['new_simulations']==0
    source=(ROOT/'examples/m72a_paper_review_figures.py').read_text();assert 'UnifiedSequenceSimulator' not in source and 'm72_policy_phase' not in source

def test_review_manifest_files_are_complete()->None:
    """Every review entry has a non-empty PNG, PDF, and plotting CSV."""
    manifest=json.loads((OUT/'figure_manifest_paper_review.json').read_text());assert len(manifest)==9
    assert all(Path(item[key]).stat().st_size>100 for item in manifest for key in ('output_png','output_pdf','plotting_csv'))

def test_review_phase_confidence_and_channel_axis()->None:
    """Phase rows preserve confidence classes and fixed-reference opportunity."""
    phase=read('fig01a_policy_phase_quality_vs_opportunity_confidence.csv');assert {'True','False'}=={row['latency_tie'] for row in phase}
    channel=read('fig03a_policy_phase_lead_vs_channels.csv');assert all(abs(float(r['reference_bandwidth_opportunity_expert_equivalents'])-float(r['opportunity'])/int(r['channels']))<1e-12 for r in channel)

def test_review_fig04_uses_only_canonical_best_rows()->None:
    """Breadth and depth maps use the exact same 89 rank-1 plans."""
    breadth=read('fig04a_best_breadth_map.csv');depth=read('fig04b_best_depth_map.csv');assert breadth==depth;assert len(breadth)==89 and all(r['search_rank']=='1' for r in breadth);assert not any(r['candidate_count']=='16' for r in breadth)

def test_review_regret_and_vek_sources_are_complete()->None:
    """Regret excludes infeasible cells and VEK is exactly 3×5."""
    regret=read('fig09b_static_policy_regret_cdf.csv');assert len([r for r in regret if r['heuristic']=='Always Full'])==193
    vek=read('fig10a_vek280_theoretical_policy_comparison.csv');assert len(vek)==15 and len({(r['predictor_quality'],r['canonical_policy']) for r in vek})==15

def test_review_tie_efficiency_is_same_latency_only()->None:
    """Figure 12 contains only the 155 exact BDS/fixed latency ties."""
    ties=read('fig12_search_tie_resource_efficiency.csv');assert len(ties)==155 and all(r['best_fixed_makespan']==r['search_makespan'] for r in ties)
