"""Validate M7.2a corrected datasets and figure provenance."""
from __future__ import annotations
import argparse,csv,json
from pathlib import Path

def read(path:Path)->list[dict[str,str]]:
    """Read a required non-empty CSV."""
    assert path.is_file() and path.stat().st_size>0,path
    with path.open() as stream:rows=list(csv.DictReader(stream))
    assert rows,path;return rows

def validate(root:Path)->dict[str,object]:
    """Assert required M7.2a files and corrected semantics."""
    required=('m72a_corrected_winners.csv','m72a_best_bds_plans.csv','m72a_search_gain_summary.csv','m72a_search_tie_efficiency.csv','m72a_phase_confidence_summary.csv','m72a_vek280_policy_matrix.csv','m72a_static_policy_regret_summary.csv','opportunity_sampling_report.csv','ocm_phase_interpretation.csv','memory_channel_slack_summary.csv')
    data={name:read(root/'data'/name) for name in required};best=data['m72a_best_bds_plans.csv'];assert all(row['search_rank']=='1' for row in best)
    matrix=data['m72a_vek280_policy_matrix.csv'];assert len(matrix)==15 and len({(r['predictor_quality'],r['canonical_policy']) for r in matrix})==15
    manifest=json.loads((root/'figure_manifest_m72a.json').read_text());assert len(manifest)==10
    for item in manifest:
        for key in ('png','pdf','plotting_csv'):
            path=Path(item[key]);assert path.is_file() and path.stat().st_size>0
    fingerprints=json.loads((root/'diagnostics/source_m72_sha256.json').read_text());assert fingerprints['unchanged'] and fingerprints['before']==fingerprints['after']
    summary=json.loads((root/'summaries/run_summary.json').read_text());assert summary['new_simulations_total']<=12 and not summary['full_m72_search_rerun']
    return {'status':'passed','figures':len(manifest),'best_bds_rows':len(best),'vek_matrix_rows':len(matrix),'new_simulations_total':summary['new_simulations_total']}

def main()->int:
    """CLI entry point."""
    parser=argparse.ArgumentParser();parser.add_argument('--input-dir',type=Path,required=True);args=parser.parse_args();result=validate(args.input_dir);print(json.dumps(result,indent=2));return 0
if __name__=='__main__':raise SystemExit(main())
