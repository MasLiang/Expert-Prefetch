"""Validate M7.2b real opportunity refinement artifacts."""
from __future__ import annotations
import argparse,csv,json
from pathlib import Path
def read(path:Path)->list[dict[str,str]]:
    """Read a required non-empty CSV."""
    assert path.is_file() and path.stat().st_size>0,path
    with path.open() as stream:rows=list(csv.DictReader(stream))
    assert rows;return rows
def validate(root:Path)->dict[str,object]:
    """Assert point count, real timing, winner rows, and bounded run count."""
    points=read(root/'m72b_opportunity_points.csv');evaluations=read(root/'m72b_policy_evaluations.csv');winners=read(root/'m72b_policy_winners.csv');assert len(points)==3 and len(evaluations)==540 and len(winners)==15
    assert all(row['synthetic_lead_time']=='False' and .381<float(row['opportunity_expert_equivalents'])<4.546 for row in points)
    assert len({(row['predictor_quality'],row['predictor_latency_cycles']) for row in winners})==15
    summary=json.loads((root/'run_summary.json').read_text());assert summary['new_simulator_runs']==543 and not summary['full_m72_search_rerun']
    for stem in ('fig_m72b_quality_vs_opportunity_refined','fig_m72b_speedup_vs_opportunity'):
        for suffix in ('.csv','.png','.pdf'):assert (root/(stem+suffix)).stat().st_size>100
    return {'status':'passed','opportunity_points':[float(row['opportunity_expert_equivalents']) for row in points],'policy_rows':len(evaluations),'winner_rows':len(winners),'new_simulator_runs':543}
def main()->int:
    """CLI entry point."""
    parser=argparse.ArgumentParser();parser.add_argument('--input-dir',type=Path,required=True);args=parser.parse_args();print(json.dumps(validate(args.input_dir),indent=2));return 0
if __name__=='__main__':raise SystemExit(main())
