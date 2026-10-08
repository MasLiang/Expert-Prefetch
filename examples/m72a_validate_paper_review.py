"""Validate the zero-simulation M7.2a paper-review figure collection."""
from __future__ import annotations
import argparse,csv,json
from collections import Counter
from pathlib import Path

def rows(path:Path)->list[dict[str,str]]:
    """Read a required review plotting CSV."""
    assert path.is_file() and path.stat().st_size>0,path
    with path.open() as stream:data=list(csv.DictReader(stream))
    assert data,path;return data

def validate(root:Path)->dict[str,object]:
    """Assert figure files, source filters, matrix shape, and zero simulations."""
    manifest=json.loads((root/'figure_manifest_paper_review.json').read_text());assert len(manifest)==9
    for item in manifest:
        for key in ('output_png','output_pdf','plotting_csv'):
            path=Path(item[key]);assert path.is_file() and path.stat().st_size>100,path
    summary=json.loads((root/'generation_summary.json').read_text());assert summary['new_simulations']==0 and summary['global_confidence']=={'total':240,'robust':46,'near':69,'exact_tie':125}
    breadth=rows(root/'data/fig04a_best_breadth_map.csv');depth=rows(root/'data/fig04b_best_depth_map.csv');assert breadth==depth and all(row['search_rank']=='1' for row in breadth);assert len(breadth)==len({row['configuration_id'] for row in breadth})==89;assert Counter(map(lambda row:int(row['candidate_count']),breadth))==Counter({1:35,2:22,4:30,8:2})
    channel=rows(root/'data/fig03a_policy_phase_lead_vs_channels.csv');assert all(abs(float(row['reference_bandwidth_opportunity_expert_equivalents'])-float(row['opportunity'])/int(row['channels']))<1e-12 for row in channel)
    vek=rows(root/'data/fig10a_vek280_theoretical_policy_comparison.csv');assert len(vek)==15 and len({(row['predictor_quality'],row['canonical_policy']) for row in vek})==15
    ties=rows(root/'data/fig12_search_tie_resource_efficiency.csv');assert len(ties)==155 and all(row['best_fixed_makespan']==row['search_makespan'] for row in ties)
    inventory=rows(root/'source_figure_inventory.csv');assert len(inventory)==10 and all(row[f'{kind}_exists']=='True' and int(row[f'{kind}_bytes'])>0 for row in inventory for kind in ('plotting_csv','png','pdf'))
    return {'status':'passed','review_figures':len(manifest),'new_simulations':0,'fig04_rows':len(breadth),'vek_rows':len(vek),'latency_tie_rows':len(ties)}

def main()->int:
    """CLI entry point."""
    parser=argparse.ArgumentParser();parser.add_argument('--input-dir',type=Path,required=True);args=parser.parse_args();result=validate(args.input_dir);print(json.dumps(result,indent=2));return 0
if __name__=='__main__':raise SystemExit(main())
