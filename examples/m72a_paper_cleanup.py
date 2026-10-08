"""M7.2a CSV-first paper-figure correction without rerunning the main DSE."""
from __future__ import annotations

import argparse, ast, csv, hashlib, json, math, sys
from collections import defaultdict
from pathlib import Path
from statistics import fmean, median

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "examples")]
import m72_policy_phase as m72

SOURCE = ROOT / "results/m72_policy_phase"
OUTPUT = ROOT / "results/m72a_paper_figure_cleanup"
LABEL = {"no_prediction":"NO_PREFETCH", "full_expert_prefetch":"FULL_EXPERT",
         "uniform_prefix":"UNIFORM_PREFIX", "ranked_prefix":"RANKED_PREFIX",
         "breadth_depth_search":"BREADTH_DEPTH_SEARCH"}


def read_csv(path: Path) -> list[dict[str, str]]:
    """Read a CSV as deterministic string-valued rows."""
    with path.open(encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    """Write rows using the union of fields and fail on an empty dataset."""
    if not rows:
        raise ValueError(f"refusing to write empty M7.2a dataset: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)


def percentile(values: list[float], fraction: float) -> float:
    """Return a deterministic linearly interpolated percentile."""
    ordered = sorted(values); position = (len(ordered) - 1) * fraction
    lo = math.floor(position); hi = math.ceil(position)
    return ordered[lo] if lo == hi else ordered[lo] + (ordered[hi] - ordered[lo]) * (position - lo)


def corrected_winners(evaluations: list[dict[str, str]], original: list[dict[str, str]]) -> list[dict[str, object]]:
    """Separate exact latency winners from deterministic cost preference."""
    by_id: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in evaluations: by_id[row["comparison_id"]].append(row)
    old = {row["comparison_id"]: row for row in original}; result=[]
    for comparison_id, rows in sorted(by_id.items()):
        bds=[row for row in rows if row['policy']=='breadth_depth_search']
        canonical=[row for row in rows if row['policy']!='breadth_depth_search']
        if bds:canonical.append(min(bds,key=lambda row:(int(row['sequence_makespan']),int(row['search_rank']))))
        minimum = min(int(row["sequence_makespan"]) for row in canonical)
        raw = [row for row in canonical if int(row["sequence_makespan"]) == minimum]
        distinct = sorted({int(row["sequence_makespan"]) for row in canonical})
        second = distinct[1] if len(distinct) > 1 else None
        preferred = min(canonical, key=lambda row: int(row["search_rank"]))
        unique = len(raw) == 1
        margin = ((second / minimum - 1) * 100) if unique and second is not None else 0.0
        policies = sorted({LABEL[row["policy"]] for row in raw})
        record = dict(old[comparison_id])
        record.update(raw_latency_winners="|".join(policies), raw_latency_winner_plan_count=len(raw),
                      latency_tie=not unique, second_best_distinct_latency=second if second is not None else "",
                      strict_winner_margin_percent=margin,
                      robust_latency_winner=unique and margin >= 1.0,
                      preferred_policy_under_tie=LABEL[preferred["policy"]],
                      preferred_policy_plan_signature=preferred["policy_plan_signature"],
                      all_feasible_policies_within_1pct=all(int(row["sequence_makespan"])/minimum <= 1.01 for row in canonical),
                      all_feasible_policies_within_5pct=all(int(row["sequence_makespan"])/minimum <= 1.05 for row in canonical))
        result.append(record)
    return result


def best_bds_rows(evaluations: list[dict[str, str]], winners: list[dict[str, object]]) -> list[dict[str, object]]:
    """Return only overall rank-1 BDS plans where BDS is cost-preferred."""
    winner = {str(row["comparison_id"]): row for row in winners}
    rows=[]
    for source in evaluations:
        if source["comparison_mode"] != "unconstrained" or source["policy"] != "breadth_depth_search" or int(source["search_rank"]) != 1:
            continue
        win = winner[source["comparison_id"]]
        if win["preferred_policy_under_tie"] != "BREADTH_DEPTH_SEARCH": continue
        row=dict(source); row.update(latency_tie=win["latency_tie"], robust_latency_winner=win["robust_latency_winner"], strict_winner_margin_percent=win["strict_winner_margin_percent"])
        rows.append(row)
    return rows


def search_analysis(evaluations: list[dict[str, str]]) -> tuple[list[dict[str, object]], list[dict[str, object]], dict[str, object]]:
    """Compare best BDS to the best feasible fixed predictive policy."""
    groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in evaluations:
        if row["comparison_mode"] == "unconstrained": groups[row["comparison_id"]].append(row)
    gains=[]; ties=[]
    for comparison_id, rows in sorted(groups.items()):
        fixed=[row for row in rows if row["policy"] in ("full_expert_prefetch","uniform_prefix","ranked_prefix")]
        search=[row for row in rows if row["policy"] == "breadth_depth_search"]
        if not fixed or not search: continue
        f=min(fixed,key=lambda row:(int(row["sequence_makespan"]),int(row["search_rank"])))
        s=min(search,key=lambda row:(int(row["sequence_makespan"]),int(row["search_rank"])))
        fm,sm=int(f["sequence_makespan"]),int(s["sequence_makespan"]); tie=fm==sm
        row={"configuration_id":f["configuration_id"],"comparison_id":comparison_id,"best_fixed_policy":f["policy"],"best_fixed_makespan":fm,"search_makespan":sm,"search_improvement_cycles":fm-sm,"search_improvement_percent":(fm/sm-1)*100,"latency_tie":tie,"allocation_delta_bytes":int(s["initial_allocated_speculative_bytes"])-int(f["initial_allocated_speculative_bytes"]),"physical_traffic_delta_bytes":int(s["physical_speculative_bytes_transferred_by_router"])-int(f["physical_speculative_bytes_transferred_by_router"]),"waste_delta_bytes":int(s["wasted_physical_speculative_bytes"])-int(f["wasted_physical_speculative_bytes"]),"peak_ocm_delta_bytes":int(s["peak_ocm_bytes"])-int(f["peak_ocm_bytes"]),"quality":f["predictor_quality"],"opportunity":f["configuration_opportunity_expert_equivalents"]}
        gains.append(row)
        if tie: ties.append(dict(row,bds_allocated_bytes=s["initial_allocated_speculative_bytes"],fixed_allocated_bytes=f["initial_allocated_speculative_bytes"],bds_issued_bytes=s["issued_speculative_bytes"],fixed_issued_bytes=f["issued_speculative_bytes"],bds_physical_bytes=s["physical_speculative_bytes_transferred_by_router"],fixed_physical_bytes=f["physical_speculative_bytes_transferred_by_router"],bds_waste_bytes=s["wasted_physical_speculative_bytes"],fixed_waste_bytes=f["wasted_physical_speculative_bytes"],bds_peak_ocm=s["peak_ocm_bytes"],fixed_peak_ocm=f["peak_ocm_bytes"]))
    wins=[row for row in gains if row["search_improvement_cycles"]>0]
    reductions=[-int(row["allocation_delta_bytes"]) for row in ties if int(row["allocation_delta_bytes"])<0]
    summary={"configurations_compared":len(gains),"strict_bds_wins":len(wins),"strict_bds_wins_gt_1pct":sum(float(row["search_improvement_percent"])>1 for row in wins),"strict_bds_wins_gt_5pct":sum(float(row["search_improvement_percent"])>5 for row in wins),"median_gain_percent_among_wins":median(float(row["search_improvement_percent"]) for row in wins),"max_gain_percent":max(float(row["search_improvement_percent"]) for row in wins),"latency_ties":len(ties),"ties_with_lower_allocation":sum(int(row["allocation_delta_bytes"])<0 for row in ties),"tie_fraction_lower_allocation":sum(int(row["allocation_delta_bytes"])<0 for row in ties)/len(ties),"median_allocated_byte_reduction_when_lower":median(reductions),"tie_fraction_lower_peak_ocm":sum(int(row["peak_ocm_delta_bytes"])<0 for row in ties)/len(ties),"tie_fraction_lower_physical_traffic":sum(int(row["physical_traffic_delta_bytes"])<0 for row in ties)/len(ties)}
    return gains,ties,summary


def regret_data(evaluations: list[dict[str, str]]) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Keep heuristic feasibility separate from feasible-only regret samples."""
    groups: dict[str,list[dict[str,str]]] = defaultdict(list)
    for row in evaluations:
        if row["comparison_mode"] == "unconstrained": groups[row["comparison_id"]].append(row)
    mapping=(('Always Full','full_expert_prefetch'),('Always Prefix','ranked_prefix'),('Always No Prefetch','no_prediction'))
    points=[];summary=[]
    for label,policy in mapping:
        feasible=[]
        for comparison_id,rows in groups.items():
            best=min(int(row["sequence_makespan"]) for row in rows);match=next((row for row in rows if row["policy"]==policy),None)
            if match is not None:
                regret=(int(match["sequence_makespan"])/best-1)*100;feasible.append(regret);points.append({"heuristic":label,"comparison_id":comparison_id,"regret_percent":regret})
        summary.append({"heuristic":label,"feasible_count":len(feasible),"total_configurations":len(groups),"feasible_fraction":len(feasible)/len(groups),"infeasible_fraction":1-len(feasible)/len(groups),"median_regret_percent":median(feasible),"p90_regret_percent":percentile(feasible,.90),"p95_regret_percent":percentile(feasible,.95),"max_regret_percent":max(feasible)})
    return points,summary


def complete_vek_matrix(source_rows: list[dict[str,str]]) -> tuple[list[dict[str,object]],int]:
    """Reuse cached VEK rows and simulate only absent simple-policy cells."""
    qualities=('perfect','medium','wrong'); simple=('no_prediction','full_expert_prefetch','uniform_prefix','ranked_prefix'); new=0;rows=list(source_rows)
    for quality in qualities:
        exemplar=next(row for row in rows if row["predictor_quality"]==quality)
        cfg={"configuration_id":exemplar["configuration_id"],"milestone":exemplar["source_milestone"],"latency":int(exemplar["predictor_latency_cycles"]),"quality":quality,"true_ranks":tuple(ast.literal_eval(exemplar["true_expert_rank_vector"])),"channels":int(exemplar["memory_channels"]),"ocm":int(exemplar["ocm_bytes"]),"top_k":int(exemplar["top_k"]),"batch_size":int(exemplar["batch_size"]),"routing_shape":exemplar.get("routing_shape","high_reuse"),"placement_policy":exemplar.get("memory_placement_policy","round_robin_expert"),"opportunity_setting":exemplar["opportunity_setting"],"opportunity":float(exemplar["configuration_opportunity_expert_equivalents"]),"quality_scalar":float(exemplar["predictor_quality_scalar"]),"demand_reference":int(exemplar["demand_working_set_reference_bytes"])}
        for policy in simple:
            if any(row["predictor_quality"]==quality and row["policy"]==policy for row in rows): continue
            new+=1
            try: row,_=m72.evaluate(cfg,policy,16,(),None,'vek280_theoretical',False,'unconstrained')
            except RuntimeError as exc:
                rows.append({"predictor_quality":quality,"policy":policy,"feasible":False,"infeasible_reason":str(exc),"hardware_evidence":"theoretical_not_measured"});continue
            row.update(feasible=True,infeasible_reason="",hardware_evidence="theoretical_not_measured");rows.append({key:str(value) for key,value in row.items()})
    matrix=[]
    for quality in qualities:
        no=next(row for row in rows if row["predictor_quality"]==quality and row["policy"]=="no_prediction")
        baseline=int(no["sequence_makespan"])
        for canonical,policy in (("No Prediction","no_prediction"),("Full","full_expert_prefetch"),("Uniform Prefix","uniform_prefix"),("Ranked Prefix","ranked_prefix")):
            matches=[row for row in rows if row["predictor_quality"]==quality and row["policy"]==policy];source=matches[0]
            feasible=str(source.get("feasible","True")).lower()!="false";matrix.append(dict(source,canonical_policy=canonical,feasible=feasible,infeasible_reason=source.get("infeasible_reason",""),speedup_vs_no_prediction=baseline/int(source["sequence_makespan"]) if feasible else ""))
        bds=min((row for row in rows if row["predictor_quality"]==quality and row["policy"]=="breadth_depth_search"),key=lambda row:int(row["sequence_makespan"]))
        matrix.append(dict(bds,canonical_policy="Best BDS",feasible=True,infeasible_reason="",bds_search_rank=1,speedup_vs_no_prediction=baseline/int(bds["sequence_makespan"])))
    return matrix,new


def main() -> int:
    """Generate corrected datasets, figures, manifest, and summaries."""
    parser=argparse.ArgumentParser();parser.add_argument('--source-dir',type=Path,default=SOURCE);parser.add_argument('--output-dir',type=Path,default=OUTPUT);args=parser.parse_args();src=args.source_dir;out=args.output_dir
    source_files=(src/'raw/m72_policy_evaluations.csv',src/'raw/m72_policy_winners.csv',src/'search/m72_breadth_depth_search.csv',src/'vek280_theoretical/m72_vek280_theoretical.csv');fingerprints={str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in source_files}
    evaluations=read_csv(source_files[0]);original=read_csv(source_files[1]);search=read_csv(source_files[2]);cached_matrix=args.output_dir/'data/m72a_vek280_policy_matrix.csv';vek=read_csv(cached_matrix) if cached_matrix.exists() else read_csv(source_files[3])
    missing=sum(not any(row['predictor_quality']==q and row['policy']==p for row in vek) for q in ('perfect','medium','wrong') for p in ('no_prediction','full_expert_prefetch','uniform_prefix','ranked_prefix'))
    print(f'Reused rows: {len(evaluations)+len(original)+len(search)+len(vek)}');print(f'New required simulations: {missing}')
    if missing>12:raise RuntimeError(f'{missing} missing VEK rows exceeds small-run limit')
    winners=corrected_winners(evaluations,original);best_bds=best_bds_rows(evaluations,winners);gains,ties,gain_summary=search_analysis(evaluations);regret_points,regret_summary=regret_data(evaluations);vek_matrix,new=complete_vek_matrix(vek);assert new==missing
    uncon=[row for row in winners if row['comparison_mode']=='unconstrained'];by_id={str(row['configuration_id']):row for row in uncon}
    def phase(name):return [dict(row,**{k:v for k,v in by_id[row['configuration_id']].items() if k not in row}) for row in read_csv(src/f'phase/{name}.csv')]
    qopp=phase('m72_phase_quality_opportunity');qocm=phase('m72_phase_quality_ocm');channels=phase('m72_phase_opportunity_channels')
    for row in channels:row['reference_bandwidth_opportunity_expert_equivalents']=float(row['opportunity'])/int(row['channels'])
    confidence=[{"metric":"total_configurations","value":len(uncon)},{"metric":"exact_latency_ties","value":sum(str(row['latency_tie'])=='True' or row['latency_tie'] is True for row in uncon)},{"metric":"tie_fraction","value":sum(str(row['latency_tie'])=='True' or row['latency_tie'] is True for row in uncon)/len(uncon)},{"metric":"unique_latency_winners","value":sum(not (str(row['latency_tie'])=='True' or row['latency_tie'] is True) for row in uncon)},{"metric":"robust_winners_ge_1pct","value":sum(str(row['robust_latency_winner'])=='True' or row['robust_latency_winner'] is True for row in uncon)},{"metric":"robust_winner_fraction","value":sum(str(row['robust_latency_winner'])=='True' or row['robust_latency_winner'] is True for row in uncon)/len(uncon)},{"metric":"all_feasible_policies_within_1pct","value":sum(bool(row['all_feasible_policies_within_1pct']) for row in uncon)},{"metric":"all_feasible_policies_within_5pct","value":sum(bool(row['all_feasible_policies_within_5pct']) for row in uncon)}]
    samples=[];group=defaultdict(list)
    for row in qopp:group[(float(row['opportunity']),row['opportunity_setting'])].append(row)
    ordered=sorted(group)
    for index,(opp,label) in enumerate(ordered):
        rows=group[(opp,label)];gap=ordered[index+1][0]-opp if index+1<len(ordered) else 0;samples.append({'opportunity':opp,'source_milestone':next(x['source_milestone'] for x in evaluations if x['configuration_id']==rows[0]['configuration_id']),'predictor_latency':next(x['predictor_latency_cycles'] for x in evaluations if x['configuration_id']==rows[0]['configuration_id']),'sample_count':len(rows),'gap_to_next':gap,'boundary_undersampled':gap>1,'recommended_intermediate_opportunity':opp+gap/2 if gap>1 else ''})
    ocm_interpret=[]
    for ratio in sorted({float(row['ocm_ratio']) for row in uncon}):
        rows=[row for row in uncon if float(row['ocm_ratio'])==ratio];strict=[row for row in rows if not (str(row['latency_tie'])=='True' or row['latency_tie'] is True)]
        ocm_interpret.append({'ocm_ratio':ratio,'configurations':len(rows),'fraction_strict_no_prefetch_wins':sum(row['raw_latency_winners']=='NO_PREFETCH' for row in strict)/len(rows),'fraction_exact_latency_ties':1-len(strict)/len(rows),'fraction_predictive_policy_strict_wins':sum(row['raw_latency_winners']!='NO_PREFETCH' for row in strict)/len(rows),'fraction_bds_strict_wins':sum(row['raw_latency_winners']=='BREADTH_DEPTH_SEARCH' for row in strict)/len(rows)})
    channel_summary=[]
    for row in channels:
        group_rows=[x for x in evaluations if x['comparison_id']==row['comparison_id']];no=next(x for x in group_rows if x['policy']=='no_prediction');predictive=[x for x in group_rows if x['policy']!='no_prediction'];best=min(predictive,key=lambda x:int(x['sequence_makespan'])) if predictive else None
        channel_summary.append({'configuration_id':row['configuration_id'],'channels':row['channels'],'opportunity_setting':row['opportunity_setting'],'reference_bandwidth_opportunity_expert_equivalents':row['reference_bandwidth_opportunity_expert_equivalents'],'no_prefetch_makespan':no['sequence_makespan'],'best_predictive_makespan':best['sequence_makespan'] if best else 'infeasible','predictive_speedup':int(no['sequence_makespan'])/int(best['sequence_makespan']) if best else '', 'winning_M':best['candidate_count'] if best else '', 'winning_depth_vector':best['candidate_depth_vector'] if best else ''})
    for row in channel_summary:
        previous=next((x for x in channel_summary if x['opportunity_setting']==row['opportunity_setting'] and int(x['channels'])==int(row['channels'])-1),None)
        if previous:
            demand_gain=int(previous['no_prefetch_makespan'])/int(row['no_prefetch_makespan'])
            if previous['best_predictive_makespan']!='infeasible' and row['best_predictive_makespan']!='infeasible':
                predictive_gain=int(previous['best_predictive_makespan'])/int(row['best_predictive_makespan']);row.update(demand_only_speedup_from_previous_channel=demand_gain,best_predictive_speedup_from_previous_channel=predictive_gain,additional_speculative_prefetch_channel_benefit=predictive_gain-demand_gain)
            else:row.update(demand_only_speedup_from_previous_channel=demand_gain,best_predictive_speedup_from_previous_channel='infeasible',additional_speculative_prefetch_channel_benefit='infeasible')
        else:row.update(demand_only_speedup_from_previous_channel='',best_predictive_speedup_from_previous_channel='',additional_speculative_prefetch_channel_benefit='')
    datasets={'m72a_corrected_winners.csv':winners,'m72a_best_bds_plans.csv':best_bds,'m72a_search_gain_summary.csv':gains,'m72a_search_tie_efficiency.csv':ties,'m72a_static_policy_regret_points.csv':regret_points,'m72a_static_policy_regret_summary.csv':regret_summary,'m72a_phase_confidence_summary.csv':confidence,'opportunity_sampling_report.csv':samples,'ocm_phase_interpretation.csv':ocm_interpret,'memory_channel_slack_summary.csv':channel_summary,'m72a_vek280_policy_matrix.csv':vek_matrix}
    for name,rows in datasets.items():write_csv(out/'data'/name,rows)
    write_csv(out/'tables/m72a_search_gain_aggregate.csv',[gain_summary])
    write_csv(out/'data/fig01a_policy_phase_quality_vs_opportunity_confidence.csv',qopp);write_csv(out/'data/fig02a_policy_phase_quality_vs_ocm_confidence.csv',qocm);write_csv(out/'data/fig03a_policy_phase_lead_vs_channels.csv',channels);write_csv(out/'data/fig04a_best_breadth_map.csv',best_bds);write_csv(out/'data/fig04b_best_depth_map.csv',best_bds);write_csv(out/'data/fig09a_static_policy_feasibility.csv',regret_summary);write_csv(out/'data/fig09b_static_policy_regret_cdf.csv',regret_points);write_csv(out/'data/fig10a_vek280_theoretical_policy_comparison.csv',vek_matrix);write_csv(out/'data/fig11_search_vs_best_fixed_gain.csv',gains);write_csv(out/'data/fig12_search_tie_resource_efficiency.csv',ties)
    from m72a_make_figures import make_figures
    manifest=make_figures(out);(out/'figure_manifest_m72a.json').write_text(json.dumps(manifest,indent=2)+'\n')
    previous_path=out/'summaries/run_summary.json';previous=json.loads(previous_path.read_text()) if previous_path.exists() else {};cumulative=int(previous.get('new_simulations_total',previous.get('new_simulations',0)))+new
    (out/'summaries').mkdir(parents=True,exist_ok=True);(out/'summaries/run_summary.json').write_text(json.dumps({'source_rows_reused':len(evaluations)+len(original)+len(search)+len(vek),'new_simulations_this_run':new,'new_simulations_total':cumulative,'full_m72_search_rerun':False,'gain_summary':gain_summary,'figures':len(manifest)},indent=2)+'\n')
    after={str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in source_files};assert after==fingerprints;(out/'diagnostics').mkdir(parents=True,exist_ok=True);(out/'diagnostics/source_m72_sha256.json').write_text(json.dumps({'before':fingerprints,'after':after,'unchanged':True},indent=2)+'\n')
    print(f'Wrote M7.2a cleanup to {out}');return 0


if __name__=='__main__':raise SystemExit(main())
