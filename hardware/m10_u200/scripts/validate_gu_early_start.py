"""Validate current RTL against a frozen RTL baseline, without running a simulator model."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

from compare_u200_alignment import intervals, numerical_fixture, read_cases, write_csv, write_json
from compare_u200_default_alignment import (PROJECT, ROOT, build_snapshot, canonical_rtl_events,
    port_metrics, rtl_metrics, run_case, verify_default_parameters)
from edge_moe_sim.alignment_boundaries import analyze_boundaries


def verify_compute_services(before: dict[str, Any], after: dict[str, Any]) -> None:
    """Require exactly the same tile set and GU/DOWN execution durations in cycles."""
    def durations(timing: dict[str, Any]) -> dict[tuple[int, int], tuple[int, int]]:
        """Extract stage service, excluding scheduling and nonlinear-to-DOWN gaps."""
        rows = timing["tile_records"]
        result = {(row["expert_id"],row["tile_id"]):
                  (row["gu_done_cycle"]-row["gu_start_cycle"],
                   row["down_done_cycle"]-row["down_start_cycle"]) for row in rows}
        if len(result) != len(rows):
            raise ValueError("duplicate tile service record")
        return result
    if durations(before) != durations(after):
        raise ValueError("control optimization changed GU/DOWN compute service")


def verify_stage_releases(events: list[dict[str, Any]], timing: dict[str, Any]) -> dict[str, int]:
    """Require one independent release per stage, after its final read/completion."""
    releases: dict[tuple[int, int, str], int] = {}
    for event in events:
        if event["event"] not in ("GU_SLOT_RELEASE", "DOWN_SLOT_RELEASE"):
            continue
        key = (event["expert_id"], event["tile_id"], event["event"])
        if key in releases:
            raise ValueError("duplicate stage release")
        releases[key] = event["cycle"]
    expected = {(r["expert_id"], r["tile_id"], kind)
                for r in timing["tile_records"] for kind in ("GU_SLOT_RELEASE", "DOWN_SLOT_RELEASE")}
    if releases.keys() != expected:
        raise ValueError("stage release tile set mismatch")
    for row in timing["tile_records"]:
        key = (row["expert_id"], row["tile_id"])
        gu = releases[(*key, "GU_SLOT_RELEASE")]
        down = releases[(*key, "DOWN_SLOT_RELEASE")]
        if not row["gu_done_cycle"] <= gu < row["down_done_cycle"] <= down:
            raise ValueError("stage release violates completion causality")
    return dict(gu_release_count=len(expected)//2, down_release_count=len(expected)//2)


def verify_pipeline_services(before: dict[str, Any], after: dict[str, Any], hidden: int, tile: int) -> None:
    """Check the same tile set and the one-chunk/cycle, seven-stage drain contract."""
    keys = [(r["expert_id"],r["tile_id"]) for r in after["tile_records"]]
    old = {(r["expert_id"],r["tile_id"]) for r in before["tile_records"]}
    if len(keys) != len(set(keys)) or set(keys) != old:
        raise ValueError("pipeline tile set mismatch")
    # Start is sampled one edge before first RAM issue; done one edge after store.
    expected = (2*hidden*tile//16+8, hidden*tile//16+8)
    for row in after["tile_records"]:
        actual = (row["gu_done_cycle"]-row["gu_start_cycle"],row["down_done_cycle"]-row["down_start_cycle"])
        if actual != expected:
            raise ValueError(f"chunk pipeline service mismatch: {actual} != {expected}")


def main() -> None:
    """Run eight full-default XSim cases; reject stale baselines and early DOWN."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,default=PROJECT/"results/m10_rtl_current")
    parser.add_argument("--baseline",type=Path,default=PROJECT/"results/m10_boundary_alignment")
    parser.add_argument("--require-cycle-reduction",action="store_true",
                        help="fail unless every case is faster than the selected baseline")
    parser.add_argument("--require-stage-release",action="store_true",
                        help="require independent GU/DOWN release events for every tile")
    parser.add_argument("--require-pipelined-service",action="store_true",
                        help="check explicit chunk-pipeline service instead of control-only invariance")
    args = parser.parse_args()
    p = yaml.safe_load((PROJECT/"configs/hardware/u200_rtl_alignment_default.yaml").read_text())["alignment"]
    cases = read_cases(ROOT/"traces/alignment_cases.jsonl",p)
    verify_default_parameters(p)
    baseline = json.loads((args.baseline/"input_manifest.json").read_text())
    if baseline["parameters"] != p or baseline["cases"] != cases:
        raise ValueError("baseline configuration or stimulus mismatch")
    fixture = numerical_fixture(p)
    output = args.output.resolve()
    if output == args.baseline.resolve():
        raise ValueError("output must not overwrite the frozen baseline")
    output.mkdir(parents=True,exist_ok=True)
    hashes = {str(path.relative_to(ROOT)):hashlib.sha256(path.read_bytes()).hexdigest()
              for path in [*sorted((ROOT/"rtl").glob("*.sv")),ROOT/"tb/tb_u200_post_synth.sv"]}
    write_json(output/"input_manifest.json",dict(parameters=p,cases=cases,source_sha256=hashes,
        execution_rule="gu_only_ready",numerical_fixture=fixture,simulator_run=False,
        compute_service_rule="chunk_pipeline" if args.require_pipelined_service else "unchanged",
        baseline=str(args.baseline.resolve())))
    build,snapshot,cost = build_snapshot(output,netlist=None)
    results: list[dict[str,Any]] = []
    for case in cases:
        directory = output/case["name"]
        text,events,runtime = run_case(case,directory,build,snapshot)
        measured = rtl_metrics(text,events)
        canonical = canonical_rtl_events(events)
        timing = analyze_boundaries(canonical,case["true_expert_ids"],16,tile_start="gu_only_ready")
        release_metrics = verify_stage_releases(events,timing) if args.require_stage_release else {}
        old = json.loads((args.baseline/case["name"]/"rtl_summary.json").read_text())
        if args.require_pipelined_service:
            verify_pipeline_services(old["canonical"],timing,p["hidden_size"],p["f_tile_size"])
            router_start = next(e["cycle"] for e in events if e["event"] == "ROUTER_START")
            expected_router_cycles = p["hidden_size"]*p["num_experts"]//16+8
            if measured["router_done_cycle"]-router_start != expected_router_cycles:
                raise ValueError("Router chunk pipeline service mismatch")
        else:
            verify_compute_services(old["canonical"],timing)
        if args.require_cycle_reduction and measured["total_cycles"] >= old["total_cycles"]:
            raise ValueError(f"no cycle reduction: {case['name']}")
        conserved_keys = ("result_y","truth","demand_bytes","speculative_bytes",
            "useful_speculative_bytes","wasted_speculative_bytes","canceled_speculative_bytes","allocations")
        # Frozen JSON uses arrays; live metrics use tuples for route/allocation records.
        changed = [key for key in conserved_keys
                   if json.loads(json.dumps(measured[key])) != old[key]]
        conserved = not changed
        if not conserved or sorted(measured["compute_order"]) != [(e,t) for e in (3,4) for t in range(16)]:
            raise ValueError(f"GU-early changed numerical/route/byte/tile semantics: {changed}")
        resource_timeline = intervals(events)
        for name in ("MAC","DDR_STRIPED"):
            spans = sorted((r["start_cycle"],r["end_cycle"]) for r in resource_timeline if r["resource"] == name)
            if any(a[1] > b[0] for a,b in zip(spans,spans[1:])):
                raise ValueError("resource overlap")
        if sum(r["bytes"] for r in resource_timeline if r["resource"] == "DDR_STRIPED") != measured["demand_bytes"]+measured["speculative_bytes"]:
            raise ValueError("DMA trace byte conservation failed")
        if (measured["peak_ocm_slots"] > p["ocm_slots"] or max(measured["peak_ocm_slots_per_bank"]) > p["slots_per_bank"] or
            measured["descriptor_peak"] > p["descriptor_entries"] or measured["demand_fifo_peak"] > p["dma_fifo_depth"]):
            raise ValueError("hardware capacity exceeded")
        first = timing["tile_records"][0]
        item = dict(name=case["name"],passed=True,before_total_cycles=old["total_cycles"],
                    after_total_cycles=measured["total_cycles"],saved_cycles=old["total_cycles"]-measured["total_cycles"],
                    first_gu_start_cycle=first["gu_start_cycle"],
                    first_pair_ready_cycle=first["pair_payload_ready_cycle"],
                    gu_started_before_down_ready=first["gu_start_cycle"] < first["pair_payload_ready_cycle"],
                    memory_wait_cycles=timing["totals"]["memory_wait_cycles"],runtime=runtime)
        item.update(release_metrics)
        item.update(before_accept_to_gu_cycles=old["canonical"]["totals"]["accept_to_gu_cycles"],
                    after_accept_to_gu_cycles=timing["totals"]["accept_to_gu_cycles"],
                    before_ready_to_accept_cycles=old["canonical"]["totals"]["ready_to_accept_cycles"],
                    after_ready_to_accept_cycles=timing["totals"]["ready_to_accept_cycles"])
        measured["canonical"] = timing
        write_json(directory/"rtl_summary.json",measured)
        write_json(directory/"port_metrics.json",port_metrics(text))
        write_csv(directory/"boundary_events.csv",canonical)
        write_csv(directory/"tile_timing.csv",timing["tile_records"])
        write_csv(directory/"stage_timing.csv",timing["stage_records"])
        write_csv(directory/"timeline.csv",resource_timeline+timing["timeline"])
        results.append(item)
        print(f"{case['name']}: {old['total_cycles']} -> {measured['total_cycles']}; saved={item['saved_cycles']} early={item['gu_started_before_down_ready']}",flush=True)
    write_json(output/"comparison.json",dict(passed=True,simulator_run=False,cases=results,build_cost=cost,
        limitation="old pair-ready M10 adapter and synthesized netlist no longer represent GU-early RTL"))
    write_csv(output/"comparison.csv",[{k:v for k,v in r.items() if k != "runtime"} for r in results])
    title = "# MAC计算流水：新旧RTL比较" if args.require_pipelined_service else "# 控制优化：新旧RTL比较"
    lines = [title, "", "本轮不运行或修改主Simulator/M10性能适配层；同一激励比较当前RTL与冻结RTL基线。",
             "", f"基线：{args.baseline.resolve()}；启动规则：{baseline.get('execution_rule','gu_and_down_ready')}。", "",
             "| 用例 | 旧cycles | 新cycles | 减少cycles | GU早于DOWN驻留 |", "", "|---|---:|---:|---:|---|"]
    lines.extend(f"| {r['name']} | {r['before_total_cycles']} | {r['after_total_cycles']} | {r['saved_cycles']} | {r['gu_started_before_down_ready']} |" for r in results)
    service_text = ("Router/GU/DOWN 按独立 chunk 数 + 8 cycles 公式检查执行时长；数值/路由/字节/tile与资源互斥守恒。"
                    if args.require_pipelined_service else "数值、真实路由、字节分类、所有tile、GU/DOWN执行时长与资源互斥检查通过。")
    lines += ["", service_text+"DOWN_START必须晚于其PAYLOAD_READY。",
              "本比较流程仅验证RTL；旧funcsim/DCP不能代表新设计，综合/STA应以另外生成的最终哈希报告为准。"]
    (output/"REPORT.md").write_text("\n".join(lines)+"\n",encoding="utf-8")


if __name__ == "__main__":
    main()
