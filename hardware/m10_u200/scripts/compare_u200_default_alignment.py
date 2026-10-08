"""Compare shared default-U200 stimuli on XSim RTL, event adapter and optional netlist."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import resource
import shutil
import subprocess
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Sequence

import yaml

from compare_u200_alignment import intervals, numerical_fixture, parse_metrics, read_cases, write_csv, write_json, require_current_tile_start
from edge_moe_sim.rtl_alignment import M10AlignmentSimulator
from edge_moe_sim.alignment_boundaries import analyze_boundaries

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parents[1]


def verify_default_parameters(parameters: dict[str, Any]) -> None:
    """Reject mismatches between manifest geometry and authoritative RTL defaults."""
    mapping = dict(hidden_size="H", f_tile_size="F_TILE", num_experts="NUM_EXPERTS",
                   max_candidates="MAX_CANDIDATES", descriptor_entries="DESCRIPTOR_TABLE_ENTRIES",
                   ocm_banks="NUM_OCM_BANKS", slots_per_bank="SLOTS_PER_BANK",
                   ocm_slot_bytes="OCM_SLOT_BYTES", dma_fifo_depth="DMA_FIFO_DEPTH",
                   max_burst_bytes="MAX_BURST_BYTES")
    source = (ROOT/"rtl/u200_moe_top.sv").read_text(encoding="utf-8")
    for key, parameter in mapping.items():
        match = re.search(rf"parameter integer {parameter}\s*=\s*(\d+)", source)
        if match is None or int(match[1]) != parameters[key]:
            raise ValueError(f"manifest/RTL default mismatch: {key}/{parameter}")
    tiles = re.search(r"parameter integer TILES_PER_EXPERT\s*=\s*(\d+)", source)
    if tiles is None or int(tiles[1])*parameters["f_tile_size"] != parameters["intermediate_size"]:
        raise ValueError("manifest/RTL intermediate tile geometry mismatch")


def resource_limit() -> None:
    """Limit each subprocess virtual address space to 24 GiB (not process-tree RSS)."""
    resource.setrlimit(resource.RLIMIT_AS, (24*1024**3, 24*1024**3))


def validate_netlist(netlist: Path, metadata: Path, parameters: dict[str, Any]) -> dict[str, Any]:
    """Require generated-netlist provenance, identical RTL sources and default geometry."""
    proof = json.loads(metadata.read_text(encoding="utf-8"))
    if proof.get("top") != "u200_moe_top" or proof.get("part") != "xcu200-fsgd2104-2-e":
        raise ValueError("netlist top/part provenance mismatch")
    if proof.get("netlist_sha256") != hashlib.sha256(netlist.read_bytes()).hexdigest():
        raise ValueError("netlist SHA256 mismatch; never use a stale or substituted netlist")
    required = ("hidden_size", "f_tile_size", "intermediate_size", "num_experts", "max_candidates",
                "ocm_banks", "slots_per_bank", "ocm_slots", "ocm_slot_bytes", "descriptor_entries",
                "dma_fifo_depth", "max_burst_bytes", "physical_memory_channels", "clock_hz")
    if any(key not in proof.get("parameters", {}) for key in required):
        raise ValueError("incomplete netlist geometry provenance")
    for key, value in proof["parameters"].items():
        if parameters.get(key) != value:
            raise ValueError(f"netlist/manifest parameter mismatch: {key}")
    actual = {path.name:hashlib.sha256(path.read_bytes()).hexdigest() for path in (ROOT/"rtl").glob("*.sv")}
    if proof.get("rtl_source_sha256") != actual:
        raise ValueError("RTL sources differ from the synthesized netlist; re-synthesize and export")
    return proof


def command_run(command: Sequence[str], cwd: Path, log: Path) -> dict[str, Any]:
    """Run a finite command, retain its exact invocation and GNU time resource report."""
    started = time.monotonic()
    resource_log = log.with_suffix(".time")
    write_json(log.with_suffix(".command.json"), dict(command=list(command), cwd=str(cwd), timeout_seconds=1200))
    measured = ["/usr/bin/time", "-v", "-o", str(resource_log),
                "timeout", "--kill-after=20s", "1200s", *command]
    with log.open("w", encoding="utf-8") as stream:
        completed = subprocess.run(measured, cwd=cwd, stdout=stream, stderr=subprocess.STDOUT,
                                   preexec_fn=resource_limit, timeout=1230, check=False)
    elapsed = time.monotonic()-started
    completed.check_returncode()
    text = resource_log.read_text(encoding="utf-8")
    rss = re.search(r"Maximum resident set size \(kbytes\): (\d+)", text)
    return dict(wall_seconds=elapsed, max_child_rss_kib=int(rss[1]) if rss else None)


def build_snapshot(output: Path, *, netlist: Path | None) -> tuple[Path, str, dict[str, Any]]:
    """Build exactly once per backend without changing any DUT parameter."""
    build = output/("netlist_build" if netlist else "rtl_build")
    build.mkdir(parents=True, exist_ok=True)
    package = ROOT/"rtl/package.sv"
    bench = ROOT/"tb/tb_u200_post_synth.sv"
    snapshot = "u200_shared_funcsim" if netlist else "u200_shared_rtl"
    compile_command = ["xvlog", "--sv"]
    if netlist:
        tool = shutil.which("xvlog")
        if tool is None:
            raise ValueError("xvlog not on PATH")
        glbl = Path(tool).parent.parent/"data/verilog/src/glbl.v"
        if not glbl.is_file():
            raise ValueError(f"Vivado glbl unavailable: {glbl}")
        compile_command += ["--define", "NETLIST", str(package), str(bench), str(netlist), str(glbl)]
    else:
        compile_command += [str(package), *map(str, sorted(p for p in (ROOT/"rtl").glob("*.sv") if p != package)), str(bench)]
    measurement = {"compile": command_run(compile_command, build, build/"compile.log")}
    elaborate = ["xelab", "--debug", "off", "--mt", "4", "--snapshot", snapshot]
    if netlist:
        elaborate += ["-L", "unisims_ver", "-L", "unimacro_ver", "-L", "secureip"]
    elaborate += ["work.tb_u200_post_synth"]
    if netlist:
        elaborate += ["work.glbl"]
    measurement["elaborate"] = command_run(elaborate, build, build/"elaborate.log")
    return build, snapshot, measurement


def run_case(case: dict[str, Any], directory: Path, build: Path, snapshot: str) -> tuple[str, list[dict[str, Any]], dict[str, Any]]:
    """Execute common plusargs and require actual PASS, not merely XSim exit zero."""
    directory.mkdir(parents=True, exist_ok=True)
    batch = directory/"run.tcl"
    batch.write_text("run 1 ms\nquit\n", encoding="utf-8")
    command = ["xsim", snapshot, "--tclbatch", str(batch), "--onerror", "quit", "--onfinish", "quit",
               "--stats", "--log", str(directory/"simulation.log")]
    for arg in (f"PREDICTED_EXPERT={case['predicted_expert']}", f"BUDGET={case['budget_bytes']}",
                f"DDR_GAPS={case['ddr_gaps']}", f"TRACE={directory/'events.csv'}"):
        command += ["--testplusarg", arg]
    measurement = command_run(command, build, directory/"console.log")
    text = (directory/"simulation.log").read_text(encoding="utf-8")
    if text.count("TEST_PASS:") != 1 or re.search(r"Fatal:|FATAL:|ERROR:|Error:", text):
        raise ValueError(f"XSim functional acceptance failed: {directory}")
    if "$finish called" not in text:
        raise ValueError("missing normal testbench finish")
    with (directory/"events.csv").open(encoding="utf-8") as stream:
        events = list(csv.DictReader(stream))
    for event in events:
        for key in ("cycle", "expert_id", "tile_id", "bytes", "value"):
            event[key] = int(event[key])
    return text, events, measurement


def port_metrics(text: str) -> dict[str, Any]:
    """Parse common RTL/netlist metrics, including numerical payload and counters."""
    rows = [line.split()[1:] for line in text.splitlines() if line.startswith("PORT_METRICS ")]
    if len(rows) != 1 or len(rows[0]) != 10:
        raise ValueError("missing/duplicate PORT_METRICS")
    keys = ("total_cycles", "speculative_bytes", "demand_bytes", "useful_speculative_bytes",
            "wasted_speculative_bytes", "canceled_speculative_bytes", "admission_count",
            "compute_count", "returned_bytes")
    result: dict[str, Any] = dict(zip(keys, map(int, rows[0][:-1])))
    packed = int(rows[0][-1], 16)
    result["result_y"] = [(packed >> (32*lane)) & 0xffffffff for lane in range(16)]
    return result


def rtl_metrics(text: str, events: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Collect observed truth, allocations, compute order and actual resource peaks."""
    result = parse_metrics(text)
    result["router_done_cycle"] = next(e["cycle"] for e in events if e["event"] == "ROUTER_DONE")
    result["truth_ready_cycle"] = max(e["cycle"] for e in events if e["event"] == "TRUTH")
    result["truth"] = [(e["expert_id"], e["value"]) for e in events if e["event"] == "TRUTH"]
    result["allocations"] = [(e["expert_id"], e["tile_id"]) for e in events
                             if e["event"] == "ADMIT" and e["matrix"] == "0" and e["request_type"] == "0"]
    result["compute_order"] = [(e["expert_id"], e["tile_id"]) for e in events if e["event"] == "GU_START"]
    rows = [line.split()[1:] for line in text.splitlines() if line.startswith("RESOURCE_METRICS ")]
    if len(rows) != 1 or len(rows[0]) != 6:
        raise ValueError("missing resource observation")
    resource_values = list(map(int, rows[0]))
    result["peak_ocm_slots_per_bank"] = resource_values[:4]
    result["demand_fifo_peak"] = resource_values[4]
    result["descriptor_peak"] = resource_values[5]
    result.update(parse_wait_metrics(text, events, result["weight_wait_cycles"]))
    return result


def wait_intervals(events: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Merge unique one-cycle idle observations into disjoint half-open intervals."""
    samples = sorted((e["cycle"], e["event"]) for e in events
                     if e["event"] in ("WAIT_MEMORY", "WAIT_CONTROL"))
    rows: list[dict[str, Any]] = []
    previous: int | None = None
    for cycle, kind in samples:
        if previous is not None and cycle == previous:
            raise ValueError("duplicate/overlapping wait observation cycle")
        operation = "memory_wait" if kind == "WAIT_MEMORY" else "ready_control_wait"
        if rows and rows[-1]["operation"] == operation and rows[-1]["end_cycle"] == cycle:
            rows[-1]["end_cycle"] = cycle+1
        else:
            rows.append(dict(resource="EXPERT_ISSUE", operation=operation, start_cycle=cycle,
                             end_cycle=cycle+1, expert_id=-1, tile_id=-1, bytes=0, request_type="none"))
        previous = cycle
    return rows


def canonical_rtl_events(events: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Normalize admission/matrix names without changing any observed timestamp."""
    result = []
    kinds = {"ROUTE_RESOLVED","TILE_ISSUE_WINDOW_OPEN","COMPUTE_REQUEST_ACCEPTED",
             "GU_START","GU_DONE","DOWN_START","DOWN_DONE","TILE_RETIRED",
             "EXPERT_COMBINE_START","EXPERT_COMBINE_DONE","MEMORY_SERVICE_START","PAYLOAD_READY"}
    for event in events:
        if event["event"] == "ADMIT":
            row = {**event,"event":"REQUEST_QUEUED","matrix":"gate_up" if event["matrix"] == "0" else "down"}
        elif event["event"] in kinds:
            row = dict(event)
        else:
            continue
        result.append(row)
    return result


def parse_wait_metrics(text: str, events: Sequence[dict[str, Any]], legacy_cycles: int) -> dict[str, int]:
    """Require nonnegative counters, conserved partitions and independent sample totals."""
    lines = [line.split()[1:] for line in text.splitlines() if line.startswith("WAIT_METRICS ")]
    keys = ("expert_issue_idle_cycles", "memory_wait_cycles", "ready_control_wait_cycles",
            "legacy_memory_wait_cycles", "legacy_control_wait_cycles", "issue_pending_wait_cycles",
            "snapshot_refresh_wait_cycles")
    if len(lines) != 1 or len(lines[0]) != len(keys):
        raise ValueError("missing/duplicate WAIT_METRICS")
    result = dict(zip(keys, map(int, lines[0])))
    if any(value < 0 for value in result.values()):
        raise ValueError("negative wait counter")
    if result["expert_issue_idle_cycles"] != result["memory_wait_cycles"]+result["ready_control_wait_cycles"]:
        raise ValueError("wait partition conservation failed")
    if legacy_cycles != result["legacy_memory_wait_cycles"]+result["legacy_control_wait_cycles"]:
        raise ValueError("legacy wait partition conservation failed")
    if (result["legacy_memory_wait_cycles"] > result["memory_wait_cycles"] or
        result["legacy_control_wait_cycles"] > result["ready_control_wait_cycles"] or
        result["issue_pending_wait_cycles"] > result["expert_issue_idle_cycles"] or
        result["snapshot_refresh_wait_cycles"] > result["ready_control_wait_cycles"]):
        raise ValueError("wait subset exceeds parent counter")
    rows = wait_intervals(events)
    for operation, key in (("memory_wait", "memory_wait_cycles"), ("ready_control_wait", "ready_control_wait_cycles")):
        if sum(r["end_cycle"]-r["start_cycle"] for r in rows if r["operation"] == operation) != result[key]:
            raise ValueError("wait timeline/counter mismatch")
    return result


def compare_case(case: dict[str, Any], p: dict[str, Any], fixture: dict[str, Any],
                 rtl: dict[str, Any], sim: dict[str, Any]) -> dict[str, Any]:
    """Compare semantic invariants exactly and retain unchanged timing tolerances."""
    keys = ("demand_bytes", "speculative_bytes", "useful_speculative_bytes", "wasted_speculative_bytes",
            "canceled_speculative_bytes", "allocations", "compute_order")
    exact = {key: rtl[key] == sim[key] for key in keys}
    exact["truth"] = rtl["truth"] == list(zip(case["true_expert_ids"], case["gates_q15"]))
    exact["numerical_output"] = rtl["result_y"] == fixture["expected_result_y"]
    exact["ocm_constraint"] = max(rtl["peak_ocm_slots"], sim["peak_ocm_slots"]) <= p["ocm_slots"]
    exact["bank_constraint"] = max(*rtl["peak_ocm_slots_per_bank"], *sim["peak_ocm_slots_per_bank"]) <= p["slots_per_bank"]
    exact["fifo_descriptor_constraints"] = rtl["demand_fifo_peak"] <= p["dma_fifo_depth"] and rtl["descriptor_peak"] <= p["descriptor_entries"]
    error = (sim["total_cycles"]-rtl["total_cycles"])/rtl["total_cycles"]
    result = dict(name=case["name"], ddr_gaps=case["ddr_gaps"], semantic_passed=all(exact.values()), exact_checks=exact,
                rtl_total_cycles=rtl["total_cycles"], simulator_total_cycles=sim["total_cycles"], relative_cycle_error=error,
                timing_within_tolerance=abs(error) <= p["total_cycle_relative_tolerance"],
                rtl_weight_wait_cycles=rtl["weight_wait_cycles"], simulator_weight_wait_cycles=sim["weight_wait_cycles"],
                weight_wait_within_tolerance=abs(rtl["weight_wait_cycles"]-sim["weight_wait_cycles"]) <= p["weight_wait_absolute_tolerance_cycles"],
                rtl_memory_wait_cycles=rtl["memory_wait_cycles"], simulator_memory_wait_cycles=sim["memory_wait_cycles"],
                memory_wait_delta_cycles=rtl["memory_wait_cycles"]-sim["memory_wait_cycles"],
                memory_wait_within_tolerance=abs(rtl["memory_wait_cycles"]-sim["memory_wait_cycles"]) <= p["weight_wait_absolute_tolerance_cycles"],
                rtl_ready_control_wait_cycles=rtl["ready_control_wait_cycles"],
                simulator_ready_control_wait_cycles=sim["ready_control_wait_cycles"],
                ready_control_wait_delta_cycles=rtl["ready_control_wait_cycles"]-sim["ready_control_wait_cycles"],
                rtl_expert_issue_idle_cycles=rtl["expert_issue_idle_cycles"],
                simulator_expert_issue_idle_cycles=sim["expert_issue_idle_cycles"],
                rtl_legacy_memory_wait_cycles=rtl["legacy_memory_wait_cycles"],
                rtl_legacy_control_wait_cycles=rtl["legacy_control_wait_cycles"],
                rtl_issue_pending_wait_cycles=rtl["issue_pending_wait_cycles"],
                rtl_snapshot_refresh_wait_cycles=rtl["snapshot_refresh_wait_cycles"],
                router_done_delta=sim["router_done_cycle"]-rtl["router_done_cycle"],
                truth_ready_delta=sim["truth_ready_cycle"]-rtl["truth_ready_cycle"],
                rtl_peak_ocm_slots=rtl["peak_ocm_slots"], simulator_peak_ocm_slots=sim["peak_ocm_slots"])
    for name in rtl["canonical"]["totals"]:
        result[f"canonical_rtl_{name}"] = rtl["canonical"]["totals"][name]
        result[f"canonical_simulator_{name}"] = sim["canonical"]["totals"][name]
    result["canonical_boundary_contract_passed"] = True  # Both strict analyzers returned normally.
    return result


def main() -> None:
    """Run complete-default XSim characterization, with optional final-netlist representative cases."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT/"configs/hardware/u200_rtl_alignment_default.yaml")
    parser.add_argument("--trace", type=Path, default=ROOT/"traces/alignment_cases.jsonl")
    parser.add_argument("--output", type=Path, default=PROJECT/"results/m10_default_alignment")
    parser.add_argument("--netlist", type=Path)
    parser.add_argument("--netlist-manifest", type=Path, help="default: netlist_manifest.json beside netlist")
    parser.add_argument("--netlist-cases", nargs="+", default=["demand_fast", "wrong_gaps"])
    parser.add_argument("--strict-timing", action="store_true")
    args = parser.parse_args()
    p = yaml.safe_load(args.config.read_text(encoding="utf-8"))["alignment"]
    if p.get("fixture_profile") != "default_u200":
        raise ValueError("this entry requires default_u200; use legacy script for small fixture")
    cases = read_cases(args.trace, p)
    verify_default_parameters(p)
    require_current_tile_start("gu_and_down_ready")
    fixture = numerical_fixture(p)
    if args.netlist and (not args.netlist.is_file() or not set(args.netlist_cases) <= {c["name"] for c in cases}):
        raise ValueError("netlist file or representative case selection invalid")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    hashes = {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
              for path in [*sorted((ROOT/"rtl").glob("*.sv")), ROOT/"tb/tb_u200_post_synth.sv"]}
    manifest = dict(parameters=p, cases=cases, numerical_fixture=fixture, source_sha256=hashes,
                    backend="XSim 2023.2 RTL / M10AlignmentSimulator", runtime_cycles_are_not_timing_signoff=True)
    if args.netlist:
        manifest["netlist_sha256"] = hashlib.sha256(args.netlist.read_bytes()).hexdigest()
        manifest["netlist_provenance"] = validate_netlist(
            args.netlist, args.netlist_manifest or args.netlist.parent/"netlist_manifest.json", p)
    write_json(output/"input_manifest.json", manifest)
    build, snapshot, cost = build_snapshot(output, netlist=None)
    comparisons = []
    costs: dict[str, Any] = {"rtl_build": cost, "rtl_cases": {}}
    ports: dict[str, dict[str, Any]] = {}
    port_events: dict[str, list[dict[str, Any]]] = {}
    common_events = {"LAYER_START", "PREDICTOR_READY", "ADMIT", "ROUTER_DONE", "COMPUTE_DONE", "LAYER_DONE"}
    for case in cases:
        directory = output/case["name"]
        directory.mkdir(exist_ok=True)
        write_json(directory/"stimulus.json", dict(case=case, numerical_fixture=fixture))
        text, events, costs["rtl_cases"][case["name"]] = run_case(case, directory, build, snapshot)
        rtl = rtl_metrics(text, events)
        canonical_events = canonical_rtl_events(events)
        tile_count = p["intermediate_size"]//p["f_tile_size"]
        rtl["canonical"] = analyze_boundaries(canonical_events,case["true_expert_ids"],tile_count)
        ports[case["name"]] = port_metrics(text)
        port_events[case["name"]] = [e for e in events if e["event"] in common_events]
        timeline = intervals(events)+wait_intervals(events)+rtl["canonical"]["timeline"]
        for resource_name in ("MAC", "DDR_STRIPED"):
            spans = sorted((e["start_cycle"], e["end_cycle"]) for e in timeline if e["resource"] == resource_name)
            if any(a[1] > b[0] for a, b in zip(spans, spans[1:])):
                raise ValueError(f"RTL {resource_name} overlap")
        if sum(e["bytes"] for e in timeline if e["resource"] == "DDR_STRIPED") != rtl["demand_bytes"]+rtl["speculative_bytes"]:
            raise ValueError("RTL trace/telemetry byte mismatch")
        write_csv(directory/"rtl_timeline.csv", timeline)
        write_json(directory/"rtl_summary.json", rtl)
        sim = asdict(M10AlignmentSimulator(p, ddr_gaps=bool(case["ddr_gaps"])).simulate(case))
        sim["canonical"] = analyze_boundaries(sim["boundary_events"],case["true_expert_ids"],tile_count)
        sim["timeline"].extend(sim["canonical"]["timeline"])
        for backend,boundary_rows,summary in (("rtl",canonical_events,rtl),("simulator",sim["boundary_events"],sim)):
            write_csv(directory/f"{backend}_boundary_events.csv",boundary_rows)
            write_csv(directory/f"{backend}_tile_timing.csv",summary["canonical"]["tile_records"])
            write_csv(directory/f"{backend}_stage_timing.csv",summary["canonical"]["stage_records"])
        write_csv(directory/"simulator_timeline.csv", sim.pop("timeline"))
        write_json(directory/"simulator_summary.json", sim)
        comparisons.append(compare_case(case, p, fixture, rtl, sim))
        print(f"{case['name']}: RTL={rtl['total_cycles']} sim={sim['total_cycles']} semantics={comparisons[-1]['semantic_passed']}", flush=True)
    for row in comparisons:
        baseline = next(r for r in comparisons if r["name"] == f"demand_{'gaps' if row['ddr_gaps'] else 'fast'}")
        row["rtl_speedup_vs_demand"] = baseline["rtl_total_cycles"]/row["rtl_total_cycles"]
        row["simulator_speedup_vs_demand"] = baseline["simulator_total_cycles"]/row["simulator_total_cycles"]
        row["speedup_within_tolerance"] = abs(row["rtl_speedup_vs_demand"]-row["simulator_speedup_vs_demand"]) <= p["speedup_absolute_tolerance"]
    netlist_results = []
    if args.netlist:
        net_build, net_snapshot, costs["netlist_build"] = build_snapshot(output, netlist=args.netlist.resolve())
        for case in cases:
            if case["name"] not in args.netlist_cases:
                continue
            directory = output/"netlist"/case["name"]
            text, events, cost = run_case(case, directory, net_build, net_snapshot)
            measured = port_metrics(text)
            same_metrics = measured == ports[case["name"]]
            same_events = [e for e in events if e["event"] in common_events] == port_events[case["name"]]
            item = dict(name=case["name"], passed=same_metrics and same_events, metrics_equal=same_metrics,
                        port_events_equal=same_events, metrics=measured, cost=cost)
            netlist_results.append(item)
            write_json(directory/"comparison.json", item)
            print(f"netlist {case['name']}: port metrics/events equal={item['passed']}", flush=True)
    report = dict(semantic_passed=all(r["semantic_passed"] for r in comparisons),
                  total_cycle_tolerance_passed=all(r["timing_within_tolerance"] for r in comparisons),
                  timing_passed=all(r["timing_within_tolerance"] and r["weight_wait_within_tolerance"] and r["speedup_within_tolerance"] for r in comparisons),
                  memory_wait_tolerance_passed=all(r["memory_wait_within_tolerance"] for r in comparisons),
                  canonical_boundary_contract_passed=all(r["canonical_boundary_contract_passed"] for r in comparisons),
                  netlist_requested=bool(args.netlist), netlist_passed=all(r["passed"] for r in netlist_results) if args.netlist else None,
                  historical_unified_simulator_aligned=False, backend="M10AlignmentSimulator; independent event-level estimates",
                  numerical_oracle="independent identity SwiGLU fixture; simulator itself is a performance model",
                  cases=comparisons, netlist_cases=netlist_results, costs=costs,
                  limitations=["single candidate, Top-2, identity weights; not arbitrary workload/policy coverage",
                               "acceptance-to-GU model stage is explicitly collapsed to zero by default; not calibrated",
                               "phase-independent burst service; control issue/retirement approximated",
                               "adapter combine_cycles charged per tile; RTL combine only at expert final tile",
                               "no SDF, PnR or real-board bandwidth claim; no performance-tolerance relaxation"])
    write_json(output/"comparison.json", report)
    write_csv(output/"comparison.csv", [{k:v for k,v in r.items() if k != "exact_checks"} for r in comparisons])
    lines = ["# 完整默认 U200 RTL / Simulator 对齐", "",
             f"语义验收：{report['semantic_passed']}；总周期容差：{report['total_cycle_tolerance_passed']}；完整性能容差：{report['timing_passed']}；网表代表点：{report['netlist_passed']}。", "",
             "配置：E64、16 tiles/expert、H16/F_TILE16、128×4096-byte OCM slots、4 DDR banks、FIFO16。", "",
             "| 用例 | RTL cycles | Sim cycles | 误差 | RTL等待 | Sim等待 | RTL speedup | Sim speedup |", "",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    lines.extend(f"| {r['name']} | {r['rtl_total_cycles']} | {r['simulator_total_cycles']} | {r['relative_cycle_error']:+.2%} | {r['rtl_weight_wait_cycles']} | {r['simulator_weight_wait_cycles']} | {r['rtl_speedup_vs_demand']:.5f} | {r['simulator_speedup_vs_demand']:.5f} |" for r in comparisons)
    lines += ["", "## 统一功能边界（cycle）", "",
              "| 用例 | 内存等待RTL/Sim | ready→accept RTL/Sim | accept→GU RTL/Sim | ready→GU RTL/Sim |",
              "|---|---:|---:|---:|---:|"]
    lines.extend("| "+r["name"]+" | "+" | ".join(
        f"{r['canonical_rtl_'+key]}/{r['canonical_simulator_'+key]}" for key in
        ("memory_wait_cycles","ready_to_accept_cycles","accept_to_gu_cycles","ready_to_gu_cycles"))+" |" for r in comparisons)
    lines += ["", "统一契约functional_boundaries_v1：请求接受=valid/ready握手，GU_START独立记录；每tile timing CSV由同一分析器生成。",
              "旧95/256控制等待不是同口径；新ready→accept和accept→GU分别比较，零长度合法、不拟合参数。",
              "Simulator未展开请求接受后启动阶段，默认accept=GU_START（collapsed/unmodeled），不能据其0cycles声称该阶段已准确建模。",
              "retire、issue窗口重开与Expert combine分开记录；历史每tile3-cycle阻塞保持不变，主模型与设计RTL均未改动。"]
    lines += ["", "共享输入见input_manifest.json；每case包含原始events、timeline、summary、命令及资源测量。",
              "", "## 等待分项（cycle）", "",
              "| 用例 | RTL内存等待 | Sim内存等待 | 差值RTL−Sim | RTL控制等待 | Sim控制等待 | 旧等待中的控制部分 |",
              "|---|---:|---:|---:|---:|---:|---:|"]
    lines.extend(f"| {r['name']} | {r['rtl_memory_wait_cycles']} | {r['simulator_memory_wait_cycles']} | {r['memory_wait_delta_cycles']} | {r['rtl_ready_control_wait_cycles']} | {r['simulator_ready_control_wait_cycles']} | {r['rtl_legacy_control_wait_cycles']} |" for r in comparisons)
    lines += ["", "新发射空闲 = 内存等待 + ready控制等待；旧等待 = 旧窗口内存等待 + 旧窗口控制等待。旧指标及验收门限保留，不用新分项替代旧验收结果。",
              "内存等待按所有真实Expert下一prefix的GU+DOWN物理OCM完整payload判断，检查epoch/generation、cancel/quarantine，不使用gu_found快照。已知truth的spec resident也算权重就绪，promotion/dispatch开销属于控制。",
              "truth ready、未running、未combine、未完成所有Expert时计数；issue_pending包含在新窗口内。其他MAC/NL运行、combine、Router和最终输出阶段不包含。",
              "RTL按posedge写回前采样；EXPERT_ISSUE timeline每个观测代表[cycle,cycle+1)。Sim按事件半开区间计数，dispatch延迟仍使用原8cycles配置；不保证控制等待相等。",
              "snapshot_refresh/issue_pending是诊断子集，不是额外等待，不能再相加。网表仅公共端口校验，不输出内部等待分项。",
              "功能真值/数值/字节/顺序要求一致；等待、总周期、speedup分别使用原10%/4cycles/0.005容差，不把大总周期掩盖的等待误差算作通过。",
              "OCM按4096-byte slot及4-bank XOR映射管理，不能按payload紧密装填。资源峰值只要求满足约束，不要求解析模型与RTL峰值逐周期相同。",
              "原历史UnifiedSequenceSimulator没有改动；本次仍是独立M10适配层，不是在线锁步仿真。数值由独立fixture oracle校验，performance simulator不执行INT8数值核。",
              "新增resolver延迟为64entries扫描+4drain、Router控制为K*E+42，由结构解释，不从RTL测量结果拟合。",
              "剩余控制发射/退休、逐bank DDR phase和selection pipeline差异保留为误差，不修改RTL/XDC或容差。",
              "当前adapter沿用每tile的3-cycle retirement/combine近似，而RTL只在Expert尾tile执行combine；这是已知时序模型差异，不据总周期测量反向拟合。",
              "本矩阵不含flush/error/multi-epoch、多个候选或其他allocator政策，不能宣称这些已对齐。"]
    (output/"REPORT.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    print(f"semantic={report['semantic_passed']} timing={report['timing_passed']} netlist={report['netlist_passed']}; {output}", flush=True)
    if not report["semantic_passed"] or (args.netlist and not report["netlist_passed"]) or (args.strict_timing and not report["timing_passed"]):
        raise SystemExit("alignment acceptance failed; see comparison.json (timing is separately reported)")


if __name__ == "__main__":
    main()
