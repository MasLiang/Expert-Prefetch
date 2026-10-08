"""Run shared M10 fixtures on real RTL and an event-level simulator adapter.

Timing comparison is diagnostic: hard semantic invariants fail the command,
while an explicitly configured timing tolerance is reported separately.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import subprocess
from dataclasses import asdict
from pathlib import Path
from typing import Any, Sequence

import yaml

from edge_moe_sim.rtl_alignment import M10AlignmentSimulator

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parents[1]


def require_current_tile_start(adapter_tile_start: str) -> None:
    """Reject legacy adapters that require DOWN payload before GU startup."""
    if adapter_tile_start != "gu_only_ready":
        raise ValueError("M10AlignmentSimulator 已失效：历史 pair-ready 规则不兼容当前 GU-ready / 跨层 RTL。请运行 make -C hardware/m10_u200 alignment-main，使用主模型 UnifiedSequenceSimulator；完整对齐状态参见新 summary.json。")


def write_json(path: Path, value: Any) -> None:
    """Write reproducible JSON with a terminal newline."""
    path.write_text(json.dumps(value, indent=2, sort_keys=True)+"\n", encoding="utf-8")


def read_cases(path: Path, parameters: dict[str, Any]) -> list[dict[str, Any]]:
    """Validate either the historical small fixture or the complete default fixture."""
    fixed = dict(hidden_size=16, intermediate_size=32, f_tile_size=16, num_experts=6,
                 top_k=2, ocm_slots=4, ocm_slot_bytes=512, reserved_demand_slots=2,
                 descriptor_entries=8, dma_fifo_depth=4, physical_memory_channels=4,
                 max_burst_bytes=256, clock_hz=200_000_000)
    if parameters.get("fixture_profile") == "default_u200":
        fixed.update(intermediate_size=256, num_experts=64, ocm_slots=128,
                     ocm_slot_bytes=4096, descriptor_entries=64, dma_fifo_depth=16,
                     max_candidates=16, ocm_banks=4, slots_per_bank=32,
                     epoch_id=1, gu_base_addr=65536, down_base_addr=131072,
                     expert_stride_bytes=16384, physical_tile_bytes=256)
    elif parameters.get("fixture_profile", "small_u200") != "small_u200":
        raise ValueError("unsupported fixture_profile")
    for key, value in fixed.items():
        if parameters.get(key) != value:
            raise ValueError(f"RTL fixture requires {key}={value}")
    cases = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    names: set[str] = set()
    for case in cases:
        if (not re.fullmatch(r"[a-z][a-z0-9_]*", case["name"]) or case["name"] in names or
            case["predicted_expert"] not in (3, 4, 5) or case["budget_bytes"] not in (0, 768) or
            case["ddr_gaps"] not in (0, 1) or case["confidence_q12"] != 3000 or
            case["true_expert_ids"] != [3, 4] or case["gates_q15"] != [16384, 16384]):
            raise ValueError(f"unsupported or duplicate first-pass fixture: {case}")
        names.add(case["name"])
    if not cases:
        raise ValueError("empty alignment trace")
    return cases


def numerical_fixture(parameters: dict[str, Any] | None = None) -> dict[str, Any]:
    """Materialize the same fixed matrices generated independently by the SV bench."""
    experts = parameters["num_experts"] if parameters else 6
    tiles = parameters["intermediate_size"]//16 if parameters else 2
    activation = [16]*16
    router = [[1 if e in (3, 4) else -1]+[0]*15 for e in range(experts)]
    identity = [[int(r == c) for c in range(16)] for r in range(16)]
    logits = [sum(x*w for x, w in zip(activation, row)) for row in router]
    ids = sorted(range(experts), key=lambda e: (-logits[e], e))[:2]
    silu_q4 = math.floor(16/(1+math.exp(-1))+0.5)
    expert_value = tiles*((silu_q4*16) >> 4)
    return dict(activation_int8=activation, router_weights_output_major=router,
                router_logits=logits, true_expert_ids=ids, gates_q15=[16384, 16384],
                gate_weights_per_tile=identity, up_weights_per_tile=identity,
                down_weights_per_tile=identity, expert_tile_count=tiles,
                gu_requant_shift=0, router_logit_shift=0, expected_result_y=[expert_value]*16)


def parse_metrics(output: str) -> dict[str, Any]:
    """Extract explicit RTL metrics; missing/duplicate lines are errors."""
    rows = [line.split()[1:] for line in output.splitlines() if line.startswith("ALIGN_METRICS ")]
    if len(rows) != 1 or len(rows[0]) != 10:
        raise ValueError("RTL must emit exactly one complete ALIGN_METRICS row")
    keys = ("total_cycles", "speculative_bytes", "demand_bytes", "useful_speculative_bytes",
            "wasted_speculative_bytes", "canceled_speculative_bytes", "controller_demand_stall_cycles",
            "weight_wait_cycles", "peak_ocm_slots")
    result: dict[str, Any] = dict(zip(keys, map(int, rows[0][:-1])))
    packed = int(rows[0][-1], 16)
    result["result_y"] = [(packed >> (32*lane)) & 0xFFFFFFFF for lane in range(16)]
    return result


def intervals(events: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Pair timestamped RTL markers into half-open resource intervals."""
    pending: dict[str, dict[str, Any]] = {}
    result = []
    starts = {"DMA_START": ("DDR_STRIPED", "DMA_DONE"), "GU_START": ("MAC", "GU_DONE"),
              "DOWN_START": ("MAC", "DOWN_DONE"), "ROUTER_START": ("MAC", "ROUTER_DONE"),
              "NL_START": ("NONLINEAR", "NL_DONE")}
    for event in events:
        kind = event["event"]
        if kind in starts:
            _, done = starts[kind]
            if done in pending:
                raise ValueError(f"overlapping RTL interval {kind}")
            pending[done] = event
        elif kind in pending:
            start = pending.pop(kind)
            resource = starts[start["event"]][0]
            if event["cycle"] <= start["cycle"]:
                raise ValueError("nonpositive RTL resource duration")
            result.append(dict(resource=resource, operation=start["event"].removesuffix("_START").lower(),
                start_cycle=start["cycle"], end_cycle=event["cycle"], expert_id=start["expert_id"],
                tile_id=start["tile_id"], bytes=start["bytes"], request_type=start["request_type"]))
            if kind == "GU_DONE":
                pending["NL_DONE"] = {**event, "event": "NL_START"}
    if pending:
        raise ValueError(f"unclosed RTL intervals: {pending.keys()}")
    return result


def write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    """Write non-empty rows in deterministic field order."""
    if not rows:
        raise ValueError(f"cannot write empty table {path}")
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run_rtl(case: dict[str, Any], output_dir: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Compile authoritative U200 numerical testbench and retain its event trace/log."""
    executable = output_dir/"simulation.out"
    sources = [ROOT/"rtl/package.sv"] + sorted(p for p in (ROOT/"rtl").glob("*.sv") if p.name != "package.sv")
    command = ["iverilog", "-g2012", "-s", "tb_u200_moe", "-o", str(executable),
        f"-Ptb_u200_moe.PREDICTED_EXPERT={case['predicted_expert']}",
        f"-Ptb_u200_moe.BUDGET={case['budget_bytes']}",
        f"-Ptb_u200_moe.DDR_GAPS={case['ddr_gaps']}",
        *map(str, sources), str(ROOT/"tb/tb_u200_moe.sv")]
    compiled = subprocess.run(command, capture_output=True, text=True, timeout=60)
    (output_dir/"compile.log").write_text(compiled.stdout+compiled.stderr, encoding="utf-8")
    compiled.check_returncode()
    run = subprocess.run(["vvp", str(executable), f"+TRACE={output_dir/'rtl_events.csv'}"],
                         capture_output=True, text=True, timeout=60)
    (output_dir/"simulation.log").write_text(run.stdout+run.stderr, encoding="utf-8")
    run.check_returncode()
    metrics = parse_metrics(run.stdout)
    with (output_dir/"rtl_events.csv").open(encoding="utf-8") as stream:
        events = list(csv.DictReader(stream))
    for row in events:
        for key in ("cycle", "expert_id", "tile_id", "bytes", "value"):
            row[key] = int(row[key])
    metrics["router_done_cycle"] = next(row["cycle"] for row in events if row["event"] == "ROUTER_DONE")
    metrics["truth_ready_cycle"] = max(row["cycle"] for row in events if row["event"] == "TRUTH")
    metrics["truth"] = [(row["expert_id"], row["value"]) for row in events if row["event"] == "TRUTH"]
    metrics["allocations"] = [(row["expert_id"], row["tile_id"]) for row in events
                              if row["event"] == "ADMIT" and row["matrix"] == "0" and row["request_type"] == "0"]
    metrics["compute_order"] = [(row["expert_id"], row["tile_id"]) for row in events if row["event"] == "GU_START"]
    timeline = intervals(events)
    dma_bytes = sum(row["bytes"] for row in timeline if row["resource"] == "DDR_STRIPED")
    if dma_bytes != metrics["demand_bytes"]+metrics["speculative_bytes"]:
        raise ValueError("RTL trace/telemetry byte mismatch")
    for resource in ("MAC", "DDR_STRIPED"):
        spans = sorted((row["start_cycle"], row["end_cycle"]) for row in timeline if row["resource"] == resource)
        if any(a[1] > b[0] for a, b in zip(spans, spans[1:])):
            raise ValueError(f"RTL {resource} intervals overlap")
    write_csv(output_dir/"rtl_timeline.csv", timeline)
    write_json(output_dir/"rtl_summary.json", metrics)
    return metrics, events


def main() -> None:
    """Run first-pass alignment and separately report semantic and timing acceptance."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT/"configs/hardware/u200_rtl_alignment.yaml")
    parser.add_argument("--trace", type=Path, default=ROOT/"traces/alignment_cases.jsonl")
    parser.add_argument("--output", type=Path, default=PROJECT/"results/m101_u200_alignment")
    parser.add_argument("--strict-timing", action="store_true", help="also fail on total/wait/speedup tolerances")
    args = parser.parse_args()
    require_current_tile_start("gu_and_down_ready")
    parameters = yaml.safe_load(args.config.read_text(encoding="utf-8"))["alignment"]
    if parameters.get("fixture_profile") == "default_u200":
        raise ValueError("complete defaults require compare_u200_default_alignment.py; this legacy entry compiles the small RTL fixture")
    cases = read_cases(args.trace, parameters)
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    fixture = numerical_fixture()
    write_json(args.output/"input_manifest.json", dict(parameters=parameters, cases=cases, numerical_fixture=fixture,
        rtl_sha256={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((ROOT/"rtl").glob("*.sv"))},
        fixture="H16/F32/E6/K2, activation=16, Router W3/W4 first byte=1 and other rows=-1; Expert identity G/U/D; GU shift=0; Router shift=0"))
    comparisons = []
    for case in cases:
        directory = args.output/case["name"]
        directory.mkdir(exist_ok=True)
        write_json(directory/"stimulus.json", dict(case=case, numerical_fixture=fixture))
        rtl, _ = run_rtl(case, directory)
        sim = asdict(M10AlignmentSimulator(parameters, ddr_gaps=bool(case["ddr_gaps"])).simulate(case))
        write_csv(directory/"simulator_timeline.csv", sim.pop("timeline"))
        write_json(directory/"simulator_summary.json", sim)
        exact_keys = ("demand_bytes", "speculative_bytes", "useful_speculative_bytes",
                      "wasted_speculative_bytes", "canceled_speculative_bytes", "allocations", "compute_order")
        exact = {key: rtl[key] == sim[key] for key in exact_keys}
        exact["truth"] = rtl["truth"] == list(zip(case["true_expert_ids"], case["gates_q15"]))
        exact["numerical_output"] = rtl["result_y"] == fixture["expected_result_y"]
        exact["ocm_constraint"] = max(rtl["peak_ocm_slots"], sim["peak_ocm_slots"]) <= parameters["ocm_slots"]
        error = (sim["total_cycles"]-rtl["total_cycles"])/rtl["total_cycles"]
        comparisons.append(dict(name=case["name"], ddr_gaps=case["ddr_gaps"],
            rtl_total_cycles=rtl["total_cycles"], simulator_total_cycles=sim["total_cycles"],
            relative_cycle_error=error, timing_within_tolerance=abs(error)<=parameters["total_cycle_relative_tolerance"],
            semantic_passed=all(exact.values()), exact_checks=exact,
            rtl_weight_wait_cycles=rtl["weight_wait_cycles"], simulator_weight_wait_cycles=sim["weight_wait_cycles"],
            weight_wait_within_tolerance=abs(rtl["weight_wait_cycles"]-sim["weight_wait_cycles"]) <=
                parameters["weight_wait_absolute_tolerance_cycles"],
            router_done_delta=sim["router_done_cycle"]-rtl["router_done_cycle"],
            truth_ready_delta=sim["truth_ready_cycle"]-rtl["truth_ready_cycle"]))
    for row in comparisons:
        baseline = next((r for r in comparisons if r["name"] == f"demand_{'gaps' if row['ddr_gaps'] else 'fast'}"), None)
        if baseline is None:
            raise ValueError("each DDR mode requires a demand baseline")
        row["rtl_speedup_vs_demand"] = baseline["rtl_total_cycles"]/row["rtl_total_cycles"]
        row["simulator_speedup_vs_demand"] = baseline["simulator_total_cycles"]/row["simulator_total_cycles"]
        row["speedup_within_tolerance"] = abs(row["rtl_speedup_vs_demand"]-row["simulator_speedup_vs_demand"]) <= parameters["speedup_absolute_tolerance"]
    write_csv(args.output/"comparison.csv", [{k: v for k, v in r.items() if k != "exact_checks"} for r in comparisons])
    report = dict(semantic_passed=all(r["semantic_passed"] for r in comparisons),
        total_cycle_tolerance_passed=all(r["timing_within_tolerance"] for r in comparisons),
        timing_passed=all(r["timing_within_tolerance"] and r["weight_wait_within_tolerance"] and
                          r["speedup_within_tolerance"] for r in comparisons),
        backend="M10AlignmentSimulator using existing EventQueue/AIEPool/MultiChannelMemory/OCM and HardwareGreedy-v1",
        service_time_source="independent analytic config, not fitted to RTL layer outcomes",
        historical_unified_simulator_aligned=False, cases=comparisons,
        limitations=["fixed small numerical fixture; no arbitrary model/Top-K stimulus yet",
            "event-level phase-independent burst service vs phase-dependent bank gaps",
            "control/retirement latencies approximated; no per-beat/cycle-accurate emulation",
            "no board DDR timing, PnR or measured bandwidth claim"])
    write_json(args.output/"comparison.json", report)
    lines = ["# U200 RTL / event simulator 首次对齐结果", "",
        f"语义验收：{report['semantic_passed']}；总周期容差：{report['total_cycle_tolerance_passed']}；"
        f"完整性能容差（含等待和 speedup）：{report['timing_passed']}。", "",
        "| 激励 | RTL cycles | Simulator cycles | 周期误差 | RTL 等待 | Simulator 等待 | RTL speedup | Simulator speedup |",
        "|---|---:|---:|---:|---:|---:|---:|---:|"]
    lines.extend(f"| {r['name']} | {r['rtl_total_cycles']} | {r['simulator_total_cycles']} | "
        f"{r['relative_cycle_error']:+.2%} | {r['rtl_weight_wait_cycles']} | {r['simulator_weight_wait_cycles']} | "
        f"{r['rtl_speedup_vs_demand']:.5f} | {r['simulator_speedup_vs_demand']:.5f} |" for r in comparisons)
    lines.extend(["", "该矩阵计算占主导，总周期误差小不代表等待/收益已准确对齐。",
        "首次配置为独立解析估计，不使用 layer 结果拟合。DMA admission、truth resolver 与",
        "retirement 的控制成本、相位相关 bank 停顿尚未精确纳入。",
        "对齐 backend 是 M10 专用适配层，复用原事件/资源引擎；不是历史 UnifiedSequenceSimulator 已通过对齐。",
        "", "每个 case 保留 stimulus.json、RTL 原始事件/时间线/日志和 simulator 时间线/summary。",
        "weight_wait 是 truth ready 后没有可执行 prefix tile 的空闲时间，排除运行中、选择寄存级与 combine；",
        "controller_demand_stall_cycles 是原控制器计数，定义不同，不强行当作同一指标。"])
    (args.output/"REPORT.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    for row in comparisons:
        print(f"{row['name']}: RTL={row['rtl_total_cycles']} sim={row['simulator_total_cycles']} "
              f"error={row['relative_cycle_error']:+.2%} semantics={row['semantic_passed']}")
    print(f"semantics={report['semantic_passed']} timing_tolerance={report['timing_passed']}; {args.output}")
    if not report["semantic_passed"] or (args.strict_timing and not report["timing_passed"]):
        raise SystemExit("alignment acceptance failed; inspect comparison.json")


if __name__ == "__main__":
    main()
