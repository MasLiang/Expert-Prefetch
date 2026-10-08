"""运行四路 MIG/RDIMM 数字时序验证，保留非校准区间的周期与服务样本。"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parents[1]


def checked_run(command: list[str], directory: Path, log: Path, timeout: int) -> None:
    """运行有 wall-time 上限的命令；完整日志写入指定文件。"""
    with log.open("w", encoding="utf-8") as stream:
        result = subprocess.run(command, cwd=directory, stdout=stream,
                                stderr=subprocess.STDOUT, timeout=timeout, check=False)
    if result.returncode:
        raise RuntimeError(f"Command failed ({result.returncode}); see {log}")


def parse_case(directory: Path) -> dict[str, Any]:
    """验证明确 PASS、四 bank 完成与 burst 边界，统计 core-cycle 服务时间。"""
    sim = directory / "xsim"
    if '-generic_top "BUDGET=0"' not in (sim / "tb_u200_moe.sh").read_text(encoding="utf-8"):
        raise ValueError("Current-layer prefetch must be disabled for the comparison fixture")
    log = (sim / "simulate.log").read_text(encoding="utf-8")
    for name in ("compile.log", "elaborate.log", "simulate.log"):
        text = (sim / name).read_text(encoding="utf-8")
        if re.search(r"(?im)^\s*(ERROR:|Fatal:|FATAL:)|timing violation", text):
            raise ValueError(f"Simulation error in {sim / name}")
    if log.count("PASS: real 3-layer MoE chain") != 1 or "$finish called" not in log:
        raise ValueError(f"Missing complete functional PASS in {sim}")
    calibrated = re.findall(r"DDR_CALIBRATED bank=(\d+) cycle=(\d+)", log)
    initialized = re.findall(r"DDR_INITIALIZED bank=(\d+) cycle=(\d+)", log)
    if {int(b) for b, _ in calibrated} != set(range(4)) or len(calibrated) != 4:
        raise ValueError("Not all four controllers calibrated")
    if {int(b) for b, _ in initialized} != set(range(4)) or len(initialized) != 4:
        raise ValueError("Not all four memories initialized and checked")
    starts = [(int(l), int(c)) for l, c in re.findall(r"LAYER_START layer=(\d+) cycle=(\d+)", log)]
    results = [(int(l), int(c), int(y)) for l, c, y in re.findall(
        r"LAYER_RESULT layer=(\d+) cycle=(\d+) y=(-?\d+)", log)]
    if [l for l, _ in starts] != [0, 1, 2] or [l for l, _, _ in results] != [0, 1, 2]:
        raise ValueError("Incomplete layer sequence")
    if [y for _, _, y in results] != [24, 60, 254]:
        raise ValueError("MoE numeric results differ from reference")
    samples: list[dict[str, int]] = []
    for bank in range(4):
        rows = list(csv.DictReader((sim / f"ddr_bank_{bank}.csv").open(encoding="utf-8")))
        if not rows:
            raise ValueError(f"Bank {bank} transferred no workload data")
        previous_done = -1
        for row in rows:
            item = {k: int(v) for k, v in row.items()}
            if item["bank"] != bank or not previous_done < item["ar_cycle"] <= item["first_r_cycle"] <= item["last_r_cycle"]:
                raise ValueError(f"Invalid/simultaneous bank burst: {item}")
            if item["first_r_cycle"] <= max(int(c) for _, c in initialized):
                raise ValueError("Initialization leaked into workload measurements")
            item["bytes"] = item["beats"] * 16
            item["first_latency_cycles"] = item["first_r_cycle"] - item["ar_cycle"]
            item["service_cycles"] = item["last_r_cycle"] - item["ar_cycle"] + 1
            item["tail_cycles"] = item["last_r_cycle"] - item["first_r_cycle"]
            if item["bytes"] > 16 * item["service_cycles"]:
                raise ValueError("Bank interface bandwidth conservation failed")
            previous_done = item["last_r_cycle"]
            samples.append(item)
    with (directory / "ddr_service_samples.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(samples[0]))
        writer.writeheader()
        writer.writerows(samples)
    dram = [{k: int(v) for k, v in zip(("bank", "reads", "activates", "refreshes", "precharges"), match)}
            for match in re.findall(r"DDR_DRAM_SUMMARY bank=(\d+) reads=(\d+) activates=(\d+) refreshes=(\d+) precharges=(\d+)", log)]
    if len(dram) != 4 or any(row["reads"] == 0 for row in dram):
        raise ValueError("Missing DDR command observations")
    latency = [row["first_latency_cycles"] for row in samples]
    if directory.name == "no_prefetch" and sum(row["bytes"] for row in samples) != 9216:
        raise ValueError("No-prefetch fixture must transfer exactly three layers of true expert weights")
    return {
        "case": directory.name, "functional_passed": True, "calibration_cycles": dict(calibrated),
        "initialization_done_cycles": dict(initialized), "layer_start_cycles": dict(starts),
        "layer_result_cycles": {l: c for l, c, _ in results},
        "total_cycles": results[-1][1] - starts[0][1],
        "measurement_definition": "first layer begin handshake to final layer result; excludes calibration, preload and post-result cleanup",
        "burst_count": len(samples), "transferred_bytes": sum(row["bytes"] for row in samples),
        "first_latency_cycles": {"min": min(latency), "max": max(latency), "mean": sum(latency) / len(latency)},
        "dram_commands": sorted(dram, key=lambda row: row["bank"]),
        "vendor_elaboration_warning_count": (sim / "elaborate.log").read_text().count("WARNING:"),
        "vendor_unsupported_assertion_count": (sim / "elaborate.log").read_text().count("XSIM 43-4398"),
    }


def run_case(directory: Path) -> dict[str, Any]:
    """从 Vivado 导出的脚本重建 snapshot，并执行 1 ms 上限的仿真。"""
    checked_run(["bash", "tb_u200_moe.sh"], directory / "xsim",
                directory / "run_console.log", 1200)
    result = parse_case(directory)
    (directory / "summary.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    """准备或重跑 DDR 验证；绝不将数字仿真标成板级性能。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=PROJECT / "results/m10_ddr4_controller")
    parser.add_argument("--jobs", type=int, default=2)
    parser.add_argument("--analyze-only", action="store_true")
    args = parser.parse_args()
    if args.jobs < 1 or args.jobs > 3:
        parser.error("--jobs must be in [1,3]")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    names = ("no_prefetch", "hit", "wrong")
    if args.analyze_only:
        # 分析必须与原运行 source manifest 一致，不能把新代码的 hash 标在旧 snapshot 上。
        previous = json.loads((output / "summary.json").read_text(encoding="utf-8"))
        for path, digest in previous["source_sha256"].items():
            if hashlib.sha256((PROJECT / path).read_bytes()).hexdigest() != digest:
                raise ValueError(f"Source changed since simulation: {path}; rerun required")
        cases = [parse_case(output / "cases" / name) for name in names]
    else:
        (output / "summary.json").write_text(json.dumps({"status": "running", "functional_passed": False}) + "\n")
        vivado = shutil.which("vivado")
        if vivado is None:
            raise RuntimeError("Vivado not found on PATH")
        checked_run([vivado, "-mode", "batch", "-source", str(ROOT / "scripts/prepare_ddr4_validation.tcl"),
                     "-tclargs", str(output)], output, output / "prepare.log", 600)
        if "DDR_PREPARE_PASS" not in (output / "prepare.log").read_text():
            raise RuntimeError("DDR project preparation did not complete")
        with ThreadPoolExecutor(max_workers=args.jobs) as pool:
            cases = list(pool.map(run_case, (output / "cases" / name for name in names)))
    source_paths = sorted((ROOT / "rtl").glob("*.sv")) + [
        ROOT / "tb/tb_u200_moe.sv", ROOT / "tb/ddr4_bank_backend.sv",
        ROOT / "configs/ddr4_0.tcl", ROOT / "configs/ddr_axi_cdc.tcl",
        ROOT / "scripts/prepare_ddr4_validation.tcl", Path(__file__).resolve()]
    by_name = {case["case"]: case for case in cases}
    summary = {
        "status": "validation_complete",
        "functional_passed": all(case["functional_passed"] for case in cases), "cases": cases,
        "fixture": {"layers": 3, "hidden_size": 16, "intermediate_size": 32, "num_experts": 6,
                    "top_k": 2, "weight_bits": 8, "current_layer_prefetch_budget_bytes": 0,
                    "future_layer_prefetch_budget_bytes": 1536},
        "hardware": {"core_clock_hz": 200000000, "dram_tck_ps": 833, "sysclk_period_ps": 3332,
                     "channels": 4, "physical_bits_per_channel": 72, "axi_bits_per_channel": 128,
                     "ecc": True, "rank_count": 1, "phy_simulation_mode": "BFM",
                     "part": "MTA18ASF2G72PZ-2G3", "cdc_synchronization_stages": 3},
        "saved_cycles_hit_vs_no_prefetch": by_name["no_prefetch"]["total_cycles"] - by_name["hit"]["total_cycles"],
        "wrong_prefetch_overhead_cycles": by_name["wrong"]["total_cycles"] - by_name["no_prefetch"]["total_cycles"],
        "board_measured": False, "simulator_calibration_applied": False,
        "source_sha256": {str(path.relative_to(PROJECT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in source_paths},
        "limitations": ["BFM PHY + controller + RDIMM digital timing, not analog/PCB timing",
                        "128-bit AXI bank interfaces and CDC are fixture choices, not the complete U200 platform shell",
                        "Small H16 F32 E6 K2 workload, not a deployed-model performance claim",
                        "Vendor unsupported nested assertions are disclosed; independent data/handshake checkers remain enabled"],
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"functional_passed": summary["functional_passed"],
                      "cycles": {case["case"]: case["total_cycles"] for case in cases},
                      "saved_cycles": summary["saved_cycles_hit_vs_no_prefetch"], "output": str(output)}, indent=2))


if __name__ == "__main__":
    main()
