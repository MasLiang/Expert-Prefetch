"""在原真实尺寸 pilot 约束下补跑完整 Expert 目标预取，不把被截断的请求当作整 Expert 命中。"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "examples")]

from edge_moe_sim.real_model_profile import load_real_moe_profiles
from real_moe_pilot import run_case, write_csv


def reference_rows(directory: Path, settings: dict[str, Any]) -> list[dict[str, str]]:
    """读取历史匹配点；核对核心/配置 hashes，允许实验 wrapper 仅新增诊断字段。"""
    summary = json.loads((directory / "summary.json").read_text())
    if summary["status"] != "complete" or summary["settings"] != settings:
        raise ValueError("reference configuration differs or is incomplete")
    checked = 0
    for name, expected in summary["source_sha256"].items():
        if name.startswith("src/") or name.startswith("configs/"):
            if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != expected:
                raise ValueError(f"reference core/config source changed: {name}")
            checked += 1
    if not checked:
        raise ValueError("reference has no core/config provenance")
    with (directory / "runs.csv").open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def main() -> int:
    """运行 24 个 full-target 点并重新运行 12 个基线，验证历史基线周期未变。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/experiments/real_moe_pilot.yaml")
    parser.add_argument("--reference", type=Path, default=ROOT / "results/real_moe_pilot")
    parser.add_argument("--output", type=Path, default=ROOT / "results/real_moe_full_expert")
    args = parser.parse_args()
    settings = yaml.safe_load(args.config.read_text())
    previous = reference_rows(args.reference, settings)
    profiles = load_real_moe_profiles(ROOT / settings["catalogue"], weight_bits=settings["weight_bits"],
                                     activation_bits=settings["activation_bits"])
    base = yaml.safe_load((ROOT / settings["alignment_base"]).read_text())
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "summary.json").write_text(json.dumps({"status": "running"}) + "\n")
    rows: list[dict[str, Any]] = []
    timings: list[dict[str, Any]] = []
    baselines: list[dict[str, Any]] = []
    for profile in profiles:
        for capacity in settings["ocm_capacity_bytes"]:
            for rate in settings["ops_per_cycle"]:
                matching = [r for r in previous if r["model"] == profile.model.name
                            and int(r["requested_ocm_bytes"]) == capacity and int(r["ops_per_cycle"]) == rate]
                baseline, _ = run_case(profile, settings, base, capacity_bytes=capacity,
                    ops_per_cycle=rate, quality="perfect", policy="no_prediction", layers=settings["simulated_layers"])
                old_base = [r for r in matching if r["policy"] == "no_prediction"]
                if len(old_base) != 1 or int(old_base[0]["total_cycles"]) != baseline["total_cycles"]:
                    raise AssertionError("rerun baseline differs from original experiment")
                baselines.append(baseline)
                for quality in settings["qualities"]:
                    row, timing = run_case(profile, settings, base, capacity_bytes=capacity,
                        ops_per_cycle=rate, quality=quality, policy="full_expert_prefetch",
                        layers=settings["simulated_layers"], diagnostics=True)
                    row["baseline_cycles"] = baseline["total_cycles"]
                    row["speedup_vs_no_prediction"] = baseline["total_cycles"] / row["total_cycles"]
                    for policy in ("uniform_prefix", "ranked_prefix"):
                        old = [r for r in matching if r["quality"] == quality and r["policy"] == policy]
                        if len(old) != 1:
                            raise ValueError(f"missing unique reference: {policy}/{quality}")
                        row[f"{policy}_cycles"] = int(old[0]["total_cycles"])
                        row[f"speedup_{policy}_vs_full_expert"] = row["total_cycles"] / int(old[0]["total_cycles"])
                    rows.append(row)
                    timings.extend(timing)
                write_csv(args.output / "runs.csv", rows)
                print(f"completed full-target {profile.model.name} OCM={capacity} ops/cycle={rate}", flush=True)
    write_csv(args.output / "per_layer.csv", timings)
    write_csv(args.output / "rerun_baselines.csv", baselines)
    sources = [args.config, ROOT / settings["catalogue"], ROOT / settings["alignment_base"],
               Path(__file__), ROOT / "examples/real_moe_pilot.py"]
    sources.extend(sorted((ROOT / "src/edge_moe_sim").rglob("*.py")))
    summary = {"status": "complete", "simulator": "UnifiedSequenceSimulator", "simulation_count": 36,
        "full_expert_target_runs": 24, "rerun_baselines_equal": 12, "settings": settings,
        "reference": str(args.reference),
        "reference_artifact_sha256": {name: hashlib.sha256((args.reference / name).read_bytes()).hexdigest()
                                      for name in ("summary.json", "runs.csv")},
        "source_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
        "all_spec_transfers_completed_expert_count": sum(r["experts_all_spec_transfers_completed_by_router"] for r in rows),
        "limits": ["existing full_expert_prefetch targets all tiles but budget/admission can truncate",
                   "not an all-or-nothing whole-expert admission baseline",
                   "completion count is transfer completion, not proof of simultaneous residency",
                   "same synthetic input and scaled hardware as original pilot; no RTL or core semantics changed"]}
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
