"""严格比较非流水完整 Expert 预取与非流水完整按需加载，保持跨层预取机会。"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

from switch_prefetch_comparison import ROOT, make_inputs, run_sample, write_csv
from edge_moe_sim.config import load_model_config


def make_plot(directory: Path, rows: list[dict[str, Any]]) -> None:
    """绘制相对非流水、不预取 baseline 的速度比。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    models = list(dict.fromkeys(row["model"] for row in rows))
    fig, axes = plt.subplots(1, len(models), figsize=(10, 4), squeeze=False)
    for model, ax in zip(models, axes[0]):
        for rate in sorted({r["ops_per_cycle"] for r in rows}):
            points = [r for r in rows if r["model"] == model and r["ops_per_cycle"] == rate
                      and r["policy"] == "nonpipeline_full_expert_prefetch"]
            ax.plot([r["ocm_bytes"] / 1048576 for r in points],
                    [r["speedup_vs_nonpipeline_demand"] for r in points], marker="o", label=f"{rate} configured ops/cycle")
        ax.axhline(1, color="gray", linestyle="--")
        ax.set_title(model)
        ax.set_xlabel("Expert OCM budget (MiB)")
        ax.set_ylabel("Speedup vs nonpipeline demand full load")
        ax.legend()
    fig.suptitle("Unpruned Switch / synthetic coverage 84.4% / configured hardware")
    fig.tight_layout()
    for extension in ("png", "pdf"):
        fig.savefig(directory / f"nonpipeline_speedup.{extension}")
    plt.close(fig)


def main() -> None:
    """运行 600 对共享激励 case，并核对旧非流水 baseline 的周期完全不变。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/experiments/switch_prefetch.yaml")
    parser.add_argument("--reference", type=Path, default=ROOT / "results/switch_prefetch_baselines")
    parser.add_argument("--output", type=Path, default=ROOT / "results/switch_nonpipeline_prefetch")
    args = parser.parse_args()
    settings = yaml.safe_load(args.config.read_text())
    reference_summary = json.loads((args.reference / "summary.json").read_text())
    if reference_summary["status"] != "complete" or reference_summary["settings"] != settings:
        raise ValueError("reference settings differ or results incomplete")
    with (args.reference / "comparison.csv").open(newline="") as stream:
        reference = {(r["model"], int(r["ocm_bytes"]), int(r["ops_per_cycle"])): int(r["total_cycles"])
                     for r in csv.DictReader(stream) if r["policy"] == "demand_full_expert"}
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "summary.json").write_text(json.dumps({"status": "running"}))
    raw, rows, timings = [], [], []
    for model_path in settings["models"]:
        model = load_model_config(ROOT / model_path)
        records = make_inputs(model, settings)
        (args.output / f"{model.name}_inputs.json").write_text(json.dumps(records, indent=2))
        for capacity in settings["ocm_capacity_bytes"]:
            for rate in settings["ops_per_cycle"]:
                group = []
                for policy in ("demand_full_expert", "nonpipeline_full_expert_prefetch"):
                    samples = [run_sample(model, settings, record, capacity, rate, policy, timings) for record in records]
                    raw.extend(samples)
                    row = {key: samples[0][key] for key in ("model", "ocm_bytes", "ops_per_cycle", "policy")}
                    for key in sorted(samples[0].keys() - row.keys() - {"sample", "peak_ocm_bytes"}):
                        row[key] = sum(sample[key] for sample in samples)
                    row["peak_ocm_bytes"] = max(sample["peak_ocm_bytes"] for sample in samples)
                    row["samples"] = len(samples)
                    group.append(row)
                baseline = group[0]["total_cycles"]
                if baseline != reference[(model.name, capacity, rate)]:
                    raise AssertionError("nonpipeline demand baseline changed")
                for row in group:
                    row["speedup_vs_nonpipeline_demand"] = baseline / row["total_cycles"]
                    row["latency_reduction_percent"] = (baseline - row["total_cycles"]) / baseline * 100
                rows.extend(group)
                print(f"completed {model.name} OCM={capacity} ops/cycle={rate}", flush=True)
    write_csv(args.output / "comparison.csv", rows)
    write_csv(args.output / "per_sample.csv", raw)
    write_csv(args.output / "per_prediction.csv", timings)
    make_plot(args.output, rows)
    sources = [Path(__file__), ROOT / "examples/switch_prefetch_comparison.py", args.config,
               *(ROOT / p for p in settings["models"]), *sorted((ROOT / "src/edge_moe_sim").rglob("*.py"))]
    summary = {"status": "complete", "settings": settings, "simulator": "UnifiedSequenceSimulator",
        "simulation_count": len(raw), "configuration_count": len(rows), "unchanged_baseline_points": len(rows) // 2,
        "reference": str(args.reference),
        "source_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
        "limits": ["full Expert readiness required in both cases; same per-tile compute startup",
            "nonpipeline means no own-Expert weight DMA/compute overlap, not no cross-layer prefetch",
            "full prefetch is a target, not all-or-nothing admission or guaranteed full residency",
            "actual-byte OCM and configured hardware; no burst arbitration; 96 KiB nonpreemptive request",
            "synthetic 84.4% two-candidate coverage; no real CEPR, pruning, attention or encoder",
            "independent cold-start single-token sequences; first layer has no prediction"]}
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
