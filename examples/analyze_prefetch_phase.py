"""只分析已完成的合成矩阵，合并曲线；不运行模拟器或改变调度。"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def load_results(directories: list[Path]) -> list[dict[str, Any]]:
    """检查输入完成状态与共享配置一致性，并读取无重复聚合配置点。"""
    rows, keys = [], set()
    common: dict[str, Any] | None = None
    axes = {"tile_groups", "bytes_per_cycle_per_channel", "ocm_expert_equivalents"}
    for directory in directories:
        summary = json.loads((directory / "summary.json").read_text())
        if summary["status"] != "complete":
            raise ValueError(f"incomplete experiment: {directory}")
        fixed = {k: v for k, v in summary["settings"].items() if k not in axes}
        if common is not None and fixed != common:
            raise ValueError("experiments differ beyond documented refinement axes")
        common = fixed
        with (directory / "comparison.csv").open(newline="") as stream:
            for source in csv.DictReader(stream):
                row = {k: v if k == "policy" else float(v) for k, v in source.items()}
                key = tuple(row[k] for k in ["tile_groups", "pe_count", "bytes_per_cycle_per_channel",
                    "ocm_expert_equivalents", "policy", "quality", "prefix_depth"])
                if key in keys:
                    raise ValueError(f"duplicate configuration: {key}")
                keys.add(key)
                rows.append(row)
    return rows


def main() -> None:
    """输出固定策略曲线及按实际采样点统计的收益阈值，不拟合通用充分条件。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", nargs="+", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = load_results(args.inputs)
    args.output.mkdir(parents=True, exist_ok=True)
    selected = sorted([r for r in rows if r["policy"] == "uniform_prefix" and
        r["quality"] == .9 and r["prefix_depth"] == 1 and r["pe_count"] == 1 and
        r["ocm_expert_equivalents"] == 4], key=lambda r: (r["tile_groups"], r["compute_to_memory_payload_ratio"]))
    with (args.output / "fixed_policy_curve.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(selected[0]))
        writer.writeheader()
        writer.writerows(selected)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9, 5))
    for groups in sorted({r["tile_groups"] for r in selected}):
        points = [r for r in selected if r["tile_groups"] == groups]
        ax.plot([r["compute_to_memory_payload_ratio"] for r in points],
                [r["speedup"] for r in points], marker="o", label=f"{int(groups)} F groups / Expert")
    ax.axhline(1, color="black", linestyle="--", label="No net benefit")
    ax.axhline(1.05, color="grey", linestyle=":", label="5% speedup")
    ax.set_xscale("log")
    ax.set_xlabel("Compute / memory payload time (ideal aggregate ratio)")
    ax.set_ylabel("Speedup vs no-prediction tile streaming")
    ax.set_title("Synthetic: PE=1; q=0.9; OCM=4 Experts; requested GU depth=1")
    ax.legend(fontsize=8)
    ax.grid(alpha=.2)
    fig.tight_layout()
    for extension in ("png", "pdf"):
        fig.savefig(args.output / f"benefit_curve.{extension}")
    plt.close(fig)
    summary = {"status": "complete", "input_directories": [str(p.resolve()) for p in args.inputs],
        "configuration_points": len(rows), "sample_results": int(sum(r["samples"] for r in rows)),
        "curve_points": len(selected), "fixed_curve": {"PE": 1, "quality": .9, "depth": 1, "OCM_experts": 4},
        "benefit_counts": {str(threshold): sum(r["speedup"] >= threshold for r in rows
            if r["policy"] == "uniform_prefix") for threshold in [1.01, 1.05, 1.1]},
        "limits": ["sampled empirical region, not a universal sufficient condition",
            "5 samples per point; fixed seed; quality jointly changes recall and rank-1 precision",
            "requested depths larger than Expert are clipped; some points are equivalent executions",
            "no simulator run in analysis; source provenance is in each input summary"]}
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
