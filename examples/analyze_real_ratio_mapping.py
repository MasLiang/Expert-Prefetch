"""只分析比例映射结果，加入原始细tile baseline以避免弱化分母的误读。"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean
from typing import Any


def compare_layouts(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    """按相同模型/PE/带宽/seed匹配细tile baseline，再汇总跨layout速度比。"""
    baselines = {(r["model"],r["pe_count"],r["ratio"],r["seed"]):int(r["total_cycles"])
        for r in rows if r["layout"]=="fine_original" and r["policy"]=="no_prediction"}
    groups: dict[tuple[str,...],list[dict[str,str]]] = {}
    for row in rows:
        groups.setdefault(tuple(row[k] for k in ["model","layout","pe_count","ratio","policy"]),[]).append(row)
    result = []
    for key,group in sorted(groups.items()):
        cross = []
        for row in group:
            baseline = baselines.get((row["model"],row["pe_count"],row["ratio"],row["seed"]))
            if baseline is not None:
                cross.append(baseline/int(row["total_cycles"]))
        if cross and len(cross)!=len(group):
            raise ValueError("missing paired fine-layout baseline")
        result.append({**dict(zip(["model","layout","pe_count","ratio","policy"],key)),
            "within_current_ocm_budget":group[0]["within_current_ocm_budget"],
            "speedup_vs_same_layout_mean":mean(float(r["speedup"]) for r in group),
            "speedup_vs_fine_original_mean":mean(cross) if cross else None,
            "speedup_vs_fine_original_min":min(cross) if cross else None,
            "speedup_vs_fine_original_max":max(cross) if cross else None})
    return result


def plot_cross_layout(directory: Path, rows: list[dict[str,Any]]) -> None:
    """画原始细tile分母下的速度比，排除无该分母的Mixtral理论点。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    models = sorted({r["model"] for r in rows if r["speedup_vs_fine_original_mean"] is not None})
    fig,axes = plt.subplots(len(models),2,figsize=(12,3*len(models)),squeeze=False)
    for i,model in enumerate(models):
        for j,policy in enumerate(["uniform_random","uniform_hits_first"]):
            ax = axes[i][j]
            for layout in dict.fromkeys(r["layout"] for r in rows if r["model"]==model):
                points = sorted([r for r in rows if r["model"]==model and r["pe_count"]=="4"
                    and r["policy"]==policy and r["layout"]==layout],key=lambda r:float(r["ratio"]))
                line, = ax.plot([float(r["ratio"]) for r in points],
                    [r["speedup_vs_fine_original_mean"] for r in points],marker="o",label=layout)
                ax.fill_between([float(r["ratio"]) for r in points],
                    [r["speedup_vs_fine_original_min"] for r in points],
                    [r["speedup_vs_fine_original_max"] for r in points],color=line.get_color(),alpha=.15)
            ax.axhline(1,color="black",linestyle="--")
            ax.set_title(f"{model}; {policy}",fontsize=9)
            ax.set_xlabel("Ideal compute / memory payload ratio")
            ax.set_ylabel("Speedup vs original fine-tile streaming")
            ax.legend(fontsize=7)
    fig.suptitle("PE=4; paired hardware / seed; OCM within 24 MiB; seed min-max")
    fig.tight_layout()
    for ext in ["png","pdf"]:
        fig.savefig(directory/f"cross_layout_vs_fine.{ext}")
    plt.close(fig)


def main() -> None:
    """输出附加分母对照，不重新运行模拟器、不改变已有实验产物。"""
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parents[1]
    parser.add_argument("--input",type=Path,default=root/"results/real_ratio_mapping")
    args = parser.parse_args()
    if json.loads((args.input/"summary.json").read_text())["status"]!="complete":
        raise ValueError("input experiment is incomplete")
    with (args.input/"per_seed.csv").open(newline="") as stream:
        rows = compare_layouts(list(csv.DictReader(stream)))
    with (args.input/"cross_layout_comparison.csv").open("w",newline="") as stream:
        writer = csv.DictWriter(stream,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    plot_cross_layout(args.input,rows)


if __name__=="__main__":
    main()
