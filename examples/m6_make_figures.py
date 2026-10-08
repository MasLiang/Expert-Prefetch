"""Regenerate Milestone 6.5 tables, figures, and measured-results report from CSV."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from statistics import fmean

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def _read(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def _write(path: Path, rows: list[dict], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = fields or list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)
    markdown = "| " + " | ".join(fields) + " |\n|" + "|".join(["---"] * len(fields)) + "|\n"
    markdown += "".join("| " + " | ".join(str(row.get(field, "")) for field in fields) + " |\n" for row in rows)
    path.with_suffix(".md").write_text(markdown, encoding="utf-8")


def _plot(rows: list[dict[str, str]], x: str, y: str, name: str, figures: Path, description: str) -> dict:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figures.mkdir(parents=True, exist_ok=True)
    usable = [row for row in rows if row.get(x, "") and row.get(y, "")]
    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in usable:
        grouped[row.get("scenario", row.get("memory_placement_policy", "data"))].append(row)
    for label, group in grouped.items():
        group.sort(key=lambda row: float(row[x]))
        ax.plot([float(row[x]) for row in group], [float(row[y]) for row in group], marker="o", label=label)
    ax.set_xlabel(x); ax.set_ylabel(y); ax.grid(True, alpha=0.3)
    if len(grouped) > 1: ax.legend(fontsize=7)
    fig.tight_layout(); png = figures / f"{name}.png"; pdf = figures / f"{name}.pdf"
    fig.savefig(png, dpi=180); fig.savefig(pdf); plt.close(fig)
    return {"figure_filename": png.name, "source_csv": "raw/" + name.replace("fig01_reuse_bandwidth_phase", "reuse_bandwidth_phase").replace("fig02_reuse_aie_allocation", "reuse_aie_allocation").replace("fig03_prefetch_window", "prefetch_window_study").replace("fig04_breadth_depth", "breadth_depth_phase").replace("fig05_full_vs_prefix_vs_dse", "policy_comparison").replace("fig06_topk_scalability", "topk_scalability").replace("fig07_memory_channels", "memory_channel_study").replace("fig08_tile_size", "tile_size_study").replace("fig09_dse_pareto", "selected_dse_results") + ".csv", "x_variable": x, "y_variable": y, "filters": "all rows; grouped by scenario where applicable", "normalization_baseline": "exact_tile_streaming same workload when normalized", "description": description}


def _warnings(datasets: dict[str, list[dict[str, str]]]) -> list[str]:
    warnings: list[str] = []
    for name, rows in datasets.items():
        for row in rows:
            if float(row.get("memory_utilization", "0")) > 1.000001: warnings.append(f"{name}: memory utilization > 100%")
            if float(row.get("aie_utilization", "0")) > 1.000001: warnings.append(f"{name}: AIE utilization > 100%")
            if float(row.get("StartableRecall", "0")) > 1.000001: warnings.append(f"{name}: StartableRecall > 1")
            if float(row.get("useful_speculative_bytes", "0")) > float(row.get("speculative_bytes", "0")): warnings.append(f"{name}: useful speculation exceeds transferred speculation")
            if float(row.get("peak_ocm_bytes", "0")) > float(row.get("OCM_bytes", "inf")): warnings.append(f"{name}: OCM capacity exceeded")
    return warnings


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=ROOT / "results" / "characterization")
    args = parser.parse_args(); raw, tables, figures = args.input_dir / "raw", args.input_dir / "tables", args.input_dir / "figures"
    mapping = {"reuse_bandwidth_phase": "fig01_reuse_bandwidth_phase", "reuse_aie_allocation": "fig02_reuse_aie_allocation", "prefetch_window_study": "fig03_prefetch_window", "breadth_depth_phase": "fig04_breadth_depth", "policy_comparison": "fig05_full_vs_prefix_vs_dse", "topk_scalability": "fig06_topk_scalability", "memory_channel_study": "fig07_memory_channels", "tile_size_study": "fig08_tile_size", "selected_dse_results": "fig09_dse_pareto"}
    datasets = {name: _read(raw / f"{name}.csv") for name in mapping if (raw / f"{name}.csv").exists()}
    if not datasets: raise SystemExit(f"No characterization CSVs found in {raw}")
    manifests = []
    choices = {"reuse_bandwidth_phase": ("mean_tokens_per_expert", "board_roofline_efficiency"), "reuse_aie_allocation": ("aie_groups_per_expert", "makespan_cycles"), "prefetch_window_study": ("prefetch_window_cycles", "useful_bytes_transferred_during_prefetch_window"), "breadth_depth_phase": ("candidate_count", "makespan_cycles"), "policy_comparison": ("mean_tokens_per_expert", "makespan_cycles"), "topk_scalability": ("top_k", "makespan_cycles"), "memory_channel_study": ("memory_channels", "achieved_useful_bandwidth_gbps"), "tile_size_study": ("F_tile_size", "makespan_cycles"), "selected_dse_results": ("speculative_bytes", "makespan_cycles")}
    for dataset, rows in datasets.items(): manifests.append(_plot(rows, *choices[dataset], mapping[dataset], figures, f"Synthetic/theoretical {dataset} characterization."))
    # Compact reproducible tables selected directly from observations.
    rb = datasets.get("reuse_bandwidth_phase", [])
    transition = []
    for key in sorted({(r["effective_bandwidth_gbps"], r["weight_bits"]) for r in rb}):
        subset = [r for r in rb if (r["effective_bandwidth_gbps"], r["weight_bits"]) == key]
        transition.append({"effective_bandwidth_gbps": key[0], "weight_bits": key[1], "first_balanced_reuse": next((r["mean_tokens_per_expert"] for r in subset if r["board_regime"] == "balanced"), "not observed"), "first_compute_bound_reuse": next((r["mean_tokens_per_expert"] for r in subset if r["board_regime"] == "compute_bound"), "not observed")})
    _write(tables / "table02_roofline_transition.csv", transition)
    ra = datasets.get("reuse_aie_allocation", []); optimal = []
    for key in sorted({(r["effective_bandwidth_gbps"], r["mean_tokens_per_expert"]) for r in ra}):
        subset = [r for r in ra if (r["effective_bandwidth_gbps"], r["mean_tokens_per_expert"]) == key]; best = min(float(r["makespan_cycles"]) for r in subset)
        near = min(int(r["aie_groups_per_expert"]) for r in subset if float(r["makespan_cycles"]) <= 1.05 * best)
        chosen = next(r for r in subset if int(r["aie_groups_per_expert"]) == near)
        optimal.append({"bandwidth_gbps": key[0], "reuse": key[1], "near_optimal_aie_groups": near, "minimum_latency": best, "starvation_fraction_at_optimum": float(chosen["aie_starvation_cycles"]) / max(1, float(chosen["makespan_cycles"]))})
    _write(tables / "table03_near_optimal_aie_allocation.csv", optimal)
    topk = datasets.get("topk_scalability", []); _write(tables / "table05_topk_scalability.csv", [{key: row.get(key, "") for key in ("top_k", "token_expert_assignments", "unique_active_experts", "reuse_factor", "expert_execution_wave_count", "peak_parallel_experts", "cycles_blocked_by_aie_capacity", "cycles_blocked_by_memory_channel", "cycles_blocked_by_ocm_capacity", "makespan_cycles")} for row in topk])
    _write(tables / "table01_hardware_profiles.csv", [{"hardware_input_class": "SYNTHETIC / THEORETICAL INPUT", "source": "configs/hardware_profile.yaml", "note": "No reported number is measured VEK280 performance."}])
    _write(tables / "table04_prefetch_policy_comparison.csv", datasets.get("policy_comparison", []))
    _write(tables / "table06_dse_best_configs.csv", sorted(datasets.get("selected_dse_results", []), key=lambda row: float(row["makespan_cycles"]))[:10])
    warnings = _warnings(datasets); (args.input_dir / "characterization_warnings.json").write_text(json.dumps(warnings, indent=2) + "\n", encoding="utf-8")
    (args.input_dir / "figure_manifest.json").write_text(json.dumps(manifests, indent=2) + "\n", encoding="utf-8")
    summary = {"note": "Results are measured from the simulated configuration using synthetic/theoretical inputs.", "datasets": {name: len(rows) for name, rows in datasets.items()}, "warning_count": len(warnings), "board_roofline_transitions": transition, "near_optimal_aie_allocations": optimal}
    (args.input_dir / "characterization_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    report = "# Characterization\n\nAll hardware inputs are **SYNTHETIC / THEORETICAL INPUT**; they are not measured VEK280 results. This report summarizes event-driven simulation observations.\n\n"
    report += "## Roofline transitions\n\n" + (tables / "table02_roofline_transition.md").read_text(encoding="utf-8") + "\n## Near-optimal AIE compute-group allocations\n\n" + (tables / "table03_near_optimal_aie_allocation.md").read_text(encoding="utf-8")
    report += "\n## Diagnostics\n\n" + ("No automatic accounting warnings were emitted.\n" if not warnings else "\n".join(f"- {warning}" for warning in warnings) + "\n")
    (args.input_dir / "CHARACTERIZATION.md").write_text(report, encoding="utf-8")
    print(f"Wrote {len(manifests)} PNG/PDF figure pairs, tables, manifest, and report under {args.input_dir}")
    return 0


if __name__ == "__main__": raise SystemExit(main())
