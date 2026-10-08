"""Generate the M7.3 simulator-fidelity audit without changing simulator semantics."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import replace
from math import ceil
from pathlib import Path
from statistics import fmean
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "examples")]

import m71_unified_characterization as m71
from edge_moe_sim.unified_schema import UnifiedLayerSequence, UnifiedPredictionSpec
from edge_moe_sim.unified_sequence_simulator import UnifiedSequenceSimulator

OUT = ROOT / "results/m73_fidelity_audit"
POLICIES = ("no_prediction", "ranked_prefix", "full_expert_prefetch", "best_bds")
POLICY_LABELS = {
    "no_prediction": "NO_PREFETCH",
    "ranked_prefix": "RANKED_PREFIX",
    "full_expert_prefetch": "FULL_EXPERT",
    "best_bds": "BREADTH_DEPTH_SEARCH",
}
BASE_OCM_BYTES = 49_152
PREDICTOR_LATENCY_CYCLES = 4_767
TRUE_RANKS = (1, 3, 7, 13)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write a deterministic, non-empty CSV with the union of row fields."""
    if not rows:
        raise ValueError(f"refusing to write empty audit artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({field for row in rows for field in row})
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _policy_spec(policy: str) -> tuple[int, str, int, tuple[int, ...]]:
    """Return the frozen M7.2b representative policy parameters."""
    if policy == "no_prediction":
        return 16, "ranked_prefix", 8, ()
    if policy == "ranked_prefix":
        return 16, "ranked_prefix", 8, ()
    if policy == "full_expert_prefetch":
        return 16, "full_expert_prefetch", 1, ()
    if policy == "best_bds":
        # Existing medium-quality M7.2b representative winner: M=4, d=(1,1,1,1).
        return 4, "uniform_prefix", 1, (1, 1, 1, 1)
    raise ValueError(f"unknown audit policy: {policy}")


def simulate_representative(
    policy: str,
    *,
    bandwidth_scale: float = 1.0,
    compute_scale: float = 1.0,
    ocm_scale: float = 1.0,
) -> tuple[dict[str, Any], object]:
    """Run one fixed M7.2b representative point under a hardware scale factor."""
    model, profile, config, layers = m71.machine(
        channels=2,
        ocm=round(BASE_OCM_BYTES * ocm_scale),
        top_k=4,
        hardware_mode="controlled_synthetic_hardware",
        batch_size=1,
        routing_shape="high_reuse",
        placement_policy="round_robin_expert",
    )
    profile = replace(
        profile,
        bandwidth_bytes_per_sec_per_channel=(
            profile.bandwidth_bytes_per_sec_per_channel * bandwidth_scale
        ),
        ops_per_cycle_per_aie=profile.ops_per_cycle_per_aie * compute_scale,
    )
    candidate_count, engine_policy, prefix_depth, candidate_depths = _policy_spec(policy)
    config = replace(config, candidate_count=candidate_count)
    predictors, predictor_trace_id = m71.predictors(
        layers, model, "medium", candidate_count, True, TRUE_RANKS, 17
    )
    predictions = () if policy == "no_prediction" else tuple(
        UnifiedPredictionSpec(
            source,
            target,
            "source_router_done",
            PREDICTOR_LATENCY_CYCLES,
            candidate_count,
            engine_policy,
            prefix_depth,
            candidate_depths=candidate_depths,
        )
        for source, target in ((0, 1), (1, 2))
    )
    result = UnifiedSequenceSimulator(model, profile).simulate(
        UnifiedLayerSequence(layers),
        config,
        predictions=predictions,
        predictors=predictors,
        trace_enabled=False,
    )
    target_layers = [result.layer_results[index] for index in (1, 2)]
    timings = [timing for layer in target_layers for timing in layer.expert_timings]
    row: dict[str, Any] = {
        "policy": POLICY_LABELS[policy],
        "policy_internal": policy,
        "candidate_count": candidate_count,
        "candidate_depth_vector": str(candidate_depths) if candidate_depths else "policy_default",
        "bandwidth_scale": bandwidth_scale,
        "compute_scale": compute_scale,
        "ocm_scale": ocm_scale,
        "configured_bandwidth_GB_per_s": profile.effective_memory_bandwidth_bytes_per_sec / 1e9,
        "configured_compute_GOPS_per_s": profile.effective_compute_peak_ops_per_sec / 1e9,
        "configured_ocm_bytes": profile.ocm_bytes,
        "sequence_makespan_cycles": result.inference_makespan_cycles,
        "aie_utilization": result.aie_metrics["board_aie_utilization"],
        "memory_utilization": fmean(result.memory_metrics["per_channel_utilization"]),
        "peak_ocm_bytes": result.peak_ocm_bytes,
        "demand_blocking_cycles": result.demand_blocked_by_future_speculation_cycles,
        "critical_path_demand_blocking_cycles": result.critical_path_future_speculation_blocking_cycles,
        "true_expert_memory_starvation_cycles": sum(int(t["memory_wait_cycles"]) for t in timings),
        "speculative_bytes": sum(p.speculative_bytes for p in result.predictions),
        "useful_speculative_bytes": sum(p.useful_bytes for p in result.predictions),
        "wasted_speculative_bytes": sum(p.wasted_bytes for p in result.predictions),
        "physical_speculative_bytes_by_router": sum(
            p.physical_speculative_bytes_transferred_by_router for p in result.predictions
        ),
        "realized_opportunity_expert_equivalents": (
            fmean(p.prefetch_opportunity_expert_equivalents for p in result.predictions)
            if result.predictions else 0.0
        ),
        "predictor_trace_id": predictor_trace_id,
        "workload_trace_id": "m72-controlled-3layer-k4-b1-high_reuse",
        "source_milestone": "source_router_done",
        "predictor_latency_cycles": PREDICTOR_LATENCY_CYCLES,
        "synthetic_lead_time": False,
    }
    return row, result


def sensitivity_rows() -> tuple[list[dict[str, Any]], dict[str, tuple[dict[str, Any], object]]]:
    """Run the independent bandwidth, compute, and capacity sensitivity sweeps."""
    cache: dict[tuple[float, float, float, str], tuple[dict[str, Any], object]] = {}

    def evaluate(bw: float, compute: float, ocm: float, policy: str) -> tuple[dict[str, Any], object]:
        key = (bw, compute, ocm, policy)
        if key not in cache:
            cache[key] = simulate_representative(
                policy, bandwidth_scale=bw, compute_scale=compute, ocm_scale=ocm
            )
        return cache[key]

    axes = (
        ("memory_bandwidth", (0.5, 0.75, 1.0, 1.5, 2.0)),
        ("compute_speed", (0.5, 0.75, 1.0, 1.5, 2.0)),
        ("ocm_capacity", (0.5, 1.0, 2.0)),
    )
    rows: list[dict[str, Any]] = []
    baseline_winner = ""
    for axis, scales in axes:
        for scale in scales:
            factors = {
                "memory_bandwidth": (scale, 1.0, 1.0),
                "compute_speed": (1.0, scale, 1.0),
                "ocm_capacity": (1.0, 1.0, scale),
            }[axis]
            group = [dict(evaluate(*factors, policy)[0]) for policy in POLICIES]
            no_prefetch = next(row for row in group if row["policy_internal"] == "no_prediction")
            minimum = min(int(row["sequence_makespan_cycles"]) for row in group)
            winners = sorted(
                row["policy"] for row in group
                if int(row["sequence_makespan_cycles"]) == minimum
            )
            winner_text = "|".join(winners)
            if axis == "memory_bandwidth" and scale == 1.0:
                baseline_winner = winner_text
            for row in group:
                row.update(
                    sensitivity_axis=axis,
                    sensitivity_scale=scale,
                    speedup_vs_no_prediction=(
                        int(no_prefetch["sequence_makespan_cycles"])
                        / int(row["sequence_makespan_cycles"])
                    ),
                    latency_winners=winner_text,
                    exact_latency_tie=len(winners) > 1,
                    full_m72_search_rerun=False,
                    bds_plan_reoptimized=False,
                )
                rows.append(row)
    for row in rows:
        row["winner_changed_vs_1x"] = row["latency_winners"] != baseline_winner
    named_cache = {
        policy: evaluate(1.0, 1.0, 1.0, policy) for policy in POLICIES
    }
    assert len(cache) == 44, f"expected 44 unique sensitivity runs, observed {len(cache)}"
    return rows, named_cache


def provenance_rows() -> list[dict[str, Any]]:
    """Return the complete hardware-parameter provenance ledger."""
    src = "examples/m71_unified_characterization.py:machine"
    rows: list[dict[str, Any]] = []

    def add(parameter: str, value: Any, unit: str, location: str, source_type: str,
            reference: str, confidence: str, notes: str) -> None:
        rows.append({
            "parameter": parameter, "value": value, "unit": unit,
            "simulator_location": location, "source_type": source_type,
            "source_reference": reference, "confidence": confidence, "notes": notes,
        })

    add("controlled clock", 1_000_000_000, "Hz", src, "ASSUMED", "M7.1 controlled profile", "medium", "Cycle-to-second conversion assumption; not measured.")
    add("controlled memory channels", "1, 2, or 3; M7.3 reference=2", "channels", src, "SYNTHETIC_WORKLOAD_PARAMETER", "M7.1/M7.2 controlled sweeps", "high", "Abstract independent channels, not a board wiring claim.")
    add("peak bandwidth per channel", 16, "GB/s/channel", src, "ASSUMED", "M7.1 controlled profile", "low", "Synthetic peak input; no board measurement.")
    add("bandwidth efficiency", 0.8, "ratio", src, "ASSUMED", "M7.1 controlled profile", "low", "Constant efficiency, independent of request mix.")
    add("effective bandwidth per channel", 12.8, "bytes/cycle/channel", "HardwareProfile.effective_bytes_per_cycle_per_channel", "DERIVED", "16 GB/s * 0.8 / 1 GHz", "high", "Arithmetic consequence of assumed inputs.")
    add("VEK theoretical aggregate bandwidth", 102.4, "GB/s", src, "ASSUMED", "M7.1 vek280_theoretical study input", "low", "Unmeasured theoretical input; not hardware validation.")
    add("DMA burst/coalescing ceiling", 4096, "bytes", src, "ASSUMED", "M7.1 controlled profile", "low", "Coalescing ceiling, not a verified AXI burst limit.")
    add("Gate-Up tile transfer granularity", 4096, "bytes", "work_model + f_tile_size", "DERIVED", "2*H*F_tile*weight_bits/8", "high", "H=64, F_tile=32, INT8.")
    add("Down tile transfer granularity", 2048, "bytes", "work_model + f_tile_size", "DERIVED", "H*F_tile*weight_bits/8", "high", "H=64, F_tile=32, INT8.")
    add("DMA setup latency", 40, "cycles/request", src, "ASSUMED", "M7.1 controlled profile", "low", "Fixed request cost; not calibrated.")
    add("maximum inflight per channel", 1, "request/channel", src, "ASSUMED", "M7.1 controlled profile", "medium", "Current implementation explicitly enforces one.")
    add("DMA arbitration", "demand/commit > background > speculative", "priority order", "hardware/multi_channel_memory.py", "ASSUMED", "simulator policy", "high", "Deterministic model behavior, not measured controller arbitration.")
    add("speculative queue ordering", "score desc, rank, layer, expert, matrix, tile, request", "lexicographic", "hardware/multi_channel_memory.py", "ASSUMED", "simulator policy", "high", "Deterministic research policy.")
    add("active DMA preemption", False, "boolean", "hardware/multi_channel_memory.py", "ASSUMED", "simulator policy", "high", "Started transfers cannot be canceled or preempted.")
    add("memory placement", "round_robin_expert", "policy", src, "ASSUMED", "M7.2 representative configuration", "medium", "Expert ID modulo channel count; no address-map validation.")
    add("channel scaling", "linear aggregate ceiling", "model", "HardwareProfile", "DERIVED", "channels * per-channel bandwidth", "high", "Single-channel requests do not automatically stripe.")
    add("reference OCM capacity", BASE_OCM_BYTES, "bytes", src, "SYNTHETIC_WORKLOAD_PARAMETER", "M7.2b representative point", "high", "1.5x the 32 KiB demand reference; not board capacity.")
    add("demand working-set reference", 32768, "bytes", "examples/m72b_opportunity_refinement.py", "DERIVED", "M7.2a smallest capacity within 1% of large-OCM demand latency", "medium", "Empirical simulator normalization, not physical OCM size.")
    add("OCM organization", "byte-capacity object store", "abstraction", "hardware/capacity_ocm.py", "ASSUMED", "simulator abstraction", "high", "No banks, ports, fragmentation, or access latency.")
    add("OCM admission", "demand guaranteed per tile; speculation may reject", "policy", "hardware/capacity_ocm.py", "ASSUMED", "simulator policy", "high", "Demand tile larger than total capacity raises an error.")
    add("OCM eviction", "non-running speculative; false/low-priority first", "policy", "hardware/capacity_ocm.py", "ASSUMED", "simulator policy", "high", "Running allocations protected.")
    add("OCM residency accounting", "allocation-to-release byte-cycles", "model", "hardware/capacity_ocm.py", "DERIVED", "allocation interval accounting", "high", "Peak allocation is distinct from bytes physically transferred.")
    add("controlled logical AIE units", 16, "logical units", src, "ASSUMED", "M7.1 controlled profile", "low", "Abstract capacity units, not physical tile mapping.")
    add("AIE raw rate", 32, "ops/cycle/logical unit", src, "ASSUMED", "M7.1 controlled profile", "low", "Synthetic effective-rate input before efficiency.")
    add("AIE efficiency", 0.75, "ratio", src, "ASSUMED", "M7.1 controlled profile", "low", "Constant across Router/Gate-Up/Down.")
    add("effective AIE rate", 24, "ops/cycle/logical unit", "HardwareProfile.compute_rate_per_cycle", "DERIVED", "32 * 0.75", "high", "Arithmetic consequence of assumed inputs.")
    add("AIE units per Expert", 4, "logical units/task", "AdvancedExecutionConfig", "SYNTHETIC_WORKLOAD_PARAMETER", "M7.2 representative configuration", "high", "Fixed allocation; no mapper calibration.")
    add("AIE task startup", 20, "cycles/tile task", src, "ASSUMED", "M7.1 controlled profile", "low", "Paid per Router/tile reservation; not measured.")
    add("AIE availability", "cumulative 16-unit pool", "model", "hardware/aie_pool.py", "ASSUMED", "simulator abstraction", "high", "Enforces capacity overlap but omits placement/routing constraints.")
    add("compute duration", "ceil(ops/(units*effective_rate))+startup", "cycles", "hardware/aie_pool.py", "DERIVED", "analytical performance model", "high", "Operation count is formula-derived; rate/startup are assumed.")
    add("F tile size", 32, "elements", "AdvancedExecutionConfig", "SYNTHETIC_WORKLOAD_PARAMETER", "M7.2 controlled workload", "high", "Eight F tiles per Expert for F=256.")
    add("nonlinear throughput", 1024, "elements/cycle", "AdvancedExecutionConfig", "ASSUMED", "default advanced configuration", "low", "Independent single nonlinear resource; uncalibrated.")
    add("nonlinear startup", 0, "cycles/task", "AdvancedExecutionConfig", "ASSUMED", "default advanced configuration", "low", "Uncalibrated.")
    add("compute/DMA overlap", "independent shared resources", "dependency model", "unified_sequence_simulator.py", "ASSUMED", "event model", "high", "Causal dependencies enforced; NoC contention omitted.")
    add("VEK theoretical AIE count", 304, "physical/logical units", src, "ASSUMED", "M7.1 vek280_theoretical study input", "low", "Unmeasured study mapping; repository helper/config variants differ.")
    add("VEK theoretical compute", 202, "TOPS", src, "ASSUMED", "M7.1 vek280_theoretical study input", "low", "MAC=2-op theoretical input; not achieved throughput.")
    add("model H/F/E/K", "64/256/32/4", "elements/experts", src, "SYNTHETIC_WORKLOAD_PARAMETER", "M7.2 controlled workload", "high", "Small characterization workload, not a production model.")
    add("batch size and layer count", "1 and 3", "tokens/layers", src, "SYNTHETIC_WORKLOAD_PARAMETER", "M7.2 controlled workload", "high", "Decode-like synthetic sequence.")
    add("predictor latency", PREDICTOR_LATENCY_CYCLES, "cycles", "M7.3 representative", "SYNTHETIC_WORKLOAD_PARAMETER", "M7.2b medium-opportunity point", "high", "Produces 2.316 Expert-equivalent opportunity at 1x.")
    return rows


def bandwidth_rows() -> list[dict[str, Any]]:
    """Calculate bandwidth plausibility and setup/payload balance tables."""
    rows: list[dict[str, Any]] = []
    profiles = [
        ("controlled_synthetic", channels, 16e9, 0.8, 1e9, 40)
        for channels in (1, 2, 3)
    ] + [("vek280_theoretical_not_measured", 2, 51.2e9, 1.0, 1e9, 40)]
    for name, channels, per_channel, efficiency, clock, setup in profiles:
        effective_channel = per_channel * efficiency / clock
        for transfer_bytes in (512, 2048, 4096, 49_152):
            payload_cycles = ceil(transfer_bytes / effective_channel)
            total_cycles = setup + payload_cycles
            rows.append({
                "profile": name,
                "source_type": "ASSUMED",
                "memory_channels": channels,
                "clock_hz": int(clock),
                "configured_peak_GB_per_s_per_channel": per_channel / 1e9,
                "bandwidth_efficiency": efficiency,
                "effective_bytes_per_cycle_per_channel": effective_channel,
                "aggregate_effective_GB_per_s": channels * per_channel * efficiency / 1e9,
                "aggregate_scaling_vs_one_channel": channels,
                "transfer_bytes": transfer_bytes,
                "request_channel_count": 1,
                "payload_cycles": payload_cycles,
                "setup_cycles": setup,
                "total_dma_cycles": total_cycles,
                "setup_fraction": setup / total_cycles,
                "dominant_term": "setup_latency" if setup > payload_cycles else "payload_bandwidth",
                "comparison_reference": (
                    "controlled 1-channel aggregate" if name == "controlled_synthetic"
                    else "repository theoretical input; not measured"
                ),
                "interpretation": (
                    "Aggregate ceiling scales linearly with independent channels; "
                    "round-robin requests remain single-channel."
                ),
            })
    return rows


def compute_rows(baselines: dict[str, tuple[dict[str, Any], object]],
                 sensitivity: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Build analytical and observed compute calibration rows."""
    no_row, no_result = baselines["no_prediction"]
    total = int(no_row["sequence_makespan_cycles"])
    aie = no_result.aie_metrics
    capacity = 16 * total
    useful = int(aie["useful_aie_cycles"])
    allocated = int(aie["allocated_aie_cycles"])
    router_ops_per_layer = 2 * 1 * 64 * 32
    router_payload = ceil(router_ops_per_layer / 24)
    gu_tile_payload = ceil((4 * 1 * 64 * 32) / (4 * 24))
    down_tile_payload = ceil((2 * 1 * 32 * 64) / (4 * 24))
    nonlinear_tile = ceil(32 / 1024)
    rows = [
        {"component": "sequence makespan", "cycles": total, "cycle_basis": "wall cycles", "source_type": "DERIVED", "source": "UnifiedSequenceResult", "percentage_of_total_runtime": 100.0},
        {"component": "Router task (each of 3 layers)", "cycles": router_payload + 20, "cycle_basis": "wall cycles/task", "source_type": "DERIVED", "source": "ceil(4096/24)+20", "percentage_of_total_runtime": 3 * (router_payload + 20) / total * 100},
        {"component": "Gate-Up tile task", "cycles": gu_tile_payload + 20, "cycle_basis": "wall cycles/task using 4 AIE units", "source_type": "DERIVED", "source": "ceil(8192/(4*24))+20", "percentage_of_total_runtime": (gu_tile_payload + 20) / total * 100},
        {"component": "Down tile task", "cycles": down_tile_payload + 20, "cycle_basis": "wall cycles/task using 4 AIE units", "source_type": "DERIVED", "source": "ceil(4096/(4*24))+20", "percentage_of_total_runtime": (down_tile_payload + 20) / total * 100},
        {"component": "Nonlinear tile service", "cycles": nonlinear_tile, "cycle_basis": "wall cycles/task", "source_type": "DERIVED", "source": "ceil(32/1024)+0", "percentage_of_total_runtime": nonlinear_tile / total * 100},
        {"component": "AIE useful payload", "cycles": useful, "cycle_basis": "AIE unit-cycles", "source_type": "DERIVED", "source": "AIEPool metrics", "percentage_of_total_runtime": useful / capacity * 100},
        {"component": "AIE task startup/allocation overhead", "cycles": allocated - useful, "cycle_basis": "AIE unit-cycles", "source_type": "ASSUMED", "source": "20-cycle fixed startup per task", "percentage_of_total_runtime": (allocated - useful) / capacity * 100},
        {"component": "AIE idle capacity", "cycles": capacity - allocated, "cycle_basis": "AIE unit-cycles", "source_type": "DERIVED", "source": "16*makespan-allocated", "percentage_of_total_runtime": (capacity - allocated) / capacity * 100},
        {"component": "true-Expert memory starvation", "cycles": no_row["true_expert_memory_starvation_cycles"], "cycle_basis": "summed Expert wall waits", "source_type": "DERIVED", "source": "per-Expert event timing", "percentage_of_total_runtime": float(no_row["true_expert_memory_starvation_cycles"]) / total * 100},
    ]
    by = {(row["sensitivity_axis"], float(row["sensitivity_scale"]), row["policy_internal"]): row for row in sensitivity}
    base = int(by[("memory_bandwidth", 1.0, "no_prediction")]["sequence_makespan_cycles"])
    twice_bw = int(by[("memory_bandwidth", 2.0, "no_prediction")]["sequence_makespan_cycles"])
    twice_compute = int(by[("compute_speed", 2.0, "no_prediction")]["sequence_makespan_cycles"])
    classification = "memory-bound" if base - twice_bw > base - twice_compute else "compute-bound"
    rows.append({
        "component": "end-to-end bottleneck classification",
        "cycles": total,
        "cycle_basis": "representative sequence",
        "source_type": "DERIVED",
        "source": "2x bandwidth vs 2x compute sensitivity",
        "percentage_of_total_runtime": 100.0,
        "analysis": classification,
        "details": f"1x={base}; 2x bandwidth={twice_bw}; 2x compute={twice_compute}",
    })
    return rows


def ocm_rows(baselines: dict[str, tuple[dict[str, Any], object]]) -> list[dict[str, Any]]:
    """Build OCM consistency, allocation, and physical-byte audit rows."""
    rows: list[dict[str, Any]] = [
        {"check": "reference capacity", "value": BASE_OCM_BYTES, "unit": "bytes", "status": "PASS", "source_type": "SYNTHETIC_WORKLOAD_PARAMETER", "source_reference": "M7.2b representative point", "interpretation": "Capacity is explicit and finite, but not a measured board capacity."},
        {"check": "demand working-set normalization", "value": 32768, "unit": "bytes", "status": "PASS_WITH_LIMITATION", "source_type": "DERIVED", "source_reference": "M7.2a large-capacity convergence", "interpretation": "Simulator-derived demand reference, not physical local-memory organization."},
        {"check": "capacity/reference ratio", "value": 1.5, "unit": "ratio", "status": "PASS", "source_type": "DERIVED", "source_reference": "49152/32768", "interpretation": "The representative point intentionally permits moderate speculative pressure."},
        {"check": "bank/port organization", "value": "not modeled", "unit": "n/a", "status": "LIMITATION", "source_type": "ASSUMED", "source_reference": "CapacityOCM abstraction", "interpretation": "No bank conflicts, ports, fragmentation, or local access latency."},
        {"check": "resident vs allocated distinction", "value": "allocation intervals tracked", "unit": "boolean", "status": "PASS", "source_type": "DERIVED", "source_reference": "CapacityOCM byte-cycle accounting", "interpretation": "Peak OCM is allocation occupancy, not transferred bytes."},
    ]
    for policy in POLICIES:
        row, result = baselines[policy]
        initial = sum(p.initial_allocated_speculative_bytes for p in result.predictions)
        physical = sum(p.physical_speculative_bytes_transferred_by_router for p in result.predictions)
        rows.append({
            "check": f"policy accounting: {POLICY_LABELS[policy]}",
            "value": row["peak_ocm_bytes"],
            "unit": "peak allocated bytes",
            "status": "PASS" if int(row["peak_ocm_bytes"]) <= BASE_OCM_BYTES else "FAIL",
            "source_type": "DERIVED",
            "source_reference": "representative unified simulation",
            "initial_speculative_allocation_bytes": initial,
            "physical_speculative_bytes_at_router": physical,
            "completed_speculative_bytes": sum(p.completed_speculative_bytes for p in result.predictions),
            "rejected_speculative_bytes": result.rejected_future_prefetch_bytes,
            "evicted_speculative_bytes": result.evicted_future_speculative_bytes,
            "interpretation": "Allocation, peak occupancy, and physical progress are separate quantities.",
        })
    return rows


def end_to_end_rows(baselines: dict[str, tuple[dict[str, Any], object]]) -> list[dict[str, Any]]:
    """Record the absence of a direct hardware calibration point without fabrication."""
    return [{
        "validation_target": "representative 3-layer sequence latency",
        "hardware_platform": "none",
        "hardware_measurement_cycles": "",
        "simulator_cycles": baselines["no_prediction"][0]["sequence_makespan_cycles"],
        "absolute_error_cycles": "",
        "relative_error_percent": "",
        "validation_status": "NOT_AVAILABLE",
        "source_reference": "repository-wide search: no measured board result found",
        "notes": "No direct hardware validation available.",
    }]


def validate_artifacts(output: Path) -> dict[str, Any]:
    """Validate required CSV existence, schema, non-emptiness, and key invariants."""
    required = {
        "parameter_provenance.csv": {"parameter", "source_type", "source_reference"},
        "bandwidth_calibration.csv": {"transfer_bytes", "total_dma_cycles", "dominant_term"},
        "compute_calibration.csv": {"component", "cycles", "percentage_of_total_runtime"},
        "ocm_fidelity.csv": {"check", "status", "interpretation"},
        "sensitivity_results.csv": {"sensitivity_axis", "sensitivity_scale", "policy", "sequence_makespan_cycles"},
        "end_to_end_validation.csv": {"validation_status", "notes"},
    }
    counts: dict[str, int] = {}
    for name, fields in required.items():
        path = output / name
        if not path.is_file():
            raise FileNotFoundError(path)
        with path.open(encoding="utf-8") as stream:
            reader = csv.DictReader(stream)
            if not fields <= set(reader.fieldnames or ()):
                raise ValueError(f"{name}: missing fields {sorted(fields - set(reader.fieldnames or ())) }")
            rows = list(reader)
        if not rows:
            raise ValueError(f"{name}: no data rows")
        counts[name] = len(rows)
    if counts["sensitivity_results.csv"] != 52:
        raise ValueError("sensitivity_results.csv must contain 13 points x 4 policies")
    return {"status": "PASS", "row_counts": counts}


def main() -> int:
    """Generate all M7.3 CSV artifacts and a machine-readable validation record."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=OUT)
    args = parser.parse_args()
    output = args.output_dir
    sensitivity, baselines = sensitivity_rows()
    write_csv(output / "parameter_provenance.csv", provenance_rows())
    write_csv(output / "bandwidth_calibration.csv", bandwidth_rows())
    write_csv(output / "compute_calibration.csv", compute_rows(baselines, sensitivity))
    write_csv(output / "ocm_fidelity.csv", ocm_rows(baselines))
    write_csv(output / "sensitivity_results.csv", sensitivity)
    write_csv(output / "end_to_end_validation.csv", end_to_end_rows(baselines))
    validation = validate_artifacts(output)
    validation.update({
        "simulator_semantics_modified": False,
        "full_m72_search_rerun": False,
        "unique_unified_simulator_runs": 44,
        "representative_point": {
            "quality": "medium", "source_milestone": "source_router_done",
            "predictor_latency_cycles": PREDICTOR_LATENCY_CYCLES,
            "ocm_bytes": BASE_OCM_BYTES, "channels": 2,
            "best_bds_candidate_count": 4,
            "best_bds_depths": [1, 1, 1, 1],
        },
    })
    (output / "artifact_validation.json").write_text(
        json.dumps(validation, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"Wrote and validated M7.3 fidelity artifacts in {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
