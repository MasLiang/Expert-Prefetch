# M7.3 Simulator Fidelity Audit Report

## 1. Executive summary

The unified simulator is a credible instrument for the M7.x paper's **controlled comparative claims**, provided the paper consistently describes the results as event-driven studies under synthetic or theoretical hardware profiles. It is not yet validated for absolute VEK280 performance claims.

The audit found no simulator-semantics bug and made no scheduling, DMA, memory, or execution change. The existing causality and resource invariants remain intact. The main fidelity gap is provenance: none of the audited performance parameters is measured, hardware-specified and verified, or calibrated against a board. A second issue is that the repository contains multiple inconsistent VEK280-like theoretical profiles; they are acceptable as separate sensitivity inputs but not as one canonical hardware mapping.

## 2. Simulator correctness vs hardware fidelity

Simulator correctness and hardware fidelity answer different questions.

Event correctness is strong: deterministic event ordering, source-to-target prediction causality, DMA/AIE resource exclusion, demand priority, commit/cancel behavior, finite OCM admission, and tile dependencies are implemented and tested. M7.1c's metamorphic invariants and the current full test suite test internal consistency.

Architectural fidelity is intentionally coarse. The simulator models finite bandwidth, setup latency, queue priority, local capacity, logical compute capacity and causal overlap. It does not model DRAM row timing, AXI/NoC packets, OCM banks/ports, AIE placement/routing, kernel instruction schedules, or firmware overhead. Therefore passing event tests cannot be presented as board validation.

The detailed mapping is in [docs/M73_SIMULATOR_FIDELITY.md](docs/M73_SIMULATOR_FIDELITY.md).

## 3. Parameter provenance summary

The complete 39-row ledger is [parameter_provenance.csv](results/m73_fidelity_audit/parameter_provenance.csv). It covers memory rates and channels, DMA transfer/setup/queues, OCM capacity/admission/residency/eviction, AIE capacity/rate/startup/duration, nonlinear service, overlap assumptions, workload dimensions, and predictor timing.

Key interpretation:

- `ASSUMED` identifies an explicit modeling input or mechanism without direct hardware validation.
- `SYNTHETIC_WORKLOAD_PARAMETER` identifies a controlled experimental coordinate, not a device characteristic.
- `DERIVED` identifies arithmetic or simulator-observed consequences. It does not upgrade the confidence of assumed inputs.
- There are no `MEASURED`, `HARDWARE_SPECIFIED`, or `CALIBRATED` entries in this audit.

The 102.4 GB/s, 202 TOPS, and 304-AIE values used by the M7.1 script are classified as unmeasured theoretical assumptions. They must be labeled “theoretical, not measured.” The separate YAML and helper profiles use different efficiencies, startup costs, logical mappings, or AIE counts and must not be combined.

## 4. Memory fidelity

[bandwidth_calibration.csv](results/m73_fidelity_audit/bandwidth_calibration.csv) confirms that units and duration calculations are internally consistent. The controlled profile yields 12.8 effective bytes/cycle/channel at 1 GHz. Its aggregate ceiling scales linearly from 12.8 to 25.6 to 38.4 GB/s for one to three channels.

That is expected model scaling, not measured channel scaling. Under round-robin Expert placement, each tile request uses one channel, so additional channels improve concurrency but not the service time of one request. At the reference channel rate:

- 512 bytes: 40 payload + 40 setup cycles; setup is 50%;
- 2,048 bytes: 160 payload + 40 setup cycles;
- 4,096 bytes: 320 payload + 40 setup cycles;
- 49,152 bytes: 3,840 payload + 40 setup cycles.

Thus latency dominates very small transfers, while bandwidth dominates the study's 2–4 KiB weight tiles and larger transfers.

## 5. Compute fidelity

[compute_calibration.csv](results/m73_fidelity_audit/compute_calibration.csv) separates analytical wall-task latency from AIE unit-cycle utilization. The operation formulas and rounding are deterministic, but 32 ops/cycle/unit, 0.75 efficiency, 20-cycle task startup, and nonlinear throughput are assumptions.

The reference no-prefetch sequence takes 27,642 cycles. Only 14.8% of board AIE capacity is allocated; 11.3% is useful payload and 3.5% is startup/allocation overhead. The large idle-capacity fraction is consistent with a memory-bound, dependency-limited decode-like workload, not evidence of achieved device utilization.

Critical-Expert timing is reasonable relative to the model because each Expert must acquire Gate-Up and Down tiles and obey nonlinear dependencies. Summed true-Expert memory starvation is 59,032 cycles and exceeds wall time because multiple Experts can wait concurrently. It must not be reported as an exclusive runtime component.

## 6. Storage fidelity

[ocm_fidelity.csv](results/m73_fidelity_audit/ocm_fidelity.csv) verifies peak occupancy is at most the configured 49,152 bytes for each representative policy. The simulator separately records speculative allocation, peak concurrent occupancy, physical bytes transferred by Router time, completed bytes, rejection, and eviction.

This accounting is adequate for pressure and policy comparisons. It does not validate a physical memory organization. The capacity object store has no bank conflicts, port limits, fragmentation, alignment overhead, local bandwidth, or local access latency. The 32 KiB demand reference is a simulator-derived normalization, and 49,152 bytes is a deliberate 1.5× experimental pressure point.

## 7. Sensitivity results

The complete 52-row table is [sensitivity_results.csv](results/m73_fidelity_audit/sensitivity_results.csv). It contains 13 independent hardware points × four policies. The fixed representative is M7.2b medium quality, source-Router-done timing, predictor latency 4,767 cycles, two channels, 49,152-byte OCM, and BDS `M=4, depths=(1,1,1,1)`. The BDS plan is not re-optimized at each point, and the full M7.2 search is not rerun.

Findings:

- Across 0.5×, 0.75×, 1×, and 2× bandwidth, fixed BDS has the lowest latency; its speedup versus no prefetch ranges from 1.0067× to 1.0187×.
- At 1.5× bandwidth, all four policies are exactly tied at 20,010 cycles. This is a phase boundary, not a robust BDS win.
- Across 0.5×–2× compute speed, fixed BDS remains the unique latency winner, with 1.0077×–1.0239× speedup.
- At 0.5× OCM, all policies tie at 27,642 cycles because the prefetched working set cannot create a latency advantage under the fixed plans.
- At 2× OCM, ranked prefix gains 1.0054×, while fixed BDS remains best at 1.0187×.
- The representative workload is memory-bound: 2× bandwidth gives a 1.7146× no-prefetch speedup, whereas 2× compute gives only 1.0116×.

Answer to the sensitivity question: the qualitative conclusion that carefully bounded prefetch can help is robust over most of the tested uncertainty range, but the identity and statistical strength of the winner are not universal. Capacity and bandwidth boundaries create exact ties. The paper should report these phase transitions and avoid claiming unconditional dominance.

## 8. End-to-end validation and limitations

[end_to_end_validation.csv](results/m73_fidelity_audit/end_to_end_validation.csv) records the only honest outcome: **No direct hardware validation available.** Repository-wide audit found no board-measured end-to-end latency, achieved bandwidth, kernel throughput, DMA setup latency, or local-memory calibration point. No values were fabricated.

Consequently, current limitations are:

- absolute cycle counts are profile-conditional;
- VEK280-like modes are theoretical studies, not measured predictions;
- local-memory and interconnect structure are under-modeled;
- AIE kernel efficiency/startup and operation-specific rates are uncalibrated;
- energy and host/system effects are outside scope;
- the sensitivity sweep is one representative point, not a replacement for the full M7.2 policy landscape.

The highest-value next validation is a small board microbenchmark set: sustained per-channel and aggregate DMA bandwidth versus transfer size, DMA setup latency, Router/Gate-Up/Down kernel cycles for the modeled tile shapes, and one end-to-end three-layer trace. Those measurements can populate `MEASURED`/`CALIBRATED` provenance rows without changing event semantics.

## 9. Paper wording recommendation

Recommended wording:

> We use a deterministic, event-driven simulator to compare causal Expert-prefetch policies under shared DMA, finite local-capacity, and logical AIE-resource constraints. Hardware rates are explicit controlled or theoretical inputs rather than measured VEK280 performance. We therefore report relative policy behavior, sensitivity, and phase boundaries; absolute board latency is outside the validated scope.

Avoid wording such as “cycle-accurate VEK280 simulator,” “measured VEK280 bandwidth,” “validated 202 TOPS,” or “BDS always wins.”

### Can the simulator support the M7.x paper claims?

**Yes, conditionally.** It supports claims about causal event behavior, resource contention, relative latency, byte traffic, OCM pressure, and policy phase changes on the declared controlled profiles. This follows from deterministic event/resource validation, explicit provenance, and qualitative sensitivity across bandwidth, compute, and OCM uncertainty.

**No for absolute hardware claims.** It cannot currently support claims of absolute VEK280 latency, achieved bandwidth/TOPS, energy, or exact physical mapping. Direct microbenchmark and end-to-end hardware calibration are missing.

## Reproducibility and validation

Run:

```bash
python examples/m73_fidelity_audit.py
python -m pytest
python -m compileall -q src examples tests
```

The generated [artifact validation record](results/m73_fidelity_audit/artifact_validation.json) checks all six required CSV schemas and row counts. It records 44 unique representative simulations, zero full-M7.2-search reruns, and no simulator semantic modification.
