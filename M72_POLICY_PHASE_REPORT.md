# M7.2 Unified Policy Phase Report

## 1. Executive Summary

M7.2 evaluated predictive Expert prefetch as a policy-selection problem, with event-derived sequence makespan as the sole performance objective. The final controlled grid contains 240 configurations, two explicitly separated comparison modes, and 12,249 feasible policy-plan evaluations. The unconstrained mode selected no prefetch in 147/240 cells, bounded breadth–depth search (BDS) in 89, full-Expert prefetch in 3, and fixed ranked prefix in 1. Only 67/240 winners beat the second choice by at least 1%; 125 cells have exact latency ties. These data therefore support policy regions and near-optimal sets, not a claim that one prefetch policy universally wins.

The most defensible controlled-workload result is that predictor quality alone is insufficient. All 48 wrong-ranking cells select no prefetch, but even perfect ranking selects no prefetch in 20/48 cells. Finite OCM and memory service matter strongly: no prefetch wins 49/60 cells at 0.75× demand reference and 68/80 one-channel cells. BDS is materially (>1%) faster than the best feasible simple policy in 36 of 212 mutually comparable cells and reaches an 18.34% maximum advantage.

All controlled-hardware findings below are category **B: unified architecture results under a controlled synthetic workload**. The VEK280 subset is category **C: theoretical projection**. There are no category-D measured-hardware results.

## 2. Simulator and Experimental Scope

Every main row asserts `simulator_scope=unified_cross_layer` and `synthetic_lead_time=false`. Prediction readiness comes from a source-layer milestone plus predictor latency in the validated `UnifiedSequenceSimulator`; no target-local ready-cycle override is used. The target Router remains authoritative. One global event queue, finite OCM, non-preemptive active DMA bursts, AIE sharing, F-tile streaming, and canonical M7.1b byte lifecycle accounting remain unchanged.

The main workload is a deterministic three-MoE-layer, B=1, K=4 synthetic sequence. Main placement is `round_robin_expert`. Controlled hardware parameters are synthetic placeholders, not VEK280 measurements. The complete experiment definition is in [`m72_experiment.json`](results/m72_policy_phase/configs/m72_experiment.json), and run accounting is in [`run_summary.json`](results/m72_policy_phase/summaries/run_summary.json).

## 3. Policy Set and Search Method

The study compares no prediction, full Expert, uniform one-tile prefix, ranked prefix, and bounded GU-only BDS. BDS explores M={1,2,4,8,16}; depths are selected from {0,1,2,4,8,full}, with all-zero requests deduplicated against no prefetch. It includes uniform, monotonic rank-weighted, and selected mixed plans. The final search accounting reports 25,920 generated candidates, 12,720 unique plans, 5,520 static feasibility/zero-plan prunes, and 7,680 exact duplicate prunes.

The winner is minimum sequence makespan. Exact ties retain all raw latency winners and use, only for the preferred tie choice: lower wasted physical bytes, lower physical bytes by Router, lower initial allocation, then lower peak OCM. A robust winner has at least 1% margin. The raw evaluations and search space are [`m72_policy_evaluations.csv`](results/m72_policy_phase/raw/m72_policy_evaluations.csv) and [`m72_breadth_depth_search.csv`](results/m72_policy_phase/search/m72_breadth_depth_search.csv).

Unconstrained and fixed-budget comparisons are separate. The fixed mode applies a total 128 KiB allocation bound across two source→target predictions; the audited maximum is exactly 131,072 B. Infeasible natural-granularity policies are recorded as absent/infeasible, not assigned fabricated latency.

## 4. Prediction Quality × Prefetch Opportunity

[`Figure 1`](results/m72_policy_phase/figures/fig01_policy_phase_quality_vs_opportunity.png) uses measured opportunity rather than label position. Across all OCM/channel cells, perfect, high, and medium profiles respectively select BDS in 27, 26, and 26 of 48 cells. Low quality selects no prefetch in 38/48; wrong quality selects it in 48/48. Full prefetch appears only at the very-low-opportunity label in this controlled grid. This counter-intuitive location is preserved: the label changes both real source timing and contention context, so nominal “more lead” does not imply free bandwidth.

Answer to RQ1/RQ2/RQ3: no prefetch dominates wrong/low quality, tight OCM, and one-channel regions; full Expert is optimal in only three unconstrained cells; partial BDS is optimal in 89 cells and is the principal predictive region. Imperfect prediction clearly shrinks the already-small full-Expert region, but the relationship is not monotone in a scalar quality alone.

## 5. OCM-Pressure Phase Diagram

The OCM reference is derived per controlled workload/channel as the smallest demand-only capacity within 1% of the 1 MiB result. Main capacities are exactly 0.75×, 1×, 1.5×, and 4× that reference. [`Figure 2`](results/m72_policy_phase/figures/fig02_policy_phase_quality_vs_ocm.png) shows the selected-opportunity/channel slice.

Across the full grid, no prefetch wins 49/60 cells at 0.75×, 46/60 at 1×, 32/60 at 1.5×, and 20/60 at 4×. Full/ranked fixed policies appear only at 4×. Thus finite OCM shrinks the full-Expert region and expands both no-prefetch and bounded partial-policy regions. This answers RQ5 affirmatively for this synthetic workload, not as a hardware-general theorem.

## 6. Memory-Capability Phase Diagram

[`Figure 3`](results/m72_policy_phase/figures/fig03_policy_phase_opportunity_vs_channels.png) fixes medium quality and 1.5× OCM. Across all other axes, BDS wins 12/80 one-channel, 33/80 two-channel, and 44/80 three-channel cells. No prefetch correspondingly drops from 68 to 45 to 34 cells. Additional memory-level parallelism therefore expands the useful-predictive-policy region (RQ6), although it does not guarantee that full Expert becomes best.

The selected placement study produced the same 54,014-cycle BDS winner for round-robin Expert and striped tile. This is one selected point, not evidence that placement is generally irrelevant.

## 7. Breadth–Depth Strategy Regions

Among the 89 BDS-winning cells, selected M is 1 in 35 cells, 2 in 22, 4 in 30, and 8 in 2; M=16 never wins the main B=1 grid. The most common vectors are `(1,1)` (18), `(1,1,1,1)` (14), `(4,)` (12), and `(1,)` (12). Using the documented derived categories, 72 winners are balanced and 17 narrow-deep; no winner meets the deliberately strict wide-shallow rule (M≥8 and mean depth≤2).

[`Figure 4`](results/m72_policy_phase/figures/fig04_breadth_depth_strategy_map.png) contains only competitive partial plans. Low opportunity does not show a categorical narrow/deep takeover: it has 24 balanced and 5 narrow-deep winners. Poor ranking more often causes no prefetch than greater breadth. These answer RQ7–RQ9 without imposing the expected intuition.

## 8. Full Expert vs Partial Prefix

The direct comparison is in [`table2_full_vs_prefix.csv`](results/m72_policy_phase/tables/table2_full_vs_prefix.csv). Full and ranked prefix are infeasible in about 19–20% of main configurations under the modeled OCM admission behavior. Where feasible, always-full mean regret is 1.79% (max 12.31%) and always-ranked-prefix mean regret is 1.50% (max 13.52%). Full Expert is therefore strong and often near-optimal, but its natural allocation makes it unavailable in more tight-capacity cells.

The full-vs-prefix result is not “prefix always saves bandwidth and wins.” Search-selected mixed plans beat the best feasible simple policy in 42/212 comparable cells and by >1% in 36; in the other cells the simple policy is tied or better.

## 9. No-Prefetch Region

No prefetch is the absolute baseline and remains feasible in all 240 cells. It wins every wrong-profile cell, 38/48 low-quality cells, 49/60 0.75×-OCM cells, and 68/80 one-channel cells. Always no-prefetch has 2.95% mean regret, 0% median regret, and 34.50% maximum regret. This wide zero-regret region explains why unconditional speculation is not justified by recall alone.

## 10. Critical-Path Explanation

Six automatically selected cases are listed in [`m72_representative_cases.csv`](results/m72_policy_phase/traces/m72_representative_cases.csv). Each references a compact event JSON; real per-Expert start, GU/Down, memory wait, startup progress/gap, finish, and selected critical-Expert flags are in [`m72_representative_expert_timing.csv`](results/m72_policy_phase/traces/m72_representative_expert_timing.csv).

The phase rows also retain physical/useful/wasted bytes, critical startup gap and blocking, rejected/evicted bytes, and peak OCM. Conservative explanation labels use `near_tie`, `insufficient_opportunity`, `full_prefetch_useful`, `prefix_bandwidth_efficient`, or `mixed_or_unclassified`; they do not overwrite measured winner status.

## 11. Startup vs Sustained Streaming

[`Figure 8`](results/m72_policy_phase/figures/fig08_startup_vs_streaming.png) contrasts critical startup gap and final makespan. Across all feasible predictive plans, 825 plans reduce the critical startup gap relative to no prefetch; 11 (1.33%) nevertheless fail to reduce sequence makespan. For the selected minimum-makespan plan, every critical-gap reduction propagated to a makespan improvement, but this is selection bias and not a general startup metric property.

Recall, StartableRecall, and critical startup gap have Pearson correlations of approximately 0.21, 0.59, and -0.17 with speedup over no prefetch across predictive rows. Hence Recall cannot predict the winner (RQ15); StartableRecall is more explanatory but not sufficient (RQ16); critical gap is causal evidence but is also not a standalone optimizer (RQ17/RQ18). Sustained streaming and resource contention remain necessary explanations. The optional “prefetched compute runway” was not added; startup gap and depth remain proxies.

## 12. Top-K Sensitivity

The reduced perfect/medium-point study selects BDS at K=1,2,4,6. Winning plans are M=1 `(1,)`, M=2 `(1,1)`, M=2 `(1,1)`, and M=2 `(1,1)`, respectively. Larger K does not produce a monotone breadth increase in this selected slice (RQ10). It does not justify extrapolation to the full four-dimensional K-specific grid.

## 13. Batch/Reuse Sensitivity

These are explicitly synthetic batched/high-reuse workloads, not prefill. For fixed high-reuse routing, B=1 selects M=4 depth 1, B=4 selects M=4 depth 4, and B=16 selects M=16 depth 1. Reuse can therefore shift either sustained depth or breadth depending on scale (RQ11). At B=4, balanced and skewed shapes both select M=4 `(2,2,1,1)` in the tested point, with reuse factors 1 and 2 respectively.

## 14. Static-Policy Regret

[`Figure 9`](results/m72_policy_phase/figures/fig09_static_policy_regret.png) and [`table4_static_policy_regret.csv`](results/m72_policy_phase/tables/table4_static_policy_regret.csv) report:

| heuristic | feasible coverage | mean regret | median | max | within 1% |
|---|---:|---:|---:|---:|---:|
| always full | 80.42% | 1.79% | 0.39% | 12.31% | 65.80% |
| always ranked prefix | 80.83% | 1.50% | 0.44% | 13.52% | 67.53% |
| always no prefetch | 100% | 2.95% | 0.00% | 34.50% | 66.67% |

These answer RQ12/RQ13. BDS is materially better than the best simple feasible policy in 36/212 comparable cells (16.98%), answering RQ19.

## 15. VEK280 Theoretical Projection

[`Figure 10`](results/m72_policy_phase/figures/fig10_vek280_theoretical_projection.png) uses `vek280_theoretical_not_measured`: 304 physical AIE-ML tiles, 202 TOPS under MAC=2-op convention, 102.4 GB/s theoretical memory bandwidth, and about 1973 ops/byte ridge. Every row is tagged `theoretical_not_measured`.

At the selected point, perfect prediction selects full prefetch at 1.4805× speedup, medium selects BDS at 1.2272×, and wrong prediction selects no prefetch at 1.0×. The qualitative quality-dependent policy transition survives in this limited projection (RQ20); the controlled-grid region sizes do not automatically transfer to VEK280.

## 16. Robust Findings

- **A — validated simulator mechanism:** Router authority; genuine source→target timing; canonical byte lifecycle; partial in-flight progress distinct from executable readiness; deterministic scheduling and tie handling.
- **B — controlled synthetic result:** wrong ranking makes no prefetch optimal throughout this grid; tight OCM and fewer channels expand the no-prefetch region; BDS provides material gains in a minority of comparable cells.
- **C — theoretical projection:** the three selected VEK280-theoretical quality points transition from full to BDS to no prefetch.
- **D — measured hardware:** none.

Boundary refinement used real `source_expert_done` latency 260 between the coarse 500- and 20-cycle settings. Ten refined cells were evaluated; eight select no prefetch and two select BDS. No recursive interpolation was used.

## 17. Trace-Sensitive Findings

Three deterministic complete-ranking variants at the nominal medium-quality point produce BDS, no-prefetch, and BDS winners: 2/3 winner consistency. Applying the nominal BDS plan gives 0%, 0.767%, and 0% regret. Thus the boundary is trace-sensitive but the nominal plan remains within 1% in all three variants (RQ14). This subset is intentionally small and is not a stochastic confidence interval.

## 18. Remaining Hardware Calibration Needs

The absolute cycles require measured effective AIE throughput by operation, DMA setup/efficiency and burst behavior, OCM capacity/reservation behavior, controller costs, and validated physical mapping/placement. The model also omits the out-of-scope attention/KV/prefill/continuous-batching/cache/NoC/bank/page/power features. Until calibrated, controlled results are mechanism studies and VEK280 results are projections only.

All ten figure sources, filters, PNG/PDF paths, and exact plot-data CSVs are indexed by [`figure_manifest_m72.json`](results/m72_policy_phase/figure_manifest_m72.json).
