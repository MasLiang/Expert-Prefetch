# M9.2 Criticality-Aware Tile Scheduling

## 1. Research question

M9.2 separates the observed distance to the truth-informed Oracle into:

1. prediction uncertainty, which limits which ranked candidates are useful;
2. allocation inefficiency, which remains after fixing the predictor output.

It also evaluates whether a small, deterministic criticality term improves the
M9 hardware-friendly greedy allocator. This is a scheduler extension and
analysis pass only. The event engine, OCM/DMA semantics, model workloads, and
all existing M7/M8/M9 schedulers are unchanged.

## 2. Existing scheduler architecture

Every tile allocator implements the pure interface:

```text
schedule(prediction, Expert tile geometry, visible hardware, timeline state)
    -> prefix-closed PrefetchPlan
```

The plan contains byte-feasible logical tiles in deterministic DMA issue order.
The existing event engine—not the allocator—performs transfers, cancellation,
residency, Gate-Up/Down compute, and metric accounting. This boundary lets M9.2
add allocation policies without changing simulator causality.

## 3. Two Oracle definitions

### Full Oracle

The existing Full Oracle knows the true routed Experts. It enumerates their
feasible prefix depths and breadth-major/Expert-major DMA orders, then selects
the minimum event-simulated latency. It is the full-information lower bound.

### Fixed Prediction Oracle

The new Fixed Prediction Oracle gets the **same candidate set**, confidence
values, and rank order as every online scheduler at that configuration. It does
not receive true Expert IDs and cannot insert a candidate, remove one from the
predictor trace, or skip higher-ranked candidates to expose a lower rank.

It enumerates rectangular rank-prefix plans `(breadth, depth)` and two legal DMA
orders:

- breadth-major: tile 0 across the selected rank prefix, then tile 1;
- Expert-major: all selected prefix tiles of rank 0, then rank 1.

The seven online plans are also included in its candidate library, guaranteeing
that the fixed-prediction bound is no worse than a reported online policy. The
offline evaluator selects among these candidates using realized event-simulated
latency. Thus it is an analysis bound conditional on the original prediction,
not a deployable scheduler. The search evaluated 16,056 Full- and
Fixed-Prediction-Oracle candidate simulations in total.

## 4. Criticality-Aware formula

For each feasible next prefix tile, the new scheduler compares:

```text
confidence × depth_priority × execution_urgency × transfer_cycles
-----------------------------------------------------------------
tile_bytes × remaining_slack × (1 + channel_load_tiles)
```

Definitions:

```text
depth_priority      = max(1, 4095 >> tile_depth)
execution_urgency   = tiles_per_expert - tile_depth
transfer_cycles     = ceil(tile_bytes / channel_bytes_per_cycle)
projected_finish    = ceil((channel_queued_bytes + tile_bytes)
                           / channel_bytes_per_cycle)
remaining_slack     = max(1, Router_lead_cycles - projected_finish + 1)
```

Confidence is quantized to 12 bits. Candidate ratios are compared by integer
cross multiplication, so the scheduling loop uses no floating-point operation
or divider. Deterministic ties use `(predicted_rank, tile_id, expert_id)`.
This model uses only online-visible prediction, geometry, lead time, bandwidth,
and channel load. It has no true-Expert input.

The criticality term approximates three effects: early prefix tiles are urgent,
a long tile transfer can create a startup gap, and low remaining lead-time slack
makes that gap more likely to reach the critical path. It remains a heuristic;
the experiment tests rather than assumes its quality.

## 5. Metrics and sign convention

The CSV retains the exact signed formulas requested:

```text
prediction_gap_cycles_full_minus_fixed
    = full_oracle_latency - fixed_prediction_oracle_latency

allocation_gap_cycles_fixed_minus_scheduler
    = fixed_prediction_oracle_latency - scheduler_latency
```

Because lower latency is better, both are normally non-positive. To prevent a
negative “gap” from being misread, plots and analysis additionally provide:

```text
prediction_loss = fixed_prediction_oracle_latency - full_oracle_latency
allocation_loss = scheduler_latency - fixed_prediction_oracle_latency
total_loss      = scheduler_latency - full_oracle_latency
```

Other metrics are:

```text
scheduler_efficiency
    = (no_prefetch_latency - scheduler_latency) / transferred_bytes

critical_path_efficiency
    = (no_prefetch_memory_stall - scheduler_memory_stall)
      / allocated_tile_transfers
```

Zero-transfer plans report zero efficiency rather than dividing by zero.

## 6. Experiment matrix

M9.2 uses all M9.1 workloads and hardware profiles:

```text
3 workloads × 3 hardware profiles × 4 prediction qualities
× 4 OCM scales = 144 configurations
144 configurations × 9 schedulers = 1,296 result rows
```

The schedulers are No Prefetch, Full Expert, Uniform Tile, Ranked Tile, Budget
Proportional, Hardware Greedy, Criticality Aware, Fixed Prediction Oracle, and
Full Oracle. M9.2 fixes bandwidth at each hardware profile's 1× setting because
the requested matrix contains an OCM sweep but no bandwidth sweep.

All workload and hardware provenance limitations from
[M9.1](M91_REALISTIC_WORKLOAD_EVALUATION.md) remain in force.

## 7. Aggregate latency

Mean speedups over No Prefetch are:

| Scheduler | Mean speedup |
|---|---:|
| Full Expert | 1.00725× |
| Uniform Tile | 1.04630× |
| Ranked Tile | 1.04541× |
| Budget Proportional | **1.04753×** |
| Hardware Greedy | 1.04518× |
| Criticality Aware | 1.02151× |
| Fixed Prediction Oracle | 1.12186× |
| Full Oracle | 1.26937× |

These are unweighted means of per-configuration speedups. They are not ratios
of summed cycles and are not real-device measurements.

## 8. How much gap comes from prediction uncertainty?

For Criticality Aware, positive prediction loss relative to Full Oracle grows
with degraded rank quality:

| Quality | Prediction loss / Full-Oracle latency | Allocation loss / Full-Oracle latency |
|---|---:|---:|
| Perfect | 1.97% | 19.68% |
| High | 9.77% | 13.81% |
| Medium | 19.69% | 5.40% |
| Low | 23.26% | 3.11% |

Across points with a nonzero total gap, prediction loss accounts on average for
57.70% of Budget Proportional's gap, 57.48% of Hardware Greedy's gap, and
52.89% of Criticality Aware's gap. The share is strongly quality-dependent:
at Perfect, allocation dominates; at Medium and Low, predictor rank uncertainty
dominates.

## 9. How much gap comes from allocation?

At Perfect quality, the Fixed Prediction Oracle is close to Full Oracle
(1.97% normalized loss), yet current heuristics retain a substantial allocation
loss: 15.15% for Budget Proportional, 15.39% for Hardware Greedy, and 19.68% for
Criticality Aware. This is the cleanest evidence that allocation and DMA order
remain important even when the candidate ranking is strong.

At Low quality the fixed-prediction search cannot bypass the long false-candidate
rank prefix, so prediction loss reaches 23.26% while the remaining Criticality
allocation loss is only 3.11%. Optimizing allocation alone cannot repair that
trace.

## 10. Does Criticality Aware outperform current heuristics?

No: the evaluated formula **does not outperform** the current heuristics in
aggregate. Relative to Hardware Greedy it is better at 11/144 points, ties at
53, and is worse at 80; mean latency improvement is -2.19%. Relative to Budget
Proportional it is better at 10 points, ties at 58, and is worse at 76; mean
improvement is -2.37%.

The result is informative rather than a validation failure. Multiplying depth
priority by urgency and inverse slack can over-concentrate budget on deep tiles
of a few urgent candidates. Prediction uncertainty often makes breadth more
valuable. A future scheduler should calibrate this tradeoff, but M9.2 deliberately
does not introduce parameter search, dynamic programming, or a complex optimizer.

The strategy does have a useful region: 8 of its 11 wins over Hardware Greedy
occur on Large, 6 occur on VEK280-theoretical, and 6 occur at 2× OCM. Its largest
benefit over Full Expert is 11.75% for Large/VEK280-theoretical/Perfect/2× OCM.

## 11. When does Full Expert remain optimal?

Full Expert is an online winner, including ties, at 38/144 points. It is the
unique online winner at only four points: Small, V80-like theoretical, Perfect
prediction, at each of the four OCM scales. This is the setting with accurate
early ranks and enough assumed bandwidth to complete an Expert efficiently.

Most Medium/Low “winner memberships” are ties with No Prefetch or other policies,
not evidence of a strong Full-Expert advantage. They occur when useful predicted
traffic cannot be admitted or reaches the Router window too late.

## 12. When is tile-level scheduling most beneficial?

Budget Proportional's mean speedup over Full Expert is 1.0555× at Perfect,
1.0690× at High, 1.0388× at Medium, and 1.0000× at Low. The largest observed
tile-policy improvement is 16.65% at Small/V80-like-theoretical/High/0.25× OCM,
using Budget Proportional.

This identifies the useful middle region: the predictor ranks true Experts
early enough to exploit but not so perfectly that complete rank-0 Experts are
always best, while constrained OCM rewards partial residency. Extremely poor
ranking removes useful candidates from the transferable prefix, so fine-grained
allocation alone provides no benefit.

## 13. Validation

The checked-in run passes:

- Full Oracle latency ≤ Fixed Prediction Oracle latency at every point;
- Full Oracle latency ≤ every scheduler at every point;
- Fixed Prediction Oracle latency ≤ every online scheduler at every point;
- identical candidate count and true-rank trace across all nine schedulers in a
  configuration;
- byte and tile conservation;
- plan budget, OCM peak, prefix, deterministic issue order, and DMA constraints;
- all existing M7/M8/M9/M9.1 regression tests.

`artifact_validation.json` deliberately records `hardware_validation=false`.

## 14. Outputs and reproduction

The result directory contains:

- `scheduler_comparison.csv`: 1,296 scheduler rows and all efficiency metrics;
- `oracle_decomposition.csv`: 144 fixed/full Oracle decompositions;
- `criticality_analysis.csv`: focused per-configuration comparisons;
- five figures in PNG and PDF;
- `summary.json`, `SUMMARY.md`, and `artifact_validation.json`.

Reproduce M9.2 from the repository root:

```bash
python examples/m92_criticality_scheduler.py
pytest -q
```

The command writes only `results/m92_criticality_scheduler/` and does not
rewrite earlier milestone results.

## 15. Claim boundary

**SUPPORTED**

- simulator-based relative scheduling comparison;
- allocation/prediction gap decomposition under the declared fixed-prediction
  search space;
- hardware-friendly scheduler trends and the negative result for this exact
  criticality formula.

**NOT CLAIMED**

- real FPGA performance;
- real predictor accuracy;
- RTL implementation or achieved PL timing;
- global optimality of the bounded Fixed Prediction Oracle;
- universal superiority of Criticality Aware or another scheduler.

