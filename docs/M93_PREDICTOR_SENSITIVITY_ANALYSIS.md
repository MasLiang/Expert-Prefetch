# M9.3 Predictor Sensitivity and Calibration Robustness

## 1. Scope

M9.3 is an evaluation-only study of how predictor ranking quality and confidence
calibration affect the existing tile schedulers. It does not modify simulator
semantics, scheduler algorithms, OCM/DMA behavior, or any M7–M9.2 result.

The study uses the three representative synthetic M9.1 workloads, the VEK280
theoretical, V80-like theoretical, and Generic FPGA streaming profiles, and five
fixed deployment strategies: No Prefetch, Full Expert, Budget Proportional,
Hardware Greedy, and Ranked Tile. Oracle is intentionally excluded as a
deployment strategy.

## 2. Experiment reuse and matrix

The experiment directly reuses `AdvancedSimulator`, `HardwareProfile`, the M9.1
workload loader, and the existing scheduler factory. Only deterministic
`PrecomputedBatchPredictor` traces are newly generated.

```text
3 workloads × 3 hardware profiles × 4 OCM scales × 3 bandwidth scales
× 9 quality levels × 4 calibration modes × 5 schedulers
= 19,440 simulations
```

The well-calibrated subset forms the 4,860-row quality sweep. Calibration
robustness is aggregated over four modes for each resource/quality/scheduler
point, producing another 4,860 rows.

## 3. Continuous quality model

Quality takes values `1.0, 0.9, ..., 0.2`. For true Expert index `i`, one-based
best rank `i+1`, and low-quality target rank approximately
`(i+1) × candidate_count / Top-K`, the trace generator uses:

```text
progress = (1 - quality) / (1 - 0.2)
rank_i = round(best_rank_i
               + progress × (low_quality_rank_i - best_rank_i))
```

Ranks are clamped to a legal strictly increasing sequence. This gives:

| Quality | Small C=16 | Medium C=32 | Large C=64 |
|---:|---|---|---|
| 1.0 | (1, 2) | (1, 2) | (1, 2) |
| 0.9 | (2, 4) | (3, 6) | (5, 10) |
| 0.8 | (3, 5) | (5, 9) | (9, 17) |
| 0.7 | (4, 7) | (7, 13) | (13, 25) |
| 0.6 | (4, 9) | (8, 17) | (16, 33) |
| 0.5 | (5, 11) | (10, 21) | (20, 41) |
| 0.4 | (6, 12) | (12, 24) | (24, 48) |
| 0.3 | (7, 14) | (14, 28) | (28, 56) |
| 0.2 | (8, 16) | (16, 32) | (32, 64) |

Candidate count is fixed within each workload. Candidate membership and true
routing are unchanged by calibration.

## 4. Calibration modes

All score curves are non-increasing, so the existing predictor aggregation
preserves exactly the same candidate ranking:

| Mode | Rank-score curve | Interpretation |
|---|---|---|
| Well calibrated | linear 1.00 → 0.20 | rank confidence prior |
| Over-confident wrong | shallow 0.99 → 0.85 | false ranked candidates remain highly trusted when ranking is imperfect |
| Under-confident correct | compressed 0.35 → 0.15 | even correctly ranked candidates receive weak confidence |
| Uniform | constant 0.50 | confidence carries no rank magnitude information |

“Well calibrated” is a controlled synthetic hypothesis, not calibration against
measurements. Likewise, the other modes are deterministic perturbations, not
outputs from a trained predictor. Scores change; candidate IDs and their list
positions do not.

## 5. Metrics

Each row records latency, speedup over No Prefetch, normalized latency, memory
stall, allocated tiles, allocated/useful/wasted/cancelled bytes, useful divided
by transferred bytes, OCM peak, bandwidth utilization, and the exact ranking
trace ID.

For each fixed resource/quality/scheduler point, robustness reports population
variance, standard deviation, and range of latency and speedup over the four
calibration modes. This measures calibration sensitivity without mixing changes
in candidate ranking.

## 6. Prediction quality threshold

Using aggregate mean speedup and the well-calibrated trace:

| Scheduler | Lowest quality with any positive mean benefit | Lowest quality with >1% mean benefit |
|---|---:|---:|
| Full Expert | 0.9 | 0.9 |
| Budget Proportional | 0.2 | 0.4 |
| Hardware Greedy | 0.2 | 0.2 |
| Ranked Tile | 0.2 | 0.2 |

Thus tile scheduling remains weakly beneficial at the lowest tested quality in
the aggregate, but Budget Proportional's gain at 0.2 is only 0.235%. The
threshold is not universal: individual workload/hardware/resource points can
tie or regress even above these aggregate thresholds.

Mean well-calibrated speedups at quality 0.2 and 1.0 are:

| Scheduler | q=0.2 | q=1.0 |
|---|---:|---:|
| Full Expert | 0.99975× | 1.08202× |
| Budget Proportional | 1.00235× | 1.08732× |
| Hardware Greedy | 1.01078× | 1.07778× |
| Ranked Tile | 1.01100× | 1.07726× |

## 7. Does better predictor quality help more than a better allocator?

Yes over the tested range. Moving quality from 0.2 to 1.0 reduces matched-point
latency by 8.45% for Budget Proportional, 6.64% for Hardware Greedy, and 6.57%
for Ranked Tile. By comparison, the mean latency spread among these three tile
allocators at a fixed well-calibrated point is only 0.96%.

This is an endpoint-versus-cross-sectional comparison, not a causal proof, but
it shows that ranking quality is the larger sensitivity axis in this matrix.
It is consistent with the prediction-loss dominance observed at Medium/Low in
M9.2.

## 8. Budget Proportional calibration sensitivity

Budget Proportional has a mean speedup standard deviation of 0.00544 across
calibration modes and a mean speedup range of 0.01210. Its worst pointwise range
is 0.10976 at Small/V80-like/0.25× OCM/1× bandwidth/quality 0.2.

Its aggregate speedups are 1.03548× (well), 1.03732× (over-confident),
1.03722× (under-confident), and 1.03790× (uniform). The non-well-calibrated
curves are not uniformly better: many points tie and some regress by as much as
0.05669 speedup. Aggregate averaging happens to favor broader/flatter allocation
for this deterministic trace mix.

The result therefore supports sensitivity to confidence magnitude, not a claim
that miscalibration is beneficial.

## 9. Hardware Greedy robustness

Hardware Greedy is more robust than Budget Proportional under these score
perturbations:

| Scheduler | Mean speedup stddev | Mean speedup range | Maximum point range |
|---|---:|---:|---:|
| Budget Proportional | 0.00544 | 0.01210 | 0.10976 |
| Hardware Greedy | **0.00197** | **0.00447** | **0.05807** |
| Ranked Tile | 0.00214 | 0.00477 | 0.05807 |

Hardware Greedy's channel-load term and fixed-point priority reduce the effect
of confidence-curve changes. It is not always faster than Budget Proportional;
this conclusion concerns variance, not mean dominance.

## 10. Useful traffic versus quality

For Budget Proportional, mean useful/transferred bytes rise from 0.0418 at
quality 0.2 to 0.3598 at quality 1.0. Hardware Greedy rises from 0.0319 to
0.3380, and Ranked Tile from 0.0383 to 0.3397. Full Expert reaches 0.3951 at
quality 1.0 but is essentially zero below quality 0.9 because the true Experts
are outside the complete-Expert prefix admitted by the lead-window budget.

The gradual traffic-efficiency increase provides a physical explanation for
the latency curves: higher ranking quality moves useful tiles into earlier DMA
positions and reduces false speculative work.

## 11. When Full Expert is preferable

On the 972 well-calibrated resource/quality points, Full Expert is a winner
including ties at 500 points but is the unique winner at only 24. Across all
four calibration modes it is uniquely best at 96/3,888 points; the same 24
resource/ranking points repeat because Full Expert does not use confidence
magnitude.

Of those 96 unique wins:

- 80 use the Small workload;
- 80 use 2× bandwidth;
- 80 have quality 1.0 and 16 have quality 0.9;
- 64 use the high-bandwidth V80-like theoretical profile.

Full Expert is therefore favored by small complete Expert weights, accurate
top ranks, and enough lead-window bandwidth. It is not the aggregate phase
winner after averaging across all workloads and hardware profiles.

## 12. Scheduler phase diagram

Under the well-calibrated aggregate:

- Budget Proportional wins the mean from quality 0.7 through 1.0 at every OCM
  scale;
- at quality 0.2–0.6, Ranked Tile and Hardware Greedy alternate according to
  OCM scale;
- Full Expert's narrow pointwise region disappears under cross-profile
  averaging.

This phase map is descriptive of the tested grid and deterministic tie-breaking,
not a universal scheduler-selection policy.

## 13. Validation

M9.3 validates:

- identical predictor trace produces identical simulation metrics and timeline;
- calibration mode does not change candidate set, membership, ranking, or true
  Expert ranks;
- true ranks move monotonically downward as quality decreases at C=16/32/64;
- allocated bytes and tiles equal useful + wasted + cancelled;
- every plan obeys its OCM/bandwidth budget and peak OCM capacity;
- the unchanged event engine continues to enforce DMA-channel exclusion;
- all previous M7–M9.2 tests remain passing.

## 14. Outputs and reproduction

Outputs are stored in `results/m93_predictor_sensitivity/`:

- `quality_sweep.csv` — 4,860 well-calibrated rows;
- `calibration_sweep.csv` — 19,440 complete rows;
- `scheduler_robustness.csv` — 4,860 variance/range rows;
- six figures in both PNG and PDF;
- `summary.json`, `SUMMARY.md`, and `artifact_validation.json`.

Reproduce from the repository root:

```bash
python examples/m93_predictor_sensitivity.py
pytest -q
```

## 15. Claim boundary

**SUPPORTED CLAIMS**

- predictor quality sensitivity in the declared deterministic trace model;
- confidence calibration effects while candidate ranking is fixed;
- relative scheduler robustness over the declared workload/resource matrix.

**NOT CLAIMED**

- real predictor accuracy;
- learned predictor implementation;
- hardware predictor overhead;
- measured VEK280, V80, or FPGA performance;
- universal prediction-quality thresholds outside this matrix.

