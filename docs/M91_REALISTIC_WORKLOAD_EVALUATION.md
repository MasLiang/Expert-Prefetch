# M9.1 Realistic Workload Characterization

## 1. Purpose and scope

M9.1 asks whether the M9 tile-allocation conclusions survive larger MoE working
sets and materially different accelerator resource envelopes. It is an
**evaluation-only** extension: the event simulator, DMA/OCM semantics, and all
seven M9 scheduler algorithms are unchanged.

This study simulates one representative decode MoE layer per point. The number
of MoE layers is workload metadata for model scale; it is not multiplied into
the reported single-layer latency. The predictor is the existing deterministic
synthetic sensitivity model. No predictor is trained and no GPU/CUDA execution
is modeled.

## 2. Workload abstraction

`WorkloadProfile` combines a validated `ModelConfig` with:

- number of MoE layers;
- an exact F-dimension tile size;
- candidate count;
- the ranks of the two true Experts at `perfect`, `high`, `medium`, and `low`
  prediction quality.

All three workloads use Top-K=2, INT8 weights and activations, and 16 logical
tiles per Expert. The logical tile contains the matching Gate-Up and Down
F-slice. Its byte and operation counts are derived from the model formulas,
never entered independently.

| Profile | H | F | Experts | MoE layers | Candidates | F/tile | Tiles/Expert | Bytes/logical tile | Bytes/Expert |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Small edge MoE | 512 | 1,024 | 64 | 8 | 16 | 64 | 16 | 98,304 | 1,572,864 |
| Medium MoE | 1,024 | 4,096 | 128 | 16 | 32 | 256 | 16 | 786,432 | 12,582,912 |
| Large MoE | 2,048 | 8,192 | 256 | 24 | 64 | 512 | 16 | 3,145,728 | 50,331,648 |

These are **representative synthetic derived from MoE scaling** profiles, not
claims that a particular published model uses these exact dimensions.

Prediction ranks scale with candidate breadth:

| Profile | Perfect | High | Medium | Low |
|---|---|---|---|---|
| Small | (1, 2) | (1, 4) | (3, 8) | (8, 16) |
| Medium | (1, 2) | (1, 8) | (6, 16) | (16, 32) |
| Large | (1, 2) | (1, 16) | (12, 32) | (32, 64) |

## 3. Accelerator profiles and provenance

| Profile | Tag | Clock | Channels | Peak BW | Assumed efficiency | Effective BW | Base tile-buffer OCM | Effective compute |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| VEK280 theoretical dataflow | `THEORETICAL_NOT_MEASURED` | 1.0 GHz | 3 | 102.4 GB/s | 70% | 71.68 GB/s | 16 MiB | 121.2 TOPS |
| V80-like theoretical streaming | `THEORETICAL_NOT_MEASURED` | 1.0 GHz | 8 | 819 GB/s | 65% | 532.35 GB/s | 64 MiB | 19.6608 TOPS |
| Generic FPGA streaming | `SYNTHETIC_PLACEHOLDER` | 0.5 GHz | 4 | 51.2 GB/s | 65% | 33.28 GB/s | 16 MiB | 2.6624 TOPS |

The VEK280 mapping uses the official VE2802 maximum of 304 AIE-ML tiles only
as a resource anchor and groups them into 38 logical compute units. The 102.4
GB/s memory value, 16 MiB usable scheduler buffer, efficiencies, and setup
costs are study assumptions. See the [AMD Versal AI Edge product selection
guide](https://docs.amd.com/api/khub/documents/NBZYHsY_yM~VQDIhrbddxw/content).

The V80-like mapping anchors only its 32 GB HBM2e, 819 GB/s HBM bandwidth, and
752 Mb total PL-memory maxima to AMD documentation. The V80 is not an AIE
device: its 64 compute groups, eight DMA channels, 64 MiB usable tile buffer,
compute rates, and efficiency factors are abstract theoretical mapping choices,
not device measurements. See [AMD UG1617 V80 product
details](https://docs.amd.com/r/en-US/ug1617-alveo-v80/Product-Details) and
[AMD DS1013 adaptive-SoC resources](https://docs.amd.com/r/en-US/ds1013-v80/Adaptive-SoC-Resources).

The generic profile is fully synthetic. Consequently, comparisons across the
three profiles are architecture-sensitivity experiments, not vendor
benchmarks.

## 4. Experiment matrix

The complete Cartesian product is evaluated:

```text
3 workloads × 3 hardware profiles × 4 prediction qualities
× 4 OCM scales × 3 bandwidth scales = 432 configurations
432 configurations × 7 schedulers = 3,024 reported rows
```

OCM scales are 0.25×, 0.5×, 1×, and 2×. Bandwidth scales are 0.5×, 1×, and
2×. Scaling changes only the profile's usable tile-buffer capacity or per-channel
peak bandwidth; it does not change scheduler logic.

The schedulers are No Prefetch, Full Expert, Uniform Tile, Ranked Tile, Budget
Proportional, Hardware Aware Greedy, and truth-informed offline Oracle. Oracle
enumerates feasible true-Expert prefix plans and uses simulated makespan as its
objective. Oracle results are cached across prediction-quality labels because
the truth set and hardware point are unchanged.

## 5. Validation

The generated matrix passes:

- exact 3,024-row coverage and seven schedulers per configuration;
- logical tile-count and workload scaling checks;
- `allocated = useful + wasted + cancelled` conservation for bytes and tiles;
- allocation-budget and peak-OCM capacity limits;
- Oracle latency no greater than any heuristic at every point;
- deterministic checked-in CSV and plot artifact existence checks.

The run evaluated 21,526 Oracle candidate plans. The artifact validation record
explicitly sets `hardware_validation=false`: no board measurement was performed.

## 6. Aggregate results

Mean speedup relative to No Prefetch across all 432 configurations is:

| Scheduler | Mean speedup |
|---|---:|
| Full Expert | 1.02475× |
| Uniform Tile | 1.04665× |
| Ranked Tile | 1.04582× |
| Budget Proportional | **1.04919×** |
| Hardware Aware Greedy | 1.04570× |
| Oracle | 1.27501× |

Relative to Full Expert, mean latency speedups are 1.02784× for Uniform Tile,
1.02696× for Ranked Tile, 1.02943× for Budget Proportional, and 1.02681× for
Hardware Aware Greedy.

## 7. Does tile scheduling consistently outperform Full Expert?

No. Each tile policy beats Full Expert at 253–276 of the 432 points, ties at
132–155 points, and loses at 24 points. All 24 reversals occur under perfect or
high prediction; 20 occur at 2× bandwidth and 20 use the small workload. The
strongest Full-Expert-favorable point is small/VEK280-theoretical/perfect,
0.25× OCM and 2× bandwidth. Thus the supported claim is aggregate and
conditional advantage, not pointwise dominance.

## 8. Scaling with Expert count and size

Mean Hardware Greedy speedup falls from 1.0778× on Small to 1.0334× on Medium
and 1.0259× on Large. Full Expert falls more sharply: 1.0666×, 1.0057×, and
1.0019×. The Expert count grows together with Expert bytes and candidate count,
so this experiment identifies their combined scaling effect; it does not
causally isolate any one variable.

The result favors tile allocation as complete Experts become expensive relative
to the lead-window transfer budget. OCM should therefore be interpreted in
Expert equivalents, which is included per row in `scheduler_results.csv`, not
only as an absolute byte count.

## 9. OCM pressure and bandwidth

Across OCM scales 0.25×/0.5×/1×/2×, Budget Proportional mean speedups are
1.0448×/1.0555×/1.0465×/1.0499×. These means are not monotonic because a larger
buffer changes residency and eviction interactions in the unchanged event
model and averages heterogeneous workloads. The data therefore supports an OCM
tradeoff, not a universal monotonic-speedup claim.

At 0.5×/1×/2× bandwidth, Full Expert averages 1.0000×/1.0073×/1.0670×, while
Budget Proportional averages 1.0331×/1.0475×/1.0670×. Low bandwidth prevents a
complete Expert from fitting in the useful Router lead window and favors
partial prefixes. At high bandwidth with accurate top ranks, Full Expert can
catch up or win.

## 10. Accelerator sensitivity

Hardware Greedy averages 1.0403× on Generic FPGA, 1.0584× on V80-like, and
1.0384× on VEK280-theoretical. The V80-like profile has the largest assumed
effective bandwidth and usable buffer, and therefore exposes more prefetch
opportunity in this mapping. Because clock, bandwidth, compute abstraction, and
buffer assumptions differ simultaneously, this is a profile-level sensitivity
result rather than an attribution to a single physical feature.

## 11. Is Hardware Aware Greedy consistently close to Oracle?

No. Its mean latency gaps to the truth Oracle are 24.11% (Small), 20.46%
(Medium), and 20.98% (Large). By prediction quality, the mean gaps grow from
18.05% at Perfect to 20.17% at High, 23.12% at Medium, and 26.06% at Low. The
worst observed point is about 72.00%. This gap includes unavailable routing
truth, not only online scheduler inefficiency, so it is not evidence that the
greedy ordering alone causes the entire gap.

## 12. Conditions favoring each granularity

Full Expert is favored by accurate top ranks, enough lead-window bandwidth to
finish a complete Expert, and sufficient buffer capacity. Tile allocation is
favored by tight transfer windows, large Expert working sets, and true Experts
deeper in a broad candidate list. Among the evaluated online policies, Budget
Proportional has the best aggregate mean, but policy ties are frequent and no
universal winner is claimed.

## 13. Figures

1. `fig01_speedup_vs_ocm`: speedup versus OCM scale.
2. `fig02_speedup_vs_experts`: scaling from 64 to 256 Experts.
3. `fig03_scheduler_by_workload`: scheduler comparison by workload.
4. `fig04_oracle_gap_vs_workload`: truth-Oracle gap by workload.
5. `fig05_hardware_comparison`: theoretical/synthetic hardware sensitivity.
6. `fig06_useful_byte_fraction`: useful divided by transferred speculative bytes.

Every figure is stored as PNG and PDF in
`results/m91_realistic_evaluation/`. Exact point-level values remain in the CSV;
figures show aggregates and must not be used as hardware measurements.

## 14. Reproduction

From the repository root:

```bash
python examples/m91_realistic_evaluation.py
pytest -q tests/test_m91_workload_profiles.py \
  tests/test_m91_realistic_evaluation.py \
  tests/test_m9_hardware_scheduler_artifacts.py
```

The experiment rewrites the M9.1 result directory deterministically. It does
not rewrite M7, M8, or M9 artifacts.

## 15. Claim boundary

**SUPPORTED CLAIMS**

- relative scheduler comparison within this simulator;
- scaling trends over the declared matrix;
- OCM/bandwidth/resource tradeoffs under these abstractions.

**NOT CLAIMED**

- real FPGA latency;
- real VEK280 performance;
- real V80 performance;
- a trained or deployable predictor;
- GPU behavior or universal dominance of any scheduler.

