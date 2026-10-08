# V80 80% resource-envelope evaluation

Official anchors: [AMD DS1013 resources](https://docs.amd.com/r/en-US/ds1013-v80/Adaptive-SoC-Resources)
and [AMD UG1617 product details](https://docs.amd.com/r/en-US/ug1617-alveo-v80/Product-Details).
HBM bandwidth is 819 GB/s; DSP engines are 10,848; BRAM is 132 Mb,
UltraRAM is 541 Mb, and total PL memory is 752 Mb.

At 80% the study uses 655.2 GB/s effective aggregate bandwidth and 8,678
DSPs (integer floor). Using decimal megabits for the rounded published memory
figures, 80% BRAM+UltraRAM is 67,300,000 bytes (64.18 MiB). The alternative
80% total PL-memory upper envelope is 75,200,000 bytes (71.72 MiB).
Total PL memory includes more than BRAM+UltraRAM; it is not automatically a
homogeneous DMA-addressable weight buffer. Neither capacity subtracts per-design
buffer fragmentation or an implementation-specific shell footprint.

Compute assumptions: one MAC/DSP/cycle, two operations/MAC, two equal abstract
Expert engines of 4,339 DSPs, clocks of 300/500 MHz. Full pipeline duty yields
5.2068/8.678 TOPS respectively. Additional 25% pipeline-duty cases test
decode/GEMV underutilization; DSP allocation remains 80% in those cases.
The 300/500 MHz clocks and MAC mapping are assumptions, not achieved V80 rates.
The existing generic pool has AIE-named fields; here they represent DSP groups.
Router uses the same operation rate as Expert GEMM and either one or both groups.

Eight abstract DMA channels and round-robin Expert placement are retained.
Each Expert is bound to one channel, with 81.9 GB/s effective payload bandwidth;
two true Experts therefore do not automatically exploit all eight channels.
DMA setup is assumed 80 cycles per burst and compute startup 30 cycles. The
existing nonlinear model is retained. No simulator or scheduler changes occur.

The experiment covers three M9.1 INT8/Top-2 workload scales, five quality levels,
five deployment schedulers, two clocks, two compute-duty factors, two memory
envelopes, and two Router resource allocations: 1,200 reported rows.
Predictions are available at the current layer start, as in M9.3. This is a
single-layer, cold-weight experiment, not the earlier-layer sequence experiment.

## Results

All tile-policy speedups are 1.0x in this matrix, and weight-wait reductions are
zero. The aggregate byte/window budget usually cannot admit even one complete
logical tile during the short Router lead window. Some underutilized Small
cases admit one tile, but it provides no layer-makespan improvement.

For 300 MHz, full compute duty, BRAM+UltraRAM capacity, and Router using both
groups:

| Workload | Expert size | No-prefetch layer latency | Tile-policy latency | Ideal no-weight-stall latency |
|---|---:|---:|---:|---:|
| Small | 1.5 MiB | 28.080 us | 28.080 us | 4.650 us |
| Medium | 12 MiB | 162.697 us | 162.697 us | 13.117 us |
| Large | 48 MiB | 624.303 us | 624.303 us | 42.280 us |

Ideal is a model lower bound with demand weights immediately ready, not achieved
prefetch performance. Large Top-2 weights total 96 MiB, exceeding either 80%
OCM envelope; Small and Medium total weights fit. The simulator starts each
layer cold and does not turn this capacity into warm-cache residency.

For an earlier-layer predictor, payload-only transfer times at 655.2 GB/s are
approximately 0.150/1.200/4.801 us per logical tile for Small/Medium/Large, or
2.401/19.205/76.819 us per full Expert. Under the present single-channel-per-Expert
mapping, these payload times are eight times larger; setup adds further cost.
These are bandwidth calculations, not a cross-layer simulation result.

The old V80-like profile had assumed Router throughput of 16 ops/cycle/group
versus Expert throughput of 512. The present mapping uses equal DSP-derived
rates and allocates a large fraction of DSPs to Router. It therefore removes
much of that artificial lead window. The result cannot be extrapolated to
earlier-layer prediction or alternative Router/DMA resource partitioning.

Reproduce:

```bash
python examples/v80_80percent_evaluation.py
```

Configuration: `configs/hardware/v80_80percent.yaml`.
Outputs: `results/v80_80percent/evaluation.csv` and `summary.json`.
Per-row checks verify byte conservation and peak OCM capacity.
