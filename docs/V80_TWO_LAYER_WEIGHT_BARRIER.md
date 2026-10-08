# Resident Router, two-layer prediction, predecessor weight barrier

The V80 hardware assumptions are those in `configs/hardware/v80_80percent.yaml`:
655.2 GB/s aggregate effective bandwidth, 8,678 DSPs, 300 MHz, one MAC/DSP/cycle,
two equal Expert compute groups, and eight abstract memory channels. This is a
simulator estimate rather than measured FPGA performance. The single-channel
per-Expert mapping still limits each Expert to 81.9 GB/s.

Router weights remain resident. Their storage is explicitly subtracted from
the 67,300,000-byte BRAM+UltraRAM weight-buffer envelope for all MoE layers in
each workload: Small 262,144 bytes, Medium 2,097,152 bytes, Large 12,582,912 bytes.
Router and Expert use the same per-group DSP-derived rate; Router uses both groups.

## Causal execution rule

For target layer L, prediction is triggered at layer L-2 start. Predictor output
is saved when ready. Speculative requests are admitted only after all true
Expert Gate-Up and Down weights of L-1 have actually completed DMA. This is a
global preceding-layer barrier across all channels, stronger than queue priority.
It does not wait for L-1 compute to finish. It does not inspect routing truth
before the L-1 Router resolves.

`UnifiedPredictionSpec.wait_for_previous_layer_weights` enables the rule;
its default is false, preserving prior experiment behavior. Prediction records
now separately report `predictor_ready_cycle` and `prefetch_release_cycle`.
Requests arriving after target Router resolve cannot retroactively prefetch.

The DMA engine retains demand priority and non-preemptive requests. Speculation
started after the predecessor barrier can still block target-layer demand if a
wrong candidate's request is in flight when the target Router resolves.

## Experimental scope

Four sequential layers are simulated, and targets 2 and 3 are predicted.
All comparisons predict the same targets, leaving the first two layers cold.
The same candidate trace is used for one-layer-gated, two-layer-gated, and
two-layer-immediate references. Layer IDs share Expert IDs but weights are keyed
by layer, so no cross-layer weight reuse is assumed.

Five ranking qualities (1.0, 0.8, 0.6, 0.4, 0.2), three prefix policies,
two predictor latencies (0/1 us), two layer gaps (0/10 us), and three timing
modes over three workloads yield 540 sequence rows and 1,080 prediction records.
The 10 us gaps stand for additional inter-layer work; they are synthetic.

Each prediction has a byte cap equal to 25% of the remaining weight OCM.
Two pending predictions together can therefore use at most half the weight
buffer. This reserves space for demand streaming. The unified full-Expert
policy remains byte-budget limited: a complete Expert that does not fit may
have only partial stage weights admitted. It is not a strict all-or-nothing
Full Expert allocator.

An unrestricted whole-OCM speculative configuration exposed an existing finite
capacity liveness limitation; the reported sweep deliberately uses the explicit
reserved-buffer configuration above. These results do not validate arbitrary
speculative buffer allocations.

## Results at perfect ranking, zero predictor latency

No added inter-layer gap:

| Workload | No-prefetch four-layer latency | Uniform prefix latency | Reduction |
|---|---:|---:|---:|
| Small | 112.320 us | 111.840 us | 0.427% |
| Medium | 650.787 us | 649.873 us | 0.140% |
| Large | 2497.213 us | 2494.793 us | 0.097% |

With 10 us between consecutive layers, the best tested perfect-ranking policy
for each workload is:

| Workload | Policy | Baseline | Predicted | Reduction |
|---|---|---:|---:|---:|
| Small | Budget-limited full Expert | 142.320 us | 121.840 us | 14.390% |
| Medium | Ranked prefix | 680.787 us | 661.693 us | 2.805% |
| Large | Uniform prefix | 2527.213 us | 2504.793 us | 0.887% |

One-layer-gated and two-layer-gated have exactly equal sequence latency for
every matched point at the tested 0/1 us prediction latencies. Both predictions
are ready before the predecessor weight barrier. Earlier prediction can help
when prediction latency exceeds the available one-layer feature-to-barrier
interval, but this sweep does not measure that regime.

The benefit is conditional on ranking and policy. At zero inter-layer gap,
some wrong-prefetch configurations increase sequence latency; deeper prefetch
can also regress even at perfect ranking because it interferes with target
demand. Waiting for the previous weights removes interference with those
transfers but does not eliminate all future DMA interference.

## Reproduction and checks

```bash
python examples/v80_two_layer_prefetch.py
pytest -q
```

`results/v80_two_layer_prefetch/comparison.csv` contains sequence latency and
weight-wait reduction. `prediction_timing.csv` records prediction-ready,
preceding-weight-done, prefetch-release, and target-start/resolve cycles.
Representative event and prediction JSON files support timeline inspection.

Every gated speculative start is checked against the previous layer's last
weight completion. Unit tests establish early prediction, observed barrier
release before compute completion, one-vs-two-layer equality, and late prediction
handling. The existing regression suite passes with the gate disabled by default.
