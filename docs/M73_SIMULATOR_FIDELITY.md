# M7.3 Simulator Fidelity and Hardware Mapping

## Scope

This document maps the unified event-driven simulator to a real accelerator data path. It audits fidelity; it does not claim that the controlled profile is a VEK280 measurement. The complete parameter ledger is in [parameter_provenance.csv](../results/m73_fidelity_audit/parameter_provenance.csv).

The research question is comparative: under the same abstract machine, how much true-Expert weight wait can a causally generated prefetch remove relative to no prefetch or exact post-Router fetching? The simulator is suitable for controlled policy and phase studies. It is not a cycle-accurate board model and cannot predict absolute VEK280 latency without calibration.

## Hardware-to-simulator mapping

| Real-hardware stage | Simulator component | Modeled behavior | Simplified behavior | Not modeled |
|---|---|---|---|---|
| External memory | `HardwareProfile` and `MultiChannelMemory` | Finite per-channel payload rate, setup cost, deterministic queues, independent channels, non-preemptive active DMA | Constant efficiency and fixed setup; abstract placement | DRAM/HBM timing, row buffers, refresh, protocol packets, backpressure, host traffic |
| DMA engines/interconnect | `MultiChannelMemory` | Demand/committed traffic priority, speculative score ordering, per-channel mutual exclusion, completion events | One in-flight request per channel; optional tile coalescing | Descriptor engines, credits, NoC contention, AXI burst legality, arbitration implementation |
| Local/on-chip memory | `CapacityOCM` | Byte capacity, allocation lifetime, peak occupancy, speculative rejection/eviction, demand protection | Object/tile store with zero local-access latency | Banks, ports, address layout, fragmentation, ECC, local bandwidth, bank conflicts |
| AIE array | `AIEPool` | Finite logical-unit capacity, concurrent reservations, operation counts, fixed effective rate, startup cost | Homogeneous logical units and analytical duration | Kernel schedule, vector instructions, cascade/stream topology, placement, routing, synchronization details |
| Nonlinear engine | `NonlinearEngine` | Independent serialized service and causal dependency | Fixed elements/cycle | DSP placement, local traffic and shared interconnect contention |
| Router/controller | unified event queue | Router-before-Top-K dependencies, source-milestone prediction, commit/cancel, deterministic same-cycle ordering | Control delays are fixed or implicit | Firmware variability, interrupts and command-queue overhead |

The abstract path is therefore:

```text
real:       memory -> DMA/interconnect -> local memory -> AIE compute
simulator:  profile -> MultiChannelMemory -> CapacityOCM -> AIEPool
                              \-> deterministic event/dependency graph <-/
```

## Validated behavior

“Validated” here means validated against analytical invariants and simulator tests, not against a physical board.

- Event causality is enforced: a target prefetch is issued only after its configured source-layer milestone plus predictor latency.
- A channel never serves overlapping requests when `max_inflight_per_channel=1`; active speculative requests are non-preemptive.
- Demand/committed requests outrank queued background and speculative requests.
- AIE reservations never exceed the configured logical-unit capacity.
- Expert compute cannot consume a weight tile before its DMA completion and OCM residency.
- OCM occupancy never exceeds configured capacity; running allocations are protected and rejected/evicted bytes are recorded.
- Results and ordering are deterministic for identical inputs.
- Work/byte formulas and integer cycle rounding are covered by the existing unit and metamorphic test suites.

These properties establish event correctness and internal accounting consistency.

## Modeling assumptions

The following behaviors are intentional assumptions rather than hardware-validated facts:

- The controlled profile uses 1 GHz, 16 GB/s/channel, 0.8 bandwidth efficiency, 40-cycle DMA setup, 16 logical AIE units, 32 ops/cycle/unit, 0.75 AIE efficiency, and 20-cycle task startup.
- Memory channels have independent, linearly additive ceilings. `round_robin_expert` maps an Expert to one channel; it does not stripe an individual request.
- DMA setup and effective payload rates do not vary with address, transfer history, contention pattern, or matrix type.
- OCM is a byte-capacity store. Allocation/residency is physically meaningful as a capacity constraint, but its organization and access timing are not modeled.
- Compute duration is `ceil(operations / (logical_units × effective_rate)) + startup`. All operation classes use the same base rate in the controlled profile.
- DMA, AIE, and nonlinear work can overlap whenever the event dependencies and their separate resource constraints permit it. Shared NoC contention is absent.
- The M7.2 workload (`H=64`, `F=256`, `E=32`, `K=4`, batch 1, three layers) and predictor timing/quality are synthetic study controls.

No audited parameter is classified as `MEASURED`, `HARDWARE_SPECIFIED`, or `CALIBRATED`. Values labeled `DERIVED` are arithmetic consequences of assumptions or simulator observations; that label does not make the underlying rate hardware-grounded.

## Memory fidelity

The [bandwidth calibration](../results/m73_fidelity_audit/bandwidth_calibration.csv) verifies dimensional consistency. At 1 GHz, the controlled 16 GB/s/channel peak with 0.8 efficiency becomes 12.8 bytes/cycle/channel. Aggregate ceilings for one, two, and three independent channels are 12.8, 25.6, and 38.4 GB/s.

This scaling is mathematically correct for the abstraction. It does not show that a board sustains linear channel scaling. With round-robin placement, a 4 KiB request still uses one channel: its payload is 320 cycles plus 40 setup cycles. For a 512-byte request, payload and setup are both 40 cycles, so setup is 50% of service time. Payload bandwidth dominates larger tile and full-Expert transfers.

The repository's `vek280_theoretical` values are unmeasured study inputs. In addition, `examples/m71_unified_characterization.py`, `configs/hardware_profile_vek280_theoretical.yaml`, and `vek280_like_theoretical_profile()` do not form one calibrated profile: their efficiency, startup, logical mapping, and in one helper the AIE count differ. They must not be mixed in a paper as a single hardware configuration.

## Compute fidelity

The [compute calibration](../results/m73_fidelity_audit/compute_calibration.csv) records both wall-task cycles and AIE unit-cycles. Those bases must not be summed together. For the controlled workload:

- one Router task is 191 cycles;
- one Gate-Up F-tile task is 106 cycles on four logical units;
- one Down F-tile task is 63 cycles on four logical units;
- one nonlinear F-tile service is one cycle.

The rates and startup terms are assumed; only operation counts, rounding, and aggregate accounting are derived. Per-Expert summed memory-wait cycles can exceed wall time because Experts wait concurrently; it is a diagnostic, not a runtime decomposition.

The representative no-prefetch run is memory-bound within this model. Doubling bandwidth improves 27,642 cycles to 16,122; doubling compute improves it only to 27,324. This supports studying prefetch policy on this controlled point, but it does not establish that a physical deployment has the same balance.

## OCM fidelity

The [OCM audit](../results/m73_fidelity_audit/ocm_fidelity.csv) distinguishes:

- initial speculative allocation requests;
- peak concurrently allocated OCM bytes;
- completed speculative bytes;
- physical speculative transfer progress at target Router resolution;
- rejected and evicted speculative bytes.

The 49,152-byte representative capacity is a synthetic pressure point, equal to 1.5 times the 32 KiB demand working-set reference derived in M7.2a. It is not the claimed size of a VEK280 memory. Speculative bytes are physically meaningful only as transferred weight payload under the abstract placement model; they do not include metadata, alignment, DMA descriptors, ECC, or bank effects.

## Acceptability for the current question

The abstraction is acceptable for causal, apples-to-apples comparisons of policies because all policies share identical resource ceilings, dependencies, queues, and capacity rules. It can test whether a prefetch is early enough, whether it competes with demand, whether it occupies finite OCM, and whether compute hides transfer time.

The abstraction is insufficient for claims about absolute board latency, achieved GB/s or TOPS, energy, detailed AIE mapping, or exact VEK280 resource feasibility. Those claims require measured kernels and board-level calibration points.

## Reproduction

Generate the audit artifacts with:

```bash
python examples/m73_fidelity_audit.py
```

The generator records that it ran 44 unique representative unified simulations, did not rerun the M7.2 search, and did not re-optimize the fixed BDS plan. Machine-readable validation is in [artifact_validation.json](../results/m73_fidelity_audit/artifact_validation.json).
