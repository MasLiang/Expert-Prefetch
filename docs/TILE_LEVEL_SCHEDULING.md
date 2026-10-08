# Resource-Aware Tile Scheduling

## Research question

The simulator now treats MoE prefetch as a resource-allocation problem:

> Given a ranked Expert prediction, finite OCM, finite memory bandwidth, DMA channels, and compute resources, how should the accelerator allocate and order independently transferable Expert tiles?

The decision variable is a prefix-closed tile allocation per candidate Expert, not merely the number of selected Experts. For logical tile (t) of Expert (e):

\[
x_{e,t}\in\{0,1\},\qquad
\sum_{e,t}x_{e,t}S_{e,t}\le C_{prefetch}\le C_{OCM}.
\]

Prefix closure, (x_{e,t}=1\Rightarrow x_{e,0..t-1}=1), follows from the current sequential F-tile execution order. It prevents allocating an unusable interior tile while an earlier dependency is missing.

This is a generic simulator abstraction. No allocator or experiment profile depends on VEK280.

## Existing engine reused

M8 does not replace the M7 simulator. A generated `PrefetchPlan` is adapted into the existing single-layer event DAG:

```text
ranked prediction
      │
      ▼
TilePrefetchScheduler
      │  logical Expert/F-axis tiles
      ▼
ordered Gate-Up and Down DMA requests
      │
      ▼
MultiChannelMemory → CapacityOCM → AIEPool → NonlinearEngine
```

The original implementation remains authoritative for:

- per-channel bandwidth, setup latency, placement, and non-preemptive DMA;
- demand priority and speculative cancellation/commit;
- finite OCM admission, residency, eviction, and peak accounting;
- Gate-Up → nonlinear → Down dependencies;
- AIE capacity reservations and DMA/compute overlap;
- event ordering, timeline, and old M7 metrics.

Old callers do not provide `tile_prefetch_scheduler`; their behavior and request ordering are unchanged.

## Tile representation

`ExpertTile` represents one logical F-axis compute tile. It contains independently transferable Gate-Up and Down fragments:

```text
size_bytes = gate_up_bytes + down_bytes
```

It also records Expert/tile identity, predictor rank, confidence, optional DMA-ready cycle, and an observable residency state. The live execution state remains in `CapacityOnChipMemory`; the immutable tile object is the scheduler's decision-time view.

The model geometry supplied to a scheduler is a mapping from Expert ID to an ordered tile sequence. This is the scheduler's model input and permits different numbers and sizes of tiles without embedding a board or model assumption in the scheduler.

## Scheduler interface

All schedulers implement:

```python
schedule(prediction, expert_tiles, hardware, timeline) -> PrefetchPlan
```

Inputs include ranked confidence, exact tile byte geometry, available OCM, effective bandwidth/channel count, current cycle/occupancy, Router deadline, and an optional explicit byte budget. The effective allocation budget is:

\[
\min(C_{OCM}-C_{occupied}, C_{explicit}, BW_{channel}\times channels\times lead\_cycles).
\]

This payload envelope is a planning upper bound. The existing DMA engine then applies per-request setup latency, placement, channel availability, and exact integer completion timing, so planned bytes need not all complete by the deadline. The output contains:

- allocated logical tiles;
- total byte budget and allocated bytes;
- prefix depth by Expert;
- deterministic DMA issue order and earliest issue cycle.

The event simulator determines actual start/end cycles after channel contention. The plan never fabricates a start time before resource arbitration.

## Implemented policies

### No prefetch

`NoPrefetchScheduler` returns an empty speculative plan. The existing demand path fetches each Gate-Up/Down tile when execution requires it.

### Full-Expert prefetch

`FullExpertPrefetchScheduler` visits candidates by prediction rank and admits an Expert only if all of its logical tiles fit. It never partially admits the next Expert. This is the coarse-grained baseline.

### Uniform tile prefetch

`UniformTilePrefetchScheduler` allocates breadth-first rounds: tile 0 to every feasible candidate, then tile 1, and so on. With capacity for 16 equal tiles and eight candidates it produces depth two for every candidate.

### Ranked tile prefetch

`RankedTilePrefetchScheduler` repeatedly selects the feasible next prefix tile with the largest confidence-weighted marginal value per byte:

\[
value(e,t)=\frac{confidence_e\,decay^t}{S_{e,t}}.
\]

The default `decay=0.5` gives higher-confidence Experts deeper prefixes while retaining breadth. Ties resolve by rank, tile ID, and deterministic ordering. This is an initial heuristic, not an optimality or learning claim.

## DMA and compute order

Logical tiles are ordered by the allocator. The adapter issues all selected Gate-Up fragments in that order, followed by all selected Down fragments in the same order. This favors startup coverage because Gate-Up tile 0 is the first compute dependency. Each physical request carries a new optional `schedule_order`; legacy requests leave it unset and retain the previous score/rank ordering.

No speculative compute occurs before Router authority. At Router resolution, true tiles commit, queued false tiles cancel, and an already-active false transfer remains non-preemptive. Compute begins only when the needed matrix fragment is resident, the prior stage/tile dependency is complete, and sufficient AIE capacity is available.

## Metrics

`AdvancedSimulationResult.tile_metrics` adds:

- transferred logical prefetch tiles and bytes;
- useful and wasted tiles/bytes;
- `useful_prefetched_bytes / prefetched_bytes` efficiency;
- startup latency from Router resolution;
- true-Expert memory-stall cycles;
- idle AIE unit-cycles;
- candidate logical-tile coverage.

`tile_prefetch_plan` separately records planned tiles/bytes, byte budget, depths, and issue order. Planned allocation and physically transferred traffic are deliberately distinct because queued false tiles may be canceled before transfer.

Conservation is checked on physical traffic:

```text
prefetched tiles = useful tiles + wasted tiles
prefetched bytes = useful bytes + wasted bytes
```

## M8 experiment

The configuration [m8_tile_scheduler.yaml](../configs/m8_tile_scheduler.yaml) defines a generic synthetic, single-token layer with 64 Experts and Top-K 2. Its medium F tile size produces 16 logical tiles per Expert. Base OCM equals two full Expert weight sets. The Router rate is an explicit synthetic control chosen to expose a nontrivial prediction/DMA window; it is not a device throughput claim.

The sweep covers:

- OCM: 0.5×, 1×, 2×;
- per-channel bandwidth: 0.5×, 1×, 2×;
- DMA channels: 1, 2, 4;
- prediction quality: perfect, high, medium, low;
- tile size: small, medium, large;
- policy: no prefetch, full Expert, uniform tile, ranked tile.

Run:

```bash
python examples/m8_tile_scheduler_experiment.py
```

This creates 1,296 simulations in [results/m8_tile_scheduler](../results/m8_tile_scheduler), including the experiment CSV, JSON/Markdown summaries, three PNG/PDF figures, and artifact validation.

## Result interpretation

Across the bounded sweep, ranked-tile allocation averages 1.1239× speedup over no prefetch and 1.0207× over full-Expert prefetch. Mean useful-byte efficiency is 18.3% for ranked, 39.1% for full Expert, and 15.6% for uniform tile. The aggregate efficiency number is dominated by the perfect/high-quality cases where full Expert correctly concentrates on complete true Experts; fine granularity is not an unconditional traffic-efficiency win.

The benefit is conditional:

- At 0.5× OCM (capacity for one full Expert), ranked tile averages 1.0962× relative to full Expert. Fine-grained prefixes exploit the strongest capacity pressure.
- At 1× and 2× OCM, ranked averages below full Expert (0.9868× and 0.9792×). Once multiple complete Experts fit, fragmentation is not automatically beneficial.
- More DMA channels increase the value of broad tile coverage in this placement model: ranked averages 0.9500×, 1.0637×, and 1.0485× versus full Expert at one, two, and four channels. A single channel serializes the broader plan.
- Smaller tiles improve scheduling freedom but pay more DMA setup events. Ranked/full speedup is 1.0195× for small, 1.0185× for medium, and 1.0242× for large tiles on average; the relationship is not monotonic because setup and coverage interact.
- Ranked is more efficient than uniform for perfect/high/medium traces, but at low quality uniform breadth is better: uniform speedup is 1.1016× versus 1.0828× for ranked. Confidence concentration can over-invest in false high-ranked Experts.

For perfect/high predictions, full Expert has higher mean useful-byte efficiency (83.6%/69.6%) and higher mean speedup than ranked. For medium/low predictions, ranked efficiency (18.4%/10.1%) exceeds full Expert (3.2%/0%) and avoids the near-zero/negative average full-Expert benefit. Thus fine-grained allocation reduces wrong-traffic exposure specifically when true Experts occur deeper in the ranking, not for every quality.

Ranked is faster than full Expert in 242 configurations, tied in 7, and slower in 75. Therefore the validated methodology conclusion is not “ranked always wins.” It is that the optimal allocation changes with OCM pressure, bandwidth parallelism, transfer granularity, and prediction rank quality.

## Validation boundary

Tests enforce allocator determinism, byte feasibility, prefix closure, tile/byte conservation, OCM capacity, DMA channel non-overlap, compute-ready dependencies, and single-tile compatibility with the legacy full-Expert policy.

The bundled rates, capacities, prediction ranks, and model are synthetic methodology controls. These results do not constitute FPGA, GPU, AIE, or VEK280 validation and must not be used as absolute hardware-performance claims.
