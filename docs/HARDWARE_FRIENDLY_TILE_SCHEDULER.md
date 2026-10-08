# Hardware-Friendly Online Tile Scheduler

## Scope and claim boundary

M9 asks whether lightweight online logic can approach the best tile allocation without dynamic programming, ILP, exhaustive search, reinforcement learning, or floating-point optimization.

The milestone implements scheduler models and evaluates them through the existing event simulator. It does **not** implement RTL or HLS, report synthesis, establish timing closure, or validate an FPGA. “Hardware friendly” means the decision process can be expressed using bounded fixed-point registers, counters, accumulators, comparisons, shifts, and a small number of arithmetic units.

The machine and workload in [m9_hardware_scheduler.yaml](../configs/m9_hardware_scheduler.yaml) are generic synthetic controls, not VEK280 parameters.

## Scheduler taxonomy

`TileAllocationScheduler` is the common online interface:

```python
schedule(prediction, expert_tiles, hardware_state, timeline_state) -> PrefetchPlan
```

Conceptually its inputs are:

- `PredictionResult`: `(expert_id, confidence)` in deterministic rank order;
- `HardwareState`: OCM capacity, effective bytes/cycle/channel, DMA channel count;
- `TimelineState`: current cycle, occupied OCM, optional explicit budget, Router deadline;
- model state: exact ordered tile geometry for every candidate Expert.

Each `PlannedTileTransfer` returns Expert ID, tile ID, byte cost and integer queue priority (`issue_order`, where lower issues first). Actual DMA start time is determined by the existing channel arbiter rather than guessed by the allocator.

The effective online planning budget is:

\[
B=\min(C_{OCM}-C_{occupied}, C_{explicit}, BW_{channel}\times channels\times lead\_cycles).
\]

All allocation remains prefix-closed because tile (t) cannot execute before tiles (0..t-1). The existing DMA engine subsequently applies setup latency, channel placement, non-preemption, demand priority, and exact completion time.

## Online schedulers

### Existing baselines

- `NoPrefetchScheduler`: no speculative allocation.
- `FullExpertPrefetchScheduler`: admits only complete Experts in rank order.
- `UniformTilePrefetchScheduler`: breadth-first equal-depth rounds.
- `RankedTilePrefetchScheduler`: retained M8 floating-point software heuristic. It is a comparison reference, not the proposed PL implementation.

### Budget Proportional

`BudgetProportionalScheduler` quantizes confidence to a configurable fixed-point integer. For candidate (e):

\[
share_e=\left\lfloor\frac{B\times q_e}{\sum_jq_j}\right\rfloor.
\]

It allocates the largest whole prefix fitting that share, then makes rank-ordered residual rounds so whole-tile rounding does not strand capacity. Required structures are a confidence accumulator, per-Expert share/depth registers, integer multiplication, and a fixed-point divider or reciprocal unit.

### Hardware-Aware Greedy

`HardwareAwareGreedyScheduler` quantizes confidence and evaluates only the next dependency-legal prefix tile of each Expert. Its conceptual score is:

\[
score(e,t)=\frac{q_e\times priority(t)}{bytes(e,t)\times(1+channel\_load(e))}.
\]

`priority(t)` is implemented by a configurable right shift with depth. The implementation never divides scores: it compares candidates (a,b) using cross multiplication,

\[
numerator_a\times cost_b > numerator_b\times cost_a.
\]

State consists of quantized confidence, next depth per Expert, and small per-channel load counters. Selection is deterministic by score, rank, tile ID, and Expert ID. When every Expert has exactly one tile, the scheduler deliberately reuses the old Ranked allocation path, satisfying single-tile compatibility.

This greedy is “hardware aware” in a limited and explicit sense: its total budget includes the bandwidth window and it penalizes repeated assignment to the same round-robin DMA channel. It does not predict future event-DAG latency.

## Offline latency oracle

`OracleTileAllocationScheduler` is explicitly marked `hardware_implementable=False`. For the Top-K=2 study it enumerates every feasible true-Expert prefix-depth pair under the same OCM/bandwidth planning budget, tests breadth-first and Expert-major issue order, and evaluates each plan using the full event simulator.

The selected objective is minimum end-to-end layer latency, with allocated bytes and plan identity as deterministic tie-breakers. Across the 12 hardware points, 3,966 Oracle candidate simulations were executed.

The Oracle knows the true Experts. Its gap therefore contains two effects:

1. allocation/scheduling regret; and
2. predictor uncertainty that no online scheduler can eliminate.

Consequently, “gap to truth-oracle” must not be interpreted as pure implementation inefficiency.

## Complexity estimation

[complexity_comparison.csv](../results/m9_hardware_scheduler/complexity_comparison.csv) records primitive operation counts and state entries for 16 candidates, 16 tiles/Expert, and a 16-tile representative allocation.

| Scheduler | Comparisons | Multiplications | Divisions | State entries | Estimated control cycles | PL classification |
|---|---:|---:|---:|---:|---:|---|
| Full Expert | 16 | 0 | 0 | 16 | 32 | simple |
| Uniform Tile | 256 | 0 | 0 | 16 | 32 | simple counters |
| Budget Proportional | 32 | 16 | 16 | 48 | 176 | fixed-point divider/reciprocal needed |
| Hardware Greedy | 240 | 480 | 0 | 32 | 112 | comparator tree and cross multipliers |
| Ranked Tile | 240 | 256 | 256 | 32 | 128 | software floating-point reference |
| Oracle | exhaustive | exhaustive | offline | offline | offline | not implementable online |

The control-cycle estimate assumes bounded pipelining/parallel comparison and is qualitative, not a synthesis result. Relative to the simulated layer latency, mean estimated overhead is about 1.58% for Hardware Greedy and 1.71% for Budget Proportional. This supports plausibility for further RTL/HLS design, not proof of area/frequency feasibility.

## Experiment

The experiment covers:

- OCM: 0.25×, 0.5×, 1×, 2×;
- per-channel bandwidth: 0.5×, 1×, 2×;
- predictor quality: perfect, high, medium, low;
- seven schedulers including the offline Oracle;
- fixed single token, batch one, Top-K 2, two DMA channels, and 16 tiles/Expert.

Run:

```bash
python examples/m9_hardware_scheduler_experiment.py
```

Outputs are in [results/m9_hardware_scheduler](../results/m9_hardware_scheduler):

- `experiment.csv`: all 336 reported scheduler rows;
- `scheduler_comparison.csv`: aggregate latency, efficiency, and Oracle gap;
- `oracle_gap_analysis.csv`: all non-Oracle pointwise gaps;
- `complexity_comparison.csv`: qualitative PL cost;
- summaries, artifact validation, and three PNG/PDF figures.

## Results and required answers

### 1. Tile allocation versus Expert-level prefetch

Mean speedup versus no prefetch is:

- Full Expert: 1.0675×;
- Uniform Tile: 1.1083×;
- Ranked Tile: 1.1165×;
- Budget Proportional: 1.1240×;
- Hardware Greedy: 1.1144×.

Thus fine-grained allocation improves the sweep average over Full Expert. The advantage is not universal: Full Expert remains an online winner in five configurations, particularly when prediction is accurate and enough complete Experts fit.

### 2. Proximity of hardware-friendly schedulers to Oracle

Mean latency gaps to the truth-informed Oracle are:

- Budget Proportional: 27.88%;
- Hardware Greedy: 28.87%;
- Ranked Tile: 28.60%;
- Uniform Tile: 29.46%;
- Full Expert: 35.74%.

Therefore the current lightweight schedulers are **not near the truth-oracle** in absolute terms. Budget Proportional and Hardware Greedy are, however, close to the best tested online scheduler at each point: their mean online regret is 2.77% and 3.69%, respectively. The remaining Oracle gap is strongly affected by not knowing the true Expert set.

### 3. When simple Greedy fails

Hardware Greedy fails or loses advantage when:

- low-quality confidence directs depth toward false early candidates;
- its channel-load penalty chooses breadth that does not lie on the true critical path;
- one channel serializes many small speculative requests;
- per-tile DMA setup overwhelms marginal coverage;
- larger OCM makes full true-Expert residency more useful than shallow breadth.

Its mean Oracle gap grows from 26.42% at perfect quality to 33.69% at low quality. It is an online winner in 21 of 48 points including ties, versus 29 for Budget Proportional.

### 4. Resource constraints and preferred policy

- At 0.25× OCM, Uniform, Ranked, and Budget Proportional are winners in all 12 associated quality/bandwidth points (ties included); Hardware Greedy wins nine. Very small capacity collapses several prefix policies to similar shallow plans.
- At 0.5× OCM, Budget Proportional wins nine points, while Ranked and Hardware Greedy win six each.
- At 1×/2× OCM, Full Expert begins to reappear, but Budget Proportional retains the largest total winner count among online policies.
- Increasing bandwidth changes the feasible tile budget and allows deeper plans, but no single online scheduler dominates every bandwidth point.

### 5. Is overhead small enough for PL?

The bounded operation/state estimates are small relative to simulated layer latency, and Hardware Greedy avoids division and floating point. This makes both new schedulers plausible candidates for FPGA-class PL implementation. It does not prove that a particular device meets area, clock, power, routing, or deadline constraints. RTL/HLS synthesis and measured scheduler latency are still required.

## Validation

Tests and artifact checks enforce:

- OCM capacity and bandwidth-window allocation limits;
- deterministic prefix-closed plans;
- `allocated = useful + wasted + cancelled` for tile transfers and bytes;
- non-overlapping per-channel DMA intervals and existing compute dependencies;
- Hardware Greedy single-tile compatibility with M8 Ranked;
- Oracle latency no greater than every heuristic at all 48 reported points;
- all prior M7/M8 behavior and tests.

These results validate the simulator methodology and online algorithm structure only. They do not claim FPGA implementation or hardware validation.
