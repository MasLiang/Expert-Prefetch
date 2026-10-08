# M9.1 Realistic-Scale Characterization Summary

All workloads are representative synthetic profiles derived from MoE scaling. Hardware profiles are theoretical or synthetic mappings. This is not hardware benchmarking.

- Formal workload/hardware/quality/resource configurations: 432
- Reported scheduler rows: 3024
- Offline Oracle candidate simulations: 21526
- Mean scheduler speedups vs no prefetch: `{'no_prefetch': 1.0, 'full_expert': 1.0247479845010825, 'uniform_tile': 1.0466468934565034, 'ranked_tile': 1.0458207500609245, 'budget_proportional': 1.0491924242861812, 'hardware_greedy': 1.0456956585189172, 'oracle': 1.2750149757090004}`
- Mean tile-policy speedups vs full Expert: `{'uniform_tile': 1.027837728744566, 'ranked_tile': 1.0269607862319907, 'budget_proportional': 1.0294326862143721, 'hardware_greedy': 1.0268108972515826}`
- Hardware Greedy truth-Oracle gap by workload: `{'small_edge_moe': 24.106793418643566, 'medium_moe': 20.461702094146666, 'large_moe': 20.979443496000183}` percent
- Online winner memberships including ties: `{'budget_proportional': 371, 'full_expert': 156, 'hardware_greedy': 328, 'no_prefetch': 132, 'ranked_tile': 331, 'uniform_tile': 342}`

## Required answers

1. **Does tile scheduling consistently outperform full Expert?** The aggregate comparison is shown above and in `scheduler_results.csv`, but “consistently” requires every point. The full report documents reversals by quality, OCM and hardware; no unconditional dominance is claimed.
2. **Scaling with Expert count/size/OCM.** Larger profiles increase logical tile bytes and Expert working sets while candidate breadth grows 16→32→64. OCM is therefore best interpreted in Expert equivalents, not raw bytes. Low Expert-equivalent capacity favors shallow tile breadth; sufficient capacity and accurate early ranks can favor complete Experts.
3. **Across accelerators.** Differences come from effective memory bandwidth, channel parallelism, usable-buffer assumption, compute capacity, and Router lead window. They are architecture-sensitivity results, not vendor benchmarks.
4. **Is Hardware Greedy consistently close to Oracle?** Its mean gaps by workload are `{'small_edge_moe': 24.106793418643566, 'medium_moe': 20.461702094146666, 'large_moe': 20.979443496000183}` percent. The truth Oracle knows the routed Experts, so this gap includes predictor uncertainty. Pointwise results determine whether “close” is justified; the report does not assume it.
5. **Favored conditions.** Full Expert is favored by accurate top ranks, adequate OCM for complete weights, and enough lead-window bandwidth. Tile allocation is favored by tight OCM, deeper true ranks, and windows that transfer useful prefixes but not complete Experts.

SUPPORTED CLAIMS: relative scheduler comparison, scaling trends, resource tradeoffs.

NOT CLAIMED: real FPGA latency, real VEK280 performance, real V80 performance.
