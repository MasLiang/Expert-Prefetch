# M9 Hardware-Friendly Tile Scheduler Summary

This study evaluates scheduler models in simulation. It does not implement or validate FPGA RTL/HLS.

- Controlled hardware/quality points: 48
- Reported policy simulations: 336
- Offline oracle candidate simulations: 3966
- Online winner memberships including ties: `{'budget_proportional': 29, 'full_expert': 5, 'hardware_greedy': 21, 'ranked_tile': 24, 'uniform_tile': 22}`
- Mean oracle gaps (%): `{'no_prefetch': 43.378082731664165, 'full_expert': 35.739934100656505, 'uniform_tile': 29.4591361700513, 'ranked_tile': 28.595761516219248, 'budget_proportional': 27.88247445922431, 'hardware_greedy': 28.874955014233162, 'oracle': 0.0}`

## Required analysis

1. **Tile allocation versus Expert prefetch.** Compare `full_expert` with uniform/ranked/proportional/greedy in `scheduler_comparison.csv`. Fine-grained policies help most when OCM cannot hold complete high-ranked Experts or prediction truth is deeper in the ranking; full Expert remains strong for perfect predictions and sufficient OCM.
2. **Proximity to oracle.** Hardware Greedy gaps by quality are `{'perfect': 26.41699165497104, 'high': 26.589904172273197, 'medium': 28.80367011655041, 'low': 33.689254113138006}` percent. Budget Proportional gaps are `{'perfect': 23.808653815408633, 'high': 25.90136640045923, 'medium': 27.99101233583133, 'low': 33.82886528519805}` percent. These are event-simulated latency gaps, not analytical proxy scores.
3. **Greedy failure modes.** Greedy can overvalue startup breadth, imbalance useful depth when confidence is poorly calibrated, or pay excessive per-tile DMA setup. One serialized DMA channel and large available OCM can favor complete Expert residency.
4. **Resource-dependent preference.** Lower OCM favors shallow breadth; higher OCM and high-quality ranks favor full Expert/deeper prefixes. More bandwidth changes how much of the planned budget completes inside the lead window.
5. **PL overhead.** `complexity_comparison.csv` gives primitive operation/state counts and a qualitative control-cycle estimate. Hardware Greedy uses quantized integers, shifts, counters, comparisons, and cross multiplications with no division. Budget Proportional requires an accumulator plus fixed-point division/reciprocal. Both are structurally implementable in FPGA-class PL, but no timing closure, area, power, RTL, or HLS claim is made.

Oracle is explicitly offline-only and may use exhaustive event-simulator evaluation. All online policies obey the same OCM, bandwidth-window, DMA, and compute-dependency constraints.
