# M7.2b Opportunity Boundary Refinement

## Scope

M7.2b adds three genuine one-layer-ahead timing points inside the old 0.381→4.546 opportunity gap. It fixes the M7.2 Figure-1 workload, routing, controlled hardware, two memory channels, 49,152-byte OCM (1.5× demand reference), K=4, B=1, placement, and complete deterministic ranking per quality. Only `source_router_done + predictor_latency_cycles` changes. Every main row uses `UnifiedSequenceSimulator`, `simulator_scope=unified_cross_layer`, and `synthetic_lead_time=false`.

## Achieved timing points

| Predictor latency | Realized lead | Opportunity |
|---:|---:|---:|
| 7,167 cycles | 2,047.0 cycles | 1.0661 Experts |
| 4,767 cycles | 4,447.0 cycles | 2.3161 Experts |
| 2,847 cycles | 6,367.0 cycles | 3.3161 Experts |

Three timing controls plus 15 quality/timing cells × 36 policy-plan evaluations produced 543 new simulator runs. The full M7.2 grid was not rerun.

## Winners

The winner pattern is identical at all three opportunity points:

| Quality | Preferred policy | Raw latency winner | Robust ≥1% | Best BDS plan |
|---|---|---|---:|---|
| perfect | BDS | BDS | yes | M=2, `(1,1)` |
| high | BDS | BDS | yes | M=2, `(1,1)` |
| medium | BDS | BDS | yes | M=4, `(1,1,1,1)` |
| low | No Prefetch | No Prefetch | no | M=1, `(1,)` |
| wrong | No Prefetch | No Prefetch | no | M=1, `(1,)` |

Perfect/high/medium BDS speedup is 1.01872× at every new point. Low/wrong no-prefetch speedup is 1.0×. Full Expert and Ranked Prefix never become the preferred policy.

## Transition interpretation

No opportunity-driven policy transition appears between 1.066 and 3.316. Predictor quality is the stronger discriminator: medium-or-better ranking selects BDS, while low/wrong selects no prefetch, independent of the three achieved opportunities. Breadth likewise does not transition with opportunity; it changes with quality from M=2 to M=4, while the non-selected best BDS alternative under low/wrong is M=1.

The coarse endpoints at 0.381 and 4.546 had the same cost-preferred policy regions. They often tied Uniform Prefix, which was optional and intentionally omitted from the refinement policy set; the new strict-vs-tie status therefore should not be compared as if the policy sets were identical. The preferred-policy conclusion is comparable and unchanged.

## Paper-use verdict

Figure 1 is now sufficiently sampled to state that no material policy transition was observed at three genuine intermediate opportunity points for this fixed machine/workload slice. It remains a discrete phase sample, not a continuously interpolated boundary proof. No further sweep is launched by this milestone.

**Intermediate opportunity refinement reveals: NO MATERIAL TRANSITION.**
