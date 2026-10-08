# M7.2a Paper Figure Correction and Result Presentation Cleanup

## Scope and provenance

M7.2a changed presentation, filtering, and derived analysis only. It reused 23,323 cached M7.2 rows and ran eight missing VEK280-theoretical simple-policy evaluations. It did not rerun the main DSE. SHA-256 fingerprints before and after processing prove the four M7.2 source CSVs were unchanged. No unified simulator, scheduler, DMA, OCM, FTileGroup, AIEPool, Router, predictor, or search-objective semantics were intentionally changed.

## 1. Was Figure 4 previously plotting all search candidates?

Yes. The old figure selected every BDS row within a regret threshold, so later large-M markers could cover smaller plans. Corrected Figures 4A/4B contain only overall `search_rank == 1` rows for which BDS is the final cost-preferred policy. There are 89 such rows, one deterministic plan per comparison cell.

## 2. True best-plan breadth

The canonical best BDS distribution is M=1: 35, M=2: 22, M=4: 30, M=8: 2, M=16: 0. Figure 4A shows breadth and Figure 4B independently shows mean prefix depth; neither overlays losing candidates.

## 3. Exact latency ties

After collapsing BDS internally to its best plan and comparing the five canonical policies, 125/240 unconstrained configurations are exact latency ties: **52.08%**. `raw_latency_winners` is retained separately from `preferred_policy_under_tie`.

## 4. Robust winners

Only 46/240 configurations have one unique canonical latency winner with at least 1% margin: **19.17%**. There are 115 unique latency winners in total; 69 are strict but below 1%. Consequently the phase maps use filled circles for robust winners, triangles for strict near-ties, and hollow squares for exact ties resolved by resource-cost preference.

## 5. How often does BDS strictly beat the best fixed predictive policy?

There are 209 configurations in which both BDS and at least one fixed predictive policy are feasible. BDS strictly wins 50/209 (**23.92%**), wins by more than 1% in 36/209 (**17.22%**), and by more than 5% in 11/209 (**5.26%**). Median gain among strict wins is 2.75%; maximum gain is 18.34%.

## 6. Same-latency BDS resource efficiency

BDS ties the best fixed predictive latency in 155 configurations. It allocates fewer speculative bytes in all 155, with a median 122,880-byte reduction. It lowers peak OCM in 43/155 (**27.74%**) and transfers fewer physical speculative bytes in 26/155 (**16.77%**). Allocation efficiency—not a fabricated latency win—is therefore the strongest tie result.

## 7. Correct low-OCM interpretation

At 0.75× OCM, 56.67% of cells are exact ties, 33.33% are strict no-prefetch wins, and 10% are strict predictive/BDS wins. At 1×, the corresponding values are 55%, 33.33%, and 11.67%. Therefore the supported wording is: **at demand-working-set-scale OCM, speculation often provides no latency advantage; extra OCM headroom enables useful future prefetch.** It is not correct to describe every cost-selected no-prefetch cell as strictly fastest.

## 8. Why opportunity × channels was coupled

The original opportunity was `effective aggregate bandwidth × realized lead time / Expert bytes`. Channel count changes aggregate bandwidth, so plotting this metric against channel count placed the swept hardware variable in both axes.

## 9. Corrected channel axis

Figure 3A uses `reference_bandwidth_opportunity_expert_equivalents = original opportunity / channel_count`, equivalent to applying one fixed single-channel effective bandwidth to each row’s genuine realized lead time. It retains timing changes caused by the machine while removing the explicit swept bandwidth multiplier. The channel summary separately reports demand-only channel speedup and additional best-predictive channel benefit. In the representative medium-quality slice, adding a second channel gives about 1.97× demand-only speedup plus roughly 0.036–0.037 additional predictive gain where comparable; the third channel adds no demand-only gain in these points but 0.05–0.177 predictive gain, showing exploitable speculative slack. One high-opportunity one-channel predictive point is explicitly infeasible.

## 10. VEK280 theoretical policy matrix

Yes. The matrix now contains exactly 15 rows: perfect, medium, and wrong quality crossed with No Prediction, Full, Uniform Prefix, Ranked Prefix, and Best BDS. All cells are explicit and all happen to be feasible. Speedups are:

| Quality | No prediction | Full | Uniform | Ranked | Best BDS |
|---|---:|---:|---:|---:|---:|
| perfect | 1.000 | 1.480 | 1.053 | 1.324 | 1.370 |
| medium | 1.000 | 1.190 | 1.053 | 1.227 | 1.227 |
| wrong | 1.000 | 0.983 | 1.000 | 0.995 | 1.000 |

This remains a **VEK280 THEORETICAL MODEL — NOT MEASURED**.

## 11. Static-policy feasibility

Always Full is feasible in 193/240 cells (80.42%), Always Ranked Prefix in 194/240 (80.83%), and Always No Prefetch in 240/240 (100%). Figure 9A presents feasibility separately from regret.

## 12. Static-policy regret distribution

Regret is calculated only on feasible cells:

| Heuristic | Median | P90 | P95 | Max |
|---|---:|---:|---:|---:|
| Always Full | 0.394% | 5.815% | 9.570% | 12.307% |
| Always Prefix | 0.443% | 5.496% | 6.487% | 13.519% |
| Always No Prefetch | 0.000% | 11.513% | 20.106% | 34.498% |

Figure 9B is the feasible-only empirical CDF; it does not turn infeasibility into zero regret.

## 13. Strongest main-paper candidates

Recommended main-body figures are Figure 1A (quality/opportunity with confidence semantics), Figure 3A (methodologically independent channel axis), Figure 4A/4B as a paired breadth/depth result, Figure 9A/9B as a paired feasibility/regret result, and Figure 12 (same-latency allocation efficiency). Figure 10A is appropriate for a clearly labeled theoretical projection section.

## 14. Undersampled phase boundaries

The largest opportunity gap is **4.165 Expert equivalents**, from 0.381 to 4.546. The report recommends a future point near 2.463 using a genuine source milestone/predictor-latency configuration. M7.2a did not launch this sweep and does not interpolate across the gap.

## 15. Recommended additional simulation after cleanup

Run a small boundary-only unified study around fixed-reference opportunity 2–3, using real source milestones/latencies, and repeat a few cells across deterministic ranking variants. No full-grid rerun is justified. Hardware calibration should precede stronger VEK280 claims.

## Outputs

Corrected data, tables, figures, diagnostics, summaries, and manifest are under `results/m72a_paper_figure_cleanup/`. Every figure has PNG, PDF, and exact plotting CSV. `figure_manifest_m72a.json` records filters, axes, grouping, comparison mode, tie semantics, and hardware evidence.

**M7.2a corrected presentation/filtering only. Unified simulator and DSE semantics were not intentionally changed.**
