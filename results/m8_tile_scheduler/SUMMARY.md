# M8 Tile Scheduler Experiment Summary

This is simulator-based methodology validation on a generic synthetic profile; it is not hardware validation.

- Controlled configurations: 324
- Simulator runs: 1296
- Mean ranked-tile speedup vs no prefetch: 1.1239×
- Mean ranked-tile speedup vs full-Expert prefetch: 1.0207×
- Winner memberships including ties: `{'full_expert': 78, 'no_prefetch': 5, 'ranked_tile': 192, 'uniform_tile': 147}`
- Mean useful-byte efficiency: `{'full_expert': 0.3910607828103539, 'uniform_tile': 0.15599627545602485, 'ranked_tile': 0.18305275368387317}`
- Ranked vs full-Expert configurations: `{'ranked_faster': 242, 'exact_tie': 7, 'ranked_slower': 75}`

## Answers to the four experiment questions

1. **Does tile scheduling outperform full-Expert prefetch?** On average, ranked tile is 1.0207× faster, but not universally: it is faster in 242 configurations, tied in 7, and slower in 75.
2. **Under what resource conditions?** Mean ranked/full speedup by OCM scale is `{'0.5': 1.0961885639167055, '1.0': 0.9867903062720351, '2.0': 0.9791959969163768}`. The strongest benefit occurs at 0.5× OCM, where only one full Expert fits but many startup prefixes fit. With more OCM, full Expert can tie or win. Multiple DMA channels make breadth easier to exploit; one channel can serialize speculative work enough to erase the advantage.
3. **How does quality affect allocation?** Speedups by quality are `{'perfect': {'full_expert': 1.3235564440103282, 'uniform_tile': 1.1218121176749913, 'ranked_tile': 1.1462938570587868}, 'high': {'full_expert': 1.1788188529898749, 'uniform_tile': 1.1216023035391207, 'ranked_tile': 1.1451475879859858}, 'medium': {'full_expert': 0.9984963768539545, 'uniform_tile': 1.1158388336450755, 'ranked_tile': 1.1214625978930584}, 'low': {'full_expert': 0.998329583302013, 'uniform_tile': 1.1016325520410517, 'ranked_tile': 1.082784222270063}}`. Ranked is strongest for useful high-ranked confidence, whereas uniform breadth is safer at low quality because it invests less depth in false early candidates.
4. **Does fine granularity reduce waste?** Mean useful-byte efficiency rises from 0.391 for full Expert to 0.156 for uniform and 0.183 for ranked allocation. This is an aggregate result; individual configurations can reverse it.

The correct conclusion is conditional: policy benefits depend on OCM, bandwidth, channel count, tile size, and true-Expert ranks. This is methodology validation on a synthetic profile, not hardware validation.

Artifacts:

- `experiment.csv`
- `summary.json`
- `fig_m8_latency_vs_ocm.png`, `fig_m8_latency_vs_ocm.pdf`, `fig_m8_efficiency_vs_quality.png`, `fig_m8_efficiency_vs_quality.pdf`, `fig_m8_ranked_vs_full_heatmap.png`, `fig_m8_ranked_vs_full_heatmap.pdf`
