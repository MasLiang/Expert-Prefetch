# M9.3 Predictor Sensitivity Summary

- Quality rows: 4860
- Calibration rows: 19440
- Robustness rows: 4860
- Beneficial quality thresholds: `{'full_expert': 0.9, 'budget_proportional': 0.2, 'hardware_greedy': 0.2, 'ranked_tile': 0.2}`
- Quality 1.0 vs 0.2 latency improvements: `{'full_expert': 8.245035620908203, 'budget_proportional': 8.454745240408307, 'hardware_greedy': 6.640473737809735, 'ranked_tile': 6.56618245731711}` percent
- Calibration mean speedups: `{'no_prefetch': {'well_calibrated': 1.0, 'overconfident_wrong': 1.0, 'underconfident_correct': 1.0, 'uniform': 1.0}, 'full_expert': {'well_calibrated': 1.010871464583296, 'overconfident_wrong': 1.010871464583296, 'underconfident_correct': 1.010871464583296, 'uniform': 1.010871464583296}, 'budget_proportional': {'well_calibrated': 1.0354786091728214, 'overconfident_wrong': 1.0373221040024259, 'underconfident_correct': 1.0372194163475477, 'uniform': 1.0379008077094682}, 'hardware_greedy': {'well_calibrated': 1.0369832170342634, 'overconfident_wrong': 1.037512632026995, 'underconfident_correct': 1.0369500449082154, 'uniform': 1.037512632026995}, 'ranked_tile': {'well_calibrated': 1.036299790937346, 'overconfident_wrong': 1.0369964165746581, 'underconfident_correct': 1.0363667478030445, 'uniform': 1.0369964165746581}}`
- Mean calibration speedup standard deviations: `{'no_prefetch': 0.0, 'full_expert': 0.0, 'budget_proportional': 0.005441354339526071, 'hardware_greedy': 0.0019697030535610785, 'ranked_tile': 0.002137907352812936}`
- Mean allocator latency spread: 0.9550%
- Full Expert winner membership: `{'including_ties': 500, 'unique': 24}`

SUPPORTED CLAIMS: predictor quality sensitivity; confidence calibration effects; scheduler robustness.

NOT CLAIMED: real predictor accuracy; learned predictor implementation; hardware predictor overhead.
