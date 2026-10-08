# M9.2 Criticality-Aware Scheduling Summary

- Configurations: 144
- Scheduler rows: 1296
- Offline candidate simulations: 16056
- Bandwidth scale: 1.0x
- Mean speedups vs No Prefetch: `{'no_prefetch': 1.0, 'full_expert': 1.007251940689976, 'uniform_tile': 1.0462988486719733, 'ranked_tile': 1.045410133101518, 'budget_proportional': 1.0475332538956685, 'hardware_greedy': 1.045181803385853, 'criticality_aware': 1.0215142882565755, 'fixed_prediction_oracle': 1.1218627745474041, 'full_oracle': 1.2693736611456867}`
- Gap decomposition by quality: `{'perfect': {'prediction_loss_percent_of_full_oracle': 1.9737951294578342, 'criticality_allocation_loss_percent_of_full_oracle': 19.677680276512262}, 'high': {'prediction_loss_percent_of_full_oracle': 9.770883775334257, 'criticality_allocation_loss_percent_of_full_oracle': 13.80972515078198}, 'medium': {'prediction_loss_percent_of_full_oracle': 19.694205228742216, 'criticality_allocation_loss_percent_of_full_oracle': 5.402948300388655}, 'low': {'prediction_loss_percent_of_full_oracle': 23.263910923636168, 'criticality_allocation_loss_percent_of_full_oracle': 3.1122210389894045}}`
- Criticality vs Hardware Greedy: `{'mean_improvement_percent': -2.1862446256268218, 'better_points': 11, 'tie_points': 53, 'worse_points': 80}`
- Criticality vs Budget Proportional: `{'mean_improvement_percent': -2.37433577616325, 'better_points': 10, 'tie_points': 58, 'worse_points': 76}`
- Full Expert online winner memberships including ties: 38/144

The exact signed formulas requested are retained in the CSV. Positive `loss`
columns are used for plots: prediction loss = fixed - full, allocation loss =
scheduler - fixed.

SUPPORTED: simulator-based relative scheduling comparison; allocation/prediction
gap decomposition; hardware-friendly scheduler trends.

NOT CLAIMED: real FPGA performance; real predictor accuracy; RTL implementation.
