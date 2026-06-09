# IntervenDB Model Report

## Summary

- Dataset: `uwf`
- Rows: `95871`
- Folds: `10`
- Max hidden drop: `0.671795`
- Max RRS: `0.500000`
- Best mean worst-intervention F1: `0.993976`

## Method Summary

| method                                      |   observed_f1_mean |   worst_intervention_f1_mean |   hidden_drop_mean |   hidden_drop_max |
|:--------------------------------------------|-------------------:|-----------------------------:|-------------------:|------------------:|
| intervendb_adaptive_openworld_selector      |           0.994327 |                     0.993976 |        0.00035062  |         0.0035062 |
| intervendb_coverage_lift_openworld_selector |           0.994327 |                     0.993976 |        0.00035062  |         0.0035062 |
| intervendb_coverage_openworld_selector      |           0.99016  |                     0.98981  |        0.00035062  |         0.0035062 |
| intervendb_minimax_selector                 |           0.938111 |                     0.938111 |        0           |         0         |
| intervendb_openworld_selector               |           0.938111 |                     0.938111 |        0           |         0         |
| intervendb_pareto_selector                  |           0.938111 |                     0.938111 |        0           |         0         |
| intervendb_shift_guard_selector             |           0.938111 |                     0.938111 |        0           |         0         |
| intervendb_stability_selector               |           0.938111 |                     0.938111 |        0           |         0         |
| intervendb_conformal_tail_guard             |           0.932534 |                     0.93171  |        0.000823669 |         0.0030118 |
| intervendb_gated_conformal_tail_guard       |           0.946886 |                     0.916096 |        0.0307898   |         0.152692  |
| intervendb_env_calibrated_ensemble          |           0.922123 |                     0.894748 |        0.0273745   |         0.267458  |
| intervendb_counterfactual_ensemble          |           0.923344 |                     0.894156 |        0.0291885   |         0.276451  |
| intervendb_observed_selector                |           0.92706  |                     0.859881 |        0.0671795   |         0.671795  |
| intervendb_tail_guard_ensemble              |           0.909513 |                     0.826867 |        0.0826459   |         0.135294  |

## Split Summary

| split          |   observed_f1_mean |   worst_intervention_f1_mean |   hidden_drop_mean |   hidden_drop_max |   rrs_mean |   rrs_max |
|:---------------|-------------------:|-----------------------------:|-------------------:|------------------:|-----------:|----------:|
| heldout_family |           0.926351 |                     0.905966 |         0.0203855  |         0.671795  | 0.0848188  | 0.5       |
| random         |           0.982562 |                     0.965301 |         0.0172608  |         0.148757  | 0.00909925 | 0.0363636 |
| source_ip      |           0.993586 |                     0.988914 |         0.00467196 |         0.0634359 | 0.00909925 | 0.0363636 |
| time           |           0.989987 |                     0.983865 |         0.00612181 |         0.0789234 | 0.155157   | 0.339422  |


