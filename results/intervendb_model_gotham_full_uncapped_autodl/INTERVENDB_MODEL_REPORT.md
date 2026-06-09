# IntervenDB Model Report

## Summary

- Dataset: `gotham`
- Rows: `5512259`
- Folds: `3`
- Max hidden drop: `0.150435`
- Max RRS: `1.162162`
- Best mean worst-intervention F1: `0.987654`

## Method Summary

| method                                      |   observed_f1_mean |   worst_intervention_f1_mean |   hidden_drop_mean |   hidden_drop_max |
|:--------------------------------------------|-------------------:|-----------------------------:|-------------------:|------------------:|
| intervendb_counterfactual_ensemble          |           0.987655 |                     0.987654 |        9.91491e-07 |       2.21614e-06 |
| intervendb_coverage_lift_openworld_selector |           0.987653 |                     0.987653 |        0           |       0           |
| intervendb_coverage_openworld_selector      |           0.987653 |                     0.987653 |        0           |       0           |
| intervendb_openworld_selector               |           0.987653 |                     0.987653 |        0           |       0           |
| intervendb_pareto_selector                  |           0.987653 |                     0.987653 |        0           |       0           |
| intervendb_stability_selector               |           0.987653 |                     0.987653 |        0           |       0           |
| intervendb_env_calibrated_ensemble          |           0.987654 |                     0.987653 |        1.05104e-06 |       1.89954e-06 |
| intervendb_conformal_tail_guard             |           0.98716  |                     0.984586 |        0.0025744   |       0.00329739  |
| intervendb_tail_guard_ensemble              |           0.975143 |                     0.974375 |        0.00076846  |       0.00125528  |
| intervendb_adaptive_openworld_selector      |           0.957891 |                     0.93185  |        0.0260412   |       0.0624252   |
| intervendb_minimax_selector                 |           0.974114 |                     0.918736 |        0.0553778   |       0.150435    |
| intervendb_observed_selector                |           0.974114 |                     0.918736 |        0.0553778   |       0.150435    |
| intervendb_shift_guard_selector             |           0.974114 |                     0.918736 |        0.0553778   |       0.150435    |
| intervendb_gated_conformal_tail_guard       |           0.919482 |                     0.88294  |        0.0365418   |       0.0624249   |

## Split Summary

| split   |   observed_f1_mean |   worst_intervention_f1_mean |   hidden_drop_mean |   hidden_drop_max |   rrs_mean |   rrs_max |
|:--------|-------------------:|-----------------------------:|-------------------:|------------------:|-----------:|----------:|
| device  |           0.97689  |                     0.93543  |         0.0414601  |         0.150435  |  0.118186  |  0.428571 |
| random  |           0.995312 |                     0.991704 |         0.00360789 |         0.0472005 |  0.0691076 |  0.327273 |
| time    |           0.958282 |                     0.953623 |         0.00465939 |         0.0156985 |  0.321049  |  1.16216  |


