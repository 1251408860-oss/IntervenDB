# Local Baseline Comparison Report

## Run Summary

- Dataset: `gotham`
- Rows: `5512259`
- Splits/folds: `3`
- Detectors run: `['cat_full', 'iforest_env_sensitive', 'lgbm_full', 'lr_full', 'pyod_copod_env_sensitive', 'pyod_ecod_env_sensitive', 'rf_full', 'xgb_full']`
- Elapsed seconds: `24901.51`
- Max hidden drop: `0.846623`
- Max RRS: `0.651996`

## Detector Summary

| detector                                    | detector_family   | feature_profile   |   observed_f1_mean |   worst_intervention_f1_mean |   hidden_drop_mean |   hidden_drop_max |
|:--------------------------------------------|:------------------|:------------------|-------------------:|-----------------------------:|-------------------:|------------------:|
| lgbm_full__val_observed_f1                  | lightgbm          | full              |           0.967069 |                     0.574556 |          0.392513  |          0.846623 |
| cat_full__val_observed_f1                   | catboost          | full              |           0.999021 |                     0.722818 |          0.276202  |          0.390789 |
| lgbm_full__val_robust_minimax               | lightgbm          | full              |           0.966672 |                     0.735349 |          0.231324  |          0.376312 |
| lgbm_full                                   | lightgbm          | full              |           0.794205 |                     0.572232 |          0.221973  |          0.331454 |
| cat_full                                    | catboost          | full              |           0.99807  |                     0.783592 |          0.214478  |          0.287133 |
| lgbm_full__oracle_env_upper_bound           | lightgbm          | full              |           0.970642 |                     0.757048 |          0.213594  |          0.322327 |
| xgb_full__val_observed_f1                   | xgboost           | full              |           0.971767 |                     0.780291 |          0.191476  |          0.310573 |
| iforest_env_sensitive__val_robust_minimax   | sklearn_anomaly   | env_sensitive     |           0.891293 |                     0.70083  |          0.190464  |          0.307514 |
| xgb_full                                    | xgboost           | full              |           0.970863 |                     0.785681 |          0.185182  |          0.308291 |
| iforest_env_sensitive__val_observed_f1      | sklearn_anomaly   | env_sensitive     |           0.683583 |                     0.500639 |          0.182944  |          0.297949 |
| iforest_env_sensitive                       | sklearn_anomaly   | env_sensitive     |           0.945349 |                     0.806814 |          0.138535  |          0.228991 |
| lr_full__val_observed_f1                    | sklearn_linear    | full              |           0.974343 |                     0.863288 |          0.111055  |          0.16809  |
| lr_full                                     | sklearn_linear    | full              |           0.957751 |                     0.853908 |          0.103843  |          0.124499 |
| lr_full__val_robust_minimax                 | sklearn_linear    | full              |           0.979737 |                     0.882417 |          0.0973202 |          0.154277 |
| lr_full__oracle_env_upper_bound             | sklearn_linear    | full              |           0.979776 |                     0.89259  |          0.087186  |          0.133125 |
| rf_full__val_observed_f1                    | sklearn_tree      | full              |           0.828433 |                     0.743433 |          0.0850004 |          0.135735 |
| pyod_ecod_env_sensitive__val_robust_minimax | pyod_ecod         | env_sensitive     |           0.737968 |                     0.669923 |          0.0680448 |          0.158645 |
| pyod_ecod_env_sensitive__val_observed_f1    | pyod_ecod         | env_sensitive     |           0.704395 |                     0.645641 |          0.0587543 |          0.122557 |
| rf_full                                     | sklearn_tree      | full              |           0.986561 |                     0.929809 |          0.0567523 |          0.159888 |
| rf_full__val_robust_minimax                 | sklearn_tree      | full              |           0.974117 |                     0.918738 |          0.0553785 |          0.150435 |

## Split Summary

| split   |   hidden_drop_mean |   hidden_drop_max |   observed_f1_mean |   rrs_mean |   rrs_max |
|:--------|-------------------:|------------------:|-------------------:|-----------:|----------:|
| device  |          0.155195  |          0.846623 |           0.882049 |   0.337183 |  0.651996 |
| random  |          0.126143  |          0.376312 |           0.883119 |   0.387922 |  0.646107 |
| time    |          0.0458826 |          0.307514 |           0.850323 |   0.208125 |  0.416073 |


