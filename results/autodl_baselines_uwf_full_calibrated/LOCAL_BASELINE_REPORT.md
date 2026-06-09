# Local Baseline Comparison Report

## Run Summary

- Dataset: `uwf`
- Rows: `95871`
- Splits/folds: `10`
- Detectors run: `['cat_full', 'gb_full', 'iforest_env_sensitive', 'lgbm_full', 'lr_full', 'pyod_copod_env_sensitive', 'pyod_ecod_env_sensitive', 'rf_full', 'xgb_full']`
- Elapsed seconds: `5109.56`
- Max hidden drop: `0.715789`
- Max RRS: `0.446129`

## Detector Summary

| detector                                         | detector_family   | feature_profile   |   observed_f1_mean |   worst_intervention_f1_mean |   hidden_drop_mean |   hidden_drop_max |
|:-------------------------------------------------|:------------------|:------------------|-------------------:|-----------------------------:|-------------------:|------------------:|
| rf_full__val_observed_f1                         | sklearn_tree      | full              |           0.95146  |                     0.648954 |          0.302506  |          0.567308 |
| iforest_env_sensitive                            | sklearn_anomaly   | env_sensitive     |           0.859243 |                     0.662997 |          0.196245  |          0.282456 |
| pyod_copod_env_sensitive__val_observed_f1        | pyod_copod        | env_sensitive     |           0.693703 |                     0.501249 |          0.192454  |          0.515875 |
| iforest_env_sensitive__val_observed_f1           | sklearn_anomaly   | env_sensitive     |           0.742848 |                     0.554644 |          0.188204  |          0.585776 |
| lr_full                                          | sklearn_linear    | full              |           0.900048 |                     0.716617 |          0.183431  |          0.532984 |
| lr_full__val_observed_f1                         | sklearn_linear    | full              |           0.870992 |                     0.692673 |          0.17832   |          0.291461 |
| rf_full                                          | sklearn_tree      | full              |           0.908417 |                     0.735329 |          0.173087  |          0.671795 |
| rf_full__val_robust_minimax                      | sklearn_tree      | full              |           0.912281 |                     0.761558 |          0.150723  |          0.671795 |
| lr_full__oracle_env_upper_bound                  | sklearn_linear    | full              |           0.953422 |                     0.803204 |          0.150218  |          0.292011 |
| pyod_ecod_env_sensitive__val_observed_f1         | pyod_ecod         | env_sensitive     |           0.693247 |                     0.550665 |          0.142582  |          0.340351 |
| lr_full__val_robust_minimax                      | sklearn_linear    | full              |           0.787805 |                     0.647939 |          0.139865  |          0.291795 |
| pyod_ecod_env_sensitive__oracle_env_upper_bound  | pyod_ecod         | env_sensitive     |           0.862097 |                     0.739108 |          0.122989  |          0.195091 |
| pyod_copod_env_sensitive                         | pyod_copod        | env_sensitive     |           0.55461  |                     0.434352 |          0.120257  |          0.438803 |
| pyod_ecod_env_sensitive__val_robust_minimax      | pyod_ecod         | env_sensitive     |           0.657417 |                     0.54538  |          0.112037  |          0.308912 |
| lgbm_full__val_observed_f1                       | lightgbm          | full              |           0.898287 |                     0.793942 |          0.104344  |          0.684755 |
| pyod_copod_env_sensitive__oracle_env_upper_bound | pyod_copod        | env_sensitive     |           0.875552 |                     0.774278 |          0.101274  |          0.192149 |
| iforest_env_sensitive__oracle_env_upper_bound    | sklearn_anomaly   | env_sensitive     |           0.9323   |                     0.832434 |          0.099866  |          0.200157 |
| cat_full__val_observed_f1                        | catboost          | full              |           0.957277 |                     0.873749 |          0.083528  |          0.667519 |
| lgbm_full__val_robust_minimax                    | lightgbm          | full              |           0.899962 |                     0.816773 |          0.0831884 |          0.715789 |
| cat_full__val_robust_minimax                     | catboost          | full              |           0.91245  |                     0.844833 |          0.0676167 |          0.559809 |

## Split Summary

| split          |   hidden_drop_mean |   hidden_drop_max |   observed_f1_mean |   rrs_mean |   rrs_max |
|:---------------|-------------------:|------------------:|-------------------:|-----------:|----------:|
| heldout_family |          0.103645  |          0.715789 |           0.805663 |   0.162696 |  0.374462 |
| random         |          0.0789914 |          0.532984 |           0.941123 |   0.240033 |  0.368337 |
| source_ip      |          0.0489812 |          0.285919 |           0.937305 |   0.247268 |  0.446129 |
| time           |          0.0738927 |          0.292116 |           0.939309 |   0.203515 |  0.373541 |


