# EDBT Experiment Completion Status - 2026-06-01

## Completed Runs

| Run | Dataset rows | Splits/folds | Runtime | Result directory |
|---|---:|---:|---:|---|
| IntervenDB self-model, UWF full | 95,871 | 10 | 514.84s | `results/intervendb_model_uwf_full_main` |
| Calibrated baselines, UWF full | 95,871 | 10 | 5,109.56s | `results/autodl_baselines_uwf_full_calibrated` |
| IntervenDB self-model, Gotham full | 5,512,259 | 3 | 11,731.63s | `results/intervendb_model_gotham_full_uncapped_autodl` |
| Calibrated baselines, Gotham full core | 5,512,259 | 3 | 24,901.51s | `results/autodl_baselines_gotham_full_core_calibrated` |

## Main Comparison

| Dataset | IntervenDB method | IntervenDB worst F1 | Best baseline | Baseline worst F1 | Absolute gain | IntervenDB hidden-drop max | Baseline hidden-drop max |
|---|---|---:|---|---:|---:|---:|---:|
| UWF full | `adaptive_openworld_selector` | 0.993976 | `cat_full` | 0.895299 | +0.098677 | 0.003506 | 0.089207 |
| Gotham full | `counterfactual_ensemble` | 0.987654 | `cat_full__val_robust_minimax` | 0.944990 | +0.042664 | 0.000003 | 0.078901 |

## Gotham Full Baseline Detectors

Completed on full Gotham 2025 with random/time/device splits:

- `lr_full`
- `rf_full`
- `xgb_full`
- `lgbm_full`
- `cat_full`
- `iforest_env_sensitive`
- `pyod_ecod_env_sensitive`
- `pyod_copod_env_sensitive`

## Notes

- Baseline script now has split-level checkpoint/resume under `_checkpoints_baseline`.
- Full Gotham baseline completed and produced `baseline_run_summary.json`.
- All key remote results were synchronized back to `<project-root>\results`.
- The strongest reported result is the robust worst-intervention F1 gap on both full datasets, especially paired with much lower hidden drop.


