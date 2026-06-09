# Reproducibility Guide

This repository contains source code, configuration files, and compact summary outputs for the IntervenDB experiments. It does not include raw datasets, large prediction dumps, model checkpoints, DuckDB databases, Parquet intermediates, or the paper source tree.

## 1. Verify A Fresh Checkout

Run the repository integrity check first:

```bash
python scripts/verify_artifact.py --compile
```

This check does not require raw datasets. It verifies the repository layout, required result summaries, basic summary-table consistency, sensitive-string hygiene, and Python syntax.

## 2. Rebuild Compact Summary Outputs

The committed result directories are enough to rebuild the local comparison pack:

```bash
python scripts/build_local_cpu_experiment_pack.py
python scripts/verify_artifact.py
```

These commands should complete without downloading raw datasets.

## 3. Re-run Full Experiments

Full experiment reproduction requires the external datasets described in `DATASETS.md`.

IntervenDB model-selection runs:

```bash
python scripts/run_intervendb_model.py configs/intervendb_model_uwf_full_main.yaml
python scripts/run_intervendb_model.py configs/intervendb_model_gotham_full_uncapped_autodl.yaml
```

Calibrated baseline runs:

```bash
python scripts/run_local_baseline_comparison.py configs/autodl_baseline_uwf_full_calibrated.yaml
python scripts/run_local_baseline_comparison.py configs/autodl_baseline_gotham_full_core_calibrated.yaml
```

Risk-cube and witness summaries:

```bash
python scripts/run_local_uwf_smoke.py configs/local_uwf_smoke.yaml
python scripts/build_risk_cube_and_witness.py configs/local_risk_cube.yaml
python scripts/build_risk_cube_and_witness.py configs/local_risk_cube_random.yaml
python scripts/run_witness_sweep.py configs/local_risk_cube.yaml 8 16 24 32 48
```

The full Gotham calibrated baseline run is the most expensive path in this repository. The committed summaries make inspection possible without rerunning that full workload.
