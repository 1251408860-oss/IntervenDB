# IntervenDB: Intervention-Aware Risk Cubes for Robust Detector Selection

This repository contains the open-source implementation and artifact package for the IntervenDB experiments.

IntervenDB studies how learned intrusion detectors behave under hidden environment shift. The code builds intervention-aware risk summaries, compares deployable detector-selection policies, and reports compact robustness results on UWF-2024 and Gotham-2025.

To facilitate reproducibility and provide a clear review path, the package is organized around the code modules, experiment entry points, dataset-access notes, and committed summary outputs needed to inspect or rerun the experiments. This public repository intentionally excludes the paper source tree.

## Repository Navigation & Artifact Mapping

| Artifact Scope | Description | Directory Link |
| :--- | :--- | :--- |
| Core IntervenDB package | Data normalization, intervention construction, metric computation, witness utilities, and shared experiment helpers | [`src/intervendb`](./src/intervendb) |
| Experiment runners and checks | Model-selection runs, calibrated baselines, local smoke tests, risk-cube construction, witness sweeps, and repository verification | [`scripts`](./scripts) |
| Experiment configurations | YAML configurations for the UWF-2024 and Gotham-2025 runs retained in this public artifact | [`configs`](./configs) |
| Compact result summaries | Committed CSV, JSON, and figure outputs used to inspect the reported robustness comparisons without rerunning the full workloads | [`results`](./results) |
| Reproduction and dataset notes | Step-by-step reproduction notes, dataset locations, and experiment completion status | [`REPRODUCIBILITY.md`](./REPRODUCIBILITY.md), [`DATASETS.md`](./DATASETS.md), [`docs`](./docs) |

## Global Environment Overview

The artifact is designed for Python-based local reproduction on Linux, macOS, or Windows. The lightweight verification and summary-rebuild paths do not require raw datasets. Full experiment reproduction requires external access to the UWF-ZeekData24 and Gotham Dataset 2025 sources described in [`DATASETS.md`](./DATASETS.md).

The repository does not redistribute raw datasets, extracted dataset files, prediction-level dumps, model checkpoints, DuckDB databases, Parquet intermediates, local logs, paper drafts, credentials, or LaTeX sources. The committed summaries are kept small so that reviewers and users can inspect the main experimental claims from a clean checkout.

## Dataset Access

If you only want to verify the artifact layout or rebuild the compact summary tables, you can skip this section.

Full experiment runs expect the datasets to be placed under the paths used by the configuration files:

```text
data/raw/uwf_zeekdata24/*.csv
data/raw/gotham_2025/processed_full/*.csv
```

The upstream dataset pages and additional placement notes are documented in [`DATASETS.md`](./DATASETS.md). The Gotham extraction helper is available through [`scripts/manage_datasets.py`](./scripts/manage_datasets.py) when the archive is already available locally.

## Quick Installation

Create and activate a Python environment:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

On Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e .
```

Optional calibrated baseline experiments require heavier model packages:

```bash
python -m pip install -e ".[baselines]"
```

### Core dependencies

- Python 3.9+
- `numpy`
- `pandas`
- `scipy`
- `scikit-learn`
- `matplotlib`
- `seaborn`
- `PyYAML`
- `duckdb`
- `pyarrow`

### Optional baseline dependencies

- `xgboost`
- `lightgbm`
- `catboost`
- `pyod`
- `interpret-core`

## Quick Artifact Verification

Verify a fresh checkout without downloading raw datasets:

```bash
python scripts/verify_artifact.py --compile
```

This check validates the required repository layout, committed result summaries, sensitive-string hygiene, and Python syntax.

## Rebuilding Committed Summaries

The committed result directories are enough to rebuild the compact comparison tables and local summary figures:

```bash
python scripts/build_local_cpu_experiment_pack.py
python scripts/verify_artifact.py
```

The main summary table is written to:

```text
results/local_cpu_experiment_pack/paper_main_robustness_results.csv
```

Detailed execution commands, expected inputs, and output locations are documented in [`REPRODUCIBILITY.md`](./REPRODUCIBILITY.md).

## Full Experiment Reproduction

Full reproduction starts from the external datasets listed in [`DATASETS.md`](./DATASETS.md).

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

The Gotham calibrated baseline run is the most expensive path in this artifact. The committed summaries make inspection possible without rerunning that full workload.

## Main Result Summary

The current committed summary reports:

- UWF-2024: +0.0987 mean worst-intervention F1 over the strongest deployable calibrated baseline.
- Gotham-2025: +0.0427 mean worst-intervention F1 over the strongest deployable calibrated baseline.

Oracle threshold policies are included only as non-deployable upper bounds.

## License

The code is released under the MIT License. Dataset licenses and access terms are controlled by the upstream dataset providers listed in [`DATASETS.md`](./DATASETS.md).
