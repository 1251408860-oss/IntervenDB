# Contributing

This repository is primarily a research-code artifact. Contributions should keep it reproducible, small, and free of private data.

## Before Opening A Pull Request

Run:

```bash
python scripts/verify_artifact.py --compile
```

If a change affects reported summaries, also run:

```bash
python scripts/build_local_cpu_experiment_pack.py
python scripts/verify_artifact.py
```

## Do Not Commit

- raw datasets or extracted dataset files;
- prediction-level dumps, checkpoints, DuckDB files, or Parquet intermediates;
- private credentials, API keys, local machine paths, or upload scripts;
- paper drafts, LaTeX auxiliary files, Python caches, virtual environments, or editor metadata.

## Preferred Changes

- keep experiment configs deterministic and documented;
- keep result summaries compact and inspectable;
- add checks to `scripts/verify_artifact.py` when adding required outputs;
- prefer small, reviewable changes over broad rewrites.
