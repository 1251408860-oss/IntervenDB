# Dataset Access

This artifact does not redistribute raw datasets. Download the datasets from their official sources and place them under the paths expected by the configs.

## UWF-ZeekData24

Expected path:

```text
data/raw/uwf_zeekdata24/*.csv
```

Source:

```text
https://datasets.uwf.edu/data/UWF-ZeekData24/
```

The paper run used eight CSV categories:

- benign
- credential_access
- defense_evasion
- exfiltration
- initial_access
- persistence
- privilege_escalation
- reconnaissance

## Gotham Dataset 2025

Expected processed path:

```text
data/raw/gotham_2025/processed_full/*.csv
```

Source:

```text
https://zenodo.org/records/14502760
```

The extraction helper in `scripts/manage_datasets.py` can extract processed CSV files from the Gotham archive when the archive is available locally.

The final paper tables in this artifact use UWF-2024 and Gotham-2025 full results.
