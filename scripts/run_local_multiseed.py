from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]


def main():
    base_config_path = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "configs" / "local_uwf_smoke.yaml"
    seeds = [int(x) for x in sys.argv[2:]] if len(sys.argv) > 2 else [40, 41, 42]

    with base_config_path.open("r", encoding="utf-8") as f:
        base_cfg = yaml.safe_load(f)

    multi_root = ROOT / "results" / "local_multiseed"
    multi_root.mkdir(parents=True, exist_ok=True)

    metric_frames = []
    ranking_frames = []
    hidden_frames = []
    summaries = []

    for seed in seeds:
        cfg = dict(base_cfg)
        cfg["random_state"] = seed
        cfg["output_dir"] = f"results/local_multiseed/seed_{seed}"
        for i, params in enumerate(cfg["interventions"].values()):
            params["seed"] = seed + i + 1

        cfg_path = multi_root / f"config_seed_{seed}.yaml"
        with cfg_path.open("w", encoding="utf-8") as f:
            yaml.safe_dump(cfg, f, sort_keys=False)

        subprocess.run(
            [sys.executable, str(ROOT / "scripts" / "run_local_uwf_smoke.py"), str(cfg_path)],
            cwd=str(ROOT),
            check=True,
        )

        out_dir = ROOT / cfg["output_dir"]
        metrics = pd.read_csv(out_dir / "metrics_by_detector_environment.csv")
        rankings = pd.read_csv(out_dir / "ranking_reversal.csv")
        hidden = pd.read_csv(out_dir / "hidden_drop_summary.csv")
        with (out_dir / "run_summary.json").open("r", encoding="utf-8") as f:
            summary = json.load(f)

        metrics["seed"] = seed
        rankings["seed"] = seed
        hidden["seed"] = seed
        summary["seed"] = seed

        metric_frames.append(metrics)
        ranking_frames.append(rankings)
        hidden_frames.append(hidden)
        summaries.append(summary)

    all_metrics = pd.concat(metric_frames, ignore_index=True)
    all_rankings = pd.concat(ranking_frames, ignore_index=True)
    all_hidden = pd.concat(hidden_frames, ignore_index=True)

    all_metrics.to_csv(multi_root / "all_metrics.csv", index=False)
    all_rankings.to_csv(multi_root / "all_ranking_reversal.csv", index=False)
    all_hidden.to_csv(multi_root / "all_hidden_drop.csv", index=False)

    metric_summary = (
        all_metrics
        .groupby(["detector", "feature_profile", "environment_id"], as_index=False)
        .agg(
            f1_mean=("f1", "mean"),
            f1_std=("f1", "std"),
            auc_mean=("auc", "mean"),
            cfs_mean=("causal_fragility_score", "mean"),
            els_mean=("environment_leakage_score", "mean"),
        )
    )
    metric_summary.to_csv(multi_root / "metric_summary.csv", index=False)

    ranking_summary = (
        all_rankings
        .groupby("environment_id", as_index=False)
        .agg(
            rrs_mean=("ranking_reversal_score", "mean"),
            rrs_std=("ranking_reversal_score", "std"),
        )
    )
    ranking_summary.to_csv(multi_root / "ranking_summary.csv", index=False)

    hidden_summary = (
        all_hidden
        .groupby("detector", as_index=False)
        .agg(
            observed_f1_mean=("observed_f1", "mean"),
            worst_intervention_f1_mean=("worst_intervention_f1", "mean"),
            hidden_drop_mean=("worst_hidden_drop", "mean"),
            hidden_drop_std=("worst_hidden_drop", "std"),
        )
        .sort_values("hidden_drop_mean", ascending=False)
    )
    hidden_summary.to_csv(multi_root / "hidden_drop_summary.csv", index=False)

    with (multi_root / "run_summaries.json").open("w", encoding="utf-8") as f:
        json.dump(summaries, f, indent=2)

    print(json.dumps({
        "seeds": seeds,
        "output_dir": str(multi_root),
        "max_mean_hidden_drop": float(hidden_summary["hidden_drop_mean"].max()),
        "max_mean_rrs": float(ranking_summary["rrs_mean"].max()),
    }, indent=2))


if __name__ == "__main__":
    main()


