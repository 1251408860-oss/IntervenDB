from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import yaml
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
SCRIPTS = ROOT / "scripts"
for path in [SRC, SCRIPTS]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from intervendb.data import dataset_profile, load_uwf_directory, write_json
from intervendb.interventions import apply_intervention, load_interventions
from intervendb.metrics import (
    causal_fragility_score,
    classification_metrics,
    environment_leakage_score,
    ranking_reversal_score,
    score_environment_leakage_score,
    summarize_ranking,
)
from run_local_uwf_smoke import build_detectors, build_preprocessor, detector_score, fit_detector


def validate_split(train_df: pd.DataFrame, test_df: pd.DataFrame, split_name: str) -> None:
    if len(train_df) == 0 or len(test_df) == 0:
        raise ValueError(f"{split_name}: empty train/test split")
    if train_df["y"].nunique() < 2:
        raise ValueError(f"{split_name}: train split must contain both classes")
    if test_df["y"].nunique() < 2:
        raise ValueError(f"{split_name}: test split must contain both classes")


def random_split(df: pd.DataFrame, test_size: float, seed: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    return train_test_split(df, test_size=test_size, stratify=df["y"], random_state=seed)


def time_split_within_family(df: pd.DataFrame, test_size: float) -> Tuple[pd.DataFrame, pd.DataFrame]:
    train_parts = []
    test_parts = []
    for _, part in df.groupby("attack_family", dropna=False):
        ordered = part.sort_values("ts")
        n_test = max(1, int(round(len(ordered) * test_size)))
        if n_test >= len(ordered):
            n_test = max(1, len(ordered) // 3)
        train_parts.append(ordered.iloc[:-n_test])
        test_parts.append(ordered.iloc[-n_test:])
    return (
        pd.concat(train_parts, ignore_index=True).sample(frac=1.0, random_state=0),
        pd.concat(test_parts, ignore_index=True).sample(frac=1.0, random_state=1),
    )


def source_ip_split_within_family(df: pd.DataFrame, test_size: float, seed: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    train_parts = []
    test_parts = []
    for _, part in df.groupby("attack_family", dropna=False):
        groups = part["src_ip_zeek"].fillna("missing").astype(str).unique().tolist()
        rng.shuffle(groups)
        target = max(1, int(round(len(part) * test_size)))
        selected = []
        count = 0
        for group in groups:
            selected.append(group)
            count += int((part["src_ip_zeek"].fillna("missing").astype(str) == group).sum())
            if count >= target:
                break
        mask = part["src_ip_zeek"].fillna("missing").astype(str).isin(selected)
        if mask.all() or (~mask).sum() == 0:
            fallback_test = part.sample(n=target, random_state=seed)
            test_parts.append(fallback_test)
            train_parts.append(part.drop(fallback_test.index))
        else:
            test_parts.append(part[mask])
            train_parts.append(part[~mask])
    return (
        pd.concat(train_parts, ignore_index=True).sample(frac=1.0, random_state=seed),
        pd.concat(test_parts, ignore_index=True).sample(frac=1.0, random_state=seed + 1),
    )


def heldout_family_splits(df: pd.DataFrame, test_size: float, seed: int) -> Iterable[Tuple[str, pd.DataFrame, pd.DataFrame]]:
    benign = df[df["attack_family"] == "benign"]
    attacks = df[df["attack_family"] != "benign"]
    families = sorted(attacks["attack_family"].astype(str).unique())
    for family in families:
        heldout = attacks[attacks["attack_family"].astype(str) == family]
        train_attacks = attacks[attacks["attack_family"].astype(str) != family]
        if len(heldout) < 10 or len(train_attacks) < 10:
            continue
        n_benign_test = min(len(benign) // 2, max(50, len(heldout)))
        benign_test = benign.sample(n=n_benign_test, random_state=seed + len(family), replace=False)
        benign_train = benign.drop(benign_test.index)
        train_df = pd.concat([benign_train, train_attacks], ignore_index=True).sample(frac=1.0, random_state=seed)
        test_df = pd.concat([benign_test, heldout], ignore_index=True).sample(frac=1.0, random_state=seed + 1)
        yield family.replace(" ", "_").lower(), train_df, test_df


def make_splits(df: pd.DataFrame, split_names: List[str], test_size: float, seed: int):
    for split_name in split_names:
        if split_name == "random":
            train_df, test_df = random_split(df, test_size, seed)
            yield split_name, "all", train_df, test_df
        elif split_name == "time":
            train_df, test_df = time_split_within_family(df, test_size)
            yield split_name, "within_family_chronological", train_df, test_df
        elif split_name == "source_ip":
            train_df, test_df = source_ip_split_within_family(df, test_size, seed)
            yield split_name, "within_family_source_ip", train_df, test_df
        elif split_name == "heldout_family":
            for fold, train_df, test_df in heldout_family_splits(df, test_size, seed):
                yield split_name, fold, train_df, test_df
        else:
            raise ValueError(f"Unknown split strategy: {split_name}")


def evaluate_split(split_name, fold, train_df, test_df, interventions, detectors, seed):
    validate_split(train_df, test_df, f"{split_name}/{fold}")
    y_train = train_df["y"].to_numpy()
    y_test = test_df["y"].to_numpy()

    metric_rows = []
    prediction_frames = []
    for detector_name, spec in detectors.items():
        fit_start = time.perf_counter()
        preprocessor = build_preprocessor(spec["profile"])
        model = fit_detector(spec["kind"], spec["estimator"], preprocessor, train_df, y_train)
        fit_seconds = time.perf_counter() - fit_start

        detector_prediction_parts = []
        for env_name, intervention in interventions.items():
            eval_start = time.perf_counter()
            eval_df, sps = apply_intervention(test_df, intervention)
            pred, score = detector_score(detector_name, model, eval_df)
            eval_seconds = time.perf_counter() - eval_start

            row = {
                "split": split_name,
                "fold": fold,
                "detector": detector_name,
                "feature_profile": spec["profile"],
                "environment_id": env_name,
                "train_rows": int(len(train_df)),
                "test_rows": int(len(test_df)),
                "train_positive_rate": float(train_df["y"].mean()),
                "test_positive_rate": float(test_df["y"].mean()),
                "fit_seconds": fit_seconds,
                "eval_seconds": eval_seconds,
                "semantic_preservation_score": sps,
                **classification_metrics(y_test, pred, score),
            }
            metric_rows.append(row)

            pred_frame = pd.DataFrame({
                "split": split_name,
                "fold": fold,
                "detector": detector_name,
                "environment_id": env_name,
                "semantic_label": eval_df["semantic_label"].astype(str).to_numpy(),
                "y": y_test,
                "prediction": pred,
                "score": score,
            })
            detector_prediction_parts.append(pred_frame)
            prediction_frames.append(pred_frame)

        detector_all = pd.concat(detector_prediction_parts, ignore_index=True)
        cfs = causal_fragility_score(detector_all)
        els = environment_leakage_score(detector_all)
        score_els = score_environment_leakage_score(detector_all)
        for row in metric_rows:
            if row["detector"] == detector_name:
                row["causal_fragility_score"] = cfs
                row["environment_leakage_score"] = els
                row["score_environment_leakage_score"] = score_els

    return pd.DataFrame(metric_rows), pd.concat(prediction_frames, ignore_index=True)


def summarize_hidden_drop(metrics_df: pd.DataFrame) -> pd.DataFrame:
    observed = (
        metrics_df[metrics_df["environment_id"] == "observed"]
        [["split", "fold", "detector", "f1"]]
        .rename(columns={"f1": "observed_f1"})
    )
    worst = (
        metrics_df
        .groupby(["split", "fold", "detector"], as_index=False)["f1"]
        .min()
        .rename(columns={"f1": "worst_intervention_f1"})
    )
    out = observed.merge(worst, on=["split", "fold", "detector"])
    out["worst_hidden_drop"] = out["observed_f1"] - out["worst_intervention_f1"]
    return out.sort_values(["split", "fold", "worst_hidden_drop"], ascending=[True, True, False])


def summarize_ranking_by_split(metrics_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (split, fold), part in metrics_df.groupby(["split", "fold"]):
        static = part[part["environment_id"] == "observed"].set_index("detector")["f1"].to_dict()
        static_top = max(static.items(), key=lambda kv: kv[1])[0]
        for env in sorted(part["environment_id"].astype(str).unique()):
            env_scores = part[part["environment_id"] == env].set_index("detector")["f1"].to_dict()
            rows.append({
                "split": split,
                "fold": fold,
                "environment_id": env,
                "ranking_reversal_score": ranking_reversal_score(static, env_scores),
                "static_top_detector": static_top,
                "environment_top_detector": max(env_scores.items(), key=lambda kv: kv[1])[0],
            })
    return pd.DataFrame(rows)


def plot_split_results(hidden_df: pd.DataFrame, ranking_df: pd.DataFrame, out_dir: Path) -> None:
    sns.set_theme(style="whitegrid")

    top_hidden = (
        hidden_df
        .groupby(["split", "detector"], as_index=False)["worst_hidden_drop"]
        .mean()
    )
    plt.figure(figsize=(11, 5))
    sns.barplot(data=top_hidden, x="split", y="worst_hidden_drop", hue="detector")
    plt.xticks(rotation=15, ha="right")
    plt.title("Mean hidden drop by split")
    plt.tight_layout()
    plt.savefig(out_dir / "hidden_drop_by_split.png", dpi=180)
    plt.close()

    rrs = ranking_df[ranking_df["environment_id"] != "observed"]
    plt.figure(figsize=(11, 5))
    sns.barplot(data=rrs, x="split", y="ranking_reversal_score", hue="environment_id")
    plt.xticks(rotation=15, ha="right")
    plt.title("Ranking reversal by split")
    plt.tight_layout()
    plt.savefig(out_dir / "ranking_reversal_by_split.png", dpi=180)
    plt.close()


def main():
    config_path = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "configs" / "uwf_split_experiments.yaml"
    with config_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    start = time.perf_counter()
    out_dir = ROOT / cfg["output_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_uwf_directory(
        raw_dir=ROOT / cfg["dataset"]["raw_dir"],
        max_rows_per_family=int(cfg["dataset"]["max_rows_per_family"]),
        random_state=int(cfg["random_state"]),
    )
    write_json(out_dir / "dataset_profile.json", dataset_profile(df))

    interventions = load_interventions(cfg["interventions"])
    detectors = build_detectors(int(cfg["random_state"]))

    all_metrics = []
    all_predictions = []
    split_profiles = []
    for split_name, fold, train_df, test_df in make_splits(
        df,
        split_names=cfg["splits"],
        test_size=float(cfg["test_size"]),
        seed=int(cfg["random_state"]),
    ):
        split_profiles.append({
            "split": split_name,
            "fold": fold,
            "train_rows": int(len(train_df)),
            "test_rows": int(len(test_df)),
            "train_positive_rate": float(train_df["y"].mean()),
            "test_positive_rate": float(test_df["y"].mean()),
            "train_families": sorted(train_df["attack_family"].astype(str).unique().tolist()),
            "test_families": sorted(test_df["attack_family"].astype(str).unique().tolist()),
        })
        metrics, predictions = evaluate_split(
            split_name,
            fold,
            train_df,
            test_df,
            interventions,
            detectors,
            seed=int(cfg["random_state"]),
        )
        all_metrics.append(metrics)
        all_predictions.append(predictions)

    metrics_df = pd.concat(all_metrics, ignore_index=True)
    predictions_df = pd.concat(all_predictions, ignore_index=True)
    hidden_df = summarize_hidden_drop(metrics_df)
    ranking_df = summarize_ranking_by_split(metrics_df)

    metrics_df.to_csv(out_dir / "split_metrics.csv", index=False)
    predictions_df.to_csv(out_dir / "split_predictions.csv", index=False)
    hidden_df.to_csv(out_dir / "split_hidden_drop.csv", index=False)
    ranking_df.to_csv(out_dir / "split_ranking_reversal.csv", index=False)
    pd.DataFrame(split_profiles).to_csv(out_dir / "split_profiles.csv", index=False)

    summary_by_split = (
        hidden_df
        .groupby("split", as_index=False)
        .agg(
            mean_hidden_drop=("worst_hidden_drop", "mean"),
            max_hidden_drop=("worst_hidden_drop", "max"),
        )
    )
    rrs_by_split = (
        ranking_df[ranking_df["environment_id"] != "observed"]
        .groupby("split", as_index=False)
        .agg(
            mean_rrs=("ranking_reversal_score", "mean"),
            max_rrs=("ranking_reversal_score", "max"),
        )
    )
    summary = summary_by_split.merge(rrs_by_split, on="split", how="left")
    summary.to_csv(out_dir / "split_summary.csv", index=False)
    plot_split_results(hidden_df, ranking_df, out_dir)

    payload = {
        "config": str(config_path),
        "output_dir": str(out_dir),
        "rows": int(len(df)),
        "splits": cfg["splits"],
        "num_split_folds": int(len(split_profiles)),
        "elapsed_seconds": time.perf_counter() - start,
        "max_hidden_drop": float(hidden_df["worst_hidden_drop"].max()),
        "max_rrs": float(ranking_df["ranking_reversal_score"].max()),
    }
    write_json(out_dir / "split_run_summary.json", payload)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()


