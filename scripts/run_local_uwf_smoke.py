from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import yaml
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, IsolationForest, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from intervendb.data import DatasetPaths, dataset_profile, feature_columns, load_uwf_directory, load_uwf_smoke_dataset, write_json
from intervendb.interventions import apply_intervention, load_interventions
from intervendb.metrics import (
    causal_fragility_score,
    classification_metrics,
    environment_leakage_score,
    score_environment_leakage_score,
    summarize_ranking,
)


def build_preprocessor(profile):
    numeric_cols, categorical_cols = feature_columns(profile)
    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    transformers = [("num", numeric_pipe, numeric_cols)]
    if categorical_cols:
        transformers.append(("cat", categorical_pipe, categorical_cols))
    return ColumnTransformer(transformers)


def build_detectors(seed: int):
    return {
        "lr_full": {
            "profile": "full",
            "kind": "supervised",
            "estimator": LogisticRegression(max_iter=1000, class_weight="balanced", n_jobs=None),
        },
        "rf_full": {
            "profile": "full",
            "kind": "supervised",
            "estimator": RandomForestClassifier(
                n_estimators=160,
                max_depth=None,
                min_samples_leaf=2,
                class_weight="balanced",
                random_state=seed,
                n_jobs=-1,
            ),
        },
        "gb_full": {
            "profile": "full",
            "kind": "supervised",
            "estimator": GradientBoostingClassifier(random_state=seed),
        },
        "lr_env_sensitive": {
            "profile": "env_sensitive",
            "kind": "supervised",
            "estimator": LogisticRegression(max_iter=1000, class_weight="balanced", n_jobs=None),
        },
        "rf_env_sensitive": {
            "profile": "env_sensitive",
            "kind": "supervised",
            "estimator": RandomForestClassifier(
                n_estimators=180,
                min_samples_leaf=2,
                class_weight="balanced",
                random_state=seed + 7,
                n_jobs=-1,
            ),
        },
        "protocol_core_rf": {
            "profile": "protocol_core",
            "kind": "supervised",
            "estimator": RandomForestClassifier(
                n_estimators=120,
                min_samples_leaf=2,
                class_weight="balanced",
                random_state=seed + 13,
                n_jobs=-1,
            ),
        },
        "iforest_env_sensitive": {
            "profile": "env_sensitive",
            "kind": "unsupervised",
            "estimator": IsolationForest(
                n_estimators=180,
                contamination="auto",
                random_state=seed,
                n_jobs=-1,
            ),
        },
    }


def detector_score(model_name, model, x):
    if model_name.startswith("iforest"):
        raw = -model.decision_function(x)
        min_v, max_v = raw.min(), raw.max()
        score = (raw - min_v) / (max_v - min_v + 1e-12)
        pred = (score >= 0.5).astype(int)
        return pred, score
    if hasattr(model, "predict_proba"):
        score = model.predict_proba(x)[:, 1]
        pred = (score >= 0.5).astype(int)
        return pred, score
    pred = model.predict(x)
    return pred, pred.astype(float)


def fit_detector(kind, estimator, preprocessor, train_df, y_train):
    if kind == "unsupervised":
        benign_train = train_df[y_train == 0]
        pipe = Pipeline([
            ("preprocessor", preprocessor),
            ("model", estimator),
        ])
        pipe.fit(benign_train)
        return pipe

    pipe = Pipeline([
        ("preprocessor", preprocessor),
        ("model", estimator),
    ])
    pipe.fit(train_df, y_train)
    return pipe


def plot_results(metrics_df: pd.DataFrame, ranking_df: pd.DataFrame, out_dir: Path):
    sns.set_theme(style="whitegrid")

    plt.figure(figsize=(10, 5))
    pivot = metrics_df.pivot(index="detector", columns="environment_id", values="f1")
    sns.heatmap(pivot, annot=True, fmt=".3f", cmap="viridis")
    plt.title("F1 under counterfactual environment interventions")
    plt.tight_layout()
    plt.savefig(out_dir / "f1_heatmap.png", dpi=180)
    plt.close()

    plt.figure(figsize=(10, 5))
    sns.barplot(data=metrics_df, x="detector", y="causal_fragility_score")
    plt.xticks(rotation=20, ha="right")
    plt.title("Causal Fragility Score by detector")
    plt.tight_layout()
    plt.savefig(out_dir / "cfs_by_detector.png", dpi=180)
    plt.close()

    plt.figure(figsize=(9, 4))
    sns.barplot(data=ranking_df, x="environment_id", y="ranking_reversal_score")
    plt.xticks(rotation=20, ha="right")
    plt.title("Ranking reversal relative to observed benchmark")
    plt.tight_layout()
    plt.savefig(out_dir / "ranking_reversal.png", dpi=180)
    plt.close()


def main():
    config_path = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "configs" / "local_uwf_smoke.yaml"
    with config_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    start = time.perf_counter()
    out_dir = ROOT / cfg["output_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)

    if "raw_dir" in cfg["dataset"]:
        df = load_uwf_directory(
            raw_dir=ROOT / cfg["dataset"]["raw_dir"],
            max_rows_per_family=int(cfg["dataset"]["max_rows_per_family"]),
            random_state=int(cfg["random_state"]),
        )
    else:
        paths = DatasetPaths(
            benign_csv=ROOT / cfg["dataset"]["benign_csv"],
            reconnaissance_csv=ROOT / cfg["dataset"]["reconnaissance_csv"],
        )
        df = load_uwf_smoke_dataset(
            paths=paths,
            max_rows_per_class=int(cfg["dataset"]["max_rows_per_class"]),
            random_state=int(cfg["random_state"]),
        )
    write_json(out_dir / "dataset_profile.json", dataset_profile(df))

    train_df, test_df = train_test_split(
        df,
        test_size=float(cfg["test_size"]),
        stratify=df["y"],
        random_state=int(cfg["random_state"]),
    )
    y_train = train_df["y"].to_numpy()
    y_test = test_df["y"].to_numpy()

    interventions = load_interventions(cfg["interventions"])
    detectors = build_detectors(int(cfg["random_state"]))

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

            base_metrics = classification_metrics(y_test, pred, score)
            row = {
                "detector": detector_name,
                "feature_profile": spec["profile"],
                "environment_id": env_name,
                "fit_seconds": fit_seconds,
                "eval_seconds": eval_seconds,
                "semantic_preservation_score": sps,
                **base_metrics,
            }
            metric_rows.append(row)

            pred_frame = pd.DataFrame({
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

    metrics_df = pd.DataFrame(metric_rows)
    ranking_df = pd.DataFrame(summarize_ranking(metric_rows))
    predictions_df = pd.concat(prediction_frames, ignore_index=True)

    metrics_df.to_csv(out_dir / "metrics_by_detector_environment.csv", index=False)
    ranking_df.to_csv(out_dir / "ranking_reversal.csv", index=False)
    predictions_df.to_csv(out_dir / "predictions_long.csv", index=False)

    observed = metrics_df[metrics_df["environment_id"] == "observed"][["detector", "f1"]].rename(columns={"f1": "observed_f1"})
    worst = metrics_df.groupby("detector", as_index=False)["f1"].min().rename(columns={"f1": "worst_intervention_f1"})
    summary = observed.merge(worst, on="detector")
    summary["worst_hidden_drop"] = summary["observed_f1"] - summary["worst_intervention_f1"]
    summary.to_csv(out_dir / "hidden_drop_summary.csv", index=False)

    plot_results(metrics_df, ranking_df, out_dir)

    run_summary = {
        "config": str(config_path),
        "output_dir": str(out_dir),
        "rows": int(len(df)),
        "test_rows": int(len(test_df)),
        "detectors": sorted(detectors.keys()),
        "interventions": sorted(interventions.keys()),
        "elapsed_seconds": time.perf_counter() - start,
        "observed_top_detector": str(
            metrics_df[metrics_df["environment_id"] == "observed"]
            .sort_values("f1", ascending=False)
            .iloc[0]["detector"]
        ),
        "max_hidden_drop": float(summary["worst_hidden_drop"].max()),
    }
    write_json(out_dir / "run_summary.json", run_summary)

    print(json.dumps(run_summary, indent=2))


if __name__ == "__main__":
    main()


