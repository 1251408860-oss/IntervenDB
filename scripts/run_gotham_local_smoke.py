from __future__ import annotations

import json
import hashlib
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
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

from intervendb.metrics import (
    causal_fragility_score,
    classification_metrics,
    environment_leakage_score,
    ranking_reversal_score,
    score_environment_leakage_score,
)


def stable_seed(seed: int, text: str) -> int:
    digest = hashlib.md5(text.encode("utf-8")).hexdigest()
    return (seed + int(digest[:8], 16)) % (2**32 - 1)


def select_csv_paths(raw_dir: Path, include_files: list[str] | None = None) -> list[Path]:
    paths = sorted(raw_dir.glob("*.csv"))
    if include_files:
        wanted = set(include_files)
        paths = [path for path in paths if path.name in wanted]
        missing = sorted(wanted - {path.name for path in paths})
        if missing:
            print(f"warning: requested Gotham files not found under {raw_dir}: {missing}")
    return paths


def read_csv_limited(path: Path, max_rows: int | None, seed: int, chunksize: int) -> pd.DataFrame:
    if max_rows is None or max_rows <= 0:
        return pd.read_csv(path, low_memory=False)

    rng = np.random.default_rng(stable_seed(seed, path.name))
    kept = None
    for chunk in pd.read_csv(path, chunksize=chunksize, low_memory=False):
        if chunk.empty:
            continue
        chunk = chunk.copy()
        chunk["_sample_key"] = rng.random(len(chunk))
        kept = chunk if kept is None else pd.concat([kept, chunk], ignore_index=True)
        if len(kept) > max_rows:
            kept = kept.nsmallest(max_rows, "_sample_key").reset_index(drop=True)
    if kept is None:
        return pd.DataFrame()
    return kept.drop(columns=["_sample_key"]).reset_index(drop=True)


def load_gotham(
    raw_dir: Path,
    max_rows_per_label: int,
    seed: int,
    include_files: list[str] | None = None,
    max_rows_per_file: int | None = None,
    chunksize: int = 100_000,
) -> pd.DataFrame:
    frames = []
    for path in select_csv_paths(raw_dir, include_files):
        frame = read_csv_limited(path, max_rows_per_file, seed, chunksize)
        if frame.empty:
            continue
        frame["device_file"] = path.name
        frame["device_type"] = path.name.replace("iotsim-", "").rsplit("-", 1)[0]
        frames.append(frame)
    if not frames:
        raise FileNotFoundError(f"No Gotham CSV files found under {raw_dir}")
    df = pd.concat(frames, ignore_index=True)
    df = df[df["label"].astype(str) != "Unknown"].copy()

    numeric_cols = [
        "frame.len",
        "ip.ttl",
        "ip.proto",
        "tcp.srcport",
        "tcp.dstport",
        "tcp.window_size_value",
        "tcp.window_size_scalefactor",
        "tcp.pdu.size",
        "udp.srcport",
        "udp.dstport",
    ]
    for col in numeric_cols:
        if col not in df.columns:
            df[col] = 0.0
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    for col in ["frame.protocols", "ip.flags", "tcp.flags", "ip.src", "ip.dst"]:
        if col not in df.columns:
            df[col] = "missing"
        df[col] = df[col].fillna("missing").astype(str)

    dt = pd.to_datetime(df["frame.time"], errors="coerce")
    ts = dt.astype("int64") / 1e9
    ts = ts.replace([np.inf, -np.inf], np.nan).fillna(ts.median())
    df["ts"] = ts
    df["ts_norm"] = (ts - ts.min()) / (ts.max() - ts.min() + 1e-9)
    df["hour_of_day"] = np.floor((ts / 3600.0) % 24).astype(int)
    df["label"] = df["label"].astype(str)
    df["y"] = (df["label"] != "Benign").astype(int)
    df["semantic_label"] = np.where(df["y"] == 1, df["label"], "benign")
    df["attack_family"] = df["semantic_label"]

    parts = []
    for label, part in df.groupby("semantic_label", dropna=False):
        n = min(max_rows_per_label, len(part))
        parts.append(part.sample(n=n, random_state=seed, replace=False))
    return pd.concat(parts, ignore_index=True).sample(frac=1.0, random_state=seed).reset_index(drop=True)


def feature_columns(profile: str):
    if profile == "full":
        return [
            "frame.len",
            "ip.ttl",
            "ip.proto",
            "tcp.srcport",
            "tcp.dstport",
            "tcp.window_size_value",
            "tcp.window_size_scalefactor",
            "tcp.pdu.size",
            "udp.srcport",
            "udp.dstport",
            "ts_norm",
            "hour_of_day",
        ], ["frame.protocols", "ip.flags", "tcp.flags", "device_type"]
    if profile == "env_sensitive":
        return [
            "frame.len",
            "ip.ttl",
            "tcp.window_size_value",
            "tcp.window_size_scalefactor",
            "tcp.pdu.size",
            "ts_norm",
            "hour_of_day",
        ], []
    if profile == "protocol_core":
        return [
            "ip.proto",
            "tcp.srcport",
            "tcp.dstport",
            "udp.srcport",
            "udp.dstport",
        ], ["frame.protocols", "ip.flags", "tcp.flags"]
    raise ValueError(f"Unknown profile: {profile}")


def build_preprocessor(profile: str):
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
        "lr_full": {"profile": "full", "kind": "supervised", "estimator": LogisticRegression(max_iter=1000, class_weight="balanced")},
        "rf_full": {"profile": "full", "kind": "supervised", "estimator": RandomForestClassifier(n_estimators=140, class_weight="balanced", min_samples_leaf=2, random_state=seed, n_jobs=-1)},
        "gb_full": {"profile": "full", "kind": "supervised", "estimator": GradientBoostingClassifier(random_state=seed)},
        "lr_env_sensitive": {"profile": "env_sensitive", "kind": "supervised", "estimator": LogisticRegression(max_iter=1000, class_weight="balanced")},
        "rf_env_sensitive": {"profile": "env_sensitive", "kind": "supervised", "estimator": RandomForestClassifier(n_estimators=140, class_weight="balanced", min_samples_leaf=2, random_state=seed + 7, n_jobs=-1)},
        "protocol_core_rf": {"profile": "protocol_core", "kind": "supervised", "estimator": RandomForestClassifier(n_estimators=120, class_weight="balanced", min_samples_leaf=2, random_state=seed + 13, n_jobs=-1)},
        "iforest_env_sensitive": {"profile": "env_sensitive", "kind": "unsupervised", "estimator": IsolationForest(n_estimators=160, random_state=seed, n_jobs=-1)},
    }


def apply_gotham_intervention(df: pd.DataFrame, name: str, params: dict):
    rng = np.random.default_rng(int(params.get("seed", 0)))
    out = df.copy()

    if "sampling_rate" in params:
        rate = float(params["sampling_rate"])
        for col in ["frame.len", "tcp.pdu.size"]:
            noise = rng.normal(1.0, 0.02, len(out))
            out[col] = (out[col].astype(float).to_numpy() * rate * noise).clip(min=0)

    if "duration_multiplier" in params:
        mult = float(params["duration_multiplier"])
        centered = out["ts_norm"].astype(float).to_numpy() - out["ts_norm"].min()
        out["ts_norm"] = (centered * mult)
        out["ts_norm"] = (out["ts_norm"] - out["ts_norm"].min()) / (out["ts_norm"].max() - out["ts_norm"].min() + 1e-9)

    if "background_byte_multiplier" in params:
        mult = float(params["background_byte_multiplier"])
        noise = rng.normal(1.0, 0.05, len(out))
        out["frame.len"] = (out["frame.len"].astype(float).to_numpy() * mult * noise).clip(min=0)

    if "packet_jitter" in params:
        jitter = float(params["packet_jitter"])
        out["frame.len"] = (out["frame.len"].astype(float).to_numpy() + rng.normal(0.0, jitter, len(out))).clip(min=0)

    if "timestamp_shift" in params:
        ts = out["ts"].astype(float).to_numpy() + float(params["timestamp_shift"])
        out["ts"] = ts
        out["hour_of_day"] = np.floor((ts / 3600.0) % 24).astype(int)

    sps = 1.0
    out["environment_id"] = name
    return out, sps


def detector_score(name, model, x):
    if name.startswith("iforest"):
        raw = -model.decision_function(x)
        score = (raw - raw.min()) / (raw.max() - raw.min() + 1e-12)
        pred = (score >= 0.5).astype(int)
        return pred, score
    score = model.predict_proba(x)[:, 1] if hasattr(model, "predict_proba") else model.predict(x).astype(float)
    pred = (score >= 0.5).astype(int)
    return pred, score


def fit_detector(kind, estimator, preprocessor, train_df, y_train):
    pipe = Pipeline([("preprocessor", preprocessor), ("model", estimator)])
    if kind == "unsupervised":
        pipe.fit(train_df[y_train == 0])
    else:
        pipe.fit(train_df, y_train)
    return pipe


def make_split(df: pd.DataFrame, split: str, test_size: float, seed: int):
    if split == "random":
        return train_test_split(df, test_size=test_size, stratify=df["y"], random_state=seed)
    if split == "device":
        devices = df["device_file"].astype(str).unique().tolist()
        rng = np.random.default_rng(seed)
        rng.shuffle(devices)
        test_devices = set(devices[: max(1, int(round(len(devices) * test_size)))])
        test = df[df["device_file"].astype(str).isin(test_devices)]
        train = df[~df["device_file"].astype(str).isin(test_devices)]
        if train["y"].nunique() < 2 or test["y"].nunique() < 2:
            return train_test_split(df, test_size=test_size, stratify=df["y"], random_state=seed)
        return train, test
    if split == "time":
        train_parts, test_parts = [], []
        for _, part in df.groupby("semantic_label", dropna=False):
            ordered = part.sort_values("ts")
            n_test = max(1, int(round(len(ordered) * test_size)))
            train_parts.append(ordered.iloc[:-n_test])
            test_parts.append(ordered.iloc[-n_test:])
        return pd.concat(train_parts).sample(frac=1, random_state=seed), pd.concat(test_parts).sample(frac=1, random_state=seed + 1)
    raise ValueError(f"Unknown split: {split}")


def evaluate(df, cfg):
    train_df, test_df = make_split(df, cfg["split"], float(cfg["test_size"]), int(cfg["random_state"]))
    y_train = train_df["y"].to_numpy()
    y_test = test_df["y"].to_numpy()
    detectors = build_detectors(int(cfg["random_state"]))
    interventions = cfg["interventions"]
    metric_rows = []
    prediction_frames = []

    for detector_name, spec in detectors.items():
        start = time.perf_counter()
        model = fit_detector(spec["kind"], spec["estimator"], build_preprocessor(spec["profile"]), train_df, y_train)
        fit_seconds = time.perf_counter() - start
        detector_parts = []
        for env_name, params in interventions.items():
            eval_df, sps = apply_gotham_intervention(test_df, env_name, params or {})
            pred, score = detector_score(detector_name, model, eval_df)
            row = {
                "dataset": "gotham_2025_sample",
                "split": cfg["split"],
                "detector": detector_name,
                "feature_profile": spec["profile"],
                "environment_id": env_name,
                "train_rows": int(len(train_df)),
                "test_rows": int(len(test_df)),
                "train_positive_rate": float(train_df["y"].mean()),
                "test_positive_rate": float(test_df["y"].mean()),
                "fit_seconds": fit_seconds,
                "semantic_preservation_score": sps,
                **classification_metrics(y_test, pred, score),
            }
            metric_rows.append(row)
            part = pd.DataFrame({
                "detector": detector_name,
                "environment_id": env_name,
                "semantic_label": eval_df["semantic_label"].astype(str).to_numpy(),
                "y": y_test,
                "prediction": pred,
                "score": score,
            })
            detector_parts.append(part)
            prediction_frames.append(part.assign(split=cfg["split"]))
        detector_all = pd.concat(detector_parts, ignore_index=True)
        cfs = causal_fragility_score(detector_all)
        els = environment_leakage_score(detector_all)
        score_els = score_environment_leakage_score(detector_all)
        for row in metric_rows:
            if row["detector"] == detector_name:
                row["causal_fragility_score"] = cfs
                row["environment_leakage_score"] = els
                row["score_environment_leakage_score"] = score_els
    return pd.DataFrame(metric_rows), pd.concat(prediction_frames, ignore_index=True)


def summarize(metrics):
    observed = metrics[metrics["environment_id"] == "observed"][["detector", "f1"]].rename(columns={"f1": "observed_f1"})
    worst = metrics.groupby("detector", as_index=False)["f1"].min().rename(columns={"f1": "worst_intervention_f1"})
    hidden = observed.merge(worst, on="detector")
    hidden["worst_hidden_drop"] = hidden["observed_f1"] - hidden["worst_intervention_f1"]

    static = metrics[metrics["environment_id"] == "observed"].set_index("detector")["f1"].to_dict()
    ranking_rows = []
    for env in sorted(metrics["environment_id"].unique()):
        env_scores = metrics[metrics["environment_id"] == env].set_index("detector")["f1"].to_dict()
        ranking_rows.append({
            "environment_id": env,
            "ranking_reversal_score": ranking_reversal_score(static, env_scores),
            "static_top_detector": max(static.items(), key=lambda kv: kv[1])[0],
            "environment_top_detector": max(env_scores.items(), key=lambda kv: kv[1])[0],
        })
    return hidden, pd.DataFrame(ranking_rows)


def plot(metrics, hidden, ranking, out_dir):
    sns.set_theme(style="whitegrid")
    pivot = metrics.pivot(index="detector", columns="environment_id", values="f1")
    plt.figure(figsize=(10, 5))
    sns.heatmap(pivot, annot=True, fmt=".3f", cmap="viridis")
    plt.title("Gotham sample F1 under interventions")
    plt.tight_layout()
    plt.savefig(out_dir / "gotham_f1_heatmap.png", dpi=180)
    plt.close()

    plt.figure(figsize=(10, 4))
    sns.barplot(data=hidden, x="detector", y="worst_hidden_drop")
    plt.xticks(rotation=20, ha="right")
    plt.title("Gotham hidden drop")
    plt.tight_layout()
    plt.savefig(out_dir / "gotham_hidden_drop.png", dpi=180)
    plt.close()

    plt.figure(figsize=(8, 4))
    sns.barplot(data=ranking, x="environment_id", y="ranking_reversal_score")
    plt.xticks(rotation=20, ha="right")
    plt.title("Gotham ranking reversal")
    plt.tight_layout()
    plt.savefig(out_dir / "gotham_ranking_reversal.png", dpi=180)
    plt.close()


def main():
    if len(sys.argv) < 2:
        print("usage: python scripts/run_gotham_local_smoke.py <config.yaml>")
        return 2
    cfg_path = Path(sys.argv[1])
    with cfg_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    start = time.perf_counter()
    out_dir = ROOT / cfg["output_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset_cfg = cfg["dataset"]
    max_rows_per_file = dataset_cfg.get("max_rows_per_file")
    max_rows_per_file = int(max_rows_per_file) if max_rows_per_file else None
    chunksize = int(dataset_cfg.get("chunksize", 100_000))
    cache_path = dataset_cfg.get("cache_path")
    if cache_path:
        cache_path = ROOT / cache_path
    if cache_path and cache_path.exists():
        df = pd.read_parquet(cache_path)
    else:
        df = load_gotham(
            ROOT / dataset_cfg["raw_dir"],
            int(dataset_cfg["max_rows_per_label"]),
            int(cfg["random_state"]),
            include_files=dataset_cfg.get("include_files"),
            max_rows_per_file=max_rows_per_file,
            chunksize=chunksize,
        )
        if cache_path:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_parquet(cache_path, index=False)
    metrics, predictions = evaluate(df, cfg)
    hidden, ranking = summarize(metrics)

    metrics.to_csv(out_dir / "gotham_metrics.csv", index=False)
    predictions.to_csv(out_dir / "gotham_predictions.csv", index=False)
    hidden.to_csv(out_dir / "gotham_hidden_drop.csv", index=False)
    ranking.to_csv(out_dir / "gotham_ranking_reversal.csv", index=False)
    plot(metrics, hidden, ranking, out_dir)

    profile = {
        "rows": int(len(df)),
        "positives": int(df["y"].sum()),
        "negatives": int((1 - df["y"]).sum()),
        "labels": df["semantic_label"].value_counts().to_dict(),
        "devices": sorted(df["device_file"].unique().tolist()),
    }
    summary = {
        "config": str(cfg_path),
        "output_dir": str(out_dir),
        "profile": profile,
        "max_hidden_drop": float(hidden["worst_hidden_drop"].max()),
        "max_rrs": float(ranking["ranking_reversal_score"].max()),
        "elapsed_seconds": time.perf_counter() - start,
    }
    with (out_dir / "gotham_run_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()


