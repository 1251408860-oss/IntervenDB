from __future__ import annotations

import hashlib
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import yaml
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, IsolationForest, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
SCRIPTS = ROOT / "scripts"
for path in [SRC, SCRIPTS]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from intervendb.data import dataset_profile, feature_columns as uwf_feature_columns, load_uwf_directory, write_json
from intervendb.interventions import apply_intervention, load_interventions
from intervendb.metrics import (
    causal_fragility_score,
    classification_metrics,
    environment_leakage_score,
    ranking_reversal_score,
    score_environment_leakage_score,
)
from run_gotham_local_smoke import (
    apply_gotham_intervention,
    feature_columns as gotham_feature_columns,
    load_gotham,
    make_split as make_gotham_split,
)
from run_uwf_split_experiments import make_splits as make_uwf_splits
from run_uwf_split_experiments import validate_split


@dataclass(frozen=True)
class DetectorSpec:
    name: str
    family: str
    profile: str
    kind: str
    estimator_factory: Callable[[int], object]


@dataclass
class PreprocessedDetector:
    preprocessor: Any
    estimator: Any


CHECKPOINT_VERSION = 1


def config_fingerprint(cfg: dict) -> str:
    payload = json.dumps(cfg, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in str(value))


def checkpoint_options(cfg: dict) -> dict:
    raw = cfg.get("checkpoint", {}) or {}
    return {
        "enabled": bool(raw.get("enabled", True)),
        "resume": bool(raw.get("resume", True)),
        "dir": str(raw.get("dir", "_checkpoints_baseline")),
    }


def split_checkpoint_dir(out_dir: Path, cfg: dict, split_name: str, fold: str) -> Path:
    opts = checkpoint_options(cfg)
    return out_dir / opts["dir"] / f"{safe_name(split_name)}__{safe_name(fold)}"


def load_split_checkpoint(
    out_dir: Path,
    cfg: dict,
    cfg_hash: str,
    split_name: str,
    fold: str,
) -> tuple[pd.DataFrame, pd.DataFrame, dict, pd.DataFrame] | None:
    opts = checkpoint_options(cfg)
    if not opts["enabled"] or not opts["resume"]:
        return None
    ckpt_dir = split_checkpoint_dir(out_dir, cfg, split_name, fold)
    manifest_path = ckpt_dir / "manifest.json"
    metrics_path = ckpt_dir / "metrics.csv"
    ranking_path = ckpt_dir / "ranking.csv"
    profile_path = ckpt_dir / "split_profile.json"
    calibration_path = ckpt_dir / "calibration_profile.csv"
    if not all(path.exists() for path in [manifest_path, metrics_path, ranking_path, profile_path]):
        return None
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    if manifest.get("checkpoint_version") != CHECKPOINT_VERSION:
        return None
    if manifest.get("config_fingerprint") != cfg_hash:
        return None
    if manifest.get("split") != split_name or str(manifest.get("fold")) != str(fold):
        return None
    metrics = pd.read_csv(metrics_path)
    ranking = pd.read_csv(ranking_path)
    profile = json.loads(profile_path.read_text(encoding="utf-8"))
    calibration = pd.read_csv(calibration_path) if calibration_path.exists() else pd.DataFrame()
    print(f"resume baseline split checkpoint {split_name}/{fold}")
    return metrics, ranking, profile, calibration


def write_split_checkpoint(
    out_dir: Path,
    cfg: dict,
    cfg_hash: str,
    split_name: str,
    fold: str,
    split_profile: dict,
    metrics: pd.DataFrame,
    ranking: pd.DataFrame,
    calibration_profile: pd.DataFrame | None = None,
) -> None:
    opts = checkpoint_options(cfg)
    if not opts["enabled"]:
        return
    ckpt_dir = split_checkpoint_dir(out_dir, cfg, split_name, fold)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(ckpt_dir / "metrics.csv", index=False)
    ranking.to_csv(ckpt_dir / "ranking.csv", index=False)
    if calibration_profile is not None and not calibration_profile.empty:
        calibration_profile.to_csv(ckpt_dir / "calibration_profile.csv", index=False)
    write_json(ckpt_dir / "split_profile.json", split_profile)
    manifest = {
        "checkpoint_version": CHECKPOINT_VERSION,
        "config_fingerprint": cfg_hash,
        "split": split_name,
        "fold": str(fold),
        "written_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    write_json(ckpt_dir / "manifest.json", manifest)


def stable_int(seed: int, *parts: object) -> int:
    text = "::".join([str(seed), *[str(part) for part in parts]])
    digest = hashlib.md5(text.encode("utf-8")).hexdigest()
    return (seed + int(digest[:8], 16)) % (2**32 - 1)


def calibration_options(cfg: dict) -> dict:
    raw = cfg.get("calibration", {}) or {}
    enabled = bool(raw.get("enabled", False))
    policies = raw.get("policies") or ["native_default"]
    if not enabled:
        policies = ["native_default"]
    return {
        "enabled": enabled,
        "policies": [str(policy) for policy in policies],
        "threshold_grid_size": int(raw.get("threshold_grid_size", 101)),
        "refit_on_full_train": bool(raw.get("refit_on_full_train", True)),
    }


def threshold_grid(size: int) -> np.ndarray:
    return np.linspace(0.01, 0.99, max(11, int(size)))


def best_threshold_for_frames(frames: list[pd.DataFrame], grid: np.ndarray) -> tuple[float, float, float]:
    best = (0.5, -1.0, -1.0)
    for threshold in grid:
        f1s = []
        for frame in frames:
            pred = (frame["score"].to_numpy(dtype=float) >= threshold).astype(int)
            f1s.append(f1_score(frame["y"].to_numpy(), pred, zero_division=0))
        min_f1 = float(np.min(f1s)) if f1s else 0.0
        mean_f1 = float(np.mean(f1s)) if f1s else 0.0
        if (min_f1, mean_f1) > (best[1], best[2]):
            best = (float(threshold), min_f1, mean_f1)
    return best


def best_threshold_for_frame(frame: pd.DataFrame, grid: np.ndarray) -> tuple[float, float]:
    best = (0.5, -1.0)
    y = frame["y"].to_numpy()
    score = frame["score"].to_numpy(dtype=float)
    for threshold in grid:
        pred = (score >= threshold).astype(int)
        value = float(f1_score(y, pred, zero_division=0))
        if value > best[1]:
            best = (float(threshold), value)
    return best


def observed_frame(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    if "observed" in frames:
        return frames["observed"]
    return next(iter(frames.values()))


def random_validation_split(train_df: pd.DataFrame, validation_size: float, seed: int):
    y = train_df["y"]
    if y.nunique() == 2 and y.value_counts().min() >= 2:
        fit_df, val_df = train_test_split(
            train_df,
            test_size=validation_size,
            stratify=y,
            random_state=seed,
        )
        return fit_df, val_df, {"strategy": "random"}
    val = train_df.sample(frac=validation_size, random_state=seed)
    fit = train_df.drop(val.index)
    return fit, val, {"strategy": "random"}


def challenge_validation_split(
    train_df: pd.DataFrame,
    validation_size: float,
    seed: int,
    cfg: dict,
    split_name: str,
    fold: str,
):
    validation_cfg = cfg.get("validation", {}) or {}
    strategy = str(validation_cfg.get("strategy", "random"))
    if strategy == "random":
        return random_validation_split(train_df, validation_size, seed)

    if strategy == "attack_family_challenge":
        group_col = str(validation_cfg.get("group_col", "attack_family"))
        positive_only = True
    elif strategy == "device_challenge":
        group_col = str(validation_cfg.get("group_col", "device_file"))
        positive_only = False
    elif strategy == "group_challenge":
        group_col = str(validation_cfg.get("group_col", "attack_family"))
        positive_only = bool(validation_cfg.get("positive_only", True))
    else:
        return random_validation_split(train_df, validation_size, seed)

    if group_col not in train_df.columns:
        return random_validation_split(train_df, validation_size, seed)

    min_group_rows = int(validation_cfg.get("min_group_rows", 30))
    max_groups = max(1, int(validation_cfg.get("max_groups", 1)))
    benign_label = str(validation_cfg.get("benign_label", "benign"))
    source = train_df[train_df["y"] == 1] if positive_only else train_df
    groups = []
    for group, part in source.groupby(group_col, dropna=False):
        group_name = str(group)
        if group_name == benign_label:
            continue
        if len(part) >= min_group_rows:
            groups.append(group_name)
    groups = sorted(set(groups))
    if len(groups) < 2:
        return random_validation_split(train_df, validation_size, seed)

    start = stable_int(seed, split_name, fold, group_col) % len(groups)
    selected_groups = [groups[(start + idx) % len(groups)] for idx in range(min(max_groups, len(groups) - 1))]
    selected_mask = train_df[group_col].astype(str).isin(selected_groups)
    if positive_only:
        selected_mask &= train_df["y"].eq(1)

    removed_group = train_df[selected_mask]
    fit_base = train_df[~selected_mask]
    if removed_group.empty or fit_base["y"].nunique() < 2:
        return random_validation_split(train_df, validation_size, seed)

    max_rows = int(validation_cfg.get("max_rows", max(100, round(len(train_df) * validation_size))))
    benign_ratio = float(validation_cfg.get("benign_ratio", 1.0))
    attack_rows = max(10, min(len(removed_group), int(max_rows / (1.0 + max(benign_ratio, 0.01)))))
    val_group = removed_group.sample(n=attack_rows, random_state=stable_int(seed, split_name, fold, "group_eval"))

    benign_pool = fit_base[fit_base["y"] == 0]
    benign_rows = min(len(benign_pool), max_rows - len(val_group), max(10, int(round(len(val_group) * benign_ratio))))
    if benign_rows <= 0:
        return random_validation_split(train_df, validation_size, seed)
    val_benign = benign_pool.sample(n=benign_rows, random_state=stable_int(seed, split_name, fold, "benign_eval"))

    fit_df = fit_base.drop(val_benign.index)
    val_df = pd.concat([val_group, val_benign], ignore_index=True).sample(
        frac=1.0,
        random_state=stable_int(seed, split_name, fold, "validation_shuffle"),
    )
    if fit_df["y"].nunique() < 2 or val_df["y"].nunique() < 2:
        return random_validation_split(train_df, validation_size, seed)

    return fit_df, val_df, {
        "strategy": strategy,
        "group_col": group_col,
        "groups": ",".join(selected_groups),
        "removed_group_rows": int(len(removed_group)),
    }


def optional_import(module: str, attr: str | None = None):
    try:
        imported = __import__(module, fromlist=[attr] if attr else [])
        return getattr(imported, attr) if attr else imported
    except Exception as exc:
        print(f"warning: optional baseline unavailable: {module}{'.' + attr if attr else ''}: {exc}")
        return None


def build_preprocessor(dataset_type: str, profile: str) -> ColumnTransformer:
    if dataset_type == "uwf":
        numeric_cols, categorical_cols = uwf_feature_columns(profile)
    elif dataset_type == "gotham":
        numeric_cols, categorical_cols = gotham_feature_columns(profile)
    else:
        raise ValueError(f"unknown dataset type: {dataset_type}")

    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", make_onehot_encoder()),
    ])
    transformers = [("num", numeric_pipe, numeric_cols)]
    if categorical_cols:
        transformers.append(("cat", categorical_pipe, categorical_cols))
    return ColumnTransformer(transformers)


def make_onehot_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def build_detector_catalog(seed: int) -> dict[str, DetectorSpec]:
    catalog: dict[str, DetectorSpec] = {
        "lr_full": DetectorSpec(
            "lr_full",
            "sklearn_linear",
            "full",
            "supervised",
            lambda s: LogisticRegression(max_iter=1000, class_weight="balanced"),
        ),
        "rf_full": DetectorSpec(
            "rf_full",
            "sklearn_tree",
            "full",
            "supervised",
            lambda s: RandomForestClassifier(
                n_estimators=180,
                min_samples_leaf=2,
                class_weight="balanced",
                random_state=s,
                n_jobs=-1,
            ),
        ),
        "gb_full": DetectorSpec(
            "gb_full",
            "sklearn_boosting",
            "full",
            "supervised",
            lambda s: GradientBoostingClassifier(n_estimators=140, learning_rate=0.06, random_state=s),
        ),
        "iforest_env_sensitive": DetectorSpec(
            "iforest_env_sensitive",
            "sklearn_anomaly",
            "env_sensitive",
            "unsupervised_iforest",
            lambda s: IsolationForest(n_estimators=180, contamination="auto", random_state=s, n_jobs=-1),
        ),
    }

    XGBClassifier = optional_import("xgboost", "XGBClassifier")
    if XGBClassifier:
        catalog["xgb_full"] = DetectorSpec(
            "xgb_full",
            "xgboost",
            "full",
            "supervised",
            lambda s: XGBClassifier(
                n_estimators=180,
                max_depth=4,
                learning_rate=0.06,
                subsample=0.9,
                colsample_bytree=0.9,
                objective="binary:logistic",
                eval_metric="logloss",
                tree_method="hist",
                random_state=s,
                n_jobs=-1,
            ),
        )

    LGBMClassifier = optional_import("lightgbm", "LGBMClassifier")
    if LGBMClassifier:
        catalog["lgbm_full"] = DetectorSpec(
            "lgbm_full",
            "lightgbm",
            "full",
            "supervised",
            lambda s: LGBMClassifier(
                n_estimators=220,
                max_depth=-1,
                learning_rate=0.04,
                num_leaves=31,
                subsample=0.9,
                colsample_bytree=0.9,
                class_weight="balanced",
                random_state=s,
                n_jobs=-1,
                verbose=-1,
            ),
        )

    CatBoostClassifier = optional_import("catboost", "CatBoostClassifier")
    if CatBoostClassifier:
        catalog["cat_full"] = DetectorSpec(
            "cat_full",
            "catboost",
            "full",
            "supervised",
            lambda s: CatBoostClassifier(
                iterations=160,
                depth=6,
                learning_rate=0.06,
                loss_function="Logloss",
                eval_metric="F1",
                random_seed=s,
                verbose=False,
                allow_writing_files=False,
                thread_count=-1,
            ),
        )

    ExplainableBoostingClassifier = optional_import("interpret.glassbox", "ExplainableBoostingClassifier")
    if ExplainableBoostingClassifier:
        catalog["ebm_full"] = DetectorSpec(
            "ebm_full",
            "interpret_ebm",
            "full",
            "supervised",
            lambda s: ExplainableBoostingClassifier(
                interactions=0,
                outer_bags=4,
                max_rounds=160,
                learning_rate=0.03,
                random_state=s,
                n_jobs=-2,
            ),
        )

    pyod_models = {
        "pyod_ecod_env_sensitive": ("pyod_ecod", "pyod.models.ecod", "ECOD", {}),
        "pyod_copod_env_sensitive": ("pyod_copod", "pyod.models.copod", "COPOD", {}),
        "pyod_hbos_env_sensitive": ("pyod_hbos", "pyod.models.hbos", "HBOS", {"contamination": 0.05}),
    }
    for name, (family, module, attr, kwargs) in pyod_models.items():
        cls = optional_import(module, attr)
        if cls:
            catalog[name] = DetectorSpec(
                name,
                family,
                "env_sensitive",
                "unsupervised_pyod",
                lambda s, cls=cls, kwargs=kwargs: cls(**kwargs),
            )

    return catalog


def load_dataset(cfg: dict) -> tuple[pd.DataFrame, dict]:
    dataset_cfg = cfg["dataset"]
    dataset_type = dataset_cfg["type"]
    if dataset_type == "uwf":
        df = load_uwf_directory(
            raw_dir=ROOT / dataset_cfg["raw_dir"],
            max_rows_per_family=int(dataset_cfg["max_rows_per_family"]),
            random_state=int(cfg["random_state"]),
        )
        return df, dataset_profile(df)

    if dataset_type == "gotham":
        cache_path = dataset_cfg.get("cache_path")
        cache = ROOT / cache_path if cache_path else None
        if cache and cache.exists():
            df = pd.read_parquet(cache)
        else:
            max_rows_per_file = dataset_cfg.get("max_rows_per_file")
            df = load_gotham(
                ROOT / dataset_cfg["raw_dir"],
                int(dataset_cfg["max_rows_per_label"]),
                int(cfg["random_state"]),
                include_files=dataset_cfg.get("include_files"),
                max_rows_per_file=int(max_rows_per_file) if max_rows_per_file else None,
                chunksize=int(dataset_cfg.get("chunksize", 100_000)),
            )
            if cache:
                cache.parent.mkdir(parents=True, exist_ok=True)
                df.to_parquet(cache, index=False)
        profile = {
            "rows": int(len(df)),
            "positives": int(df["y"].sum()),
            "negatives": int((1 - df["y"]).sum()),
            "labels": df["semantic_label"].value_counts().to_dict(),
            "devices": sorted(df["device_file"].astype(str).unique().tolist()),
        }
        return df, profile

    raise ValueError(f"unknown dataset type: {dataset_type}")


def make_splits(df: pd.DataFrame, cfg: dict):
    dataset_type = cfg["dataset"]["type"]
    split_names = cfg["splits"]
    test_size = float(cfg["test_size"])
    seed = int(cfg["random_state"])
    max_folds = cfg.get("max_folds_per_split", {}) or {}

    if dataset_type == "uwf":
        counts: dict[str, int] = {}
        for split_name, fold, train_df, test_df in make_uwf_splits(df, split_names, test_size, seed):
            counts[split_name] = counts.get(split_name, 0) + 1
            if int(max_folds.get(split_name, 10**9)) < counts[split_name]:
                continue
            yield split_name, fold, train_df, test_df
        return

    if dataset_type == "gotham":
        for split_name in split_names:
            train_df, test_df = make_gotham_split(df, split_name, test_size, seed)
            yield split_name, "all", train_df, test_df
        return

    raise ValueError(f"unknown dataset type: {dataset_type}")


def fit_detector(spec: DetectorSpec, dataset_type: str, train_df: pd.DataFrame, y_train: np.ndarray, seed: int):
    preprocessor = build_preprocessor(dataset_type, spec.profile)
    estimator = spec.estimator_factory(seed)
    if spec.kind == "unsupervised_pyod":
        normal_df = train_df[y_train == 0]
        x_normal = preprocessor.fit_transform(normal_df)
        estimator.fit(x_normal)
        return PreprocessedDetector(preprocessor=preprocessor, estimator=estimator)

    pipe = Pipeline([
        ("preprocessor", preprocessor),
        ("model", estimator),
    ])
    if spec.kind.startswith("unsupervised"):
        pipe.fit(train_df[y_train == 0])
    else:
        pipe.fit(train_df, y_train)
    return pipe


def minmax(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    return (values - np.nanmin(values)) / (np.nanmax(values) - np.nanmin(values) + 1e-12)


def detector_score(spec: DetectorSpec, model, x: pd.DataFrame):
    if spec.kind == "unsupervised_pyod":
        transformed = model.preprocessor.transform(x)
        raw = model.estimator.decision_function(transformed)
        score = minmax(raw)
        threshold = float(getattr(model.estimator, "threshold_", np.nan))
        if np.isfinite(threshold):
            pred = (np.asarray(raw, dtype=float) > threshold).astype(int)
        else:
            pred = (score >= 0.5).astype(int)
        return pred, score

    if spec.kind.startswith("unsupervised"):
        raw = model.decision_function(x)
        if spec.kind == "unsupervised_iforest":
            raw = -raw
        score = minmax(raw)
        pred_raw = model.predict(x)
        unique = set(np.unique(pred_raw).tolist())
        if unique <= {-1, 1}:
            pred = (pred_raw == -1).astype(int)
        elif unique <= {0, 1}:
            pred = pred_raw.astype(int)
        else:
            pred = (score >= 0.5).astype(int)
        return pred, score

    if hasattr(model, "predict_proba"):
        score = model.predict_proba(x)[:, 1]
        pred = (score >= 0.5).astype(int)
        return pred, score
    pred = model.predict(x).astype(int)
    return pred, pred.astype(float)


def apply_dataset_intervention(dataset_type: str, test_df: pd.DataFrame, env_name: str, params):
    if dataset_type == "uwf":
        return apply_intervention(test_df, params)
    if dataset_type == "gotham":
        return apply_gotham_intervention(test_df, env_name, params or {})
    raise ValueError(f"unknown dataset type: {dataset_type}")


def score_detector_on_envs(
    spec: DetectorSpec,
    model,
    dataset_type: str,
    df: pd.DataFrame,
    interventions: dict,
) -> dict[str, pd.DataFrame]:
    frames = {}
    for env_name, intervention in interventions.items():
        eval_df, sps = apply_dataset_intervention(dataset_type, df, env_name, intervention)
        pred, score = detector_score(spec, model, eval_df)
        frames[str(env_name)] = pd.DataFrame({
            "environment_id": str(env_name),
            "semantic_label": eval_df["semantic_label"].astype(str).to_numpy(),
            "y": eval_df["y"].to_numpy(),
            "native_prediction": pred,
            "score": score,
            "semantic_preservation_score": sps,
        })
    return frames


def threshold_plan_for_policy(
    policy: str,
    val_frames: dict[str, pd.DataFrame],
    test_frames: dict[str, pd.DataFrame],
    grid: np.ndarray,
) -> dict:
    if policy == "native_default":
        return {
            "policy": policy,
            "threshold_mode": "native",
            "threshold": np.nan,
            "validation_min_f1": np.nan,
            "validation_mean_f1": np.nan,
        }

    if policy == "val_observed_f1":
        threshold, f1 = best_threshold_for_frame(observed_frame(val_frames), grid)
        return {
            "policy": policy,
            "threshold_mode": "single_validation_observed",
            "threshold": threshold,
            "validation_min_f1": f1,
            "validation_mean_f1": f1,
        }

    if policy == "val_robust_minimax":
        threshold, min_f1, mean_f1 = best_threshold_for_frames(list(val_frames.values()), grid)
        return {
            "policy": policy,
            "threshold_mode": "single_validation_minimax",
            "threshold": threshold,
            "validation_min_f1": min_f1,
            "validation_mean_f1": mean_f1,
        }

    if policy == "oracle_env_upper_bound":
        thresholds = {}
        f1s = []
        for env_name, frame in test_frames.items():
            threshold, f1 = best_threshold_for_frame(frame, grid)
            thresholds[env_name] = threshold
            f1s.append(f1)
        return {
            "policy": policy,
            "threshold_mode": "per_test_environment_oracle",
            "threshold": np.nan,
            "env_thresholds": thresholds,
            "validation_min_f1": float(np.min(f1s)) if f1s else np.nan,
            "validation_mean_f1": float(np.mean(f1s)) if f1s else np.nan,
        }

    raise ValueError(f"unknown calibration policy: {policy}")


def apply_threshold_plan(frame: pd.DataFrame, plan: dict) -> np.ndarray:
    if plan["threshold_mode"] == "native":
        return frame["native_prediction"].to_numpy(dtype=int)
    if plan["threshold_mode"] == "per_test_environment_oracle":
        env = str(frame["environment_id"].iloc[0])
        threshold = float(plan["env_thresholds"][env])
    else:
        threshold = float(plan["threshold"])
    return (frame["score"].to_numpy(dtype=float) >= threshold).astype(int)


def variant_name(detector_name: str, policy: str) -> str:
    return detector_name if policy == "native_default" else f"{detector_name}__{policy}"


def evaluate_one_split(
    cfg: dict,
    split_name: str,
    fold: str,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    detectors: dict[str, DetectorSpec],
) -> tuple[pd.DataFrame, pd.DataFrame, list[pd.DataFrame], pd.DataFrame]:
    validate_split(train_df, test_df, f"{split_name}/{fold}")
    dataset_type = cfg["dataset"]["type"]
    seed = int(cfg["random_state"])
    y_train = train_df["y"].to_numpy()
    interventions = (
        load_interventions(cfg["interventions"])
        if dataset_type == "uwf"
        else cfg["interventions"]
    )
    cal_opts = calibration_options(cfg)
    if cal_opts["enabled"]:
        fit_df, val_df, validation_info = challenge_validation_split(
            train_df,
            float(cfg.get("validation_size", 0.25)),
            seed,
            cfg,
            split_name,
            fold,
        )
        validate_split(fit_df, val_df, f"{split_name}/{fold}/baseline_calibration")
        grid = threshold_grid(cal_opts["threshold_grid_size"])
    else:
        fit_df, val_df, validation_info = train_df, None, {"strategy": "none"}
        grid = threshold_grid(cal_opts["threshold_grid_size"])

    metric_rows = []
    ranking_rows = []
    prediction_frames = []
    calibration_rows = []

    for detector_name, spec in detectors.items():
        print(f"fit {split_name}/{fold}/{detector_name}")
        fit_start = time.perf_counter()
        if cal_opts["enabled"]:
            model = fit_detector(spec, dataset_type, fit_df, fit_df["y"].to_numpy(), seed)
            val_frames = score_detector_on_envs(spec, model, dataset_type, val_df, interventions)
            if cal_opts["refit_on_full_train"]:
                print(f"refit {split_name}/{fold}/{detector_name}")
                model = fit_detector(spec, dataset_type, train_df, y_train, seed)
            test_frames = score_detector_on_envs(spec, model, dataset_type, test_df, interventions)
        else:
            model = fit_detector(spec, dataset_type, train_df, y_train, seed)
            val_frames = {}
            test_frames = score_detector_on_envs(spec, model, dataset_type, test_df, interventions)
        fit_seconds = time.perf_counter() - fit_start

        variant_prediction_parts: dict[str, list[pd.DataFrame]] = {}
        for policy in cal_opts["policies"]:
            plan = threshold_plan_for_policy(policy, val_frames, test_frames, grid)
            det_variant = variant_name(detector_name, policy)
            calibration_rows.append({
                "dataset": dataset_type,
                "split": split_name,
                "fold": fold,
                "detector": det_variant,
                "base_detector": detector_name,
                "detector_family": spec.family,
                "feature_profile": spec.profile,
                "calibration_policy": policy,
                "threshold_mode": plan["threshold_mode"],
                "threshold": plan["threshold"],
                "env_thresholds": json.dumps(plan.get("env_thresholds", {}), sort_keys=True),
                "validation_strategy": str(validation_info.get("strategy", "")),
                "validation_group_col": str(validation_info.get("group_col", "")),
                "validation_groups": str(validation_info.get("groups", "")),
                "validation_removed_group_rows": int(validation_info.get("removed_group_rows", 0)),
                "validation_min_f1": plan["validation_min_f1"],
                "validation_mean_f1": plan["validation_mean_f1"],
                "fit_rows": int(len(fit_df)),
                "validation_rows": int(len(val_df)) if val_df is not None else 0,
                "train_rows": int(len(train_df)),
            })

            detector_prediction_parts = []
            for env_name, frame in test_frames.items():
                eval_start = time.perf_counter()
                pred = apply_threshold_plan(frame, plan)
                score = frame["score"].to_numpy(dtype=float)
                y_eval = frame["y"].to_numpy()
                eval_seconds = time.perf_counter() - eval_start
                sps = float(frame["semantic_preservation_score"].iloc[0])
                row = {
                    "dataset": dataset_type,
                    "split": split_name,
                    "fold": fold,
                    "detector": det_variant,
                    "base_detector": detector_name,
                    "detector_family": spec.family,
                    "feature_profile": spec.profile,
                    "calibration_policy": policy,
                    "environment_id": env_name,
                    "train_rows": int(len(train_df)),
                    "test_rows": int(len(test_df)),
                    "train_positive_rate": float(train_df["y"].mean()),
                    "test_positive_rate": float(test_df["y"].mean()),
                    "fit_seconds": fit_seconds,
                    "eval_seconds": eval_seconds,
                    "semantic_preservation_score": sps,
                    **classification_metrics(y_eval, pred, score),
                }
                metric_rows.append(row)

                pred_frame = pd.DataFrame({
                    "dataset": dataset_type,
                    "split": split_name,
                    "fold": fold,
                    "detector": det_variant,
                    "base_detector": detector_name,
                    "calibration_policy": policy,
                    "environment_id": env_name,
                    "semantic_label": frame["semantic_label"].astype(str).to_numpy(),
                    "y": y_eval,
                    "prediction": pred,
                    "score": score,
                })
                detector_prediction_parts.append(pred_frame)
                if cfg.get("save_predictions", False):
                    prediction_frames.append(pred_frame)
            variant_prediction_parts[det_variant] = detector_prediction_parts

        for det_variant, detector_prediction_parts in variant_prediction_parts.items():
            detector_all = pd.concat(detector_prediction_parts, ignore_index=True)
            cfs = causal_fragility_score(detector_all)
            els = environment_leakage_score(detector_all)
            score_els = score_environment_leakage_score(detector_all)
            for row in metric_rows:
                if row["detector"] == det_variant:
                    row["causal_fragility_score"] = cfs
                    row["environment_leakage_score"] = els
                    row["score_environment_leakage_score"] = score_els

    metrics_df = pd.DataFrame(metric_rows)
    static = metrics_df[metrics_df["environment_id"] == "observed"].set_index("detector")["f1"].to_dict()
    for env in sorted(metrics_df["environment_id"].astype(str).unique()):
        env_scores = metrics_df[metrics_df["environment_id"] == env].set_index("detector")["f1"].to_dict()
        ranking_rows.append({
            "dataset": dataset_type,
            "split": split_name,
            "fold": fold,
            "environment_id": env,
            "ranking_reversal_score": ranking_reversal_score(static, env_scores),
            "static_top_detector": max(static.items(), key=lambda kv: kv[1])[0],
            "environment_top_detector": max(env_scores.items(), key=lambda kv: kv[1])[0],
        })

    return metrics_df, pd.DataFrame(ranking_rows), prediction_frames, pd.DataFrame(calibration_rows)


def summarize_hidden_drop(metrics_df: pd.DataFrame) -> pd.DataFrame:
    observed = (
        metrics_df[metrics_df["environment_id"] == "observed"]
        [["dataset", "split", "fold", "detector", "detector_family", "feature_profile", "f1"]]
        .rename(columns={"f1": "observed_f1"})
    )
    worst = (
        metrics_df
        .groupby(["dataset", "split", "fold", "detector"], as_index=False)["f1"]
        .min()
        .rename(columns={"f1": "worst_intervention_f1"})
    )
    out = observed.merge(worst, on=["dataset", "split", "fold", "detector"])
    out["worst_hidden_drop"] = out["observed_f1"] - out["worst_intervention_f1"]
    return out.sort_values(["worst_hidden_drop", "observed_f1"], ascending=[False, False])


def summarize_aggregates(metrics_df: pd.DataFrame, hidden_df: pd.DataFrame, ranking_df: pd.DataFrame):
    detector_summary = (
        hidden_df
        .groupby(["detector", "detector_family", "feature_profile"], as_index=False)
        .agg(
            observed_f1_mean=("observed_f1", "mean"),
            worst_intervention_f1_mean=("worst_intervention_f1", "mean"),
            hidden_drop_mean=("worst_hidden_drop", "mean"),
            hidden_drop_max=("worst_hidden_drop", "max"),
        )
        .sort_values(["hidden_drop_mean", "observed_f1_mean"], ascending=[False, False])
    )
    split_summary = (
        hidden_df
        .groupby("split", as_index=False)
        .agg(
            hidden_drop_mean=("worst_hidden_drop", "mean"),
            hidden_drop_max=("worst_hidden_drop", "max"),
            observed_f1_mean=("observed_f1", "mean"),
        )
    )
    rrs_summary = (
        ranking_df
        .groupby("split", as_index=False)
        .agg(rrs_mean=("ranking_reversal_score", "mean"), rrs_max=("ranking_reversal_score", "max"))
    )
    split_summary = split_summary.merge(rrs_summary, on="split", how="left")
    env_summary = (
        metrics_df
        .groupby(["environment_id", "detector"], as_index=False)
        .agg(f1_mean=("f1", "mean"), auc_mean=("auc", "mean"))
    )
    return detector_summary, split_summary, env_summary


def plot_outputs(hidden_df: pd.DataFrame, split_summary: pd.DataFrame, out_dir: Path) -> None:
    sns.set_theme(style="whitegrid")
    top = hidden_df.groupby("detector", as_index=False)["worst_hidden_drop"].mean()
    top = top.sort_values("worst_hidden_drop", ascending=False)
    plt.figure(figsize=(11, 5))
    sns.barplot(data=top, x="detector", y="worst_hidden_drop")
    plt.xticks(rotation=25, ha="right")
    plt.title("Mean hidden drop by local baseline")
    plt.tight_layout()
    plt.savefig(out_dir / "baseline_hidden_drop_by_detector.png", dpi=180)
    plt.close()

    plt.figure(figsize=(8, 4))
    sns.barplot(data=split_summary, x="split", y="hidden_drop_max")
    plt.xticks(rotation=15, ha="right")
    plt.title("Max hidden drop by split")
    plt.tight_layout()
    plt.savefig(out_dir / "baseline_hidden_drop_by_split.png", dpi=180)
    plt.close()


def write_report(out_dir: Path, summary: dict, detector_summary: pd.DataFrame, split_summary: pd.DataFrame) -> None:
    lines = [
        "# Local Baseline Comparison Report",
        "",
        "## Run Summary",
        "",
        f"- Dataset: `{summary['dataset_type']}`",
        f"- Rows: `{summary['rows']}`",
        f"- Splits/folds: `{summary['num_folds']}`",
        f"- Detectors run: `{summary['detectors_run']}`",
        f"- Elapsed seconds: `{summary['elapsed_seconds']:.2f}`",
        f"- Max hidden drop: `{summary['max_hidden_drop']:.6f}`",
        f"- Max RRS: `{summary['max_rrs']:.6f}`",
        "",
        "## Detector Summary",
        "",
        detector_summary.head(20).to_markdown(index=False),
        "",
        "## Split Summary",
        "",
        split_summary.to_markdown(index=False),
        "",
    ]
    (out_dir / "LOCAL_BASELINE_REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    cfg_path = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "configs" / "local_baseline_uwf.yaml"
    with cfg_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    start = time.perf_counter()
    out_dir = ROOT / cfg["output_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg_hash = config_fingerprint(cfg)

    df, profile = load_dataset(cfg)
    write_json(out_dir / "dataset_profile.json", profile)

    catalog = build_detector_catalog(int(cfg["random_state"]))
    requested = cfg.get("detectors") or sorted(catalog)
    detectors = {name: catalog[name] for name in requested if name in catalog}
    skipped = sorted(set(requested) - set(detectors))
    if skipped:
        pd.DataFrame({"detector": skipped, "reason": "optional dependency unavailable"}).to_csv(
            out_dir / "skipped_detectors.csv",
            index=False,
        )
    if not detectors:
        raise SystemExit("No detectors available to run.")

    metric_frames = []
    ranking_frames = []
    prediction_frames = []
    calibration_frames = []
    split_profiles = []

    for split_name, fold, train_df, test_df in make_splits(df, cfg):
        split_profile = {
            "split": split_name,
            "fold": fold,
            "train_rows": int(len(train_df)),
            "test_rows": int(len(test_df)),
            "train_positive_rate": float(train_df["y"].mean()),
            "test_positive_rate": float(test_df["y"].mean()),
        }
        cached = load_split_checkpoint(out_dir, cfg, cfg_hash, split_name, fold)
        if cached is not None:
            metrics, ranking, cached_profile, calibration = cached
            metric_frames.append(metrics)
            ranking_frames.append(ranking)
            if not calibration.empty:
                calibration_frames.append(calibration)
            split_profiles.append(cached_profile)
            continue

        split_profiles.append(split_profile)
        metrics, ranking, predictions, calibration = evaluate_one_split(cfg, split_name, fold, train_df, test_df, detectors)
        metric_frames.append(metrics)
        ranking_frames.append(ranking)
        if not calibration.empty:
            calibration_frames.append(calibration)
        prediction_frames.extend(predictions)
        write_split_checkpoint(out_dir, cfg, cfg_hash, split_name, fold, split_profile, metrics, ranking, calibration)

    metrics_df = pd.concat(metric_frames, ignore_index=True)
    ranking_df = pd.concat(ranking_frames, ignore_index=True)
    calibration_df = pd.concat(calibration_frames, ignore_index=True) if calibration_frames else pd.DataFrame()
    hidden_df = summarize_hidden_drop(metrics_df)
    detector_summary, split_summary, env_summary = summarize_aggregates(metrics_df, hidden_df, ranking_df)

    metrics_df.to_csv(out_dir / "baseline_metrics.csv", index=False)
    ranking_df.to_csv(out_dir / "baseline_ranking_reversal.csv", index=False)
    hidden_df.to_csv(out_dir / "baseline_hidden_drop.csv", index=False)
    detector_summary.to_csv(out_dir / "baseline_detector_summary.csv", index=False)
    split_summary.to_csv(out_dir / "baseline_split_summary.csv", index=False)
    env_summary.to_csv(out_dir / "baseline_environment_summary.csv", index=False)
    if not calibration_df.empty:
        calibration_df.to_csv(out_dir / "baseline_calibration_profile.csv", index=False)
    pd.DataFrame(split_profiles).to_csv(out_dir / "baseline_split_profiles.csv", index=False)
    if cfg.get("save_predictions", False) and prediction_frames:
        pd.concat(prediction_frames, ignore_index=True).to_csv(out_dir / "baseline_predictions.csv", index=False)
    plot_outputs(hidden_df, split_summary, out_dir)

    summary = {
        "config": str(cfg_path),
        "output_dir": str(out_dir),
        "dataset_type": cfg["dataset"]["type"],
        "rows": int(len(df)),
        "num_folds": int(len(split_profiles)),
        "detectors_run": sorted(detectors),
        "detectors_skipped": skipped,
        "calibration": calibration_options(cfg),
        "max_hidden_drop": float(hidden_df["worst_hidden_drop"].max()),
        "max_rrs": float(ranking_df["ranking_reversal_score"].max()),
        "elapsed_seconds": time.perf_counter() - start,
    }
    write_json(out_dir / "baseline_run_summary.json", summary)
    write_report(out_dir, summary, detector_summary, split_summary)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


