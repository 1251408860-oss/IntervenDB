from __future__ import annotations

import json
import hashlib
import shutil
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import yaml
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
SCRIPTS = ROOT / "scripts"
for path in [SRC, SCRIPTS]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from intervendb.data import write_json
from intervendb.metrics import (
    causal_fragility_score,
    classification_metrics,
    environment_leakage_score,
    ranking_reversal_score,
    score_environment_leakage_score,
)
from run_local_baseline_comparison import (
    DetectorSpec,
    apply_dataset_intervention,
    build_detector_catalog,
    detector_score,
    fit_detector,
    load_dataset,
    make_splits,
)
from run_uwf_split_experiments import validate_split


CHECKPOINT_VERSION = 1


def extend_candidate_catalog(seed: int) -> dict[str, DetectorSpec]:
    catalog = build_detector_catalog(seed)
    catalog["lr_env_sensitive"] = DetectorSpec(
        "lr_env_sensitive",
        "intervendb_internal_linear",
        "env_sensitive",
        "supervised",
        lambda s: LogisticRegression(max_iter=1000, class_weight="balanced"),
    )
    catalog["rf_env_sensitive"] = DetectorSpec(
        "rf_env_sensitive",
        "intervendb_internal_tree",
        "env_sensitive",
        "supervised",
        lambda s: RandomForestClassifier(
            n_estimators=180,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=s + 7,
            n_jobs=-1,
        ),
    )
    catalog["protocol_core_rf"] = DetectorSpec(
        "protocol_core_rf",
        "intervendb_internal_protocol",
        "protocol_core",
        "supervised",
        lambda s: RandomForestClassifier(
            n_estimators=140,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=s + 13,
            n_jobs=-1,
        ),
    )
    return catalog


def stable_int(seed: int, *parts: object) -> int:
    text = "::".join([str(seed), *[str(part) for part in parts]])
    digest = hashlib.md5(text.encode("utf-8")).hexdigest()
    return (seed + int(digest[:8], 16)) % (2**32 - 1)


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
        "dir": str(raw.get("dir", "_checkpoints")),
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
    keep_predictions: bool,
):
    opts = checkpoint_options(cfg)
    if not opts["enabled"] or not opts["resume"]:
        return None
    ckpt_dir = split_checkpoint_dir(out_dir, cfg, split_name, fold)
    manifest_path = ckpt_dir / "manifest.json"
    if not manifest_path.exists():
        return None
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if manifest.get("status") != "complete":
        return None
    if manifest.get("checkpoint_version") != CHECKPOINT_VERSION:
        return None
    if manifest.get("config_fingerprint") != cfg_hash:
        return None
    metrics_path = ckpt_dir / "metrics.csv"
    candidate_path = ckpt_dir / "candidate_validation.csv"
    split_profile_path = ckpt_dir / "split_profile.json"
    if not (metrics_path.exists() and candidate_path.exists() and split_profile_path.exists()):
        return None
    if keep_predictions and not (ckpt_dir / "predictions.csv").exists():
        return None
    print(f"resume intervendb split {split_name}/{fold} from {ckpt_dir}")
    metrics = pd.read_csv(metrics_path)
    candidate_profile = pd.read_csv(candidate_path)
    split_profile = json.loads(split_profile_path.read_text(encoding="utf-8"))
    predictions = pd.read_csv(ckpt_dir / "predictions.csv") if keep_predictions else pd.DataFrame()
    return metrics, predictions, candidate_profile, split_profile


def write_split_checkpoint(
    out_dir: Path,
    cfg: dict,
    cfg_hash: str,
    split_name: str,
    fold: str,
    metrics: pd.DataFrame,
    predictions: pd.DataFrame,
    candidate_profile: pd.DataFrame,
    split_profile: dict,
    keep_predictions: bool,
) -> None:
    opts = checkpoint_options(cfg)
    if not opts["enabled"]:
        return
    ckpt_dir = split_checkpoint_dir(out_dir, cfg, split_name, fold)
    tmp_dir = ckpt_dir.with_name(ckpt_dir.name + ".tmp")
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(tmp_dir / "metrics.csv", index=False)
    candidate_profile.to_csv(tmp_dir / "candidate_validation.csv", index=False)
    (tmp_dir / "split_profile.json").write_text(json.dumps(split_profile, indent=2), encoding="utf-8")
    if keep_predictions:
        predictions.to_csv(tmp_dir / "predictions.csv", index=False)
    manifest = {
        "status": "complete",
        "checkpoint_version": CHECKPOINT_VERSION,
        "config_fingerprint": cfg_hash,
        "split": split_name,
        "fold": fold,
        "written_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "metric_rows": int(len(metrics)),
        "prediction_rows": int(len(predictions)) if keep_predictions else 0,
        "candidate_rows": int(len(candidate_profile)),
    }
    (tmp_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    if ckpt_dir.exists():
        shutil.rmtree(ckpt_dir)
    tmp_dir.rename(ckpt_dir)


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


def intervention_map(dataset_type: str, cfg: dict):
    if dataset_type == "uwf":
        from intervendb.interventions import load_interventions

        return load_interventions(cfg["interventions"])
    return cfg["interventions"]


def threshold_grid(size: int) -> np.ndarray:
    return np.linspace(0.01, 0.99, max(11, int(size)))


def best_common_threshold(frames: list[pd.DataFrame], grid: np.ndarray) -> tuple[float, float, float]:
    best = (0.5, -1.0, -1.0)
    for threshold in grid:
        f1s = []
        for frame in frames:
            pred = (frame["score"].to_numpy(dtype=float) >= threshold).astype(int)
            f1s.append(f1_score(frame["y"].to_numpy(), pred, zero_division=0))
        min_f1 = float(np.min(f1s))
        mean_f1 = float(np.mean(f1s))
        if (min_f1, mean_f1) > (best[1], best[2]):
            best = (float(threshold), min_f1, mean_f1)
    return best


def best_env_thresholds(frames: list[pd.DataFrame], grid: np.ndarray) -> dict[str, float]:
    thresholds = {}
    for frame in frames:
        env = str(frame["environment_id"].iloc[0])
        best_t, best_f1 = 0.5, -1.0
        for threshold in grid:
            pred = (frame["score"].to_numpy(dtype=float) >= threshold).astype(int)
            value = f1_score(frame["y"].to_numpy(), pred, zero_division=0)
            if value > best_f1:
                best_t, best_f1 = float(threshold), float(value)
        thresholds[env] = best_t
    return thresholds


def apply_thresholds(frames: list[pd.DataFrame], common_threshold: float, env_thresholds: dict[str, float] | None = None):
    out = []
    for frame in frames:
        env = str(frame["environment_id"].iloc[0])
        threshold = env_thresholds.get(env, common_threshold) if env_thresholds else common_threshold
        part = frame.copy()
        part["prediction"] = (part["score"].to_numpy(dtype=float) >= threshold).astype(int)
        out.append(part)
    return out


def min_frame_f1(frames: list[pd.DataFrame]) -> float:
    values = []
    for frame in frames:
        values.append(f1_score(frame["y"].to_numpy(), frame["prediction"].to_numpy(), zero_division=0))
    return float(np.min(values)) if values else 0.0


def mean_frame_f1(frames: list[pd.DataFrame]) -> float:
    values = []
    for frame in frames:
        values.append(f1_score(frame["y"].to_numpy(), frame["prediction"].to_numpy(), zero_division=0))
    return float(np.mean(values)) if values else 0.0


def frame_by_environment(frames: list[pd.DataFrame], environment_id: str) -> pd.DataFrame:
    for frame in frames:
        if str(frame["environment_id"].iloc[0]) == environment_id:
            return frame
    raise ValueError(f"Missing environment frame: {environment_id}")


def prediction_rate(frames: list[pd.DataFrame], environment_id: str = "observed") -> float:
    frame = frame_by_environment(frames, environment_id)
    return float(frame["prediction"].to_numpy(dtype=float).mean())


def coverage_lift_decision(
    primary_frames: list[pd.DataFrame],
    gated_frames: list[pd.DataFrame],
    train_positive_rate: float,
    cfg: dict,
) -> dict:
    selection = cfg.get("selection", {}) or {}
    coverage_rate = prediction_rate(primary_frames, "observed")
    gated_coverage_rate = prediction_rate(gated_frames, "observed")
    coverage_floor = float(selection.get("coverage_floor_fraction", 0.55)) * float(train_positive_rate)
    coverage_gap = max(0.0, coverage_floor - coverage_rate)
    coverage_lift = max(0.0, gated_coverage_rate - coverage_rate)
    min_gap = float(selection.get("coverage_min_gap", 0.0))
    lift_floor = max(
        float(selection.get("coverage_lift_min_absolute", 0.03)),
        float(selection.get("coverage_lift_gap_fraction", 0.30)) * coverage_gap,
    )
    recovery_floor = float(selection.get("coverage_recovery_fraction", 0.95)) * coverage_floor
    switch_to_gated = bool(
        coverage_gap > min_gap
        and coverage_lift >= lift_floor
        and gated_coverage_rate >= recovery_floor
    )
    return {
        "coverage_rate": coverage_rate,
        "coverage_floor": coverage_floor,
        "coverage_gap": coverage_gap,
        "gated_coverage_rate": gated_coverage_rate,
        "coverage_lift": coverage_lift,
        "coverage_lift_floor": lift_floor,
        "coverage_recovery_floor": recovery_floor,
        "switch_to_gated": switch_to_gated,
    }


def method_threshold(profile_df: pd.DataFrame, candidate_name: str, cfg: dict, method_name: str) -> float:
    threshold = float(profile_df[profile_df["candidate"] == candidate_name]["threshold"].iloc[0])
    if method_name != "intervendb_observed_selector":
        cap = cfg.get("selection", {}).get("robust_threshold_cap")
        if cap is not None:
            threshold = min(threshold, float(cap))
    return threshold


def candidate_validation_profile(name: str, frames: list[pd.DataFrame], grid: np.ndarray, cfg: dict) -> dict:
    threshold, min_f1, mean_f1 = best_common_threshold(frames, grid)
    calibrated = apply_thresholds(frames, threshold)
    all_frame = pd.concat(calibrated, ignore_index=True)
    metrics = []
    for frame in calibrated:
        env = str(frame["environment_id"].iloc[0])
        metrics.append({
            "environment_id": env,
            **classification_metrics(frame["y"].to_numpy(), frame["prediction"].to_numpy(), frame["score"].to_numpy()),
        })
    metric_df = pd.DataFrame(metrics)
    cfs = causal_fragility_score(all_frame)
    alert_gap = float((metric_df["alert_budget_inflation"] - 1.0).abs().mean())
    observed_f1 = float(metric_df[metric_df["environment_id"] == "observed"]["f1"].iloc[0])
    utility = (
        min_f1
        + 0.20 * mean_f1
        - float(cfg["selection"].get("stability_cfs_penalty", 0.35)) * cfs
        - float(cfg["selection"].get("stability_alert_penalty", 0.08)) * alert_gap
    )
    return {
        "candidate": name,
        "threshold": threshold,
        "observed_f1": observed_f1,
        "min_f1": min_f1,
        "mean_f1": mean_f1,
        "cfs": cfs,
        "alert_gap": alert_gap,
        "utility": float(utility),
    }


def softmax_weights(profiles: pd.DataFrame, temperature: float) -> dict[str, float]:
    utilities = profiles.set_index("candidate")["utility"].astype(float)
    scaled = (utilities - utilities.max()) * float(temperature)
    raw = np.exp(scaled.to_numpy())
    raw = raw / max(raw.sum(), 1e-12)
    return dict(zip(utilities.index.tolist(), raw.tolist()))


def add_selection_scores(profiles: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    selection = cfg.get("selection", {}) or {}
    out = profiles.copy()
    out["shift_guard_score"] = (
        out["min_f1"].astype(float)
        + float(selection.get("shift_guard_observed_weight", 0.15)) * out["observed_f1"].astype(float)
        + float(selection.get("shift_guard_mean_weight", 0.10)) * out["mean_f1"].astype(float)
        - float(selection.get("shift_guard_alert_penalty", 0.03)) * out["alert_gap"].astype(float)
    )
    observed_best = float(out["observed_f1"].max())
    margin = float(selection.get("pareto_observed_margin", 0.15))
    eligible = out[out["observed_f1"].astype(float) >= observed_best - margin].copy()
    if eligible.empty:
        eligible = out.copy()
    floor = float(selection.get("pareto_min_f1_floor", 0.0))
    floored = eligible[eligible["min_f1"].astype(float) >= floor]
    if not floored.empty:
        eligible = floored
    eligible["pareto_score"] = (
        eligible["min_f1"].astype(float)
        + float(selection.get("pareto_observed_weight", 0.25)) * eligible["observed_f1"].astype(float)
        + float(selection.get("pareto_mean_weight", 0.10)) * eligible["mean_f1"].astype(float)
        - float(selection.get("pareto_cfs_penalty", 0.08)) * eligible["cfs"].astype(float)
        - float(selection.get("pareto_alert_penalty", 0.03)) * eligible["alert_gap"].astype(float)
    )
    out["pareto_score"] = np.nan
    out.loc[eligible.index, "pareto_score"] = eligible["pareto_score"]
    return out


def ensemble_frames(score_frames: dict[str, dict[str, pd.DataFrame]], weights: dict[str, float]) -> list[pd.DataFrame]:
    envs = sorted(next(iter(score_frames.values())).keys())
    out = []
    for env in envs:
        first = next(iter(score_frames.values()))[env]
        score = np.zeros(len(first), dtype=float)
        for candidate, weight in weights.items():
            score += float(weight) * score_frames[candidate][env]["score"].to_numpy(dtype=float)
        frame = first[["dataset", "split", "fold", "environment_id", "semantic_label", "y"]].copy()
        frame["detector"] = "intervendb_counterfactual_ensemble"
        frame["score"] = score
        frame["prediction"] = (score >= 0.5).astype(int)
        out.append(frame)
    return out


def tail_guard_frames(
    primary_frames: list[pd.DataFrame],
    anomaly_frames: list[pd.DataFrame],
    primary_threshold: float,
    anomaly_threshold: float,
) -> list[pd.DataFrame]:
    anomaly_by_env = {str(frame["environment_id"].iloc[0]): frame for frame in anomaly_frames}
    out = []
    for primary in primary_frames:
        env = str(primary["environment_id"].iloc[0])
        anomaly = anomaly_by_env[env]
        primary_score = primary["score"].to_numpy(dtype=float)
        anomaly_score = anomaly["score"].to_numpy(dtype=float)
        part = primary.copy()
        part["detector"] = "intervendb_tail_guard_ensemble"
        part["score"] = np.maximum(primary_score, anomaly_score)
        part["prediction"] = (
            (primary_score >= primary_threshold)
            | (anomaly_score >= anomaly_threshold)
        ).astype(int)
        out.append(part)
    return out


def gated_tail_guard_frames(
    primary_frames: list[pd.DataFrame],
    anomaly_frames: list[pd.DataFrame],
    primary_threshold: float,
    anomaly_threshold: float,
    primary_gate: float,
) -> list[pd.DataFrame]:
    anomaly_by_env = {str(frame["environment_id"].iloc[0]): frame for frame in anomaly_frames}
    out = []
    for primary in primary_frames:
        env = str(primary["environment_id"].iloc[0])
        anomaly = anomaly_by_env[env]
        primary_score = primary["score"].to_numpy(dtype=float)
        anomaly_score = anomaly["score"].to_numpy(dtype=float)
        part = primary.copy()
        part["detector"] = "intervendb_gated_conformal_tail_guard"
        part["score"] = np.maximum(primary_score, anomaly_score)
        part["prediction"] = (
            (primary_score >= primary_threshold)
            | ((anomaly_score >= anomaly_threshold) & (primary_score >= primary_gate))
        ).astype(int)
        out.append(part)
    return out


def raw_anomaly_score(spec, model, x: pd.DataFrame) -> np.ndarray:
    if not spec.kind.startswith("unsupervised"):
        raise ValueError(f"{spec.name} is not an unsupervised anomaly detector")
    raw = model.decision_function(x)
    if spec.kind == "unsupervised_iforest":
        raw = -raw
    return np.asarray(raw, dtype=float)


def empirical_tail_score(reference: np.ndarray, values: np.ndarray) -> np.ndarray:
    ref = np.sort(np.asarray(reference, dtype=float))
    if ref.size == 0:
        return np.zeros_like(values, dtype=float)
    ranks = np.searchsorted(ref, np.asarray(values, dtype=float), side="right")
    return ranks.astype(float) / float(ref.size)


def score_conformal_anomaly_on_envs(
    model,
    spec,
    dataset_type: str,
    split_name: str,
    fold: str,
    df: pd.DataFrame,
    interventions: dict,
    reference_scores: np.ndarray,
):
    frames = {}
    for env_name, intervention in interventions.items():
        eval_df, _ = apply_dataset_intervention(dataset_type, df, env_name, intervention)
        raw = raw_anomaly_score(spec, model, eval_df)
        score = empirical_tail_score(reference_scores, raw)
        frames[env_name] = pd.DataFrame({
            "dataset": dataset_type,
            "split": split_name,
            "fold": fold,
            "environment_id": env_name,
            "semantic_label": eval_df["semantic_label"].astype(str).to_numpy(),
            "y": df["y"].to_numpy(),
            "score": score,
        })
    return frames


def best_tail_guard_threshold(
    primary_frames: list[pd.DataFrame],
    anomaly_frames: list[pd.DataFrame],
    primary_threshold: float,
    grid: np.ndarray,
    cfg: dict,
) -> tuple[float, float, float]:
    max_alert = float(cfg["selection"].get("tail_guard_max_alert_inflation", 1.35))
    best = (float(cfg["selection"].get("tail_guard_conformal_threshold", 0.95)), -1.0, -float("inf"))
    anomaly_by_env = {str(frame["environment_id"].iloc[0]): frame for frame in anomaly_frames}
    for threshold in grid:
        f1s = []
        alerts = []
        for primary in primary_frames:
            env = str(primary["environment_id"].iloc[0])
            anomaly = anomaly_by_env[env]
            pred = (
                (primary["score"].to_numpy(dtype=float) >= primary_threshold)
                | (anomaly["score"].to_numpy(dtype=float) >= threshold)
            ).astype(int)
            y = primary["y"].to_numpy()
            f1s.append(f1_score(y, pred, zero_division=0))
            alerts.append(float(pred.mean() / max(y.mean(), 1e-12)))
        min_f1 = float(np.min(f1s))
        alert_gap = float(max(0.0, np.max(alerts) - max_alert))
        score = min_f1 - 0.25 * alert_gap
        if (score, min_f1) > (best[2], best[1]):
            best = (float(threshold), min_f1, score)
    return best


def score_candidate_on_envs(model, spec, dataset_type: str, split_name: str, fold: str, df: pd.DataFrame, interventions: dict):
    frames = {}
    for env_name, intervention in interventions.items():
        eval_df, _ = apply_dataset_intervention(dataset_type, df, env_name, intervention)
        _, score = detector_score(spec, model, eval_df)
        frames[env_name] = pd.DataFrame({
            "dataset": dataset_type,
            "split": split_name,
            "fold": fold,
            "environment_id": env_name,
            "semantic_label": eval_df["semantic_label"].astype(str).to_numpy(),
            "y": df["y"].to_numpy(),
            "score": score,
        })
    return frames


def evaluate_method_frames(
    method_name: str,
    frames: list[pd.DataFrame],
    fit_seconds: float,
    selection_info: dict,
    keep_predictions: bool,
):
    all_frame = pd.concat(frames, ignore_index=True)
    cfs = causal_fragility_score(all_frame)
    els = environment_leakage_score(all_frame)
    score_els = score_environment_leakage_score(all_frame)
    rows = []
    for frame in frames:
        env = str(frame["environment_id"].iloc[0])
        rows.append({
            "dataset": frame["dataset"].iloc[0],
            "split": frame["split"].iloc[0],
            "fold": frame["fold"].iloc[0],
            "method": method_name,
            "environment_id": env,
            "fit_seconds": fit_seconds,
            "selected_candidate": selection_info.get("selected_candidate", ""),
            "common_threshold": selection_info.get("common_threshold", np.nan),
            "anomaly_threshold": selection_info.get("anomaly_threshold", np.nan),
            "primary_gate": selection_info.get("primary_gate", np.nan),
            "coverage_rate": selection_info.get("coverage_rate", np.nan),
            "coverage_floor": selection_info.get("coverage_floor", np.nan),
            "coverage_gap": selection_info.get("coverage_gap", np.nan),
            "gated_coverage_rate": selection_info.get("gated_coverage_rate", np.nan),
            "coverage_lift": selection_info.get("coverage_lift", np.nan),
            "coverage_lift_floor": selection_info.get("coverage_lift_floor", np.nan),
            "coverage_recovery_floor": selection_info.get("coverage_recovery_floor", np.nan),
            "causal_fragility_score": cfs,
            "environment_leakage_score": els,
            "score_environment_leakage_score": score_els,
            **classification_metrics(frame["y"].to_numpy(), frame["prediction"].to_numpy(), frame["score"].to_numpy()),
        })
    if keep_predictions:
        predictions = all_frame.copy()
        predictions["method"] = method_name
    else:
        predictions = pd.DataFrame()
    return rows, predictions


def summarize_hidden_drop(metrics_df: pd.DataFrame) -> pd.DataFrame:
    observed = (
        metrics_df[metrics_df["environment_id"] == "observed"]
        [["dataset", "split", "fold", "method", "f1"]]
        .rename(columns={"f1": "observed_f1"})
    )
    worst = (
        metrics_df
        .groupby(["dataset", "split", "fold", "method"], as_index=False)["f1"]
        .min()
        .rename(columns={"f1": "worst_intervention_f1"})
    )
    out = observed.merge(worst, on=["dataset", "split", "fold", "method"])
    out["worst_hidden_drop"] = out["observed_f1"] - out["worst_intervention_f1"]
    return out.sort_values(["worst_hidden_drop", "observed_f1"], ascending=[False, False])


def summarize_ranking(metrics_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (dataset, split, fold), part in metrics_df.groupby(["dataset", "split", "fold"]):
        static = part[part["environment_id"] == "observed"].set_index("method")["f1"].to_dict()
        for env in sorted(part["environment_id"].astype(str).unique()):
            env_scores = part[part["environment_id"] == env].set_index("method")["f1"].to_dict()
            rows.append({
                "dataset": dataset,
                "split": split,
                "fold": fold,
                "environment_id": env,
                "ranking_reversal_score": ranking_reversal_score(static, env_scores),
                "static_top_method": max(static.items(), key=lambda kv: kv[1])[0],
                "environment_top_method": max(env_scores.items(), key=lambda kv: kv[1])[0],
            })
    return pd.DataFrame(rows)


def plot_outputs(hidden_df: pd.DataFrame, split_summary: pd.DataFrame, out_dir: Path) -> None:
    sns.set_theme(style="whitegrid")
    method_drop = hidden_df.groupby("method", as_index=False)["worst_hidden_drop"].mean()
    method_drop = method_drop.sort_values("worst_hidden_drop", ascending=False)
    plt.figure(figsize=(10, 4))
    sns.barplot(data=method_drop, x="method", y="worst_hidden_drop")
    plt.xticks(rotation=20, ha="right")
    plt.title("IntervenDB method hidden drop")
    plt.tight_layout()
    plt.savefig(out_dir / "intervendb_hidden_drop_by_method.png", dpi=180)
    plt.close()

    plt.figure(figsize=(8, 4))
    sns.barplot(data=split_summary, x="split", y="hidden_drop_max")
    plt.xticks(rotation=15, ha="right")
    plt.title("IntervenDB max hidden drop by split")
    plt.tight_layout()
    plt.savefig(out_dir / "intervendb_hidden_drop_by_split.png", dpi=180)
    plt.close()


def evaluate_split(cfg: dict, split_name: str, fold: str, train_df: pd.DataFrame, test_df: pd.DataFrame):
    dataset_type = cfg["dataset"]["type"]
    seed = int(cfg["random_state"])
    validate_split(train_df, test_df, f"{split_name}/{fold}")
    fit_df, val_df, validation_info = challenge_validation_split(
        train_df,
        float(cfg.get("validation_size", 0.25)),
        seed,
        cfg,
        split_name,
        fold,
    )
    validate_split(fit_df, val_df, f"{split_name}/{fold}/validation")
    interventions = intervention_map(dataset_type, cfg)
    grid = threshold_grid(int(cfg["selection"].get("threshold_grid_size", 101)))

    catalog = extend_candidate_catalog(seed)
    requested = cfg["candidate_detectors"]
    candidates = {name: catalog[name] for name in requested if name in catalog}
    if not candidates:
        raise ValueError("No IntervenDB candidate detectors are available.")

    candidate_profiles = []
    val_scores = {}
    test_scores = {}
    val_conformal_scores = {}
    test_conformal_scores = {}
    fit_start = time.perf_counter()
    refit_on_full_train = bool(cfg["selection"].get("refit_on_full_train", True))
    anomaly_name = str(cfg["selection"].get("tail_guard_candidate", "iforest_env_sensitive"))
    for name, spec in candidates.items():
        print(f"fit intervendb candidate {split_name}/{fold}/{name}")
        model = fit_detector(spec, dataset_type, fit_df, fit_df["y"].to_numpy(), seed)
        val_scores[name] = score_candidate_on_envs(model, spec, dataset_type, split_name, fold, val_df, interventions)
        if name == anomaly_name and spec.kind.startswith("unsupervised"):
            ref_scores = raw_anomaly_score(spec, model, fit_df[fit_df["y"] == 0])
            val_conformal_scores[name] = score_conformal_anomaly_on_envs(
                model,
                spec,
                dataset_type,
                split_name,
                fold,
                val_df,
                interventions,
                ref_scores,
            )
            if not refit_on_full_train:
                test_conformal_scores[name] = score_conformal_anomaly_on_envs(
                    model,
                    spec,
                    dataset_type,
                    split_name,
                    fold,
                    test_df,
                    interventions,
                    ref_scores,
                )
        if not refit_on_full_train:
            test_scores[name] = score_candidate_on_envs(model, spec, dataset_type, split_name, fold, test_df, interventions)
        candidate_profiles.append(
            candidate_validation_profile(name, list(val_scores[name].values()), grid, cfg)
        )

    if refit_on_full_train:
        for name, spec in candidates.items():
            print(f"refit intervendb candidate {split_name}/{fold}/{name}")
            model = fit_detector(spec, dataset_type, train_df, train_df["y"].to_numpy(), seed)
            test_scores[name] = score_candidate_on_envs(model, spec, dataset_type, split_name, fold, test_df, interventions)
            if name == anomaly_name and spec.kind.startswith("unsupervised"):
                ref_scores = raw_anomaly_score(spec, model, train_df[train_df["y"] == 0])
                test_conformal_scores[name] = score_conformal_anomaly_on_envs(
                    model,
                    spec,
                    dataset_type,
                    split_name,
                    fold,
                    test_df,
                    interventions,
                    ref_scores,
                )

    fit_seconds = time.perf_counter() - fit_start
    profile_df = add_selection_scores(pd.DataFrame(candidate_profiles), cfg)

    observed_name = profile_df.sort_values(["observed_f1", "mean_f1"], ascending=False).iloc[0]["candidate"]
    minimax_name = profile_df.sort_values(["min_f1", "mean_f1"], ascending=False).iloc[0]["candidate"]
    stability_name = profile_df.sort_values(["utility", "min_f1"], ascending=False).iloc[0]["candidate"]
    shift_guard_name = profile_df.sort_values(["shift_guard_score", "min_f1"], ascending=False).iloc[0]["candidate"]
    pareto_name = profile_df.sort_values(["pareto_score", "min_f1"], ascending=False, na_position="last").iloc[0]["candidate"]

    method_frames = {}
    for method_name, candidate_name in [
        ("intervendb_observed_selector", observed_name),
        ("intervendb_minimax_selector", minimax_name),
        ("intervendb_stability_selector", stability_name),
        ("intervendb_shift_guard_selector", shift_guard_name),
        ("intervendb_pareto_selector", pareto_name),
    ]:
        threshold = method_threshold(profile_df, candidate_name, cfg, method_name)
        method_frames[method_name] = (
            apply_thresholds(list(test_scores[candidate_name].values()), threshold),
            {"selected_candidate": candidate_name, "common_threshold": threshold},
        )

    if anomaly_name in test_scores:
        primary_threshold = method_threshold(profile_df, stability_name, cfg, "intervendb_tail_guard_ensemble")
        anomaly_threshold = float(cfg["selection"].get("tail_guard_anomaly_threshold", 0.5))
        method_frames["intervendb_tail_guard_ensemble"] = (
            tail_guard_frames(
                list(test_scores[stability_name].values()),
                list(test_scores[anomaly_name].values()),
                primary_threshold,
                anomaly_threshold,
            ),
            {
                "selected_candidate": f"{stability_name}+{anomaly_name}",
                "common_threshold": primary_threshold,
                "anomaly_threshold": anomaly_threshold,
            },
        )
        if anomaly_name in test_conformal_scores:
            conformal_grid = np.linspace(
                float(cfg["selection"].get("tail_guard_conformal_grid_min", 0.50)),
                float(cfg["selection"].get("tail_guard_conformal_grid_max", 0.995)),
                int(cfg["selection"].get("tail_guard_conformal_grid_size", 80)),
            )
            conformal_threshold, _, _ = best_tail_guard_threshold(
                list(val_scores[stability_name].values()),
                list(val_conformal_scores[anomaly_name].values()),
                primary_threshold,
                conformal_grid,
                cfg,
            )
            method_frames["intervendb_conformal_tail_guard"] = (
                tail_guard_frames(
                    list(test_scores[stability_name].values()),
                    list(test_conformal_scores[anomaly_name].values()),
                    primary_threshold,
                    conformal_threshold,
                ),
                {
                    "selected_candidate": f"{stability_name}+conformal_{anomaly_name}",
                    "common_threshold": primary_threshold,
                    "anomaly_threshold": conformal_threshold,
                },
            )
            gated_threshold = float(cfg["selection"].get("tail_guard_conformal_threshold", 0.50))
            primary_gate = float(cfg["selection"].get("tail_guard_primary_gate", 0.10))
            method_frames["intervendb_gated_conformal_tail_guard"] = (
                gated_tail_guard_frames(
                    list(test_scores[stability_name].values()),
                    list(test_conformal_scores[anomaly_name].values()),
                    primary_threshold,
                    gated_threshold,
                    primary_gate,
                ),
                {
                    "selected_candidate": f"{stability_name}+gated_conformal_{anomaly_name}",
                    "common_threshold": primary_threshold,
                    "anomaly_threshold": gated_threshold,
                    "primary_gate": primary_gate,
                },
            )
            stability_val_frames = apply_thresholds(list(val_scores[stability_name].values()), primary_threshold)
            gated_val_frames = gated_tail_guard_frames(
                list(val_scores[stability_name].values()),
                list(val_conformal_scores[anomaly_name].values()),
                primary_threshold,
                gated_threshold,
                primary_gate,
            )
            stability_val_score = (min_frame_f1(stability_val_frames), mean_frame_f1(stability_val_frames))
            gated_val_score = (min_frame_f1(gated_val_frames), mean_frame_f1(gated_val_frames))
            if gated_val_score >= stability_val_score:
                selector_frames = method_frames["intervendb_gated_conformal_tail_guard"][0]
                selector_info = {
                    "selected_candidate": f"openworld:gated_conformal_{anomaly_name}",
                    "common_threshold": primary_threshold,
                    "anomaly_threshold": gated_threshold,
                    "primary_gate": primary_gate,
                }
            else:
                selector_frames = method_frames["intervendb_stability_selector"][0]
                selector_info = {
                    "selected_candidate": f"openworld:{stability_name}",
                    "common_threshold": primary_threshold,
                }
            method_frames["intervendb_openworld_selector"] = (selector_frames, selector_info)
            observed_stability = next(
                frame for frame in method_frames["intervendb_stability_selector"][0]
                if str(frame["environment_id"].iloc[0]) == "observed"
            )
            coverage_rate = float(observed_stability["prediction"].mean())
            coverage_floor = float(cfg["selection"].get("coverage_floor_fraction", 0.55)) * float(train_df["y"].mean())
            if coverage_rate < coverage_floor:
                coverage_frames = method_frames["intervendb_gated_conformal_tail_guard"][0]
                coverage_info = {
                    "selected_candidate": f"coverage:gated_conformal_{anomaly_name}",
                    "common_threshold": primary_threshold,
                    "anomaly_threshold": gated_threshold,
                    "primary_gate": primary_gate,
                    "coverage_rate": coverage_rate,
                    "coverage_floor": coverage_floor,
                }
            else:
                coverage_frames = method_frames["intervendb_stability_selector"][0]
                coverage_info = {
                    "selected_candidate": f"coverage:{stability_name}",
                    "common_threshold": primary_threshold,
                    "coverage_rate": coverage_rate,
                    "coverage_floor": coverage_floor,
                }
            method_frames["intervendb_coverage_openworld_selector"] = (coverage_frames, coverage_info)

            lift_decision = coverage_lift_decision(
                method_frames["intervendb_stability_selector"][0],
                method_frames["intervendb_gated_conformal_tail_guard"][0],
                float(train_df["y"].mean()),
                cfg,
            )
            if lift_decision["switch_to_gated"]:
                coverage_lift_frames = method_frames["intervendb_gated_conformal_tail_guard"][0]
                coverage_lift_info = {
                    "selected_candidate": f"coverage_lift:gated_conformal_{anomaly_name}",
                    "common_threshold": primary_threshold,
                    "anomaly_threshold": gated_threshold,
                    "primary_gate": primary_gate,
                    **lift_decision,
                }
            else:
                coverage_lift_frames = method_frames["intervendb_stability_selector"][0]
                coverage_lift_info = {
                    "selected_candidate": f"coverage_lift:{stability_name}",
                    "common_threshold": primary_threshold,
                    **lift_decision,
                }
            coverage_lift_info.pop("switch_to_gated", None)
            method_frames["intervendb_coverage_lift_openworld_selector"] = (
                coverage_lift_frames,
                coverage_lift_info,
            )

            adaptive_primary_threshold = method_threshold(
                profile_df,
                shift_guard_name,
                cfg,
                "intervendb_adaptive_openworld_selector",
            )
            adaptive_gated_frames = gated_tail_guard_frames(
                list(test_scores[shift_guard_name].values()),
                list(test_conformal_scores[anomaly_name].values()),
                adaptive_primary_threshold,
                gated_threshold,
                primary_gate,
            )
            adaptive_decision = coverage_lift_decision(
                method_frames["intervendb_shift_guard_selector"][0],
                adaptive_gated_frames,
                float(train_df["y"].mean()),
                cfg,
            )
            if adaptive_decision["switch_to_gated"]:
                adaptive_frames = adaptive_gated_frames
                adaptive_info = {
                    "selected_candidate": f"adaptive:gated_conformal_{anomaly_name}",
                    "common_threshold": adaptive_primary_threshold,
                    "anomaly_threshold": gated_threshold,
                    "primary_gate": primary_gate,
                    **adaptive_decision,
                }
            else:
                adaptive_frames = method_frames["intervendb_shift_guard_selector"][0]
                adaptive_info = {
                    "selected_candidate": f"adaptive:{shift_guard_name}",
                    "common_threshold": adaptive_primary_threshold,
                    **adaptive_decision,
                }
            adaptive_info.pop("switch_to_gated", None)
            method_frames["intervendb_adaptive_openworld_selector"] = (
                adaptive_frames,
                adaptive_info,
            )

    excluded = set(cfg["selection"].get("ensemble_exclude_candidates", []) or [])
    ensemble_profile_df = profile_df[~profile_df["candidate"].isin(excluded)].copy()
    if ensemble_profile_df.empty:
        ensemble_profile_df = profile_df
    weights = softmax_weights(ensemble_profile_df, float(cfg["selection"].get("ensemble_temperature", 10.0)))
    val_ensemble = ensemble_frames(val_scores, weights)
    test_ensemble = ensemble_frames(test_scores, weights)
    ensemble_threshold, _, _ = best_common_threshold(val_ensemble, grid)
    env_thresholds = best_env_thresholds(val_ensemble, grid)
    method_frames["intervendb_counterfactual_ensemble"] = (
        apply_thresholds(test_ensemble, ensemble_threshold),
        {"selected_candidate": "weighted_pool", "common_threshold": ensemble_threshold},
    )
    method_frames["intervendb_env_calibrated_ensemble"] = (
        apply_thresholds(test_ensemble, ensemble_threshold, env_thresholds),
        {"selected_candidate": "weighted_pool_env_calibrated", "common_threshold": ensemble_threshold},
    )

    keep_predictions = bool(cfg.get("save_predictions", True))
    metric_rows = []
    prediction_frames = []
    for method_name, (frames, info) in method_frames.items():
        rows, predictions = evaluate_method_frames(method_name, frames, fit_seconds, info, keep_predictions)
        metric_rows.extend(rows)
        if keep_predictions:
            prediction_frames.append(predictions)

    profile_df["split"] = split_name
    profile_df["fold"] = fold
    for candidate, weight in weights.items():
        profile_df.loc[profile_df["candidate"] == candidate, "ensemble_weight"] = weight

    split_profile = {
        "split": split_name,
        "fold": fold,
        "fit_rows": int(len(fit_df)),
        "validation_rows": int(len(val_df)),
        "test_rows": int(len(test_df)),
        "fit_positive_rate": float(fit_df["y"].mean()),
        "validation_positive_rate": float(val_df["y"].mean()),
        "test_positive_rate": float(test_df["y"].mean()),
        "validation_strategy": str(validation_info.get("strategy", "")),
        "validation_group_col": str(validation_info.get("group_col", "")),
        "validation_groups": str(validation_info.get("groups", "")),
        "validation_removed_group_rows": int(validation_info.get("removed_group_rows", 0)),
        "observed_selector_candidate": str(observed_name),
        "minimax_selector_candidate": str(minimax_name),
        "stability_selector_candidate": str(stability_name),
        "shift_guard_selector_candidate": str(shift_guard_name),
        "pareto_selector_candidate": str(pareto_name),
    }
    predictions_df = pd.concat(prediction_frames, ignore_index=True) if keep_predictions else pd.DataFrame()
    return pd.DataFrame(metric_rows), predictions_df, profile_df, split_profile


def write_report(out_dir: Path, summary: dict, method_summary: pd.DataFrame, split_summary: pd.DataFrame) -> None:
    lines = [
        "# IntervenDB Model Report",
        "",
        "## Summary",
        "",
        f"- Dataset: `{summary['dataset_type']}`",
        f"- Rows: `{summary['rows']}`",
        f"- Folds: `{summary['num_folds']}`",
        f"- Max hidden drop: `{summary['max_hidden_drop']:.6f}`",
        f"- Max RRS: `{summary['max_rrs']:.6f}`",
        f"- Best mean worst-intervention F1: `{summary['best_mean_worst_intervention_f1']:.6f}`",
        "",
        "## Method Summary",
        "",
        method_summary.to_markdown(index=False),
        "",
        "## Split Summary",
        "",
        split_summary.to_markdown(index=False),
        "",
    ]
    (out_dir / "INTERVENDB_MODEL_REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    cfg_path = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "configs" / "intervendb_model_uwf.yaml"
    with cfg_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    start = time.perf_counter()
    out_dir = ROOT / cfg["output_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg_hash = config_fingerprint(cfg)
    keep_predictions = bool(cfg.get("save_predictions", True))

    df, profile = load_dataset(cfg)
    write_json(out_dir / "dataset_profile.json", profile)

    metric_frames, prediction_frames, profile_frames, split_profiles = [], [], [], []
    for split_name, fold, train_df, test_df in make_splits(df, cfg):
        checkpoint = load_split_checkpoint(out_dir, cfg, cfg_hash, split_name, fold, keep_predictions)
        if checkpoint is None:
            metrics, predictions, candidate_profile, split_profile = evaluate_split(
                cfg,
                split_name,
                fold,
                train_df,
                test_df,
            )
            write_split_checkpoint(
                out_dir,
                cfg,
                cfg_hash,
                split_name,
                fold,
                metrics,
                predictions,
                candidate_profile,
                split_profile,
                keep_predictions,
            )
        else:
            metrics, predictions, candidate_profile, split_profile = checkpoint
        metric_frames.append(metrics)
        if keep_predictions:
            prediction_frames.append(predictions)
        profile_frames.append(candidate_profile)
        split_profiles.append(split_profile)

    metrics_df = pd.concat(metric_frames, ignore_index=True)
    predictions_df = pd.concat(prediction_frames, ignore_index=True) if keep_predictions else pd.DataFrame()
    candidate_df = pd.concat(profile_frames, ignore_index=True)
    split_profile_df = pd.DataFrame(split_profiles)
    hidden_df = summarize_hidden_drop(metrics_df)
    ranking_df = summarize_ranking(metrics_df)
    method_summary = (
        hidden_df
        .groupby("method", as_index=False)
        .agg(
            observed_f1_mean=("observed_f1", "mean"),
            worst_intervention_f1_mean=("worst_intervention_f1", "mean"),
            hidden_drop_mean=("worst_hidden_drop", "mean"),
            hidden_drop_max=("worst_hidden_drop", "max"),
        )
        .sort_values(["worst_intervention_f1_mean", "hidden_drop_mean"], ascending=[False, True])
    )
    split_summary = (
        hidden_df
        .groupby("split", as_index=False)
        .agg(
            observed_f1_mean=("observed_f1", "mean"),
            worst_intervention_f1_mean=("worst_intervention_f1", "mean"),
            hidden_drop_mean=("worst_hidden_drop", "mean"),
            hidden_drop_max=("worst_hidden_drop", "max"),
        )
    )
    rrs_summary = ranking_df.groupby("split", as_index=False).agg(
        rrs_mean=("ranking_reversal_score", "mean"),
        rrs_max=("ranking_reversal_score", "max"),
    )
    split_summary = split_summary.merge(rrs_summary, on="split", how="left")

    metrics_df.to_csv(out_dir / "intervendb_metrics.csv", index=False)
    if keep_predictions:
        predictions_df.to_csv(out_dir / "intervendb_predictions.csv", index=False)
    hidden_df.to_csv(out_dir / "intervendb_hidden_drop.csv", index=False)
    ranking_df.to_csv(out_dir / "intervendb_ranking_reversal.csv", index=False)
    method_summary.to_csv(out_dir / "intervendb_method_summary.csv", index=False)
    split_summary.to_csv(out_dir / "intervendb_split_summary.csv", index=False)
    candidate_df.to_csv(out_dir / "intervendb_candidate_validation.csv", index=False)
    split_profile_df.to_csv(out_dir / "intervendb_split_profiles.csv", index=False)
    plot_outputs(hidden_df, split_summary, out_dir)

    summary = {
        "config": str(cfg_path),
        "output_dir": str(out_dir),
        "dataset_type": cfg["dataset"]["type"],
        "rows": int(len(df)),
        "num_folds": int(len(split_profile_df)),
        "methods": sorted(metrics_df["method"].unique().tolist()),
        "max_hidden_drop": float(hidden_df["worst_hidden_drop"].max()),
        "max_rrs": float(ranking_df["ranking_reversal_score"].max()),
        "best_mean_worst_intervention_f1": float(method_summary["worst_intervention_f1_mean"].max()),
        "elapsed_seconds": time.perf_counter() - start,
    }
    write_json(out_dir / "intervendb_run_summary.json", summary)
    write_report(out_dir, summary, method_summary, split_summary)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


