from __future__ import annotations

from typing import Dict, Iterable, List

import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon
from scipy.stats import kendalltau
from sklearn.metrics import accuracy_score, f1_score, mutual_info_score, precision_score, recall_score, roc_auc_score


def classification_metrics(y_true, y_pred, y_score) -> Dict[str, float]:
    out = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }
    if len(np.unique(y_true)) == 2 and len(np.unique(y_score)) > 1:
        out["auc"] = float(roc_auc_score(y_true, y_score))
    else:
        out["auc"] = float("nan")
    true_attacks = max(int(np.sum(y_true)), 1)
    out["alert_budget_inflation"] = float(np.sum(y_pred) / true_attacks)
    return out


def _hist(values, bins):
    hist, _ = np.histogram(values, bins=bins)
    hist = hist.astype(float) + 1e-12
    return hist / hist.sum()


def causal_fragility_score(frame: pd.DataFrame, score_col: str = "score", env_col: str = "environment_id", semantic_col: str = "semantic_label") -> float:
    values = []
    for semantic in sorted(frame[semantic_col].astype(str).unique()):
        part = frame[frame[semantic_col].astype(str) == semantic]
        envs = sorted(part[env_col].astype(str).unique())
        if len(envs) < 2:
            continue
        all_scores = part[score_col].to_numpy(dtype=float)
        if np.max(all_scores) - np.min(all_scores) < 1e-12:
            values.append(0.0)
            continue
        bins = np.linspace(np.min(all_scores), np.max(all_scores) + 1e-12, 21)
        hists = {
            env: _hist(part[part[env_col].astype(str) == env][score_col].to_numpy(dtype=float), bins)
            for env in envs
        }
        max_div = 0.0
        for i in range(len(envs)):
            for j in range(i + 1, len(envs)):
                max_div = max(max_div, float(jensenshannon(hists[envs[i]], hists[envs[j]])))
        values.append(max_div)
    return float(np.mean(values)) if values else 0.0


def environment_leakage_score(frame: pd.DataFrame, pred_col: str = "prediction", env_col: str = "environment_id", semantic_col: str = "semantic_label") -> float:
    total = 0.0
    n = len(frame)
    if n == 0:
        return 0.0
    for semantic in sorted(frame[semantic_col].astype(str).unique()):
        part = frame[frame[semantic_col].astype(str) == semantic]
        if len(part) <= 1:
            continue
        total += (len(part) / n) * mutual_info_score(part[pred_col].astype(str), part[env_col].astype(str))
    return float(total)


def score_environment_leakage_score(
    frame: pd.DataFrame,
    score_col: str = "score",
    env_col: str = "environment_id",
    semantic_col: str = "semantic_label",
    bins: int = 10,
) -> float:
    """Estimate I(discretized_score; E | S).

    Label-level ELS can be zero when the final predicted label is stable but the
    detector confidence shifts under an intervention. This score-level variant
    captures that hidden environment dependence.
    """
    total = 0.0
    n = len(frame)
    if n == 0:
        return 0.0

    for semantic in sorted(frame[semantic_col].astype(str).unique()):
        part = frame[frame[semantic_col].astype(str) == semantic].copy()
        if len(part) <= 2 or part[score_col].nunique() <= 1:
            continue
        ranks = part[score_col].rank(method="average", pct=True)
        score_bins = np.minimum((ranks.to_numpy() * bins).astype(int), bins - 1)
        total += (len(part) / n) * mutual_info_score(score_bins, part[env_col].astype(str))
    return float(total)


def ranking_reversal_score(static_by_detector: Dict[str, float], intervention_by_detector: Dict[str, float]) -> float:
    detectors = sorted(set(static_by_detector) & set(intervention_by_detector))
    if len(detectors) < 2:
        return 0.0
    static_values = [static_by_detector[d] for d in detectors]
    intervention_values = [intervention_by_detector[d] for d in detectors]
    tau, _ = kendalltau(static_values, intervention_values)
    if tau is None or np.isnan(tau):
        return 1.0
    return float(1.0 - tau)


def top_detector(scores: Dict[str, float]) -> str:
    return max(scores.items(), key=lambda kv: kv[1])[0]


def summarize_ranking(rows: Iterable[Dict[str, object]]) -> List[Dict[str, object]]:
    df = pd.DataFrame(list(rows))
    static = (
        df[df["environment_id"] == "observed"]
        .set_index("detector")["f1"]
        .to_dict()
    )
    output = []
    for env in sorted(df["environment_id"].astype(str).unique()):
        env_scores = df[df["environment_id"] == env].set_index("detector")["f1"].to_dict()
        output.append({
            "environment_id": env,
            "ranking_reversal_score": ranking_reversal_score(static, env_scores),
            "static_top_detector": top_detector(static),
            "environment_top_detector": top_detector(env_scores),
        })
    return output


