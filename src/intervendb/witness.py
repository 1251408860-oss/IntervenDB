from __future__ import annotations

from typing import Iterable, List

import numpy as np
import pandas as pd


def build_score_stratified_witness(
    frame: pd.DataFrame,
    group_cols: Iterable[str],
    score_col: str = "score",
    max_witness_per_group: int = 32,
) -> pd.DataFrame:
    """Select score-stratified representatives with weights.

    The full prediction table can be much larger than the representative table.
    For every cube cell, this function sorts rows by detector score and partitions
    them into equal-sized buckets. The median row of each bucket becomes a witness
    and receives a weight equal to the bucket size.
    """
    group_cols = list(group_cols)
    witnesses: List[pd.DataFrame] = []

    for _, part in frame.groupby(group_cols, dropna=False, sort=False):
        if len(part) <= max_witness_per_group:
            sample = part.copy()
            sample["weight"] = 1.0
            witnesses.append(sample)
            continue

        ordered = part.sort_values(score_col).reset_index(drop=True)
        indices = np.array_split(np.arange(len(ordered)), max_witness_per_group)
        rows = []
        for bucket in indices:
            if len(bucket) == 0:
                continue
            mid = bucket[len(bucket) // 2]
            row = ordered.iloc[[mid]].copy()
            row["weight"] = float(len(bucket))
            rows.append(row)
        witnesses.append(pd.concat(rows, ignore_index=True))

    if not witnesses:
        return frame.head(0).copy()

    out = pd.concat(witnesses, ignore_index=True)
    return out


def build_random_witness(
    frame: pd.DataFrame,
    group_cols: Iterable[str],
    max_witness_per_group: int = 32,
    random_state: int = 0,
) -> pd.DataFrame:
    """Random weighted witness baseline for ablation."""
    group_cols = list(group_cols)
    witnesses: List[pd.DataFrame] = []

    for _, part in frame.groupby(group_cols, dropna=False, sort=False):
        if len(part) <= max_witness_per_group:
            sample = part.copy()
            sample["weight"] = 1.0
            witnesses.append(sample)
            continue

        sample = part.sample(n=max_witness_per_group, replace=False, random_state=random_state).copy()
        sample["weight"] = float(len(part)) / float(max_witness_per_group)
        witnesses.append(sample)

    if not witnesses:
        return frame.head(0).copy()

    return pd.concat(witnesses, ignore_index=True)


def add_observation_columns(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["obs_idx"] = out.groupby(["detector", "environment_id"]).cumcount()
    out["is_correct"] = (out["y"].astype(int) == out["prediction"].astype(int)).astype(int)
    out["tp"] = ((out["y"].astype(int) == 1) & (out["prediction"].astype(int) == 1)).astype(int)
    out["fp"] = ((out["y"].astype(int) == 0) & (out["prediction"].astype(int) == 1)).astype(int)
    out["tn"] = ((out["y"].astype(int) == 0) & (out["prediction"].astype(int) == 0)).astype(int)
    out["fn"] = ((out["y"].astype(int) == 1) & (out["prediction"].astype(int) == 0)).astype(int)
    out["weight"] = 1.0
    return out


