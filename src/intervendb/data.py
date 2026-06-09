from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd


NUMERIC_COLUMNS = [
    "duration",
    "src_port_zeek",
    "dest_port_zeek",
    "missed_bytes",
    "orig_bytes",
    "orig_ip_bytes",
    "orig_pkts",
    "resp_bytes",
    "resp_ip_bytes",
    "resp_pkts",
    "ts",
]

CATEGORICAL_COLUMNS = [
    "conn_state",
    "history",
    "proto",
    "service",
]

IDENTIFIER_COLUMNS = [
    "community_id",
    "uid",
    "src_ip_zeek",
    "dest_ip_zeek",
    "datetime",
    "label_cve",
]


@dataclass(frozen=True)
class DatasetPaths:
    benign_csv: Path
    reconnaissance_csv: Path


def load_uwf_smoke_dataset(paths: DatasetPaths, max_rows_per_class: int, random_state: int) -> pd.DataFrame:
    benign = pd.read_csv(paths.benign_csv)
    recon = pd.read_csv(paths.reconnaissance_csv)

    benign = benign.sample(
        n=min(max_rows_per_class, len(benign)),
        random_state=random_state,
        replace=False,
    )
    recon = recon.sample(
        n=min(max_rows_per_class, len(recon)),
        random_state=random_state,
        replace=False,
    )

    df = pd.concat([benign, recon], ignore_index=True)
    df = df.sample(frac=1.0, random_state=random_state).reset_index(drop=True)
    return normalize_zeek_frame(df)


def load_uwf_directory(raw_dir: Path, max_rows_per_family: int, random_state: int) -> pd.DataFrame:
    frames = []
    for path in sorted(raw_dir.glob("*.csv")):
        frame = pd.read_csv(path)
        frame["source_file"] = path.name
        frame = normalize_zeek_frame(frame)
        frames.append(frame)

    if not frames:
        raise FileNotFoundError(f"No CSV files found under {raw_dir}")

    df = pd.concat(frames, ignore_index=True)
    sampled_parts = []
    for family, part in df.groupby("attack_family", dropna=False):
        n = min(max_rows_per_family, len(part))
        sampled_parts.append(part.sample(n=n, random_state=random_state, replace=False))
    out = pd.concat(sampled_parts, ignore_index=True)
    return out.sample(frac=1.0, random_state=random_state).reset_index(drop=True)


def normalize_zeek_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    for col in NUMERIC_COLUMNS:
        if col not in out.columns:
            out[col] = 0.0
        out[col] = pd.to_numeric(out[col], errors="coerce")

    for col in CATEGORICAL_COLUMNS:
        if col not in out.columns:
            out[col] = "missing"
        out[col] = out[col].fillna("missing").astype(str)

    out["duration"] = out["duration"].fillna(0.0).clip(lower=0.0)
    for col in [
        "missed_bytes",
        "orig_bytes",
        "orig_ip_bytes",
        "orig_pkts",
        "resp_bytes",
        "resp_ip_bytes",
        "resp_pkts",
    ]:
        out[col] = out[col].fillna(0.0).clip(lower=0.0)

    out["ts"] = out["ts"].fillna(out["ts"].median())
    label_binary = out["label_binary"].astype(str).str.lower()
    label_tactic = out["label_tactic"].fillna("none").astype(str).str.lower()
    out["label_binary"] = label_binary.isin(["true", "1", "yes", "duplicate"])
    out["y"] = ((label_tactic != "none") | out["label_binary"]).astype(int)
    out["semantic_label"] = np.where(out["y"] == 1, out["label_technique"].fillna("attack"), "benign")
    out["attack_family"] = np.where(out["y"] == 1, out["label_tactic"].fillna("attack"), "benign")

    out["total_bytes"] = out["orig_bytes"] + out["resp_bytes"]
    out["total_ip_bytes"] = out["orig_ip_bytes"] + out["resp_ip_bytes"]
    out["total_pkts"] = out["orig_pkts"] + out["resp_pkts"]
    out["duration_safe"] = out["duration"].replace(0.0, 1e-3)
    out["bytes_per_sec"] = out["total_bytes"] / out["duration_safe"]
    out["pkts_per_sec"] = out["total_pkts"] / out["duration_safe"]
    out["orig_resp_byte_ratio"] = (out["orig_bytes"] + 1.0) / (out["resp_bytes"] + 1.0)
    out["orig_resp_pkt_ratio"] = (out["orig_pkts"] + 1.0) / (out["resp_pkts"] + 1.0)
    out["is_common_dest_port"] = out["dest_port_zeek"].isin([22, 53, 80, 443, 445, 3389]).astype(int)

    ts = out["ts"].to_numpy(dtype=float)
    if len(ts) > 1:
        ts_norm = (ts - np.nanmin(ts)) / (np.nanmax(ts) - np.nanmin(ts) + 1e-9)
    else:
        ts_norm = np.zeros_like(ts)
    out["ts_norm"] = ts_norm
    out["collector_window"] = np.floor(ts_norm * 24).astype(int)
    out["hour_of_day"] = np.floor((out["ts"].to_numpy(dtype=float) / 3600.0) % 24).astype(int)

    return out


def feature_columns(profile: str = "full") -> Tuple[List[str], List[str]]:
    full_numeric = [
        "duration",
        "src_port_zeek",
        "dest_port_zeek",
        "missed_bytes",
        "orig_bytes",
        "orig_ip_bytes",
        "orig_pkts",
        "resp_bytes",
        "resp_ip_bytes",
        "resp_pkts",
        "total_bytes",
        "total_ip_bytes",
        "total_pkts",
        "bytes_per_sec",
        "pkts_per_sec",
        "orig_resp_byte_ratio",
        "orig_resp_pkt_ratio",
        "is_common_dest_port",
        "ts_norm",
        "collector_window",
        "hour_of_day",
    ]
    full_categorical = CATEGORICAL_COLUMNS

    if profile == "full":
        return full_numeric, full_categorical
    if profile == "env_sensitive":
        return [
            "duration",
            "missed_bytes",
            "orig_bytes",
            "orig_ip_bytes",
            "orig_pkts",
            "resp_bytes",
            "resp_ip_bytes",
            "resp_pkts",
            "total_bytes",
            "total_ip_bytes",
            "total_pkts",
            "bytes_per_sec",
            "pkts_per_sec",
            "orig_resp_byte_ratio",
            "orig_resp_pkt_ratio",
            "ts_norm",
            "collector_window",
            "hour_of_day",
        ], []
    if profile == "protocol_core":
        return [
            "src_port_zeek",
            "dest_port_zeek",
            "is_common_dest_port",
        ], ["proto", "service", "conn_state"]

    raise ValueError(f"Unknown feature profile: {profile}")


def dataset_profile(df: pd.DataFrame) -> Dict[str, object]:
    return {
        "rows": int(len(df)),
        "positives": int(df["y"].sum()),
        "negatives": int((1 - df["y"]).sum()),
        "attack_families": sorted(df["attack_family"].astype(str).unique().tolist()),
        "techniques": sorted(df["semantic_label"].astype(str).unique().tolist()),
        "protocols": sorted(df["proto"].astype(str).unique().tolist()),
    }


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, payload: Dict[str, object]) -> None:
    import json

    ensure_parent(path)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def write_csv(path: Path, rows: Iterable[Dict[str, object]]) -> None:
    ensure_parent(path)
    pd.DataFrame(list(rows)).to_csv(path, index=False)


