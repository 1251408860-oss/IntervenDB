from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Intervention:
    name: str
    sampling_rate: float = 1.0
    duration_multiplier: float = 1.0
    background_byte_multiplier: float = 1.0
    packet_jitter: float = 0.0
    timestamp_shift: float = 0.0
    seed: int = 0


SEMANTIC_LOCK_COLUMNS = [
    "y",
    "label_binary",
    "label_tactic",
    "label_technique",
    "semantic_label",
    "attack_family",
    "proto",
    "service",
    "dest_port_zeek",
]


def apply_intervention(df: pd.DataFrame, intervention: Intervention) -> Tuple[pd.DataFrame, float]:
    rng = np.random.default_rng(intervention.seed)
    out = df.copy()

    original = out.copy()

    if intervention.sampling_rate < 1.0:
        scale = max(intervention.sampling_rate, 1e-6)
        byte_cols = ["orig_bytes", "resp_bytes", "orig_ip_bytes", "resp_ip_bytes"]
        pkt_cols = ["orig_pkts", "resp_pkts"]
        for col in byte_cols + pkt_cols:
            noise = rng.normal(loc=1.0, scale=0.03, size=len(out))
            out[col] = np.floor(out[col].to_numpy(dtype=float) * scale * noise).clip(min=0)
        dropped_bytes = original["total_ip_bytes"].to_numpy(dtype=float) - (
            out["orig_ip_bytes"].to_numpy(dtype=float) + out["resp_ip_bytes"].to_numpy(dtype=float)
        )
        out["missed_bytes"] = out["missed_bytes"].to_numpy(dtype=float) + dropped_bytes.clip(min=0)

    if intervention.duration_multiplier != 1.0:
        base = out["duration"].replace(0.0, 1e-3).to_numpy(dtype=float)
        jitter = rng.normal(loc=1.0, scale=0.02, size=len(out))
        out["duration"] = (base * intervention.duration_multiplier * jitter).clip(min=0.0)

    if intervention.background_byte_multiplier != 1.0:
        mult = intervention.background_byte_multiplier
        byte_cols = ["orig_ip_bytes", "resp_ip_bytes"]
        for col in byte_cols:
            noise = rng.normal(loc=1.0, scale=0.04, size=len(out))
            out[col] = np.floor(out[col].to_numpy(dtype=float) * mult * noise).clip(min=0)

    if intervention.packet_jitter > 0.0:
        for col in ["orig_pkts", "resp_pkts"]:
            vals = out[col].to_numpy(dtype=float)
            jitter = rng.normal(loc=0.0, scale=intervention.packet_jitter, size=len(out))
            out[col] = np.floor(vals + jitter).clip(min=0)

    if intervention.timestamp_shift != 0.0:
        out["ts"] = out["ts"].to_numpy(dtype=float) + intervention.timestamp_shift
        ts = out["ts"].to_numpy(dtype=float)
        out["ts_norm"] = (ts - ts.min()) / (ts.max() - ts.min() + 1e-9)
        out["collector_window"] = np.floor(out["ts_norm"] * 24).astype(int)
        out["hour_of_day"] = np.floor((ts / 3600.0) % 24).astype(int)

    out["total_bytes"] = out["orig_bytes"] + out["resp_bytes"]
    out["total_ip_bytes"] = out["orig_ip_bytes"] + out["resp_ip_bytes"]
    out["total_pkts"] = out["orig_pkts"] + out["resp_pkts"]
    out["duration_safe"] = out["duration"].replace(0.0, 1e-3)
    out["bytes_per_sec"] = out["total_bytes"] / out["duration_safe"]
    out["pkts_per_sec"] = out["total_pkts"] / out["duration_safe"]
    out["orig_resp_byte_ratio"] = (out["orig_bytes"] + 1.0) / (out["resp_bytes"] + 1.0)
    out["orig_resp_pkt_ratio"] = (out["orig_pkts"] + 1.0) / (out["resp_pkts"] + 1.0)

    sps = semantic_preservation_score(original, out)
    out["environment_id"] = intervention.name
    return out, sps


def semantic_preservation_score(original: pd.DataFrame, counterfactual: pd.DataFrame) -> float:
    scores = []
    for col in SEMANTIC_LOCK_COLUMNS:
        if col in original.columns and col in counterfactual.columns:
            scores.append(float((original[col].astype(str).to_numpy() == counterfactual[col].astype(str).to_numpy()).mean()))
    if not scores:
        return 0.0
    return float(np.mean(scores))


def load_interventions(config: Dict[str, Dict[str, object]]) -> Dict[str, Intervention]:
    interventions = {}
    for name, params in config.items():
        interventions[name] = Intervention(name=name, **params)
    return interventions


