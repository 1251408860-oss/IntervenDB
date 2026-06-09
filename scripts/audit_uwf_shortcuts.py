from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import yaml
from sklearn.metrics import mutual_info_score, normalized_mutual_info_score

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from intervendb.data import feature_columns, load_uwf_directory


def discretize(series: pd.Series, bins: int) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        clean = pd.to_numeric(series, errors="coerce").fillna(series.median() if series.notna().any() else 0)
        if clean.nunique() <= bins:
            return clean.astype(str)
        ranked = clean.rank(method="average", pct=True)
        values = np.minimum((ranked.to_numpy() * bins).astype(int), bins - 1).astype(str)
        return pd.Series(values, index=series.index)
    return series.fillna("missing").astype(str)


def cramers_v(x: pd.Series, y: pd.Series) -> float:
    table = pd.crosstab(x, y)
    if table.empty:
        return 0.0
    observed = table.to_numpy(dtype=float)
    row_sum = observed.sum(axis=1, keepdims=True)
    col_sum = observed.sum(axis=0, keepdims=True)
    total = observed.sum()
    expected = row_sum @ col_sum / max(total, 1.0)
    chi2 = np.divide((observed - expected) ** 2, expected, out=np.zeros_like(observed), where=expected > 0).sum()
    n = total
    r, k = observed.shape
    denom = n * max(min(k - 1, r - 1), 1)
    return float(np.sqrt(chi2 / denom)) if denom > 0 else 0.0


def main():
    config_path = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "configs" / "local_uwf_smoke.yaml"
    with config_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    out_dir = ROOT / "results" / "shortcut_audit"
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_uwf_directory(
        raw_dir=ROOT / cfg["dataset"]["raw_dir"],
        max_rows_per_family=int(cfg["dataset"]["max_rows_per_family"]),
        random_state=int(cfg["random_state"]),
    )

    numeric, categorical = feature_columns("full")
    candidates = numeric + categorical + ["src_ip_zeek", "dest_ip_zeek"]
    rows = []
    y = df["y"].astype(str)
    family = df["attack_family"].astype(str)
    technique = df["semantic_label"].astype(str)

    for col in candidates:
        if col not in df.columns:
            continue
        x = discretize(df[col], bins=12)
        rows.append({
            "feature": col,
            "n_unique": int(x.nunique()),
            "mi_label": float(mutual_info_score(x, y)),
            "nmi_label": float(normalized_mutual_info_score(x, y)),
            "mi_attack_family": float(mutual_info_score(x, family)),
            "nmi_attack_family": float(normalized_mutual_info_score(x, family)),
            "mi_technique": float(mutual_info_score(x, technique)),
            "nmi_technique": float(normalized_mutual_info_score(x, technique)),
            "cramers_v_label": cramers_v(x, y),
            "top_values": json.dumps(x.value_counts().head(5).to_dict(), ensure_ascii=False),
        })

    audit = pd.DataFrame(rows).sort_values("nmi_label", ascending=False)
    audit.to_csv(out_dir / "shortcut_audit.csv", index=False)

    top = audit.head(18)
    plt.figure(figsize=(10, 6))
    sns.barplot(data=top, x="nmi_label", y="feature")
    plt.title("Feature-label shortcut audit")
    plt.tight_layout()
    plt.savefig(out_dir / "feature_label_shortcut_audit.png", dpi=180)
    plt.close()

    heat = audit.head(18).set_index("feature")[["nmi_label", "nmi_attack_family", "nmi_technique"]]
    plt.figure(figsize=(8, 7))
    sns.heatmap(heat, annot=True, fmt=".3f", cmap="mako")
    plt.title("Normalized mutual information shortcut heatmap")
    plt.tight_layout()
    plt.savefig(out_dir / "shortcut_nmi_heatmap.png", dpi=180)
    plt.close()

    summary = {
        "rows": int(len(df)),
        "features_audited": int(len(audit)),
        "top_label_shortcuts": audit.head(8)[["feature", "nmi_label", "cramers_v_label"]].to_dict(orient="records"),
    }
    with (out_dir / "shortcut_audit_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()


