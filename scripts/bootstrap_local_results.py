from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import f1_score


ROOT = Path(__file__).resolve().parents[1]


def ci(values, alpha=0.05):
    return (
        float(np.mean(values)),
        float(np.quantile(values, alpha / 2)),
        float(np.quantile(values, 1 - alpha / 2)),
    )


def main():
    config_path = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "configs" / "local_bootstrap.yaml"
    with config_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    pred_path = ROOT / cfg["input_predictions_csv"]
    out_dir = ROOT / cfg["output_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(pred_path)
    rng = np.random.default_rng(int(cfg["random_state"]))
    n_boot = int(cfg["n_bootstrap"])

    f1_rows = []
    for (detector, env), part in df.groupby(["detector", "environment_id"], sort=False):
        y = part["y"].to_numpy()
        pred = part["prediction"].to_numpy()
        values = []
        for _ in range(n_boot):
            idx = rng.integers(0, len(part), len(part))
            values.append(f1_score(y[idx], pred[idx], zero_division=0))
        mean, low, high = ci(values)
        f1_rows.append({
            "detector": detector,
            "environment_id": env,
            "f1_mean": mean,
            "f1_ci_low": low,
            "f1_ci_high": high,
            "ci_width": high - low,
        })

    f1_ci = pd.DataFrame(f1_rows)
    f1_ci.to_csv(out_dir / "f1_bootstrap_ci.csv", index=False)

    hidden_rows = []
    for detector, part in f1_ci.groupby("detector"):
        observed = part[part["environment_id"] == "observed"]["f1_mean"].iloc[0]
        worst = part["f1_mean"].min()
        hidden_rows.append({
            "detector": detector,
            "observed_f1_boot_mean": observed,
            "worst_f1_boot_mean": worst,
            "hidden_drop_boot_mean": observed - worst,
        })

    hidden = pd.DataFrame(hidden_rows).sort_values("hidden_drop_boot_mean", ascending=False)
    hidden.to_csv(out_dir / "hidden_drop_bootstrap_summary.csv", index=False)

    summary = {
        "input_predictions_csv": str(pred_path),
        "n_bootstrap": n_boot,
        "max_hidden_drop_boot_mean": float(hidden["hidden_drop_boot_mean"].max()),
        "output_dir": str(out_dir),
    }
    with (out_dir / "bootstrap_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()


