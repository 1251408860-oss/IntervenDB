from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import yaml


ROOT = Path(__file__).resolve().parents[1]


def main():
    base_config_path = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "configs" / "local_risk_cube.yaml"
    values = [int(x) for x in sys.argv[2:]] if len(sys.argv) > 2 else [8, 16, 24, 32, 48]

    with base_config_path.open("r", encoding="utf-8") as f:
        base_cfg = yaml.safe_load(f)

    out_root = ROOT / "results" / "witness_sweep"
    out_root.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    latency_frames = []

    for k in values:
        cfg = dict(base_cfg)
        cfg["output_dir"] = f"results/witness_sweep/k_{k}"
        cfg["witness"] = dict(base_cfg["witness"])
        cfg["witness"]["max_per_cell"] = k

        cfg_path = out_root / f"config_k_{k}.yaml"
        with cfg_path.open("w", encoding="utf-8") as f:
            yaml.safe_dump(cfg, f, sort_keys=False)

        subprocess.run(
            [sys.executable, str(ROOT / "scripts" / "build_risk_cube_and_witness.py"), str(cfg_path)],
            cwd=str(ROOT),
            check=True,
        )

        run_dir = ROOT / cfg["output_dir"]
        with (run_dir / "risk_cube_summary.json").open("r", encoding="utf-8") as f:
            summary = json.load(f)
        summary["max_per_cell"] = k
        summary_rows.append(summary)

        latency = pd.read_csv(run_dir / "query_latency.csv")
        latency["max_per_cell"] = k
        latency_frames.append(latency)

    summary_df = pd.DataFrame(summary_rows).sort_values("max_per_cell")
    latency_df = pd.concat(latency_frames, ignore_index=True)

    summary_df.to_csv(out_root / "witness_sweep_summary.csv", index=False)
    latency_df.to_csv(out_root / "witness_sweep_latency.csv", index=False)

    sns.set_theme(style="whitegrid")

    plt.figure(figsize=(8, 4))
    sns.lineplot(data=summary_df, x="max_per_cell", y="compression_ratio", marker="o")
    plt.title("Witness compression ratio")
    plt.tight_layout()
    plt.savefig(out_root / "compression_ratio_curve.png", dpi=180)
    plt.close()

    plt.figure(figsize=(8, 4))
    sns.lineplot(data=summary_df, x="max_per_cell", y="mean_abs_f1_error", marker="o", label="mean abs F1 error")
    sns.lineplot(data=summary_df, x="max_per_cell", y="max_abs_f1_error", marker="o", label="max abs F1 error")
    plt.title("Witness approximation error")
    plt.tight_layout()
    plt.savefig(out_root / "approximation_error_curve.png", dpi=180)
    plt.close()

    witness_latency = latency_df[latency_df["table"] == "witness"]
    plt.figure(figsize=(8, 4))
    sns.lineplot(data=witness_latency, x="max_per_cell", y="mean_ms", hue="query", marker="o")
    plt.title("Witness query latency")
    plt.tight_layout()
    plt.savefig(out_root / "witness_latency_curve.png", dpi=180)
    plt.close()

    print(json.dumps({
        "output_dir": str(out_root),
        "values": values,
        "best_mean_abs_f1_error": float(summary_df["mean_abs_f1_error"].min()),
        "best_compression_ratio": float(summary_df["compression_ratio"].max()),
    }, indent=2))


if __name__ == "__main__":
    main()


