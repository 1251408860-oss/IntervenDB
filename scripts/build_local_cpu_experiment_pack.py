from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
OUT = RESULTS / "local_cpu_experiment_pack"


DATASETS = {
    "UWF-2024": {
        "dataset_key": "uwf",
        "model_dir": RESULTS / "intervendb_model_uwf_full_main",
        "baseline_dir": RESULTS / "autodl_baselines_uwf_full_calibrated",
        "baseline_status": "calibrated_full",
        "rows": 95_871,
    },
    "Gotham-2025": {
        "dataset_key": "gotham",
        "model_dir": RESULTS / "intervendb_model_gotham_full_uncapped_autodl",
        "baseline_dir": RESULTS / "autodl_baselines_gotham_full_core_calibrated",
        "baseline_status": "calibrated_full",
        "rows": 5_512_259,
    },
}


METHOD_ROLES = {
    "intervendb_adaptive_openworld_selector": "adaptive coverage/open-world selection",
    "intervendb_coverage_lift_openworld_selector": "coverage-lift open-world selector",
    "intervendb_coverage_openworld_selector": "coverage open-world selector",
    "intervendb_counterfactual_ensemble": "counterfactual weighted ensemble",
    "intervendb_env_calibrated_ensemble": "environment-calibrated ensemble",
    "intervendb_stability_selector": "stability-only selector",
    "intervendb_openworld_selector": "open-world selector",
    "intervendb_pareto_selector": "Pareto utility selector",
    "intervendb_shift_guard_selector": "shift guard selector",
    "intervendb_minimax_selector": "minimax-only selector",
    "intervendb_observed_selector": "observed-only selector",
    "intervendb_conformal_tail_guard": "conformal tail guard",
    "intervendb_gated_conformal_tail_guard": "gated conformal tail guard",
    "intervendb_tail_guard_ensemble": "tail-guard ensemble",
}


def ensure_out() -> None:
    OUT.mkdir(parents=True, exist_ok=True)


def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def best_model_row(summary: pd.DataFrame) -> pd.Series:
    return summary.sort_values(
        ["worst_intervention_f1_mean", "hidden_drop_mean", "observed_f1_mean"],
        ascending=[False, True, False],
    ).iloc[0]


def best_baseline_row(summary: pd.DataFrame, include_oracle: bool = False) -> pd.Series:
    work = summary.copy()
    if not include_oracle:
        work = work[~work["detector"].astype(str).str.contains("__oracle_env_upper_bound", regex=False)]
    return work.sort_values(
        ["worst_intervention_f1_mean", "hidden_drop_mean", "observed_f1_mean"],
        ascending=[False, True, False],
    ).iloc[0]


def policy_from_detector(detector: str) -> str:
    if "__" not in detector:
        return "native_default"
    return detector.split("__", 1)[1]


def bootstrap_ci(values: np.ndarray, rng: np.random.Generator, n_boot: int = 10_000) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return np.nan, np.nan, np.nan
    if values.size == 1:
        return float(values[0]), float(values[0]), float(values[0])
    means = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        idx = rng.integers(0, values.size, values.size)
        means[i] = float(values[idx].mean())
    return float(values.mean()), float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def paired_tests(values: np.ndarray) -> dict:
    values = np.asarray(values, dtype=float)
    nonzero = values[np.abs(values) > 1e-12]
    out = {
        "n": int(values.size),
        "positive_count": int((values > 0).sum()),
        "negative_count": int((values < 0).sum()),
        "zero_count": int((np.abs(values) <= 1e-12).sum()),
        "paired_t_p_greater": np.nan,
        "wilcoxon_p_greater": np.nan,
        "sign_p_greater": np.nan,
    }
    if values.size >= 2 and np.std(values) > 0:
        out["paired_t_p_greater"] = float(stats.ttest_1samp(values, popmean=0.0, alternative="greater").pvalue)
    if nonzero.size >= 1:
        try:
            out["wilcoxon_p_greater"] = float(stats.wilcoxon(nonzero, alternative="greater").pvalue)
        except ValueError:
            pass
        out["sign_p_greater"] = float(stats.binomtest(int((nonzero > 0).sum()), int(nonzero.size), 0.5, alternative="greater").pvalue)
    return out


def build_main_tables(rng: np.random.Generator) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    main_rows = []
    paired_rows = []
    test_rows = []
    ablation_frames = []

    for dataset_name, cfg in DATASETS.items():
        model_dir = cfg["model_dir"]
        baseline_dir = cfg["baseline_dir"]
        model_summary = pd.read_csv(model_dir / "intervendb_method_summary.csv")
        baseline_summary = pd.read_csv(baseline_dir / "baseline_detector_summary.csv")
        model_hidden = pd.read_csv(model_dir / "intervendb_hidden_drop.csv")
        baseline_hidden = pd.read_csv(baseline_dir / "baseline_hidden_drop.csv")

        model_best = best_model_row(model_summary)
        baseline_best = best_baseline_row(baseline_summary, include_oracle=False)
        oracle_best = None
        oracle_candidates = baseline_summary[
            baseline_summary["detector"].astype(str).str.contains("__oracle_env_upper_bound", regex=False)
        ]
        if not oracle_candidates.empty:
            oracle_best = best_baseline_row(baseline_summary, include_oracle=True)

        main_rows.append({
            "dataset": dataset_name,
            "rows": cfg["rows"],
            "baseline_status": cfg["baseline_status"],
            "model": model_best["method"],
            "model_observed_f1": float(model_best["observed_f1_mean"]),
            "model_worst_f1": float(model_best["worst_intervention_f1_mean"]),
            "model_hidden_drop_mean": float(model_best["hidden_drop_mean"]),
            "model_hidden_drop_max": float(model_best["hidden_drop_max"]),
            "best_deployable_baseline": baseline_best["detector"],
            "baseline_policy": policy_from_detector(str(baseline_best["detector"])),
            "baseline_observed_f1": float(baseline_best["observed_f1_mean"]),
            "baseline_worst_f1": float(baseline_best["worst_intervention_f1_mean"]),
            "baseline_hidden_drop_mean": float(baseline_best["hidden_drop_mean"]),
            "baseline_hidden_drop_max": float(baseline_best["hidden_drop_max"]),
            "worst_f1_gain": float(model_best["worst_intervention_f1_mean"] - baseline_best["worst_intervention_f1_mean"]),
            "hidden_drop_reduction": float(baseline_best["hidden_drop_mean"] - model_best["hidden_drop_mean"]),
            "hidden_drop_max_reduction": float(baseline_best["hidden_drop_max"] - model_best["hidden_drop_max"]),
        })
        if oracle_best is not None:
            main_rows[-1]["best_oracle_baseline"] = oracle_best["detector"]
            main_rows[-1]["oracle_worst_f1"] = float(oracle_best["worst_intervention_f1_mean"])
            main_rows[-1]["model_gain_over_oracle"] = float(
                model_best["worst_intervention_f1_mean"] - oracle_best["worst_intervention_f1_mean"]
            )

        method_name = str(model_best["method"])
        baseline_name = str(baseline_best["detector"])
        model_pair = model_hidden[model_hidden["method"].eq(method_name)][
            ["split", "fold", "observed_f1", "worst_intervention_f1", "worst_hidden_drop"]
        ].rename(columns={
            "observed_f1": "model_observed_f1",
            "worst_intervention_f1": "model_worst_f1",
            "worst_hidden_drop": "model_hidden_drop",
        })
        baseline_pair = baseline_hidden[baseline_hidden["detector"].eq(baseline_name)][
            ["split", "fold", "observed_f1", "worst_intervention_f1", "worst_hidden_drop"]
        ].rename(columns={
            "observed_f1": "baseline_observed_f1",
            "worst_intervention_f1": "baseline_worst_f1",
            "worst_hidden_drop": "baseline_hidden_drop",
        })
        paired = model_pair.merge(baseline_pair, on=["split", "fold"], how="inner")
        paired["dataset"] = dataset_name
        paired["model"] = method_name
        paired["baseline"] = baseline_name
        paired["worst_f1_gain"] = paired["model_worst_f1"] - paired["baseline_worst_f1"]
        paired["hidden_drop_reduction"] = paired["baseline_hidden_drop"] - paired["model_hidden_drop"]
        paired_rows.append(paired)

        for metric in ["worst_f1_gain", "hidden_drop_reduction"]:
            mean, low, high = bootstrap_ci(paired[metric].to_numpy(), rng)
            tests = paired_tests(paired[metric].to_numpy())
            test_rows.append({
                "dataset": dataset_name,
                "comparison": f"{method_name} vs {baseline_name}",
                "metric": metric,
                "mean": mean,
                "ci95_low": low,
                "ci95_high": high,
                **tests,
            })

        best_worst = float(model_summary["worst_intervention_f1_mean"].max())
        best_drop = float(model_summary["hidden_drop_mean"].min())
        ablation = model_summary.copy()
        ablation["dataset"] = dataset_name
        ablation["role"] = ablation["method"].map(METHOD_ROLES).fillna("selector variant")
        ablation["delta_worst_f1_from_best"] = best_worst - ablation["worst_intervention_f1_mean"]
        ablation["extra_hidden_drop_from_best"] = ablation["hidden_drop_mean"] - best_drop
        ablation_frames.append(ablation)

    main = pd.DataFrame(main_rows)
    paired_all = pd.concat(paired_rows, ignore_index=True)
    tests = pd.DataFrame(test_rows)
    ablation_all = pd.concat(ablation_frames, ignore_index=True)
    return main, paired_all, tests, ablation_all


def build_calibration_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    baseline = pd.read_csv(DATASETS["UWF-2024"]["baseline_dir"] / "baseline_detector_summary.csv")
    baseline = baseline.copy()
    baseline["policy"] = baseline["detector"].astype(str).map(policy_from_detector)
    baseline["base_detector"] = baseline["detector"].astype(str).str.split("__", n=1).str[0]
    leaderboard = baseline.sort_values(
        ["worst_intervention_f1_mean", "hidden_drop_mean", "observed_f1_mean"],
        ascending=[False, True, False],
    )
    best_by_policy = leaderboard.groupby("policy", as_index=False).head(1)
    return leaderboard, best_by_policy


def build_shortcut_table() -> pd.DataFrame:
    path = RESULTS / "shortcut_audit" / "shortcut_audit.csv"
    if not path.exists():
        return pd.DataFrame()
    audit = pd.read_csv(path)
    cols = ["feature", "n_unique", "nmi_label", "cramers_v_label", "nmi_attack_family", "nmi_technique", "top_values"]
    return audit[cols].sort_values("nmi_label", ascending=False).head(15)


def plot_outputs(main: pd.DataFrame, ablation: pd.DataFrame, best_by_policy: pd.DataFrame, shortcut: pd.DataFrame) -> None:
    sns.set_theme(style="whitegrid")

    plot_rows = []
    for _, row in main.iterrows():
        plot_rows.append({"dataset": row["dataset"], "method": "IntervenDB", "worst_f1": row["model_worst_f1"]})
        plot_rows.append({"dataset": row["dataset"], "method": "Best deployable baseline", "worst_f1": row["baseline_worst_f1"]})
        if pd.notna(row.get("oracle_worst_f1", np.nan)):
            plot_rows.append({"dataset": row["dataset"], "method": "Oracle baseline upper bound", "worst_f1": row["oracle_worst_f1"]})
    plot_df = pd.DataFrame(plot_rows)
    plt.figure(figsize=(8, 4.5))
    sns.barplot(data=plot_df, x="dataset", y="worst_f1", hue="method")
    plt.ylim(0.75, 1.01)
    plt.ylabel("Worst-intervention F1")
    plt.title("Worst-case robustness comparison")
    plt.tight_layout()
    plt.savefig(OUT / "main_worst_f1_comparison.png", dpi=180)
    plt.close()

    hidden_rows = []
    for _, row in main.iterrows():
        hidden_rows.append({"dataset": row["dataset"], "method": "IntervenDB", "hidden_drop": row["model_hidden_drop_mean"]})
        hidden_rows.append({"dataset": row["dataset"], "method": "Best deployable baseline", "hidden_drop": row["baseline_hidden_drop_mean"]})
    hidden_df = pd.DataFrame(hidden_rows)
    plt.figure(figsize=(8, 4.5))
    sns.barplot(data=hidden_df, x="dataset", y="hidden_drop", hue="method")
    plt.ylabel("Mean hidden drop")
    plt.title("Hidden drop comparison")
    plt.tight_layout()
    plt.savefig(OUT / "hidden_drop_comparison.png", dpi=180)
    plt.close()

    top_ablation = ablation.sort_values(["dataset", "worst_intervention_f1_mean"], ascending=[True, False])
    plt.figure(figsize=(11, 6))
    sns.barplot(data=top_ablation, x="worst_intervention_f1_mean", y="method", hue="dataset")
    plt.xlabel("Mean worst-intervention F1")
    plt.ylabel("")
    plt.title("IntervenDB selector/ablation variants")
    plt.tight_layout()
    plt.savefig(OUT / "intervendb_ablation_worst_f1.png", dpi=180)
    plt.close()

    if not best_by_policy.empty:
        plt.figure(figsize=(9, 4.5))
        sns.barplot(data=best_by_policy, x="policy", y="worst_intervention_f1_mean")
        plt.xticks(rotation=20, ha="right")
        plt.ylabel("Best UWF baseline worst F1")
        plt.title("UWF baseline threshold-calibration policies")
        plt.tight_layout()
        plt.savefig(OUT / "uwf_calibration_policy_best.png", dpi=180)
        plt.close()

    if not shortcut.empty:
        plt.figure(figsize=(9, 5))
        sns.barplot(data=shortcut.head(10), x="nmi_label", y="feature")
        plt.xlabel("NMI with label")
        plt.ylabel("")
        plt.title("UWF shortcut/leakage audit")
        plt.tight_layout()
        plt.savefig(OUT / "uwf_shortcut_top_features.png", dpi=180)
        plt.close()


def main() -> int:
    ensure_out()
    rng = np.random.default_rng(20260601)
    main_table, paired, tests, ablation = build_main_tables(rng)
    leaderboard, best_by_policy = build_calibration_tables()
    shortcut = build_shortcut_table()

    main_table.to_csv(OUT / "paper_main_robustness_results.csv", index=False)
    paired.to_csv(OUT / "paired_splitfold_model_vs_baseline.csv", index=False)
    tests.to_csv(OUT / "paired_statistical_tests.csv", index=False)
    ablation.to_csv(OUT / "intervendb_selector_ablation_summary.csv", index=False)
    leaderboard.to_csv(OUT / "uwf_calibrated_baseline_leaderboard.csv", index=False)
    best_by_policy.to_csv(OUT / "uwf_calibrated_policy_winners.csv", index=False)
    shortcut.to_csv(OUT / "uwf_full_shortcut_audit_top.csv", index=False)

    plot_outputs(main_table, ablation, best_by_policy, shortcut)

    summary = {
        "output_dir": str(OUT),
        "tables": [
            "paper_main_robustness_results.csv",
            "paired_statistical_tests.csv",
            "intervendb_selector_ablation_summary.csv",
            "uwf_calibrated_baseline_leaderboard.csv",
            "uwf_full_shortcut_audit_top.csv",
        ],
        "figures": [
            "main_worst_f1_comparison.png",
            "hidden_drop_comparison.png",
            "intervendb_ablation_worst_f1.png",
            "uwf_calibration_policy_best.png",
            "uwf_shortcut_top_features.png",
        ],
    }
    (OUT / "local_cpu_experiment_pack_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


