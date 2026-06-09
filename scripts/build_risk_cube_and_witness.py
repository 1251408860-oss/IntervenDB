from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import duckdb
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import yaml

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from intervendb.witness import add_observation_columns, build_random_witness, build_score_stratified_witness


FULL_METRICS_SQL = """
WITH agg AS (
    SELECT
        detector,
        environment_id,
        SUM(tp * weight) AS tp,
        SUM(fp * weight) AS fp,
        SUM(fn * weight) AS fn,
        SUM(tn * weight) AS tn,
        SUM(is_correct * weight) AS correct,
        SUM(weight) AS n,
        AVG(score) AS avg_score
    FROM {table_name}
    GROUP BY detector, environment_id
)
SELECT
    detector,
    environment_id,
    correct / NULLIF(n, 0) AS accuracy,
    tp / NULLIF(tp + fp, 0) AS precision,
    tp / NULLIF(tp + fn, 0) AS recall,
    2 * tp / NULLIF(2 * tp + fp + fn, 0) AS f1,
    avg_score,
    n
FROM agg
ORDER BY detector, environment_id
"""


SEMANTIC_CUBE_SQL = """
WITH agg AS (
    SELECT
        detector,
        environment_id,
        semantic_label,
        SUM(tp * weight) AS tp,
        SUM(fp * weight) AS fp,
        SUM(fn * weight) AS fn,
        SUM(tn * weight) AS tn,
        SUM(is_correct * weight) AS correct,
        SUM(weight) AS n,
        AVG(score) AS avg_score
    FROM {table_name}
    GROUP BY detector, environment_id, semantic_label
)
SELECT
    detector,
    environment_id,
    semantic_label,
    correct / NULLIF(n, 0) AS accuracy,
    tp / NULLIF(tp + fp, 0) AS precision,
    tp / NULLIF(tp + fn, 0) AS recall,
    2 * tp / NULLIF(2 * tp + fp + fn, 0) AS f1,
    avg_score,
    n
FROM agg
ORDER BY detector, environment_id, semantic_label
"""


RANKING_SQL = """
WITH metrics AS (
    {metrics_sql}
),
ranked AS (
    SELECT
        environment_id,
        detector,
        f1,
        RANK() OVER (PARTITION BY environment_id ORDER BY f1 DESC) AS f1_rank
    FROM metrics
)
SELECT * FROM ranked
ORDER BY environment_id, f1_rank, detector
"""


def sql_path(path: Path) -> str:
    return str(path).replace("\\", "/")


def benchmark_query(con: duckdb.DuckDBPyConnection, sql: str, repeats: int) -> dict:
    con.execute(sql).fetchall()
    timings = []
    rows = 0
    for _ in range(repeats):
        start = time.perf_counter()
        rows = len(con.execute(sql).fetchall())
        timings.append((time.perf_counter() - start) * 1000.0)
    return {
        "rows": rows,
        "mean_ms": sum(timings) / len(timings),
        "min_ms": min(timings),
        "max_ms": max(timings),
    }


def plot_outputs(full_metrics, witness_metrics, errors, latencies, out_dir: Path):
    sns.set_theme(style="whitegrid")

    merged = full_metrics.merge(
        witness_metrics,
        on=["detector", "environment_id"],
        suffixes=("_full", "_witness"),
    )

    plt.figure(figsize=(6, 6))
    sns.scatterplot(data=merged, x="f1_full", y="f1_witness", hue="environment_id")
    lim_min = min(merged["f1_full"].min(), merged["f1_witness"].min()) - 0.02
    lim_max = max(merged["f1_full"].max(), merged["f1_witness"].max()) + 0.02
    plt.plot([lim_min, lim_max], [lim_min, lim_max], "k--", linewidth=1)
    plt.xlim(lim_min, lim_max)
    plt.ylim(lim_min, lim_max)
    plt.title("Full risk vs witness risk")
    plt.tight_layout()
    plt.savefig(out_dir / "full_vs_witness_f1.png", dpi=180)
    plt.close()

    plt.figure(figsize=(10, 5))
    sns.barplot(data=errors, x="detector", y="abs_f1_error")
    plt.xticks(rotation=20, ha="right")
    plt.title("Witness absolute F1 error by detector")
    plt.tight_layout()
    plt.savefig(out_dir / "witness_error_by_detector.png", dpi=180)
    plt.close()

    plt.figure(figsize=(8, 4))
    sns.barplot(data=latencies, x="query", y="mean_ms", hue="table")
    plt.title("DuckDB query latency")
    plt.tight_layout()
    plt.savefig(out_dir / "query_latency.png", dpi=180)
    plt.close()


def main():
    config_path = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "configs" / "local_risk_cube.yaml"
    with config_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    out_dir = ROOT / cfg["output_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)

    input_csv = ROOT / cfg["input_predictions_csv"]
    full_parquet = out_dir / "risk_observations_full.parquet"
    witness_parquet = out_dir / "risk_observations_witness.parquet"
    db_path = out_dir / "intervendb_local.duckdb"

    raw = pd.read_csv(input_csv)
    full = add_observation_columns(raw)
    strategy = cfg["witness"].get("strategy", "score_stratified")
    if strategy == "score_stratified":
        witness = build_score_stratified_witness(
            full,
            group_cols=["detector", "environment_id", "semantic_label"],
            max_witness_per_group=int(cfg["witness"]["max_per_cell"]),
        )
    elif strategy == "random":
        witness = build_random_witness(
            full,
            group_cols=["detector", "environment_id", "semantic_label"],
            max_witness_per_group=int(cfg["witness"]["max_per_cell"]),
            random_state=int(cfg["witness"].get("random_state", 0)),
        )
    else:
        raise ValueError(f"Unknown witness strategy: {strategy}")

    full.to_parquet(full_parquet, index=False)
    witness.to_parquet(witness_parquet, index=False)

    con = duckdb.connect(str(db_path))
    con.execute(f"CREATE OR REPLACE VIEW risk_full AS SELECT * FROM read_parquet('{sql_path(full_parquet)}')")
    con.execute(f"CREATE OR REPLACE VIEW risk_witness AS SELECT * FROM read_parquet('{sql_path(witness_parquet)}')")

    full_metrics_sql = FULL_METRICS_SQL.format(table_name="risk_full")
    witness_metrics_sql = FULL_METRICS_SQL.format(table_name="risk_witness")
    full_semantic_sql = SEMANTIC_CUBE_SQL.format(table_name="risk_full")
    witness_semantic_sql = SEMANTIC_CUBE_SQL.format(table_name="risk_witness")

    full_metrics = con.execute(full_metrics_sql).fetchdf()
    witness_metrics = con.execute(witness_metrics_sql).fetchdf()
    full_semantic = con.execute(full_semantic_sql).fetchdf()
    witness_semantic = con.execute(witness_semantic_sql).fetchdf()

    full_metrics.to_csv(out_dir / "risk_metrics_full.csv", index=False)
    witness_metrics.to_csv(out_dir / "risk_metrics_witness.csv", index=False)
    full_semantic.to_csv(out_dir / "semantic_cube_full.csv", index=False)
    witness_semantic.to_csv(out_dir / "semantic_cube_witness.csv", index=False)

    comparison = full_metrics.merge(
        witness_metrics,
        on=["detector", "environment_id"],
        suffixes=("_full", "_witness"),
    )
    comparison["abs_f1_error"] = (comparison["f1_full"] - comparison["f1_witness"]).abs()
    comparison["abs_accuracy_error"] = (comparison["accuracy_full"] - comparison["accuracy_witness"]).abs()
    comparison.to_csv(out_dir / "witness_approximation_error.csv", index=False)

    repeats = int(cfg["benchmark_repeats"])
    latency_rows = []
    for table_name, table_sql_name in [("full", "risk_full"), ("witness", "risk_witness")]:
        for query_name, query_sql in [
            ("environment_metrics", FULL_METRICS_SQL.format(table_name=table_sql_name)),
            ("semantic_cube", SEMANTIC_CUBE_SQL.format(table_name=table_sql_name)),
        ]:
            row = benchmark_query(con, query_sql, repeats)
            row.update({"table": table_name, "query": query_name})
            latency_rows.append(row)

    latencies = pd.DataFrame(latency_rows)
    latencies.to_csv(out_dir / "query_latency.csv", index=False)

    plot_outputs(full_metrics, witness_metrics, comparison, latencies, out_dir)

    full_rows = int(len(full))
    witness_rows = int(len(witness))
    summary = {
        "input_predictions_csv": str(input_csv),
        "output_dir": str(out_dir),
        "duckdb_path": str(db_path),
        "full_rows": full_rows,
        "witness_rows": witness_rows,
        "compression_ratio": full_rows / max(witness_rows, 1),
        "witness_strategy": strategy,
        "max_abs_f1_error": float(comparison["abs_f1_error"].max()),
        "mean_abs_f1_error": float(comparison["abs_f1_error"].mean()),
        "benchmark_repeats": repeats,
    }
    with (out_dir / "risk_cube_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    con.close()
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()


