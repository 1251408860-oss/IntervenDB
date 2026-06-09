from __future__ import annotations

import argparse
import compileall
import csv
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


REQUIRED_PATHS = [
    "README.md",
    "DATASETS.md",
    "LICENSE",
    "pyproject.toml",
    "requirements_local.txt",
    "requirements_baselines.txt",
    "src/intervendb/data.py",
    "src/intervendb/interventions.py",
    "src/intervendb/metrics.py",
    "src/intervendb/witness.py",
    "configs/intervendb_model_uwf_full_main.yaml",
    "configs/intervendb_model_gotham_full_uncapped_autodl.yaml",
    "configs/autodl_baseline_uwf_full_calibrated.yaml",
    "configs/autodl_baseline_gotham_full_core_calibrated.yaml",
    "configs/local_risk_cube.yaml",
    "results/intervendb_model_uwf_full_main/intervendb_method_summary.csv",
    "results/intervendb_model_gotham_full_uncapped_autodl/intervendb_method_summary.csv",
    "results/autodl_baselines_uwf_full_calibrated/baseline_detector_summary.csv",
    "results/autodl_baselines_gotham_full_core_calibrated/baseline_detector_summary.csv",
    "results/local_cpu_experiment_pack/paper_main_robustness_results.csv",
    "results/local_cpu_experiment_pack/paired_statistical_tests.csv",
    "results/local_cpu_experiment_pack/intervendb_selector_ablation_summary.csv",
    "results/local_cpu_experiment_pack/local_cpu_experiment_pack_summary.json",
    "results/risk_cube/risk_cube_summary.json",
    "results/witness_sweep/witness_sweep_summary.csv",
]


FORBIDDEN_PUBLIC_STRINGS = [
    "password=",
    "api_key=",
    "secret_key=",
    "submission_gap",
    "reviewer_defense",
    "acceptance probability",
    "drafts/",
    "sections/",
]


TEXT_SUFFIXES = {
    ".bib",
    ".cfg",
    ".csv",
    ".json",
    ".md",
    ".py",
    ".toml",
    ".txt",
    ".yaml",
    ".yml",
}


def fail(message: str) -> None:
    raise AssertionError(message)


def rel(path: Path) -> str:
    return str(path.relative_to(ROOT)).replace("\\", "/")


def check_required_paths() -> None:
    missing = [path for path in REQUIRED_PATHS if not (ROOT / path).exists()]
    if missing:
        fail("missing required paths: " + ", ".join(missing))


def check_summary_tables() -> None:
    main_path = ROOT / "results/local_cpu_experiment_pack/paper_main_robustness_results.csv"
    with main_path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    datasets = {row.get("dataset", "") for row in rows}
    if datasets != {"UWF-2024", "Gotham-2025"}:
        fail(f"unexpected main result datasets: {sorted(datasets)}")

    if len(rows) != 2:
        fail(f"expected 2 main result rows, found {len(rows)}")


def check_forbidden_strings() -> None:
    hits: list[str] = []
    self_path = Path(__file__).resolve()
    for path in ROOT.rglob("*"):
        if not path.is_file():
            continue
        if path.resolve() == self_path:
            continue
        if ".git" in path.parts or "__pycache__" in path.parts:
            continue
        if path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="ignore").lower()
        except OSError:
            continue
        for needle in FORBIDDEN_PUBLIC_STRINGS:
            if needle.lower() in text:
                hits.append(f"{rel(path)}: {needle}")
    if hits:
        fail("forbidden public strings found:\n" + "\n".join(hits))


def check_compile() -> None:
    ok = compileall.compile_dir(str(ROOT / "src"), quiet=1)
    ok = compileall.compile_dir(str(ROOT / "scripts"), quiet=1) and ok
    if not ok:
        fail("Python compile check failed")


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify the public IntervenDB repository layout.")
    parser.add_argument("--compile", action="store_true", help="also run Python compile checks")
    args = parser.parse_args()

    checks = [
        check_required_paths,
        check_summary_tables,
        check_forbidden_strings,
    ]
    if args.compile:
        checks.append(check_compile)

    for check in checks:
        check()
        print(f"ok: {check.__name__}")
    print("artifact verification passed")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except AssertionError as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1)
