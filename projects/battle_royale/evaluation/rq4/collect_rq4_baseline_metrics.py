"""Aggregate RQ4 prior-work/basic-tabular baseline results across cohorts."""

from __future__ import annotations

# Support explicit direct-file execution as well as package entry points.
if not __package__:
    import sys as _pkg_sys
    from pathlib import Path as _PackagePath

    _workspace = _PackagePath(__file__).resolve().parents[4]
    _pkg_sys.path.insert(0, str(_workspace))
    _pkg_sys.path.insert(0, str(_workspace / "src"))

import argparse
import json
from pathlib import Path

import pandas as pd


def _read_csvs(root: Path, filename: str) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for path in sorted(root.glob(f"*/{filename}")):
        try:
            df = pd.read_csv(path)
        except pd.errors.EmptyDataError:
            continue
        df.insert(0, "cohort_slug", path.parent.name)
        frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def run(root: Path) -> dict:
    root.mkdir(parents=True, exist_ok=True)
    metrics = _read_csvs(root, "feature_ablation_metrics.csv")
    phase = _read_csvs(root, "feature_ablation_phase_metrics.csv")
    paired = _read_csvs(root, "feature_ablation_paired_tests.csv")

    outputs: dict[str, str] = {}
    if not metrics.empty:
        metrics_path = root / "rq4_prior_baseline_metrics_all_cohorts.csv"
        metrics.to_csv(metrics_path, index=False)
        outputs["metrics_all_cohorts"] = str(metrics_path)

        summary = (
            metrics.groupby("model", dropna=False)
            .agg(
                cohorts=("cohort_slug", "nunique"),
                median_auc=("auc", "median"),
                mean_auc=("auc", "mean"),
                median_ap=("ap", "median"),
                mean_ap=("ap", "mean"),
                median_bce=("bce", "median"),
                mean_bce=("bce", "mean"),
                median_n=("n", "median"),
            )
            .reset_index()
            .sort_values(["median_auc", "mean_auc"], ascending=False)
        )
    else:
        summary = pd.DataFrame()

    if not summary.empty:
        summary_path = root / "rq4_prior_baseline_metrics_summary.csv"
        summary.to_csv(summary_path, index=False)
        outputs["metrics_summary"] = str(summary_path)

    if not phase.empty:
        phase_path = root / "rq4_prior_baseline_phase_metrics_all_cohorts.csv"
        phase.to_csv(phase_path, index=False)
        outputs["phase_metrics_all_cohorts"] = str(phase_path)

    if not paired.empty:
        paired_path = root / "rq4_prior_baseline_paired_tests_all_cohorts.csv"
        paired.to_csv(paired_path, index=False)
        outputs["paired_tests_all_cohorts"] = str(paired_path)

    manifest = {
        "root": str(root),
        "cohorts": int(metrics["cohort_slug"].nunique()) if not metrics.empty else 0,
        "models": sorted(metrics["model"].dropna().unique().tolist()) if not metrics.empty else [],
        "outputs": outputs,
    }
    manifest_path = root / "rq4_prior_baseline_summary.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--root",
        type=Path,
        default=Path("model_stgcn/checkpoints/br_rq4_prior_baseline_required_v1"),
    )
    return ap.parse_args()


if __name__ == "__main__":
    print(json.dumps(run(parse_args().root), indent=2))
