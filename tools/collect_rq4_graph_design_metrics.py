"""Aggregate RQ4 graph-design matrix metrics across required cohorts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def run(root: Path) -> dict:
    root.mkdir(parents=True, exist_ok=True)
    frames: list[pd.DataFrame] = []
    for path in sorted(root.glob("*/matrix_metrics.csv")):
        df = pd.read_csv(path)
        df.insert(0, "cohort_slug", path.parent.name)
        frames.append(df)
    metrics = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    outputs: dict[str, str] = {}
    if not metrics.empty:
        metrics_path = root / "rq4_graph_design_metrics_all_cohorts.csv"
        metrics.to_csv(metrics_path, index=False)
        outputs["metrics_all_cohorts"] = str(metrics_path)

        group_col = "name" if "name" in metrics.columns else "experiment"
        numeric_cols = [c for c in ["hazard_auc", "rank_spearman", "top10_auc", "bce"] if c in metrics.columns]
        aggregations = {"cohorts": ("cohort_slug", "nunique")}
        for col in numeric_cols:
            aggregations[f"median_{col}"] = (col, "median")
            aggregations[f"mean_{col}"] = (col, "mean")
        summary = metrics.groupby(group_col, dropna=False).agg(**aggregations).reset_index()
        sort_col = "median_hazard_auc" if "median_hazard_auc" in summary.columns else None
        if sort_col:
            summary = summary.sort_values(sort_col, ascending=False)
    else:
        summary = pd.DataFrame()

    if not summary.empty:
        summary_path = root / "rq4_graph_design_metrics_summary.csv"
        summary.to_csv(summary_path, index=False)
        outputs["metrics_summary"] = str(summary_path)

    manifest = {
        "root": str(root),
        "cohorts": int(metrics["cohort_slug"].nunique()) if not metrics.empty else 0,
        "outputs": outputs,
    }
    manifest_path = root / "rq4_graph_design_summary.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path("model_stgcn/checkpoints/br_rq4_graph_design_required_v1"))
    return ap.parse_args()


if __name__ == "__main__":
    print(json.dumps(run(parse_args().root), indent=2))
