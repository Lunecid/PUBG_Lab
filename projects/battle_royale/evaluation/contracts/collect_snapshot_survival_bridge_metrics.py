"""Aggregate snapshot cumulative-survival bridge metrics across cohorts."""

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
from typing import Any

import pandas as pd


METRIC_FIELDS = {
    "hazard": ("auc", "ap", "bce", "brier", "ece", "n", "pos_rate"),
    "top5_from_survival": ("auc", "ap", "bce", "brier", "n", "pos_rate"),
    "top10_from_survival": ("auc", "ap", "bce", "brier", "n", "pos_rate"),
    "rank_from_survival": ("spearman", "mae_norm", "pairwise_acc", "pairwise_n", "n"),
}


def _flatten_metrics(data: dict[str, Any]) -> dict[str, Any]:
    row: dict[str, Any] = {
        "model_variant": data.get("model_variant"),
        "checkpoint": data.get("checkpoint"),
        "split": data.get("split"),
        "prediction_rows": (data.get("split_counts") or {}).get("prediction_rows"),
        "matches_loaded": (data.get("split_counts") or {}).get("matches_loaded"),
        "model_family": (data.get("channel_contract") or {}).get("model_family"),
        "temporal_resolution": (data.get("channel_contract") or {}).get("temporal_resolution"),
        "graph_resolution": (data.get("channel_contract") or {}).get("graph_resolution"),
        "snapshot_feature_preset": (data.get("channel_contract") or {}).get(
            "snapshot_feature_preset"
        ),
        "bridge_metric_is_headline_safe": data.get("bridge_metric_is_headline_safe"),
    }
    for block, fields in METRIC_FIELDS.items():
        metrics = data.get(block) or {}
        for field in fields:
            row[f"{block}_{field}"] = metrics.get(field)
    latest = data.get("latest_cut_bridge") or {}
    row["latest_cut_bridge_n"] = latest.get("n")
    row["latest_cut_bridge_n_matches"] = latest.get("n_matches")
    for block, fields in (
        ("top5_from_survival", ("auc", "ap", "bce", "brier", "n", "pos_rate")),
        ("top10_from_survival", ("auc", "ap", "bce", "brier", "n", "pos_rate")),
        ("rank_from_survival", ("spearman", "mae_norm", "pairwise_acc", "pairwise_n", "n")),
    ):
        metrics = latest.get(block) or {}
        for field in fields:
            row[f"latest_{block}_{field}"] = metrics.get(field)
    return row


def run(root: Path) -> dict[str, Any]:
    root.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for path in sorted(root.glob("*/*/survival_bridge_test.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        row = {"cohort_slug": path.parent.parent.name, **_flatten_metrics(data)}
        row["artifact_dir"] = str(path.parent)
        row["survival_bridge_json"] = str(path)
        row["prediction_csv"] = data.get("prediction_csv") or str(path.with_suffix(".pred.csv"))
        rows.append(row)
    for path in sorted(root.glob("*/snapshot_tabular*/snapshot_tabular_metrics.csv")):
        metrics = pd.read_csv(path)
        for metric_row in metrics.to_dict(orient="records"):
            rows.append(
                {
                    "cohort_slug": path.parent.parent.name,
                    "model_variant": metric_row.get("model_variant"),
                    "checkpoint": float("nan"),
                    "split": "test",
                    "prediction_rows": metric_row.get("n"),
                    "matches_loaded": float("nan"),
                    "model_family": path.parent.name,
                    "temporal_resolution": "10s_snapshot_tabular_reduction",
                    "graph_resolution": "none",
                    "snapshot_feature_preset": "full",
                    "hazard_auc": metric_row.get("auc"),
                    "hazard_ap": metric_row.get("ap"),
                    "hazard_bce": metric_row.get("bce"),
                    "hazard_brier": metric_row.get("brier"),
                    "hazard_ece": float("nan"),
                    "hazard_n": metric_row.get("n"),
                    "hazard_pos_rate": metric_row.get("pos_rate"),
                    "top5_from_survival_auc": float("nan"),
                    "top5_from_survival_ap": float("nan"),
                    "top5_from_survival_bce": float("nan"),
                    "top5_from_survival_brier": float("nan"),
                    "top5_from_survival_n": float("nan"),
                    "top5_from_survival_pos_rate": float("nan"),
                    "top10_from_survival_auc": float("nan"),
                    "top10_from_survival_ap": float("nan"),
                    "top10_from_survival_bce": float("nan"),
                    "top10_from_survival_brier": float("nan"),
                    "top10_from_survival_n": float("nan"),
                    "top10_from_survival_pos_rate": float("nan"),
                    "rank_from_survival_spearman": float("nan"),
                    "rank_from_survival_mae_norm": float("nan"),
                    "rank_from_survival_pairwise_acc": float("nan"),
                    "rank_from_survival_pairwise_n": float("nan"),
                    "rank_from_survival_n": float("nan"),
                    "latest_cut_bridge_n": float("nan"),
                    "latest_cut_bridge_n_matches": float("nan"),
                    "latest_top5_from_survival_auc": float("nan"),
                    "latest_top5_from_survival_ap": float("nan"),
                    "latest_top5_from_survival_bce": float("nan"),
                    "latest_top5_from_survival_brier": float("nan"),
                    "latest_top5_from_survival_n": float("nan"),
                    "latest_top5_from_survival_pos_rate": float("nan"),
                    "latest_top10_from_survival_auc": float("nan"),
                    "latest_top10_from_survival_ap": float("nan"),
                    "latest_top10_from_survival_bce": float("nan"),
                    "latest_top10_from_survival_brier": float("nan"),
                    "latest_top10_from_survival_n": float("nan"),
                    "latest_top10_from_survival_pos_rate": float("nan"),
                    "latest_rank_from_survival_spearman": float("nan"),
                    "latest_rank_from_survival_mae_norm": float("nan"),
                    "latest_rank_from_survival_pairwise_acc": float("nan"),
                    "latest_rank_from_survival_pairwise_n": float("nan"),
                    "latest_rank_from_survival_n": float("nan"),
                    "artifact_dir": str(path.parent),
                    "survival_bridge_json": float("nan"),
                    "prediction_csv": str(path.parent / "snapshot_tabular_predictions.csv"),
                }
            )

    outputs: dict[str, str] = {}
    metrics = pd.DataFrame(rows)
    if not metrics.empty:
        metrics_path = root / "snapshot_survival_bridge_metrics_all_cohorts.csv"
        metrics.to_csv(metrics_path, index=False)
        outputs["metrics_all_cohorts"] = str(metrics_path)

        summary = (
            metrics.groupby("model_variant", dropna=False)
            .agg(
                cohorts=("cohort_slug", "nunique"),
                median_hazard_auc=("hazard_auc", "median"),
                mean_hazard_auc=("hazard_auc", "mean"),
                median_hazard_ap=("hazard_ap", "median"),
                mean_hazard_ap=("hazard_ap", "mean"),
                median_top5_auc=("top5_from_survival_auc", "median"),
                mean_top5_auc=("top5_from_survival_auc", "mean"),
                median_top10_auc=("top10_from_survival_auc", "median"),
                mean_top10_auc=("top10_from_survival_auc", "mean"),
                median_rank_spearman=("rank_from_survival_spearman", "median"),
                mean_rank_spearman=("rank_from_survival_spearman", "mean"),
                median_latest_top5_auc=("latest_top5_from_survival_auc", "median"),
                mean_latest_top5_auc=("latest_top5_from_survival_auc", "mean"),
                median_latest_top10_auc=("latest_top10_from_survival_auc", "median"),
                mean_latest_top10_auc=("latest_top10_from_survival_auc", "mean"),
                median_latest_rank_spearman=("latest_rank_from_survival_spearman", "median"),
                mean_latest_rank_spearman=("latest_rank_from_survival_spearman", "mean"),
                median_prediction_rows=("prediction_rows", "median"),
            )
            .reset_index()
            .sort_values(["median_hazard_auc", "mean_hazard_auc"], ascending=False)
        )
        summary_path = root / "snapshot_survival_bridge_metrics_summary.csv"
        summary.to_csv(summary_path, index=False)
        outputs["metrics_summary"] = str(summary_path)

    manifest = {
        "root": str(root),
        "cohorts": int(metrics["cohort_slug"].nunique()) if not metrics.empty else 0,
        "model_variants": (
            sorted(metrics["model_variant"].dropna().unique().tolist()) if not metrics.empty else []
        ),
        "outputs": outputs,
    }
    manifest_path = root / "snapshot_survival_bridge_summary.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("model_stgcn/checkpoints/br_rq4_hier_survival_required_v1"),
    )
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(run(parse_args().root), indent=2))
