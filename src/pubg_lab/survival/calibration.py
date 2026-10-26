"""Calibration diagnostics for hazard prediction CSVs.

Inputs are prediction tables with at least model/control, y, and prob columns.
The script writes:

* calibration_metrics.csv: Brier, BCE, ECE, MCE by model/group
* calibration_bins.csv: reliability-curve bin data

Example:
    python -m model_stgcn.calibration \
      --predictions model_stgcn/checkpoints/phase_result_analysis/hazard_predictions_test_aligned.csv \
      --out-dir model_stgcn/checkpoints/required_experiments/calibration
"""

from __future__ import annotations

# Support explicit direct-file execution as well as package entry points.
if not __package__:
    import sys as _pkg_sys
    from pathlib import Path as _PackagePath

    _pkg_sys.path.insert(0, str(_PackagePath(__file__).resolve().parents[2]))

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss


def _auto_col(df: pd.DataFrame, preferred: str | None, candidates: list[str]) -> str:
    if preferred:
        if preferred not in df.columns:
            raise ValueError(f"column not found: {preferred}")
        return preferred
    for col in candidates:
        if col in df.columns:
            return col
    raise ValueError(f"none of these columns found: {candidates}")


def _brier(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean((p - y) ** 2)) if y.size else float("nan")


def _ece_bins(
    y: np.ndarray,
    p: np.ndarray,
    *,
    n_bins: int,
    prefix: dict[str, Any],
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    y = y.astype(float)
    p = np.clip(p.astype(float), 1e-7, 1 - 1e-7)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    rows: list[dict[str, Any]] = []
    ece = 0.0
    mce = 0.0
    ace_terms: list[float] = []
    for i in range(n_bins):
        lo = edges[i]
        hi = edges[i + 1]
        if i == n_bins - 1:
            mask = (p >= lo) & (p <= hi)
        else:
            mask = (p >= lo) & (p < hi)
        count = int(mask.sum())
        if count:
            conf = float(p[mask].mean())
            acc = float(y[mask].mean())
            gap = abs(acc - conf)
            weight = count / max(len(y), 1)
            ece += weight * gap
            mce = max(mce, gap)
            ace_terms.append(gap)
        else:
            conf = float("nan")
            acc = float("nan")
            gap = float("nan")
            weight = 0.0
        rows.append(
            {
                **prefix,
                "bin": i,
                "bin_left": float(lo),
                "bin_right": float(hi),
                "count": count,
                "weight": float(weight),
                "mean_prob": conf,
                "empirical_rate": acc,
                "abs_gap": gap,
            }
        )
    summary = {
        "ece": float(ece),
        "mce": float(mce),
        "ace": float(np.mean(ace_terms)) if ace_terms else float("nan"),
    }
    return summary, rows


def _metrics_for_group(
    group: pd.DataFrame,
    *,
    y_col: str,
    prob_col: str,
    n_bins: int,
    prefix: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    y = group[y_col].to_numpy(dtype=float)
    p = np.clip(group[prob_col].to_numpy(dtype=float), 1e-7, 1 - 1e-7)
    calib, bins = _ece_bins(y, p, n_bins=n_bins, prefix=prefix)
    metric = {
        **prefix,
        "n": int(len(group)),
        "pos_rate": float(y.mean()) if y.size else float("nan"),
        "mean_prob": float(p.mean()) if p.size else float("nan"),
        "brier": _brier(y, p),
        "bce": float(log_loss(y, p, labels=[0, 1])) if y.size else float("nan"),
        **calib,
    }
    return metric, bins


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                fields.append(key)
                seen.add(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def run(args: argparse.Namespace) -> dict[str, Any]:
    df = pd.read_csv(args.predictions)
    y_col = _auto_col(df, args.y_col, ["y", "label", "target"])
    prob_col = _auto_col(df, args.prob_col, ["prob", "y_prob", "hazard_prob"])
    model_col = _auto_col(df, args.model_col, ["model", "edge_mode", "control"])

    group_cols = [model_col]
    for col in args.group_cols:
        if col not in df.columns:
            raise ValueError(f"group column not found: {col}")
        if col not in group_cols:
            group_cols.append(col)

    metric_rows: list[dict[str, Any]] = []
    bin_rows: list[dict[str, Any]] = []
    for key, group in df.groupby(group_cols, dropna=False):
        if not isinstance(key, tuple):
            key = (key,)
        prefix = dict(zip(group_cols, key))
        metric, bins = _metrics_for_group(
            group,
            y_col=y_col,
            prob_col=prob_col,
            n_bins=args.bins,
            prefix=prefix,
        )
        metric_rows.append(metric)
        bin_rows.extend(bins)

    if args.include_overall and len(group_cols) > 1:
        for model_value, group in df.groupby(model_col, dropna=False):
            prefix = {
                model_col: model_value,
                **{col: "ALL" for col in group_cols if col != model_col},
            }
            metric, bins = _metrics_for_group(
                group,
                y_col=y_col,
                prob_col=prob_col,
                n_bins=args.bins,
                prefix=prefix,
            )
            metric_rows.append(metric)
            bin_rows.extend(bins)

    out_dir = Path(args.out_dir)
    metrics_path = out_dir / "calibration_metrics.csv"
    bins_path = out_dir / "calibration_bins.csv"
    summary_path = out_dir / "calibration_summary.json"
    _write_csv(metrics_path, metric_rows)
    _write_csv(bins_path, bin_rows)
    summary = {
        "predictions": args.predictions,
        "y_col": y_col,
        "prob_col": prob_col,
        "group_cols": group_cols,
        "bins": args.bins,
        "outputs": {
            "metrics": str(metrics_path),
            "bins": str(bins_path),
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--predictions",
        default="model_stgcn/checkpoints/phase_result_analysis/hazard_predictions_test_aligned.csv",
    )
    ap.add_argument("--out-dir", default="model_stgcn/checkpoints/required_experiments/calibration")
    ap.add_argument("--bins", type=int, default=10)
    ap.add_argument("--model-col", default=None)
    ap.add_argument("--y-col", default=None)
    ap.add_argument("--prob-col", default=None)
    ap.add_argument("--group-cols", nargs="*", default=[])
    ap.add_argument("--include-overall", action="store_true")
    return ap.parse_args()


if __name__ == "__main__":
    run(parse_args())
