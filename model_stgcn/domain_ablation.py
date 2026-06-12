"""Domain-stratified ablation summaries.

This script joins model prediction CSVs with STDB map/mode metadata and reports
whether the same gains hold across map, perspective, game mode, and
map-perspective domains.

It can be rerun after larger cross-map conversion/training without changing the
modeling code.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from model_stgcn.feature_ablation import _binary_metrics, _paired_test
from model_stgcn.splits import load_or_create_split
from stdb.queries import list_phase_feature_match_ids
from model_stgcn.baselines import load_phase_rows

logger = logging.getLogger("model_stgcn.domain_ablation")


def _perspective(game_mode: str) -> str:
    return "FPP" if str(game_mode).endswith("-fpp") else "TPP"


def _team_mode(game_mode: str) -> str:
    return str(game_mode).replace("-fpp", "")


def _load_domain_meta(test_ids: list[str]) -> pd.DataFrame:
    rows, _cols = load_phase_rows(test_ids)
    meta = pd.DataFrame(
        [
            {
                "match_id": row["match_id"],
                "team_id": int(row["team_id"]),
                "phase_idx": int(row["phase_idx"]),
                "map_name": row["map_name"],
                "game_mode": row["game_mode"],
            }
            for row in rows
        ]
    ).drop_duplicates()
    meta["perspective"] = meta["game_mode"].map(_perspective)
    meta["team_mode"] = meta["game_mode"].map(_team_mode)
    meta["map_perspective"] = meta["map_name"].astype(str) + "/" + meta["perspective"]
    meta["map_mode"] = meta["map_name"].astype(str) + "/" + meta["game_mode"].astype(str)
    return meta


def _load_predictions(paths: list[str]) -> pd.DataFrame:
    frames = []
    for path in paths:
        p = Path(path)
        if not p.exists():
            logger.warning("missing prediction file: %s", p)
            continue
        frames.append(pd.read_csv(p))
    if not frames:
        raise FileNotFoundError("no prediction CSVs found")
    preds = pd.concat(frames, ignore_index=True)
    preds["team_id"] = preds["team_id"].astype(int)
    preds["phase_idx"] = preds["phase_idx"].astype(int)
    preds["y"] = preds["y"].astype(int)
    return preds.drop_duplicates(["model", "match_id", "team_id", "phase_idx"])


def _metric_rows(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for key, g in df.groupby(group_cols, dropna=False):
        if not isinstance(key, tuple):
            key = (key,)
        row = dict(zip(group_cols, key))
        row.update(_binary_metrics(g["y"].to_numpy(), g["prob"].to_numpy()))
        row["n_matches"] = int(g["match_id"].nunique())
        rows.append(row)
    return pd.DataFrame(rows)


def _paired_metric_by_match(df: pd.DataFrame, domain_col: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (domain, model, match_id), g in df.groupby([domain_col, "model", "match_id"]):
        metrics = _binary_metrics(g["y"].to_numpy(), g["prob"].to_numpy())
        rows.append({domain_col: domain, "model": model, "match_id": match_id, **metrics})
    return pd.DataFrame(rows)


def _domain_deltas(metrics: pd.DataFrame, domain_col: str, comparisons: list[tuple[str, str]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for left, right in comparisons:
        left_df = metrics[metrics["model"] == left].set_index(domain_col)
        right_df = metrics[metrics["model"] == right].set_index(domain_col)
        for domain in sorted(set(left_df.index) & set(right_df.index)):
            l = left_df.loc[domain]
            r = right_df.loc[domain]
            rows.append(
                {
                    "domain_col": domain_col,
                    "domain": domain,
                    "left_model": left,
                    "right_model": right,
                    "n": int(l["n"]),
                    "n_matches": int(l["n_matches"]),
                    "pos_rate": float(l["pos_rate"]),
                    "delta_auc": float(l["auc"] - r["auc"]),
                    "delta_ap": float(l["ap"] - r["ap"]),
                    "bce_improvement": float(r["bce"] - l["bce"]),
                    "left_auc": float(l["auc"]),
                    "right_auc": float(r["auc"]),
                    "left_ap": float(l["ap"]),
                    "right_ap": float(r["ap"]),
                    "left_bce": float(l["bce"]),
                    "right_bce": float(r["bce"]),
                }
            )
    return pd.DataFrame(rows)


def _domain_tests(df: pd.DataFrame, domain_col: str, comparisons: list[tuple[str, str]], min_matches: int) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    per_match = _paired_metric_by_match(df, domain_col)
    for domain, domain_pm in per_match.groupby(domain_col):
        if domain_pm["match_id"].nunique() < min_matches:
            continue
        available = set(domain_pm["model"])
        for left, right in comparisons:
            if left not in available or right not in available:
                continue
            for metric, larger in [("bce", False), ("auc", True), ("ap", True)]:
                out = _paired_test(domain_pm, left_model=left, right_model=right, metric=metric, larger_is_better=larger)
                out["domain_col"] = domain_col
                out["domain"] = domain
                rows.append(out)
    return pd.DataFrame(rows)


def run(args: argparse.Namespace) -> dict[str, Any]:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ids = list_phase_feature_match_ids(args.map, args.mode, args.limit)
    _train_ids, _val_ids, test_ids = load_or_create_split(
        ids,
        split_json=args.split_json,
        seed=args.seed,
        val_frac=args.val_frac,
        test_frac=args.test_frac,
        metadata={"source": "model_stgcn.domain_ablation", "limit": args.limit},
    )
    logger.info("test matches=%d", len(test_ids))

    meta = _load_domain_meta(test_ids)
    preds = _load_predictions(args.predictions)
    merged = preds.merge(meta, on=["match_id", "team_id", "phase_idx"], how="inner")
    logger.info("pred rows=%d merged=%d models=%d", len(preds), len(merged), merged["model"].nunique())

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    comparisons = [
        ("conventional_plus_stdb_spatial", "conventional"),
        ("full_node_features", "prior_raw"),
        ("full_node_features", "prior_scores"),
        ("dynamic_full_e12", "dynamic_none_e12"),
        ("dynamic_full_tuned", "tabular_baseline_hgb"),
    ]
    domain_cols = ["perspective", "map_name", "game_mode", "map_perspective", "map_mode"]

    all_metrics = []
    all_deltas = []
    all_tests = []
    for domain_col in domain_cols:
        metrics = _metric_rows(merged, ["model", domain_col])
        deltas = _domain_deltas(metrics, domain_col, comparisons)
        tests = _domain_tests(merged, domain_col, comparisons, args.min_matches)
        all_metrics.append(metrics.assign(domain_col=domain_col))
        all_deltas.append(deltas)
        if not tests.empty:
            all_tests.append(tests)

    metrics_df = pd.concat(all_metrics, ignore_index=True) if all_metrics else pd.DataFrame()
    deltas_df = pd.concat(all_deltas, ignore_index=True) if all_deltas else pd.DataFrame()
    tests_df = pd.concat(all_tests, ignore_index=True) if all_tests else pd.DataFrame()

    merged_path = out_dir / "domain_predictions_joined.csv"
    metrics_path = out_dir / "domain_metrics.csv"
    deltas_path = out_dir / "domain_deltas.csv"
    tests_path = out_dir / "domain_paired_tests.csv"
    summary_path = out_dir / "domain_ablation_summary.json"

    merged.to_csv(merged_path, index=False)
    metrics_df.to_csv(metrics_path, index=False)
    deltas_df.to_csv(deltas_path, index=False)
    tests_df.to_csv(tests_path, index=False)

    summary = {
        "split": {"candidate_matches": len(ids), "test_matches": len(test_ids)},
        "prediction_files": args.predictions,
        "merged_rows": int(len(merged)),
        "models": sorted(merged["model"].unique().tolist()),
        "outputs": {
            "joined_predictions": str(merged_path),
            "domain_metrics": str(metrics_path),
            "domain_deltas": str(deltas_path),
            "domain_paired_tests": str(tests_path),
        },
        "domain_counts": (
            merged[["match_id", "map_name", "game_mode", "perspective", "map_perspective"]]
            .drop_duplicates()
            .groupby(["map_name", "game_mode", "perspective"])
            .size()
            .reset_index(name="test_matches")
            .sort_values("test_matches", ascending=False)
            .to_dict(orient="records")
        ),
        "key_deltas": deltas_df.to_dict(orient="records"),
    }
    summary_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    logger.info("wrote %s", summary_path)
    return summary


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--map", default=None)
    ap.add_argument("--mode", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--test-frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--split-json", default="model_stgcn/checkpoints/split_phase_all_seed42.json")
    ap.add_argument(
        "--predictions",
        nargs="+",
        default=[
            "model_stgcn/checkpoints/feature_ablation/feature_ablation_predictions.csv",
            "model_stgcn/checkpoints/phase_result_analysis/hazard_predictions_test_aligned.csv",
        ],
    )
    ap.add_argument("--out-dir", default="model_stgcn/checkpoints/domain_ablation")
    ap.add_argument("--min-matches", type=int, default=30)
    return ap.parse_args()


if __name__ == "__main__":
    run(parse_args())
