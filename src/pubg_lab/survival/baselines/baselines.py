"""
Sklearn baselines for phase-level PUBG survival/rank prediction.

These baselines intentionally ignore graph tensors. They use the same
phase-level feature table as DynamicSTGNN and provide a sanity/performance
floor before claiming graph gains.

Usage:
    python -m model_stgcn.baselines --limit 100
    python -m model_stgcn.baselines
"""

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
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    mean_absolute_error,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from pubg_lab.survival.data.split_access import load_or_create_split
from pubg_lab.data.stdb.connection.conn import connect
from pubg_lab.data.stdb.queries.queries import list_phase_eligible_match_ids

logger = logging.getLogger("pubg_lab.survival.baselines.baselines")


EXCLUDE_COLUMNS = {
    "match_id",
    "team_id",
    "target_defined",
    "hazard_next",
    "survives_next_phase",
    "final_rank",
    "final_rank_norm",
    "final_top3",
    "final_top5",
    "final_top10",
    "final_winner",
    "last_phase_alive",
}

CATEGORICAL_COLUMNS = ["map_name", "game_mode", "combat_initiated"]


def load_phase_rows(match_ids: list[str]) -> tuple[list[dict], list[str]]:
    if not match_ids:
        return [], []
    with connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT *
                  FROM stdb.v_team_phase_feature_long
                 WHERE match_id = ANY(%s)
                   AND target_defined
                """,
                (match_ids,),
            )
            cols = [d[0] for d in cur.description]
            rows = [dict(zip(cols, r)) for r in cur.fetchall()]
    return rows, cols


def _matrix(rows: list[dict], feature_cols: list[str]):
    X = pd.DataFrame([{c: row.get(c) for c in feature_cols} for row in rows])
    y_hazard = np.array([int(row["hazard_next"]) for row in rows], dtype=np.int64)
    y_rank = np.array([float(row["final_rank_norm"]) for row in rows], dtype=np.float32)
    y_top5 = np.array([int(bool(row["final_top5"])) for row in rows], dtype=np.int64)
    y_top10 = np.array([int(bool(row["final_top10"])) for row in rows], dtype=np.int64)
    return X, y_hazard, y_rank, y_top5, y_top10


def _preprocessor(feature_cols: list[str]) -> ColumnTransformer:
    cat_cols = [c for c in CATEGORICAL_COLUMNS if c in feature_cols]
    num_cols = [c for c in feature_cols if c not in cat_cols]
    return ColumnTransformer(
        transformers=[
            ("num", SimpleImputer(strategy="median"), num_cols),
            (
                "cat",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                cat_cols,
            ),
        ],
        remainder="drop",
    )


def _classifier(feature_cols: list[str]) -> Pipeline:
    return Pipeline(
        [
            ("prep", _preprocessor(feature_cols)),
            (
                "model",
                HistGradientBoostingClassifier(
                    max_iter=200,
                    learning_rate=0.06,
                    l2_regularization=0.01,
                    random_state=42,
                ),
            ),
        ]
    )


def _regressor(feature_cols: list[str]) -> Pipeline:
    return Pipeline(
        [
            ("prep", _preprocessor(feature_cols)),
            (
                "model",
                HistGradientBoostingRegressor(
                    max_iter=200,
                    learning_rate=0.06,
                    l2_regularization=0.01,
                    random_state=42,
                ),
            ),
        ]
    )


def _binary_metrics(model: Pipeline, X, y) -> dict[str, float]:
    pred = model.predict(X)
    out = {
        "acc": float(accuracy_score(y, pred)),
        "pos_rate": float(np.mean(y)),
    }
    if hasattr(model, "predict_proba"):
        prob = model.predict_proba(X)[:, 1]
        try:
            out["auc"] = float(roc_auc_score(y, prob))
        except ValueError:
            out["auc"] = float("nan")
        try:
            out["ap"] = float(average_precision_score(y, prob))
        except ValueError:
            out["ap"] = float("nan")
    return out


def run(args: argparse.Namespace) -> dict:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ids = list_phase_eligible_match_ids(args.map, args.mode, args.limit)
    if len(ids) < 20:
        raise SystemExit(f"not enough matches: {len(ids)}")
    train_ids, val_ids, test_ids = load_or_create_split(
        ids,
        split_json=args.split_json,
        seed=args.seed,
        val_frac=args.val_frac,
        test_frac=args.test_frac,
        metadata={
            "map": args.map,
            "mode": args.mode,
            "limit": args.limit,
            "source": "pubg_lab.survival.baselines.baselines",
        },
    )
    logger.info("split: train=%d val=%d test=%d", len(train_ids), len(val_ids), len(test_ids))

    train_rows, cols = load_phase_rows(train_ids)
    val_rows, _ = load_phase_rows(val_ids)
    test_rows, _ = load_phase_rows(test_ids)
    feature_cols = [c for c in cols if c not in EXCLUDE_COLUMNS]
    logger.info(
        "rows: train=%d val=%d test=%d features=%d",
        len(train_rows),
        len(val_rows),
        len(test_rows),
        len(feature_cols),
    )

    X_train, yh_train, yr_train, y5_train, y10_train = _matrix(train_rows, feature_cols)
    X_val, yh_val, yr_val, y5_val, y10_val = _matrix(val_rows, feature_cols)
    X_test, yh_test, yr_test, y5_test, y10_test = _matrix(test_rows, feature_cols)

    hazard = _classifier(feature_cols)
    top5 = _classifier(feature_cols)
    top10 = _classifier(feature_cols)
    rank = _regressor(feature_cols)

    logger.info("fit hazard")
    hazard.fit(X_train, yh_train)
    logger.info("fit top5")
    top5.fit(X_train, y5_train)
    logger.info("fit top10")
    top10.fit(X_train, y10_train)
    logger.info("fit rank")
    rank.fit(X_train, yr_train)

    def eval_split(name: str, X, yh, yr, y5, y10) -> dict:
        rank_pred = rank.predict(X)
        return {
            "hazard": _binary_metrics(hazard, X, yh),
            "top5": _binary_metrics(top5, X, y5),
            "top10": _binary_metrics(top10, X, y10),
            "rank_mae_norm": float(mean_absolute_error(yr, rank_pred)),
            "n_rows": len(yh),
            "name": name,
        }

    result = {
        "n_matches": len(ids),
        "splits": {
            "train": len(train_ids),
            "val": len(val_ids),
            "test": len(test_ids),
        },
        "n_features": len(feature_cols),
        "feature_cols": feature_cols,
        "val": eval_split("val", X_val, yh_val, yr_val, y5_val, y10_val),
        "test": eval_split("test", X_test, yh_test, yr_test, y5_test, y10_test),
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2, default=str))
    logger.info("wrote %s", out_path)
    logger.info("val=%s", result["val"])
    logger.info("test=%s", result["test"])
    return result


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--map", default=None)
    ap.add_argument("--mode", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--test-frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--split-json",
        default=None,
        help="Load this match split manifest, or create it if it does not exist",
    )
    ap.add_argument("--out", default="model_stgcn/checkpoints/baseline_phase_100.json")
    return ap.parse_args()


if __name__ == "__main__":
    run(parse_args())
