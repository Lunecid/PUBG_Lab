"""Same-condition tabular baselines for snapshot phase-survival models.

These baselines use the exact snapshot tensor surface consumed by the no-graph
deep models: 10s team-state features, phase ids, masks, and phase hazard
labels.  They deliberately do not consume the phase-aggregate habitat/STDB
feature views used by the stronger RQ4 representation ablation.  The purpose is
to test whether temporal deep models add value beyond tabular reductions of the
same official snapshot telemetry.
"""

from __future__ import annotations

# Support explicit direct-file execution as well as package entry points.
if not __package__:
    import sys as _pkg_sys
    from pathlib import Path as _PackagePath

    _pkg_sys.path.insert(0, str(_PackagePath(__file__).resolve().parents[2]))

import argparse
import json
import logging
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from model_stgcn.splits import load_or_create_split
from pubg_lab.data.stdb.conn import connect
from pubg_lab.data.stdb.queries import PHASE_T_MAX, list_phase_eligible_match_ids
from pubg_lab.data.stdb.snapshot_queries import SNAP_FEATURE_NAMES, SNAP_FEATURE_SPECS

logger = logging.getLogger("model_stgcn.snapshot_tabular_baseline")

CONFIGS = ("snapshot_last_hgb", "snapshot_summary_hgb")
LEARNER_ALIASES = {
    "logistic": "lr",
    "logreg": "lr",
    "random_forest": "rf",
    "randomforest": "rf",
    "hist_gradient_boosting": "hgb",
    "histgradientboosting": "hgb",
    "xgboost": "xgb",
    "lgbm": "lightgbm",
}
ALL_LEARNERS = ("lr", "rf", "hgb", "xgb", "mlp")


def _safe_float(value: Any) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return 0.0
    return out if np.isfinite(out) else 0.0


def _binary_metric(y: np.ndarray, p: np.ndarray) -> dict[str, float | int]:
    y = np.asarray(y).astype(np.int64)
    p = np.clip(np.asarray(p).astype(np.float64), 1e-7, 1.0 - 1e-7)
    pred = (p >= 0.5).astype(np.int64)
    return {
        "n": int(y.size),
        "pos_rate": float(y.mean()) if y.size else float("nan"),
        "acc_at_0_5": float(accuracy_score(y, pred)) if y.size else float("nan"),
        "bce": float(log_loss(y, p, labels=[0, 1])) if y.size else float("nan"),
        "brier": float(brier_score_loss(y, p)) if y.size else float("nan"),
        "auc": float(roc_auc_score(y, p)) if y.size and y.min() != y.max() else float("nan"),
        "ap": float(average_precision_score(y, p))
        if y.size and y.min() != y.max()
        else float("nan"),
    }


def _phase_cut_index(
    phase_of_snap: np.ndarray, node_mask_s: np.ndarray, cut_phase: int
) -> int | None:
    valid_snap = node_mask_s.any(axis=-1)
    pid = np.clip(phase_of_snap.astype(np.int64), 1, PHASE_T_MAX)
    eligible = valid_snap & (pid <= cut_phase)
    idx = np.flatnonzero(eligible)
    return int(idx[-1]) if idx.size else None


def _base_row(item: dict, *, team_idx: int, cut_phase: int, cut_idx: int) -> dict[str, Any]:
    team_id = int(item["team_ids"][team_idx])
    snapshot_ts = item["snapshot_ts"].numpy()
    cut_ts = int(snapshot_ts[cut_idx]) if cut_idx < len(snapshot_ts) else -1
    return {
        "match_id": str(item["match_id"]),
        "team_id": team_id,
        "team_idx": int(team_idx),
        "phase_idx": int(cut_phase),
        "cut_phase": int(cut_phase),
        "cut_snapshot_idx": int(cut_idx),
        "cut_snapshot_t_norm": float(max(cut_ts, 0) / 3600.0),
        "hazard_next": int(item["y_hazard"][cut_phase - 1, team_idx].item()),
        "final_rank_norm": float(item["y_rank"][team_idx].item()),
        "final_top5": int(item["y_topk"][team_idx, 0].item()),
        "final_top10": int(item["y_topk"][team_idx, 1].item()),
    }


def _add_last_features(
    row: dict[str, Any], X_s: np.ndarray, *, cut_idx: int, team_idx: int
) -> None:
    x_last = X_s[cut_idx, team_idx]
    for feature_idx, name in enumerate(SNAP_FEATURE_NAMES):
        row[f"last_{name}"] = _safe_float(x_last[feature_idx])


def _add_summary_features(
    row: dict[str, Any],
    X_s: np.ndarray,
    phase_of_snap: np.ndarray,
    node_mask_s: np.ndarray,
    *,
    cut_phase: int,
    cut_idx: int,
    team_idx: int,
) -> None:
    pid = np.clip(phase_of_snap.astype(np.int64), 1, PHASE_T_MAX)
    t = np.arange(X_s.shape[0])
    observed_until_cut = (t <= cut_idx) & node_mask_s[:, team_idx] & (pid <= cut_phase)
    observed_current_phase = observed_until_cut & (pid == cut_phase)
    row["obs_count_until_cut"] = int(observed_until_cut.sum())
    row["obs_count_current_phase"] = int(observed_current_phase.sum())
    row["has_current_phase_snapshot"] = int(observed_current_phase.any())
    row["cut_phase_norm"] = float(cut_phase / max(PHASE_T_MAX - 1, 1))

    if not observed_until_cut.any():
        seq = np.zeros((1, X_s.shape[-1]), dtype=np.float32)
    else:
        seq = X_s[observed_until_cut, team_idx]
    first = seq[0]
    last = seq[-1]
    mean = seq.mean(axis=0)
    mn = seq.min(axis=0)
    mx = seq.max(axis=0)
    std = seq.std(axis=0)
    delta = last - first

    if observed_current_phase.any():
        cur_seq = X_s[observed_current_phase, team_idx]
    else:
        cur_seq = np.zeros((1, X_s.shape[-1]), dtype=np.float32)
    cur_mean = cur_seq.mean(axis=0)

    for feature_idx, name in enumerate(SNAP_FEATURE_NAMES):
        row[f"mean_{name}"] = _safe_float(mean[feature_idx])
        row[f"min_{name}"] = _safe_float(mn[feature_idx])
        row[f"max_{name}"] = _safe_float(mx[feature_idx])
        row[f"std_{name}"] = _safe_float(std[feature_idx])
        row[f"delta_{name}"] = _safe_float(delta[feature_idx])
        row[f"current_mean_{name}"] = _safe_float(cur_mean[feature_idx])


def _rows_for_item(item: dict, configs: set[str]) -> list[dict[str, Any]]:
    X_s = item["X_s"].numpy().astype(np.float32, copy=False)
    phase_of_snap = item["phase_of_snap"].numpy()
    node_mask_s = item["node_mask_s"].numpy().astype(bool, copy=False)
    target_mask = item["target_mask"].numpy().astype(bool, copy=False)
    rows: list[dict[str, Any]] = []
    for cut_phase in range(1, PHASE_T_MAX):
        cut_idx = _phase_cut_index(phase_of_snap, node_mask_s, cut_phase)
        if cut_idx is None:
            continue
        for team_idx, team_id in enumerate(item["team_ids"]):
            if int(team_id) == 0 or not bool(target_mask[cut_phase - 1, team_idx]):
                continue
            row = _base_row(item, team_idx=team_idx, cut_phase=cut_phase, cut_idx=cut_idx)
            _add_last_features(row, X_s, cut_idx=cut_idx, team_idx=team_idx)
            if "snapshot_summary_hgb" in configs:
                _add_summary_features(
                    row,
                    X_s,
                    phase_of_snap,
                    node_mask_s,
                    cut_phase=cut_phase,
                    cut_idx=cut_idx,
                    team_idx=team_idx,
                )
            rows.append(row)
    return rows


def _sql_scaled_expr(alias: str, column: str, scale: float) -> str:
    if column == "inside_blue":
        return f"(CASE WHEN {alias}.{column} THEN 1.0 ELSE 0.0 END)"
    return f"(COALESCE({alias}.{column}, 0)::float8 / {float(scale)})"


def _snapshot_tabular_sql() -> str:
    snap_selects: list[str] = []
    agg_selects: list[str] = []
    outer_selects: list[str] = [
        "l.match_id",
        "l.team_id",
        "l.team_idx",
        "l.cut_phase AS phase_idx",
        "l.cut_phase",
        "COALESCE(last_s.cut_snapshot_idx, 0) AS cut_snapshot_idx",
        "COALESCE(last_s.cut_snapshot_t_norm, 0.0) AS cut_snapshot_t_norm",
        "l.hazard_next",
        "l.final_rank_norm",
        "l.final_top5::int AS final_top5",
        "l.final_top10::int AS final_top10",
        "COALESCE(agg.obs_count_until_cut, 0) AS obs_count_until_cut",
        "COALESCE(agg.obs_count_current_phase, 0) AS obs_count_current_phase",
        "(COALESCE(agg.obs_count_current_phase, 0) > 0)::int AS has_current_phase_snapshot",
        "l.cut_phase::float8 / 7.0 AS cut_phase_norm",
    ]
    for name, column, scale in SNAP_FEATURE_SPECS:
        scaled = _sql_scaled_expr("sf", column, scale)
        snap_selects.append(f"{scaled} AS x_{name}")
        agg_selects.extend(
            [
                f"AVG(x_{name}) AS mean_{name}",
                f"MIN(x_{name}) AS min_{name}",
                f"MAX(x_{name}) AS max_{name}",
                f"COALESCE(STDDEV_POP(x_{name}), 0.0) AS std_{name}",
                f"AVG(x_{name}) FILTER (WHERE in_current_phase) AS current_mean_{name}",
            ]
        )
        outer_selects.extend(
            [
                f"COALESCE(last_s.last_{name}, 0.0) AS last_{name}",
                f"COALESCE(agg.mean_{name}, 0.0) AS mean_{name}",
                f"COALESCE(agg.min_{name}, 0.0) AS min_{name}",
                f"COALESCE(agg.max_{name}, 0.0) AS max_{name}",
                f"COALESCE(agg.std_{name}, 0.0) AS std_{name}",
                f"(COALESCE(last_s.last_{name}, 0.0) - COALESCE(first_s.first_{name}, 0.0)) AS delta_{name}",
                f"COALESCE(agg.current_mean_{name}, 0.0) AS current_mean_{name}",
            ]
        )

    return f"""
    WITH base_labels AS (
        SELECT
            -- deterministic row ids / row order (2026-10-05 HGB determinism fix):
            ROW_NUMBER() OVER (ORDER BY l.match_id, l.team_id, l.cut_phase) AS rid,
            l.match_id,
            l.team_id,
            DENSE_RANK() OVER (PARTITION BY l.match_id ORDER BY l.team_id) - 1 AS team_idx,
            l.cut_phase,
            l.hazard_next,
            l.final_rank_norm,
            l.final_top5,
            l.final_top10,
            b.phase_start_s,
            b.phase_end_s
          FROM stdb.team_phase_labels l
          JOIN stdb.match_phase_global b
            ON b.match_id = l.match_id
           AND b.phase_idx = l.cut_phase
         WHERE l.match_id = ANY(%s)
           AND l.target_defined
           AND l.cut_phase BETWEEN 1 AND 7
           AND EXISTS (
               SELECT 1
                 FROM stdb.team_snapshot_features sf
                WHERE sf.match_id = l.match_id
                  AND sf.team_id = l.team_id
                  AND sf.snapshot_t <= b.phase_end_s
           )
    ),
    snap AS (
        SELECT
            l.rid,
            sf.snapshot_t,
            ROW_NUMBER() OVER (PARTITION BY l.rid ORDER BY sf.snapshot_t) - 1 AS snapshot_idx,
            GREATEST(sf.snapshot_t, 0)::float8 / 3600.0 AS snapshot_t_norm,
            (sf.snapshot_t >= l.phase_start_s AND sf.snapshot_t <= l.phase_end_s) AS in_current_phase,
            {", ".join(snap_selects)}
          FROM base_labels l
          JOIN stdb.team_snapshot_features sf
            ON sf.match_id = l.match_id
           AND sf.team_id = l.team_id
           AND sf.snapshot_t <= l.phase_end_s
    ),
    agg AS (
        SELECT
            rid,
            COUNT(*) AS obs_count_until_cut,
            COUNT(*) FILTER (WHERE in_current_phase) AS obs_count_current_phase,
            {", ".join(agg_selects)}
          FROM snap
         GROUP BY rid
    ),
    last_s AS (
        SELECT DISTINCT ON (rid)
            rid,
            snapshot_idx AS cut_snapshot_idx,
            snapshot_t_norm AS cut_snapshot_t_norm,
            {", ".join(f"x_{name} AS last_{name}" for name in SNAP_FEATURE_NAMES)}
          FROM snap
         ORDER BY rid, snapshot_t DESC
    ),
    first_s AS (
        SELECT DISTINCT ON (rid)
            rid,
            {", ".join(f"x_{name} AS first_{name}" for name in SNAP_FEATURE_NAMES)}
          FROM snap
         ORDER BY rid, snapshot_t ASC
    )
    SELECT
        {", ".join(outer_selects)}
      FROM base_labels l
      JOIN agg ON agg.rid = l.rid
      JOIN last_s ON last_s.rid = l.rid
      JOIN first_s ON first_s.rid = l.rid
     ORDER BY l.match_id, l.team_id, l.cut_phase
    """


def load_snapshot_tabular_rows(match_ids: list[str], configs: set[str]) -> pd.DataFrame:
    del configs
    if not match_ids:
        return pd.DataFrame()
    with connect() as conn:
        with conn.cursor() as cur:
            cur.execute(_snapshot_tabular_sql(), (match_ids,))
            cols = [desc[0] for desc in cur.description]
            rows = [dict(zip(cols, row)) for row in cur.fetchall()]
    df = pd.DataFrame(rows)
    if not df.empty:
        df["phase_idx"] = df["phase_idx"].astype(int)
        df["cut_phase"] = df["cut_phase"].astype(int)
        df["team_id"] = df["team_id"].astype(int)
        df["team_idx"] = df["team_idx"].astype(int)
        df["hazard_next"] = df["hazard_next"].astype(int)
    return df


def _feature_cols(df: pd.DataFrame, config: str) -> list[str]:
    common = ["phase_idx", "cut_phase", "cut_snapshot_idx", "cut_snapshot_t_norm"]
    last = [f"last_{name}" for name in SNAP_FEATURE_NAMES]
    if config == "snapshot_last_hgb":
        return [c for c in common + last if c in df.columns]
    if config == "snapshot_summary_hgb":
        summary_prefixes = ("mean_", "min_", "max_", "std_", "delta_", "current_mean_")
        summary = [
            c
            for c in df.columns
            if c.startswith(summary_prefixes)
            or c
            in {
                "obs_count_until_cut",
                "obs_count_current_phase",
                "has_current_phase_snapshot",
                "cut_phase_norm",
            }
        ]
        return [c for c in common + last + sorted(summary) if c in df.columns]
    raise ValueError(f"unknown config={config}")


def _normalise_learner(name: str) -> str:
    key = name.strip().lower().replace("-", "_")
    return LEARNER_ALIASES.get(key, key)


def _resolve_learners(spec: str) -> tuple[list[str], list[str]]:
    requested = [_normalise_learner(s) for s in spec.split(",") if s.strip()]
    if not requested or requested == ["default"]:
        requested = ["hgb"]
    if "all" in requested:
        requested = list(ALL_LEARNERS)
    out: list[str] = []
    unavailable: list[str] = []
    for learner in requested:
        if learner == "lightgbm":
            unavailable.append(learner)
            continue
        if learner not in ALL_LEARNERS:
            raise ValueError(f"unknown learner={learner}")
        if learner == "xgb":
            try:
                import xgboost  # noqa: F401
            except Exception:
                unavailable.append(learner)
                continue
        if learner not in out:
            out.append(learner)
    if not out:
        raise ValueError(f"no available learners from spec={spec!r}; unavailable={unavailable}")
    return out, unavailable


def _finite_numeric_array(values: Any) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    return np.where(np.isfinite(arr), arr, np.nan)


def _model_variant(config: str, learner: str) -> str:
    if learner == "hgb":
        return config
    base = config[:-4] if config.endswith("_hgb") else config
    return f"{base}_{learner}"


def _classifier(args: argparse.Namespace, *, learner: str = "hgb") -> Pipeline:
    learner = _normalise_learner(learner)
    steps: list[tuple[str, object]] = [
        ("finite", FunctionTransformer(_finite_numeric_array, validate=False)),
        ("impute", SimpleImputer(strategy="median")),
    ]
    if learner in {"lr", "mlp"}:
        steps.append(("scale", StandardScaler(with_mean=True)))
    if learner == "hgb":
        model = HistGradientBoostingClassifier(
            max_iter=args.max_iter,
            learning_rate=args.learning_rate,
            l2_regularization=args.l2_regularization,
            # 'auto' switches early stopping on for n > 10_000 rows and then holds out a
            # row-order-dependent 10% validation split: the source of the run-to-run HGB
            # drift (up to 0.0048 AUC per cohort).  Off => fixed max_iter, deterministic.
            early_stopping=False,
            random_state=args.seed,
        )
    elif learner == "lr":
        model = LogisticRegression(
            max_iter=args.max_iter,
            solver="liblinear",
            random_state=args.seed,
        )
    elif learner == "rf":
        workers = None if args.n_jobs == 0 else args.n_jobs
        model = RandomForestClassifier(
            n_estimators=args.n_estimators,
            min_samples_leaf=2,
            random_state=args.seed,
            n_jobs=workers,
        )
    elif learner == "xgb":
        from xgboost import XGBClassifier

        workers = None if args.n_jobs == 0 else args.n_jobs
        model = XGBClassifier(
            n_estimators=args.n_estimators,
            max_depth=4,
            learning_rate=args.learning_rate,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method="hist",
            random_state=args.seed,
            n_jobs=workers,
        )
    elif learner == "mlp":
        model = MLPClassifier(
            hidden_layer_sizes=(64, 32),
            alpha=1e-4,
            batch_size=256,
            early_stopping=True,
            max_iter=args.max_iter,
            random_state=args.seed,
        )
    else:
        raise ValueError(f"unknown learner={learner}")
    steps.append(("model", model))
    return Pipeline(steps)


def _fit_predict(
    *,
    config: str,
    learner: str,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    feature_cols = _feature_cols(train_df, config)
    if not feature_cols:
        raise RuntimeError(f"{config} has no feature columns")
    model = _classifier(args, learner=learner)
    model_variant = _model_variant(config, learner)
    y_train = train_df["hazard_next"].astype(int).to_numpy()
    y_test = test_df["hazard_next"].astype(int).to_numpy()
    logger.info(
        "fit %-24s learner=%-4s rows=%d test=%d features=%d",
        config,
        learner,
        len(train_df),
        len(test_df),
        len(feature_cols),
    )
    with warnings.catch_warnings():
        if learner in {"lr", "mlp"}:
            warnings.filterwarnings("ignore", category=RuntimeWarning, module=r"sklearn\..*")
        model.fit(train_df[feature_cols], y_train)
        prob = model.predict_proba(test_df[feature_cols])[:, 1]
    if not np.isfinite(prob).all():
        raise RuntimeError(f"{model_variant} produced non-finite probabilities")
    pred = test_df[
        [
            "match_id",
            "team_id",
            "team_idx",
            "phase_idx",
            "cut_phase",
            "final_rank_norm",
            "final_top5",
            "final_top10",
        ]
    ].copy()
    pred["model_variant"] = model_variant
    pred["hazard_true"] = y_test
    pred["hazard_prob"] = prob
    metrics = {
        "model_variant": model_variant,
        "feature_config": config,
        "learner": learner,
        "n_features": len(feature_cols),
        "feature_cols": feature_cols,
        **_binary_metric(y_test, prob),
    }
    return pred[
        [
            "model_variant",
            "match_id",
            "team_id",
            "team_idx",
            "phase_idx",
            "cut_phase",
            "hazard_true",
            "hazard_prob",
            "final_rank_norm",
            "final_top5",
            "final_top10",
        ]
    ], metrics


def run(args: argparse.Namespace) -> dict[str, Any]:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    configs = {c.strip() for c in args.configs.split(",") if c.strip()}
    unknown = configs - set(CONFIGS)
    if unknown:
        raise SystemExit(f"unknown configs: {sorted(unknown)}")
    learners, unavailable_learners = _resolve_learners(args.learners)
    if unavailable_learners:
        logger.warning("skipping unavailable learners: %s", unavailable_learners)
    ids = list_phase_eligible_match_ids(args.map, args.mode, args.limit)
    train_ids, val_ids, test_ids = load_or_create_split(
        ids,
        split_json=args.split_json,
        seed=args.seed,
        val_frac=args.val_frac,
        test_frac=args.test_frac,
        metadata={
            "source": "model_stgcn.snapshot_tabular_baseline",
            "map": args.map,
            "mode": args.mode,
            "limit": args.limit,
            "input_resolution": "10s_snapshot_tabular_reduction",
        },
    )
    logger.info(
        "split total=%d train=%d val=%d test=%d",
        len(ids),
        len(train_ids),
        len(val_ids),
        len(test_ids),
    )
    train_df = load_snapshot_tabular_rows(train_ids, configs)
    test_df = load_snapshot_tabular_rows(test_ids, configs)
    if train_df.empty or test_df.empty:
        raise SystemExit(f"empty train/test rows: train={len(train_df)} test={len(test_df)}")
    logger.info("rows train=%d test=%d", len(train_df), len(test_df))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pred_frames: list[pd.DataFrame] = []
    metrics_rows: list[dict[str, Any]] = []
    feature_manifest: dict[str, list[str]] = {}
    for config in sorted(configs):
        for learner in learners:
            preds, metrics = _fit_predict(
                config=config,
                learner=learner,
                train_df=train_df,
                test_df=test_df,
                args=args,
            )
            pred_frames.append(preds)
            metrics_rows.append(metrics)
            feature_manifest[str(metrics["model_variant"])] = metrics.pop("feature_cols")

    predictions = pd.concat(pred_frames, ignore_index=True)
    metrics_df = pd.DataFrame(metrics_rows)
    phase_rows: list[dict[str, Any]] = []
    for (model_variant, cut_phase), group in predictions.groupby(["model_variant", "cut_phase"]):
        phase_rows.append(
            {
                "model_variant": model_variant,
                "cut_phase": int(cut_phase),
                **_binary_metric(
                    group["hazard_true"].to_numpy(),
                    group["hazard_prob"].to_numpy(),
                ),
            }
        )
    phase_df = pd.DataFrame(phase_rows)

    pred_path = out_dir / "snapshot_tabular_predictions.csv"
    metrics_path = out_dir / "snapshot_tabular_metrics.csv"
    phase_path = out_dir / "snapshot_tabular_phase_metrics.csv"
    features_path = out_dir / "snapshot_tabular_features.json"
    summary_path = out_dir / "snapshot_tabular_summary.json"

    predictions.to_csv(pred_path, index=False)
    metrics_df.to_csv(metrics_path, index=False)
    phase_df.to_csv(phase_path, index=False)
    features_path.write_text(json.dumps(feature_manifest, indent=2), encoding="utf-8")
    summary = {
        "split": {
            "total": len(ids),
            "train": len(train_ids),
            "val": len(val_ids),
            "test": len(test_ids),
            "train_rows": len(train_df),
            "test_rows": len(test_df),
        },
        "configs": sorted(configs),
        "learners": learners,
        "unavailable_learners": unavailable_learners,
        "feature_surface": "official_10s_snapshot_team_state_only",
        "outputs": {
            "predictions": str(pred_path),
            "metrics": str(metrics_path),
            "phase_metrics": str(phase_path),
            "feature_manifest": str(features_path),
        },
        "metrics": metrics_df.to_dict(orient="records"),
    }
    summary_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    logger.info("wrote %s", summary_path)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--map", default=None)
    parser.add_argument("--mode", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--split-json", default=None)
    parser.add_argument("--out-dir", default="model_stgcn/checkpoints/snapshot_tabular_baseline")
    parser.add_argument("--configs", default=",".join(CONFIGS))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-frac", type=float, default=0.1)
    parser.add_argument("--test-frac", type=float, default=0.1)
    parser.add_argument("--max-iter", type=int, default=200)
    parser.add_argument("--n-estimators", type=int, default=300)
    parser.add_argument("--n-jobs", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=0.06)
    parser.add_argument("--l2-regularization", type=float, default=0.01)
    parser.add_argument(
        "--learners", default="hgb", help="Comma-separated learners or 'all': lr,rf,hgb,xgb,mlp"
    )
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
