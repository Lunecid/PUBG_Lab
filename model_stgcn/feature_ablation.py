"""Feature-group ablation for phase-level PUBG hazard prediction.

This script answers the novelty question directly:

* How far can a prior-paper style FightScore/ActiveScore baseline go?
* How much do conventional telemetry features help?
* How much do STDB trajectory/exposure features add on top?
* How much do F6 Tier-1 network role features add on top of those
  (FEATURE_DESIGN_V2 §7, table stdb.team_phase_network_features)?
* How much do F7 Tier-1 habitat-pressure features add
  (FEATURE_DESIGN_V2 §8, table stdb.team_phase_habitat_features)?

The target is phase-level hazard_next in stdb.v_team_phase_feature_long.
F6/F7 columns come from the V2 manifest (single source of truth) and are
merged on (match_id, team_id, phase_idx); if a family table is absent or
empty its configs are skipped so the script still runs on pre-V2 databases.
All comparisons use the existing match-level split and can optionally align
test rows to the DynamicSTGNN analysis key set.
"""

from __future__ import annotations

import argparse
import json
import logging
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import rankdata, wilcoxon
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, average_precision_score, roc_auc_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.preprocessing import OneHotEncoder

from model_stgcn.baselines import CATEGORICAL_COLUMNS, EXCLUDE_COLUMNS
from model_stgcn.splits import load_or_create_split
from model_stgcn.v3.cohorts import resolve_map_mode
from stdb.conn import connect
from stdb.queries import list_phase_feature_match_ids

logger = logging.getLogger("model_stgcn.feature_ablation")

KEY_COLS = ["match_id", "team_id", "phase_idx"]

V2_MANIFEST_PATH = Path("model_stgcn/feature_manifests/pubg_representation_v2.json")
NETWORK_FAMILY = "f6_network_phase_role_tier1"
NETWORK_TABLE = "stdb.team_phase_network_features"
HABITAT_FAMILY = "f7_habitat_pressure_tier1"
HABITAT_TABLE = "stdb.team_phase_habitat_features"
DEFAULT_FEATURE_VIEW = "stdb.v_team_phase_feature_long"
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
DEFAULT_LEARNERS = ("hgb",)
ALL_LEARNERS = ("lr", "rf", "hgb", "xgb", "mlp")


AGGRESSIVENESS_COLS = [
    "damage_events_dealt",
    "kills_dealt",
    "damage_dealt",
    "groggy_dealt",
    "distinct_opponents_damaged",
]

ACTIVENESS_COLS = [
    "path_length_m",
    "avg_speed_mps",
    "avg_enemy_density_100m",
    "pickup_count",
    "unique_item_count",
    "carepackage_pickup_count",
]

GLOBAL_COLS = [
    "phase_idx",
    "map_name",
    "game_mode",
    "match_team_size",
    "phase_start_s",
    "phase_end_s",
    "phase_duration_s",
    "avg_alive_teams",
    "avg_alive_players",
    "safe_radius_start_m",
    "safe_radius_end_m",
    "safe_radius_delta_m",
    "safe_center_shift_m",
    "poison_radius_start_m",
    "poison_radius_end_m",
    "zone_gap_start_m",
    "zone_gap_end_m",
]

STATE_COLS = [
    "team_size",
    "alive_players_start",
    "alive_players_end",
    "alive_players_min",
    "alive_players_avg",
    "mean_health_avg",
    "min_health_min",
    "health_slope",
    "team_spread_avg_m",
    "team_spread_max_m",
    "vehicle_ratio",
    "vehicle_speed_mean",
    "vehicle_speed_max",
]

TRAJECTORY_COLS = [
    "path_length_m",
    "displacement_m",
    "sinuosity",
    "avg_speed_mps",
    "rotation_timing",
    "safe_alignment",
    "zone_entry_time",
    "outside_safe_duration_s",
    "boundary_distance_slope",
]

EXPOSURE_COLS = [
    "mean_dist_to_boundary_m",
    "min_dist_to_boundary_m",
    "outside_safe_ratio",
    "avg_enemy_density_100m",
    "min_nearest_enemy_m",
    "path_near_enemy_ratio",
    "recent_damage_taken",
    "combat_zone_pass_count",
]

COMBAT_COLS = [
    "damage_dealt",
    "damage_taken",
    "damage_events_dealt",
    "damage_events_taken",
    "groggy_dealt",
    "groggy_taken",
    "kills_dealt",
    "kills_taken",
    "distinct_opponents_damaged",
    "distinct_opponents_damaged_by",
    "combat_partner_count",
    "combat_initiated",
    "nearby_enemy_combat_events_200m",
    "third_party_exposure_count",
]

LOOT_COLS = [
    "pickup_count",
    "equip_count",
    "use_count",
    "unique_item_count",
    "weapon_pickup_count",
    "weapon_equip_count",
    "ammo_pickup_count",
    "armor_pickup_count",
    "throwable_pickup_count",
    "heal_use_count",
    "boost_use_count",
    "carepackage_pickup_count",
    "lootbox_pickup_count",
]

PRIOR_SCORE_COLS = [
    "phase_idx",
    "fight_score",
    "active_score",
    "strategy_type",
    "type_survival_rate",
]


def _dedupe(cols: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for col in cols:
        if col not in seen:
            out.append(col)
            seen.add(col)
    return out


def _rows_to_df(rows: list[dict[str, Any]]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    if not df.empty:
        df["phase_idx"] = df["phase_idx"].astype(int)
        df["team_id"] = df["team_id"].astype(int)
        df["hazard_next"] = df["hazard_next"].astype(int)
    return df


def _load_manifest(path: Path = V2_MANIFEST_PATH) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _split_db_object(name: str) -> tuple[str, str]:
    parts = name.split(".", 1)
    if len(parts) != 2 or not parts[0] or not parts[1]:
        raise ValueError(f"database object must be schema.name, got {name!r}")
    return parts[0], parts[1]


def load_phase_rows_from_view(match_ids: list[str], view_name: str) -> tuple[list[dict], list[str]]:
    if not match_ids:
        return [], []
    schema, rel = _split_db_object(view_name)
    quoted_view = f'"{schema}"."{rel}"'
    with connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                f"""
                SELECT *
                  FROM {quoted_view}
                 WHERE match_id = ANY(%s)
                   AND target_defined
                   AND phase_idx BETWEEN 1 AND 7
                 ORDER BY match_id, team_id, phase_idx
                """,
                (match_ids,),
            )
            cols = [d[0] for d in cur.description]
            rows = [dict(zip(cols, r)) for r in cur.fetchall()]
    return rows, cols


def _safe_metric_auc(y: np.ndarray, p: np.ndarray) -> float:
    try:
        return float(roc_auc_score(y, p))
    except ValueError:
        return float("nan")


def _safe_metric_ap(y: np.ndarray, p: np.ndarray) -> float:
    try:
        return float(average_precision_score(y, p))
    except ValueError:
        return float("nan")


def _bce(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    if y_true.size == 0:
        return float("nan")
    p = np.clip(y_prob.astype(float), 1e-7, 1.0 - 1e-7)
    y = y_true.astype(float)
    return float(-(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)).mean())


def _binary_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> dict[str, float | int]:
    y_true = y_true.astype(int)
    y_pred = (y_prob >= 0.5).astype(int)
    return {
        "n": int(y_true.size),
        "pos_rate": float(np.mean(y_true)) if y_true.size else float("nan"),
        "acc": float(accuracy_score(y_true, y_pred)) if y_true.size else float("nan"),
        "bce": _bce(y_true, y_prob),
        "auc": _safe_metric_auc(y_true, y_prob),
        "ap": _safe_metric_ap(y_true, y_prob),
    }


def _metric_rows(df: pd.DataFrame, group_cols: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for key, g in df.groupby(group_cols, dropna=False):
        if not isinstance(key, tuple):
            key = (key,)
        row = dict(zip(group_cols, key))
        row.update(_binary_metrics(g["y"].to_numpy(), g["prob"].to_numpy()))
        rows.append(row)
    return rows


def _paired_metric_by_match(preds: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (model, match_id), g in preds.groupby(["model", "match_id"]):
        rows.append(
            {
                "model": model,
                "match_id": match_id,
                **_binary_metrics(g["y"].to_numpy(), g["prob"].to_numpy()),
            }
        )
    return pd.DataFrame(rows)


def _paired_test(
    per_match: pd.DataFrame,
    *,
    left_model: str,
    right_model: str,
    metric: str,
    larger_is_better: bool,
) -> dict[str, Any]:
    left = per_match[per_match["model"] == left_model][["match_id", metric]].rename(columns={metric: "left"})
    right = per_match[per_match["model"] == right_model][["match_id", metric]].rename(columns={metric: "right"})
    joined = left.merge(right, on="match_id", how="inner").dropna()
    if joined.empty:
        return {
            "left_model": left_model,
            "right_model": right_model,
            "metric": metric,
            "larger_is_better": larger_is_better,
            "n": 0,
            "left_mean": float("nan"),
            "right_mean": float("nan"),
            "median_improvement": float("nan"),
            "mean_improvement": float("nan"),
            "p_value_greater": float("nan"),
            "rank_biserial": float("nan"),
        }

    improvement = (
        joined["left"].to_numpy() - joined["right"].to_numpy()
        if larger_is_better
        else joined["right"].to_numpy() - joined["left"].to_numpy()
    )
    improvement = improvement[np.isfinite(improvement)]
    nonzero = improvement[improvement != 0]
    if nonzero.size == 0:
        p_value = 1.0
        rank_biserial = 0.0
    else:
        stat = wilcoxon(nonzero, alternative="greater", zero_method="wilcox", mode="auto")
        p_value = float(stat.pvalue)
        ranks = rankdata(np.abs(nonzero))
        pos_sum = float(ranks[nonzero > 0].sum())
        neg_sum = float(ranks[nonzero < 0].sum())
        total = pos_sum + neg_sum
        rank_biserial = (pos_sum - neg_sum) / total if total else 0.0

    return {
        "left_model": left_model,
        "right_model": right_model,
        "metric": metric,
        "larger_is_better": larger_is_better,
        "n": int(improvement.size),
        "left_mean": float(joined["left"].mean()),
        "right_mean": float(joined["right"].mean()),
        "median_improvement": float(np.median(improvement)) if improvement.size else float("nan"),
        "mean_improvement": float(np.mean(improvement)) if improvement.size else float("nan"),
        "p_value_greater": p_value,
        "rank_biserial": float(rank_biserial),
    }


def _available(cols: list[str], df: pd.DataFrame) -> list[str]:
    return [c for c in _dedupe(cols) if c in df.columns]


def _finite_numeric_array(values: Any) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    return np.where(np.isfinite(arr), arr, np.nan)


def _preprocessor(
    feature_cols: list[str],
    categorical_cols: list[str] | None = None,
    *,
    scale_numeric: bool = False,
) -> ColumnTransformer:
    categorical_cols = categorical_cols or CATEGORICAL_COLUMNS + ["strategy_type"]
    cat_cols = [c for c in categorical_cols if c in feature_cols]
    num_cols = [c for c in feature_cols if c not in cat_cols]
    finite_step = ("finite", FunctionTransformer(_finite_numeric_array, validate=False))
    if scale_numeric:
        num_transformer = Pipeline(
            [
                finite_step,
                ("impute", SimpleImputer(strategy="median")),
                ("scale", StandardScaler(with_mean=True)),
            ]
        )
    else:
        num_transformer = Pipeline(
            [
                finite_step,
                ("impute", SimpleImputer(strategy="median")),
            ]
        )
    return ColumnTransformer(
        transformers=[
            ("num", num_transformer, num_cols),
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


def _normalise_learner(name: str) -> str:
    key = name.strip().lower().replace("-", "_")
    return LEARNER_ALIASES.get(key, key)


def _resolve_learners(spec: str) -> tuple[list[str], list[str]]:
    requested = [_normalise_learner(s) for s in spec.split(",") if s.strip()]
    if not requested or requested == ["default"]:
        requested = list(DEFAULT_LEARNERS)
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


def _model_name(config: str, learner: str, *, suffix: bool) -> str:
    return f"{config}__{learner}" if suffix else config


def _classifier(
    feature_cols: list[str],
    categorical_cols: list[str] | None = None,
    *,
    learner: str = "hgb",
    seed: int = 42,
    max_iter: int = 200,
    n_estimators: int = 300,
    n_jobs: int = 0,
) -> Pipeline:
    learner = _normalise_learner(learner)
    scale_numeric = learner in {"lr", "mlp"}
    if learner == "hgb":
        model = HistGradientBoostingClassifier(
            max_iter=max_iter,
            learning_rate=0.06,
            l2_regularization=0.01,
            # see snapshot_tabular_baseline.py: 'auto' early stopping (n > 10_000) holds out a
            # row-order-dependent validation split; off => deterministic fixed-iteration fit.
            early_stopping=False,
            random_state=seed,
        )
    elif learner == "lr":
        model = LogisticRegression(
            max_iter=max_iter,
            solver="liblinear",
            random_state=seed,
        )
    elif learner == "rf":
        workers = None if n_jobs == 0 else n_jobs
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            min_samples_leaf=2,
            random_state=seed,
            n_jobs=workers,
        )
    elif learner == "xgb":
        from xgboost import XGBClassifier

        workers = None if n_jobs == 0 else n_jobs
        model = XGBClassifier(
            n_estimators=n_estimators,
            max_depth=4,
            learning_rate=0.06,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method="hist",
            random_state=seed,
            n_jobs=workers,
        )
    elif learner == "mlp":
        model = MLPClassifier(
            hidden_layer_sizes=(64, 32),
            alpha=1e-4,
            batch_size=256,
            early_stopping=True,
            max_iter=max_iter,
            random_state=seed,
        )
    else:
        raise ValueError(f"unknown learner={learner}")
    return Pipeline(
        [
            ("prep", _preprocessor(feature_cols, categorical_cols, scale_numeric=scale_numeric)),
            ("model", model),
        ]
    )


def _robust_stats(df: pd.DataFrame, cols: list[str]) -> dict[str, tuple[float, float]]:
    stats: dict[str, tuple[float, float]] = {}
    for col in cols:
        values = pd.to_numeric(df[col], errors="coerce")
        median = float(values.median()) if values.notna().any() else 0.0
        q75 = float(values.quantile(0.75)) if values.notna().any() else 1.0
        q25 = float(values.quantile(0.25)) if values.notna().any() else 0.0
        iqr = q75 - q25
        if not np.isfinite(iqr) or iqr <= 1e-6:
            iqr = 1.0
        stats[col] = (median, iqr)
    return stats


def _scaled_sum(df: pd.DataFrame, cols: list[str], stats: dict[str, tuple[float, float]]) -> pd.Series:
    total = pd.Series(np.zeros(len(df), dtype=float), index=df.index)
    for col in cols:
        median, iqr = stats[col]
        values = pd.to_numeric(df[col], errors="coerce").fillna(median)
        total += (values - median) / iqr
    return total


def fit_prior_transform(train_df: pd.DataFrame) -> dict[str, Any]:
    prior_cols = _available(AGGRESSIVENESS_COLS + ACTIVENESS_COLS, train_df)
    stats = _robust_stats(train_df, prior_cols)
    tmp = train_df.copy()
    tmp["fight_score"] = _scaled_sum(tmp, _available(AGGRESSIVENESS_COLS, tmp), stats)
    tmp["active_score"] = _scaled_sum(tmp, _available(ACTIVENESS_COLS, tmp), stats)
    medians = (
        tmp.groupby("phase_idx")[["fight_score", "active_score"]]
        .median()
        .rename(columns={"fight_score": "fight_median", "active_score": "active_median"})
    )
    global_fight = float(tmp["fight_score"].median())
    global_active = float(tmp["active_score"].median())

    tmp = add_prior_features(
        train_df,
        {
            "stats": stats,
            "phase_medians": medians,
            "global_fight_median": global_fight,
            "global_active_median": global_active,
            "survival_rate": {},
            "global_survival_rate": float(1.0 - train_df["hazard_next"].mean()),
        },
        with_survival_rate=False,
    )
    rates = (
        tmp.groupby(["phase_idx", "strategy_type"])["hazard_next"]
        .mean()
        .rename("hazard_rate")
        .reset_index()
    )
    survival_rate = {
        (int(row.phase_idx), str(row.strategy_type)): float(1.0 - row.hazard_rate)
        for row in rates.itertuples(index=False)
    }
    return {
        "stats": stats,
        "phase_medians": medians,
        "global_fight_median": global_fight,
        "global_active_median": global_active,
        "survival_rate": survival_rate,
        "global_survival_rate": float(1.0 - train_df["hazard_next"].mean()),
    }


def add_prior_features(df: pd.DataFrame, transform: dict[str, Any], *, with_survival_rate: bool = True) -> pd.DataFrame:
    out = df.copy()
    stats: dict[str, tuple[float, float]] = transform["stats"]
    out["fight_score"] = _scaled_sum(out, _available(AGGRESSIVENESS_COLS, out), stats)
    out["active_score"] = _scaled_sum(out, _available(ACTIVENESS_COLS, out), stats)

    medians: pd.DataFrame = transform["phase_medians"]
    phase_medians = medians.reindex(out["phase_idx"]).reset_index(drop=True)
    fight_cut = phase_medians["fight_median"].fillna(transform["global_fight_median"]).to_numpy()
    active_cut = phase_medians["active_median"].fillna(transform["global_active_median"]).to_numpy()
    fight_high = out["fight_score"].to_numpy() > fight_cut
    active_high = out["active_score"].to_numpy() > active_cut

    strategy = np.full(len(out), "camper", dtype=object)
    strategy[fight_high & active_high] = "hunter"
    strategy[fight_high & ~active_high] = "sniper"
    strategy[~fight_high & active_high] = "traveler"
    out["strategy_type"] = strategy

    if with_survival_rate:
        rates: dict[tuple[int, str], float] = transform["survival_rate"]
        fallback = float(transform["global_survival_rate"])
        out["type_survival_rate"] = [
            rates.get((int(phase), str(stype)), fallback)
            for phase, stype in zip(out["phase_idx"], out["strategy_type"])
        ]
    else:
        out["type_survival_rate"] = float(transform["global_survival_rate"])
    return out


def load_family_cols(family: str, manifest_path: Path = V2_MANIFEST_PATH) -> list[str]:
    manifest = _load_manifest(manifest_path)
    return list(manifest["families"][family]["columns"])


def load_family_rows(table: str, match_ids: list[str], cols: list[str]) -> pd.DataFrame:
    if not match_ids:
        return pd.DataFrame(columns=KEY_COLS + cols)
    quoted = ", ".join(f'"{c}"' for c in cols)
    with connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                f"""
                SELECT match_id, team_id, phase_idx, {quoted}
                  FROM {table}
                 WHERE match_id = ANY(%s)
                """,
                (match_ids,),
            )
            out_cols = [d[0] for d in cur.description]
            df = pd.DataFrame(cur.fetchall(), columns=out_cols)
    if not df.empty:
        df["team_id"] = df["team_id"].astype(int)
        df["phase_idx"] = df["phase_idx"].astype(int)
    return df


def attach_family_features(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    *,
    family: str,
    table: str,
    etl_module: str,
    train_ids: list[str],
    test_ids: list[str],
    skip: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """LEFT-merge one manifest family from its table; [] when unavailable/skipped."""
    if skip:
        logger.info("%s skipped by flag", family)
        return train_df, test_df, []
    try:
        cols = load_family_cols(family)
        if all(c in train_df.columns for c in cols) and all(c in test_df.columns for c in cols):
            logger.info("%s already present in feature view cols=%d", family, len(cols))
            return train_df, test_df, cols
        fam_train = load_family_rows(table, train_ids, cols)
        fam_test = load_family_rows(table, test_ids, cols)
    except Exception:
        logger.exception("%s unavailable — skipping its configs", family)
        return train_df, test_df, []
    if fam_train.empty or fam_test.empty:
        logger.warning("%s empty for this split — skipping %s configs", table, family)
        return train_df, test_df, []
    train_df = train_df.merge(fam_train, on=KEY_COLS, how="left")
    test_df = test_df.merge(fam_test, on=KEY_COLS, how="left")
    coverage = float(train_df[cols[0]].notna().mean())
    logger.info("%s merged cols=%d train_coverage=%.4f", family, len(cols), coverage)
    if coverage < 0.99:
        logger.warning(
            "%s coverage %.4f < 0.99 — run %s to completion",
            family, coverage, etl_module,
        )
    return train_df, test_df, cols


def filter_to_aligned_keys(test_df: pd.DataFrame, align_path: str | None) -> tuple[pd.DataFrame, dict[str, Any]]:
    if not align_path:
        return test_df, {"enabled": False}
    path = Path(align_path)
    if not path.exists():
        return test_df, {"enabled": False, "missing": str(path)}

    key_cols = ["match_id", "team_id", "phase_idx"]
    keys = pd.read_csv(path, usecols=key_cols).drop_duplicates()
    before = len(test_df)
    out = test_df.merge(keys, on=key_cols, how="inner")
    return out, {
        "enabled": True,
        "path": str(path),
        "before_rows": int(before),
        "after_rows": int(len(out)),
        "keys": int(len(keys)),
    }


def make_feature_configs(
    df: pd.DataFrame,
    network_cols: list[str] | None = None,
    habitat_cols: list[str] | None = None,
) -> dict[str, list[str]]:
    network_cols = [c for c in (network_cols or []) if c in df.columns]
    habitat_cols = [c for c in (habitat_cols or []) if c in df.columns]
    # full_node_features keeps its historical meaning (view + prior scores,
    # no F6/F7) so results stay comparable across runs; new variants are explicit.
    new_cols = set(network_cols) | set(habitat_cols)
    all_cols = [c for c in df.columns if c not in EXCLUDE_COLUMNS and c not in new_cols]
    conventional = _dedupe(GLOBAL_COLS + STATE_COLS + COMBAT_COLS + LOOT_COLS)
    stdb_spatial = _dedupe(TRAJECTORY_COLS + EXPOSURE_COLS)
    configs = {
        "prior_scores": _available(PRIOR_SCORE_COLS, df),
        "prior_raw": _available(["phase_idx"] + AGGRESSIVENESS_COLS + ACTIVENESS_COLS, df),
        "global_state": _available(GLOBAL_COLS + STATE_COLS, df),
        "conventional": _available(conventional, df),
        "stdb_spatial": _available(["phase_idx"] + stdb_spatial, df),
        "conventional_plus_trajectory": _available(conventional + TRAJECTORY_COLS, df),
        "conventional_plus_exposure": _available(conventional + EXPOSURE_COLS, df),
        "conventional_plus_stdb_spatial": _available(conventional + stdb_spatial, df),
        "full_node_features": _available(all_cols, df),
    }
    if network_cols:
        configs["network_tier1"] = _available(["phase_idx"] + network_cols, df)
        configs["conventional_plus_network"] = _available(conventional + network_cols, df)
        configs["conventional_stdb_plus_network"] = _available(
            conventional + stdb_spatial + network_cols, df
        )
        configs["full_plus_network"] = _available(all_cols + network_cols, df)
    if habitat_cols:
        configs["habitat_tier1"] = _available(["phase_idx"] + habitat_cols, df)
        configs["conventional_stdb_plus_habitat"] = _available(
            conventional + stdb_spatial + habitat_cols, df
        )
    if network_cols and habitat_cols:
        configs["conventional_stdb_network_habitat"] = _available(
            conventional + stdb_spatial + network_cols + habitat_cols, df
        )
    return configs


def make_manifest_feature_configs(
    df: pd.DataFrame,
    *,
    manifest_path: Path = V2_MANIFEST_PATH,
) -> dict[str, list[str]]:
    manifest = _load_manifest(manifest_path)
    families = manifest["families"]
    configs: dict[str, list[str]] = {}
    for config_name, family_names in manifest.get("feature_only_ablations", {}).items():
        cols: list[str] = []
        for family_name in family_names:
            if family_name not in families:
                logger.warning("manifest config %s references unknown family %s", config_name, family_name)
                continue
            cols.extend(families[family_name].get("columns", []))
        configs[config_name] = _available(cols, df)
    return {k: v for k, v in configs.items() if v}


def train_predict_config(
    *,
    name: str,
    feature_cols: list[str],
    learner: str,
    output_name: str,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    args: argparse.Namespace,
    categorical_cols: list[str] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if not feature_cols:
        raise ValueError(f"no features for {name}")
    model = _classifier(
        feature_cols,
        categorical_cols,
        learner=learner,
        seed=args.seed,
        max_iter=args.max_iter,
        n_estimators=args.n_estimators,
        n_jobs=args.n_jobs,
    )
    X_train = train_df[feature_cols]
    y_train = train_df["hazard_next"].astype(int).to_numpy()
    X_test = test_df[feature_cols]
    y_test = test_df["hazard_next"].astype(int).to_numpy()

    logger.info(
        "fit %-34s learner=%-4s rows=%d features=%d",
        name,
        learner,
        len(train_df),
        len(feature_cols),
    )
    with warnings.catch_warnings():
        if learner in {"lr", "mlp"}:
            warnings.filterwarnings("ignore", category=RuntimeWarning, module=r"sklearn\..*")
        model.fit(X_train, y_train)
        prob = model.predict_proba(X_test)[:, 1]
    if not np.isfinite(prob).all():
        raise RuntimeError(f"{output_name} produced non-finite probabilities")
    preds = test_df[["match_id", "team_id", "phase_idx"]].copy()
    preds["model"] = output_name
    preds["y"] = y_test
    preds["prob"] = prob
    metrics = {
        "model": output_name,
        "feature_config": name,
        "learner": learner,
        "n_features": len(feature_cols),
        "feature_cols": feature_cols,
        **_binary_metrics(y_test, prob),
    }
    return preds[["model", "match_id", "team_id", "phase_idx", "y", "prob"]], metrics


def run(args: argparse.Namespace) -> dict[str, Any]:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    manifest = _load_manifest(args.manifest)
    learners, unavailable_learners = _resolve_learners(args.learners)
    if unavailable_learners:
        logger.warning("skipping unavailable learners: %s", unavailable_learners)
    suffix_models = len(learners) > 1 or args.suffix_model_names
    categorical_cols = _dedupe(list(manifest.get("categorical_columns", [])) + CATEGORICAL_COLUMNS + ["strategy_type"])
    map_name, mode, cohort_spec = resolve_map_mode(
        cohort=args.cohort,
        map_name=args.map,
        mode=args.mode,
    )
    ids = list_phase_feature_match_ids(map_name, mode, args.limit)
    train_ids, val_ids, test_ids = load_or_create_split(
        ids,
        split_json=args.split_json,
        seed=args.seed,
        val_frac=args.val_frac,
        test_frac=args.test_frac,
        metadata={"source": "model_stgcn.feature_ablation", "limit": args.limit},
    )
    logger.info("split total=%d train=%d val=%d test=%d", len(ids), len(train_ids), len(val_ids), len(test_ids))

    train_rows, _cols = load_phase_rows_from_view(train_ids, args.feature_view)
    test_rows, _ = load_phase_rows_from_view(test_ids, args.feature_view)
    train_df = _rows_to_df(train_rows)
    test_df = _rows_to_df(test_rows)

    prior_transform = fit_prior_transform(train_df)
    train_df = add_prior_features(train_df, prior_transform)
    test_df = add_prior_features(test_df, prior_transform)
    train_df, test_df, network_cols = attach_family_features(
        train_df,
        test_df,
        family=NETWORK_FAMILY,
        table=NETWORK_TABLE,
        etl_module="stdb.network_feature_etl",
        train_ids=train_ids,
        test_ids=test_ids,
        skip=args.skip_network,
    )
    train_df, test_df, habitat_cols = attach_family_features(
        train_df,
        test_df,
        family=HABITAT_FAMILY,
        table=HABITAT_TABLE,
        etl_module="stdb.habitat_feature_etl",
        train_ids=train_ids,
        test_ids=test_ids,
        skip=args.skip_habitat,
    )
    test_df, alignment = filter_to_aligned_keys(test_df, args.align_to)
    logger.info("rows train=%d test=%d alignment=%s", len(train_df), len(test_df), alignment)

    if args.config_style == "legacy":
        feature_configs = make_feature_configs(train_df, network_cols, habitat_cols)
    elif args.config_style == "manifest":
        feature_configs = make_manifest_feature_configs(train_df, manifest_path=args.manifest)
    else:
        feature_configs = {
            **make_feature_configs(train_df, network_cols, habitat_cols),
            **make_manifest_feature_configs(train_df, manifest_path=args.manifest),
        }
    if args.configs:
        wanted = [c.strip() for c in args.configs.split(",") if c.strip()]
        feature_configs = {k: v for k, v in feature_configs.items() if k in wanted}

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pred_frames: list[pd.DataFrame] = []
    overall_rows: list[dict[str, Any]] = []
    feature_manifest: dict[str, list[str]] = {}
    for name, feature_cols in feature_configs.items():
        for learner in learners:
            output_name = _model_name(name, learner, suffix=suffix_models)
            preds, metrics = train_predict_config(
                name=name,
                feature_cols=feature_cols,
                learner=learner,
                output_name=output_name,
                train_df=train_df,
                test_df=test_df,
                args=args,
                categorical_cols=categorical_cols,
            )
            pred_frames.append(preds)
            overall_rows.append(metrics)
            feature_manifest[output_name] = feature_cols

    predictions = pd.concat(pred_frames, ignore_index=True)
    overall = pd.DataFrame(overall_rows).drop(columns=["feature_cols"])
    phase = pd.DataFrame(_metric_rows(predictions, ["model", "phase_idx"]))
    per_match = _paired_metric_by_match(predictions)

    tests: list[dict[str, Any]] = []
    comparisons = [
        ("full_node_features", "prior_scores"),
        ("full_node_features", "prior_raw"),
        ("conventional_plus_stdb_spatial", "conventional"),
        ("conventional_plus_trajectory", "conventional"),
        ("conventional_plus_exposure", "conventional"),
        ("stdb_spatial", "prior_scores"),
        ("full_node_features", "conventional_plus_stdb_spatial"),
        # F6 network-role increments (FEATURE_DESIGN_V2 §11)
        ("conventional_stdb_plus_network", "conventional_plus_stdb_spatial"),
        ("conventional_plus_network", "conventional"),
        ("network_tier1", "prior_scores"),
        ("full_plus_network", "full_node_features"),
        # F7 habitat-pressure increments
        ("conventional_stdb_plus_habitat", "conventional_plus_stdb_spatial"),
        ("habitat_tier1", "prior_scores"),
        ("conventional_stdb_network_habitat", "conventional_plus_stdb_spatial"),
        ("conventional_stdb_network_habitat", "conventional_stdb_plus_network"),
        ("conventional_stdb_network_habitat", "conventional_stdb_plus_habitat"),
        # Manifest-defined A0-A11 comparisons.
        ("a3_current_conventional", "a0_official_raw"),
        ("a5_raw_plus_stdb", "a0_official_raw"),
        ("a6_raw_plus_network", "a0_official_raw"),
        ("a7_raw_plus_habitat", "a0_official_raw"),
        ("a8_raw_stdb_network", "a5_raw_plus_stdb"),
        ("a9_raw_stdb_habitat", "a5_raw_plus_stdb"),
        ("a10_raw_network_habitat", "a0_official_raw"),
        ("a11_full_representation", "a5_raw_plus_stdb"),
        ("a11_full_representation", "a8_raw_stdb_network"),
        ("a11_full_representation", "a9_raw_stdb_habitat"),
    ]
    available = set(per_match["model"])
    for learner in learners:
        for left, right in comparisons:
            left_model = _model_name(left, learner, suffix=suffix_models)
            right_model = _model_name(right, learner, suffix=suffix_models)
            if left_model not in available or right_model not in available:
                continue
            tests.append(_paired_test(per_match, left_model=left_model, right_model=right_model, metric="bce", larger_is_better=False))
            tests.append(_paired_test(per_match, left_model=left_model, right_model=right_model, metric="auc", larger_is_better=True))
            tests.append(_paired_test(per_match, left_model=left_model, right_model=right_model, metric="ap", larger_is_better=True))
    tests_df = pd.DataFrame(tests)

    predictions_path = out_dir / "feature_ablation_predictions.csv"
    overall_path = out_dir / "feature_ablation_metrics.csv"
    phase_path = out_dir / "feature_ablation_phase_metrics.csv"
    per_match_path = out_dir / "feature_ablation_per_match_metrics.csv"
    tests_path = out_dir / "feature_ablation_paired_tests.csv"
    manifest_path = out_dir / "feature_ablation_features.json"
    summary_path = out_dir / "feature_ablation_summary.json"

    predictions.to_csv(predictions_path, index=False)
    overall.to_csv(overall_path, index=False)
    phase.to_csv(phase_path, index=False)
    per_match.to_csv(per_match_path, index=False)
    tests_df.to_csv(tests_path, index=False)
    manifest_path.write_text(json.dumps(feature_manifest, indent=2, default=str), encoding="utf-8")

    summary = {
        "split": {
            "total": len(ids),
            "train": len(train_ids),
            "val": len(val_ids),
            "test": len(test_ids),
        },
        "cohort": {
            "label": cohort_spec.label if cohort_spec else None,
            "slug": cohort_spec.slug if cohort_spec else None,
            "map_name": map_name,
            "mode": mode,
        },
        "alignment": alignment,
        "feature_view": args.feature_view,
        "config_style": args.config_style,
        "learners": learners,
        "unavailable_learners": unavailable_learners,
        "outputs": {
            "predictions": str(predictions_path),
            "overall_metrics": str(overall_path),
            "phase_metrics": str(phase_path),
            "per_match_metrics": str(per_match_path),
            "paired_tests": str(tests_path),
            "feature_manifest": str(manifest_path),
        },
        "overall_metrics": overall.to_dict(orient="records"),
        "paired_tests": tests,
    }
    summary_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    logger.info("wrote %s", summary_path)
    logger.info("overall=%s", overall.to_dict(orient="records"))
    logger.info("paired=%s", tests)
    return summary


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", default=None, help="Named required cohort, e.g. 'Erangel TPP squad'")
    ap.add_argument("--map", default=None)
    ap.add_argument("--mode", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--test-frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--learners", default="hgb", help="Comma-separated learners or 'all': lr,rf,hgb,xgb,mlp")
    ap.add_argument("--max-iter", type=int, default=200)
    ap.add_argument("--n-estimators", type=int, default=300)
    ap.add_argument("--n-jobs", type=int, default=0)
    ap.add_argument(
        "--suffix-model-names",
        action="store_true",
        help="Append learner suffix even when only one learner is run.",
    )
    ap.add_argument("--split-json", default="model_stgcn/checkpoints/split_phase_all_seed42.json")
    ap.add_argument("--align-to", default="model_stgcn/checkpoints/phase_result_analysis/hazard_predictions_test_aligned.csv")
    ap.add_argument("--out-dir", default="model_stgcn/checkpoints/feature_ablation")
    ap.add_argument("--feature-view", default=DEFAULT_FEATURE_VIEW)
    ap.add_argument("--manifest", type=Path, default=V2_MANIFEST_PATH)
    ap.add_argument(
        "--config-style",
        default="legacy",
        choices=["legacy", "manifest", "both"],
        help="legacy keeps historical configs; manifest runs A0-A11 from the V2 manifest.",
    )
    ap.add_argument("--configs", default=None, help="Comma-separated subset of feature configs")
    ap.add_argument(
        "--skip-network",
        action="store_true",
        help="Do not merge F6 network features even when the table exists",
    )
    ap.add_argument(
        "--skip-habitat",
        action="store_true",
        help="Do not merge F7 habitat features even when the table exists",
    )
    return ap.parse_args()


if __name__ == "__main__":
    run(parse_args())
