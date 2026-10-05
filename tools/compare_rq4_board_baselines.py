"""Compare RQ4 phase-transition risk/advantage boards against baselines.

The saved model artifacts already contain row-level phase hazard predictions.
This tool reuses those predictions, derives board scores, and evaluates all
arms with one board-metric contract.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-codex")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from tools.build_dynamic_probability_fluctuation import build_trajectories


DEFAULT_LSTM_TRAJECTORIES = Path(
    "reports/rq4_dynamic_probability_fluctuation/dynamic_probability_trajectories.csv"
)
DEFAULT_PRIOR_ROOT = Path("model_stgcn/checkpoints/br_rq4_prior_model_zoo_required_v1")
DEFAULT_OUT = Path("reports/rq4_board_baseline_comparison")
DEFAULT_FEATURE_CACHE = DEFAULT_OUT / "board_baseline_feature_rows.csv"
TOP_K = 5

PROPOSED_VARIANT = "lstm_realtime_board"

PRIOR_MODEL_MAP = {
    "a2_prior_scores__hgb": "sagong_prior_score_hgb_board",
    "a0_official_raw__hgb": "official_raw_hgb_board",
    "conventional_stdb_network_habitat__hgb": "pressure_habitat_hgb_board",
}

VARIANT_LABELS = {
    "random_alive_board": "Random alive",
    "health_alive_count_board": "Health/alive-count",
    "zone_distance_board": "Zone-distance",
    "sagong_prior_score_hgb_board": "Sagong-style adapted HGB",
    "official_raw_hgb_board": "Official-raw HGB",
    "pressure_habitat_hgb_board": "Pressure-habitat HGB",
    PROPOSED_VARIANT: "LSTM realtime board",
}

FEATURE_COLUMNS = [
    "match_id",
    "team_id",
    "phase_idx",
    "alive_players_end",
    "alive_players_avg",
    "mean_health_avg",
    "min_health_min",
    "outside_safe_ratio",
    "outside_safe_duration_s",
    "zone_entry_time",
    "safe_alignment",
    "zone_gap_end_m",
    "mean_dist_to_boundary_m",
    "min_dist_to_boundary_m",
    "hazard_next",
    "final_rank_norm",
    "final_top5",
    "final_top10",
]

BOARD_METRICS = [
    "risk_top1_hit",
    "risk_top3_hit",
    f"risk_top{TOP_K}_hit",
    "risk_mrr",
    "adv_top_final_top5",
    "adv_top_final_top10",
    f"adv_top{TOP_K}_contains_top10",
    f"adv_ndcg_top{TOP_K}_top10",
    "risk_delta_top1_hit",
    "risk_delta_mrr",
    "adv_delta_top_final_top10",
    f"adv_delta_ndcg_top{TOP_K}_top10",
]


def _to_numeric(series: pd.Series, default: float = 0.0) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    if values.notna().any():
        fill = float(values.median())
    else:
        fill = default
    return values.fillna(fill).astype(float)


def _robust_z(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    if not values.notna().any():
        return pd.Series(np.zeros(len(series), dtype=float), index=series.index)
    median = float(values.median())
    iqr = float(values.quantile(0.75) - values.quantile(0.25))
    if not np.isfinite(iqr) or iqr <= 1e-9:
        iqr = 1.0
    return (values.fillna(median).astype(float) - median) / iqr


def _stable_unit(seed: int, *parts: object) -> float:
    text = "|".join([str(seed), *(str(part) for part in parts)])
    digest = hashlib.blake2b(text.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big") / float(2**64)


def _reciprocal_rank(sorted_labels: np.ndarray) -> float:
    hits = np.flatnonzero(sorted_labels.astype(int) == 1)
    if hits.size == 0:
        return float("nan")
    return float(1.0 / (int(hits[0]) + 1))


def _hit_at(sorted_labels: np.ndarray, k: int) -> float:
    if sorted_labels.size == 0 or int(sorted_labels.sum()) == 0:
        return float("nan")
    return float(sorted_labels[:k].sum() > 0)


def _precision_at(sorted_labels: np.ndarray, k: int) -> float:
    if sorted_labels.size == 0:
        return float("nan")
    top = sorted_labels[: min(k, sorted_labels.size)]
    if top.size == 0:
        return float("nan")
    return float(top.mean())


def _ndcg_at(sorted_relevance: np.ndarray, k: int) -> float:
    rel = sorted_relevance[: min(k, sorted_relevance.size)].astype(float)
    if rel.size == 0:
        return float("nan")
    discount = 1.0 / np.log2(np.arange(2, rel.size + 2))
    dcg = float((rel * discount).sum())
    ideal = np.sort(sorted_relevance.astype(float))[::-1][: rel.size]
    idcg = float((ideal * discount).sum())
    if idcg <= 0.0:
        return float("nan")
    return float(dcg / idcg)


def _rank_metrics(frame: pd.DataFrame, *, score_col: str, label_col: str, top_k: int) -> dict[str, float]:
    if frame.empty:
        return {
            "top1_hit": float("nan"),
            "top3_hit": float("nan"),
            f"top{top_k}_hit": float("nan"),
            "mrr": float("nan"),
            "precision_at_3": float("nan"),
            f"precision_at_{top_k}": float("nan"),
        }
    sorted_frame = frame.sort_values(score_col, ascending=False)
    labels = sorted_frame[label_col].astype(int).to_numpy()
    if int(labels.sum()) == 0:
        return {
            "top1_hit": float("nan"),
            "top3_hit": float("nan"),
            f"top{top_k}_hit": float("nan"),
            "mrr": float("nan"),
            "precision_at_3": _precision_at(labels, 3),
            f"precision_at_{top_k}": _precision_at(labels, top_k),
        }
    return {
        "top1_hit": float(labels[0]) if labels.size else float("nan"),
        "top3_hit": _hit_at(labels, 3),
        f"top{top_k}_hit": _hit_at(labels, top_k),
        "mrr": _reciprocal_rank(labels),
        "precision_at_3": _precision_at(labels, 3),
        f"precision_at_{top_k}": _precision_at(labels, top_k),
    }


def _normalise_key_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["match_id"] = out["match_id"].astype(str)
    out["team_id"] = out["team_id"].astype(str)
    if "phase_slot" in out.columns:
        out["phase_slot"] = pd.to_numeric(out["phase_slot"], errors="coerce").astype("Int64")
    if "phase_idx" in out.columns:
        out["phase_idx"] = pd.to_numeric(out["phase_idx"], errors="coerce").astype("Int64")
    return out


def filter_common_score_universe(
    score_rows: pd.DataFrame,
    *,
    proposed_variant: str = PROPOSED_VARIANT,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Keep only cohort/match/phase/team rows available for every variant."""
    if score_rows.empty:
        return score_rows.copy(), {
            "input_rows": 0,
            "output_rows": 0,
            "variants": [],
            "common_row_keys": 0,
            "input_board_keys": 0,
            "output_board_keys": 0,
            "dropped_rows_by_variant": {},
        }
    key_cols = ["cohort_slug", "match_id", "phase_slot", "team_id"]
    board_cols = ["cohort_slug", "match_id", "phase_slot"]
    out = _normalise_key_columns(score_rows)
    out["model_variant"] = out["model_variant"].astype(str)
    variants = sorted(out["model_variant"].dropna().unique().tolist())
    if proposed_variant in variants:
        variants = [proposed_variant, *[v for v in variants if v != proposed_variant]]
    key_variant = out[key_cols + ["model_variant"]].drop_duplicates()
    common_keys = key_variant.groupby(key_cols, dropna=False)["model_variant"].nunique()
    common_keys = common_keys[common_keys == len(variants)].reset_index()[key_cols]
    filtered = out.merge(common_keys, on=key_cols, how="inner")
    before = out.groupby("model_variant").size()
    after = filtered.groupby("model_variant").size()
    dropped = {
        variant: int(before.get(variant, 0) - after.get(variant, 0))
        for variant in variants
    }
    coverage = {
        "input_rows": int(len(out)),
        "output_rows": int(len(filtered)),
        "variants": variants,
        "common_row_keys": int(len(common_keys)),
        "input_board_keys": int(out[board_cols].drop_duplicates().shape[0]),
        "output_board_keys": int(filtered[board_cols].drop_duplicates().shape[0]),
        "dropped_rows_by_variant": dropped,
        "contract": "all variants evaluated on intersected cohort/match/phase/team rows",
    }
    return filtered, coverage


def load_lstm_score_rows(path: Path) -> pd.DataFrame:
    traj = pd.read_csv(path)
    traj = _normalise_key_columns(traj)
    if "phase_slot" not in traj.columns and "phase_idx" in traj.columns:
        traj = traj.rename(columns={"phase_idx": "phase_slot"})
    traj["model_variant"] = PROPOSED_VARIANT
    traj["risk_score"] = pd.to_numeric(traj["hazard_prob"], errors="coerce")
    traj["advantage_score"] = pd.to_numeric(traj["dynamic_win_prob"], errors="coerce")
    if "observed_readout" not in traj.columns:
        traj["observed_readout"] = 1
    return traj[
        [
            "cohort_slug",
            "model_variant",
            "match_id",
            "team_idx",
            "team_id",
            "phase_slot",
            "cut_phase",
            "hazard_true",
            "risk_score",
            "advantage_score",
            "final_rank_norm",
            "final_top5",
            "final_top10",
            "observed_readout",
        ]
    ].copy()


def load_prior_predictions(root: Path, model_map: dict[str, str]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for path in sorted(root.glob("*/feature_ablation_predictions.csv")):
        cohort = path.parent.name
        df = pd.read_csv(path)
        df = df[df["model"].isin(model_map)].copy()
        if df.empty:
            continue
        df["cohort_slug"] = cohort
        df["model_variant"] = df["model"].map(model_map)
        df = df.rename(
            columns={
                "phase_idx": "phase_slot",
                "y": "hazard_true",
                "prob": "hazard_prob",
            }
        )
        df["cut_phase"] = df["phase_slot"]
        frames.append(df)
    if not frames:
        raise SystemExit(f"No selected prior prediction rows found under {root}")
    out = pd.concat(frames, ignore_index=True)
    return _normalise_key_columns(out)


def build_match_cohort_map(*frames: pd.DataFrame) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for frame in frames:
        if frame.empty or "cohort_slug" not in frame.columns:
            continue
        pairs = frame[["match_id", "cohort_slug"]].dropna().drop_duplicates()
        for row in pairs.itertuples(index=False):
            mapping.setdefault(str(row.match_id), str(row.cohort_slug))
    return mapping


def load_feature_rows_from_db(match_ids: list[str]) -> pd.DataFrame:
    from stdb.conn import connect

    if not match_ids:
        return pd.DataFrame(columns=FEATURE_COLUMNS)
    quoted = ", ".join(f'"{col}"' for col in FEATURE_COLUMNS)
    with connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                f"""
                SELECT {quoted}
                  FROM stdb.v_team_phase_feature_long
                 WHERE match_id = ANY(%s)
                   AND target_defined
                """,
                (match_ids,),
            )
            cols = [desc[0] for desc in cur.description]
            return pd.DataFrame(cur.fetchall(), columns=cols)


def prepare_feature_rows(
    *,
    feature_csv: Path,
    match_to_cohort: dict[str, str],
    match_ids: list[str],
    refresh: bool,
) -> pd.DataFrame:
    if feature_csv.exists() and not refresh:
        features = pd.read_csv(feature_csv)
    else:
        features = load_feature_rows_from_db(match_ids)
        feature_csv.parent.mkdir(parents=True, exist_ok=True)
        features.to_csv(feature_csv, index=False)

    features = _normalise_key_columns(features)
    if "cohort_slug" not in features.columns:
        features["cohort_slug"] = features["match_id"].map(match_to_cohort)
    features = features[features["cohort_slug"].notna()].copy()
    features["phase_slot"] = pd.to_numeric(features["phase_idx"], errors="coerce").astype(int)
    features["cut_phase"] = features["phase_slot"]
    features["hazard_true"] = pd.to_numeric(features["hazard_next"], errors="coerce")
    for col in ["final_rank_norm", "final_top5", "final_top10"]:
        features[col] = pd.to_numeric(features[col], errors="coerce")
    features = features.dropna(
        subset=["cohort_slug", "match_id", "team_id", "phase_slot", "hazard_true", "final_rank_norm"]
    ).copy()
    features = features.sort_values(["cohort_slug", "match_id", "team_id", "phase_slot"])
    team_keys = features[["cohort_slug", "match_id", "team_id"]].drop_duplicates().copy()
    team_keys["team_idx"] = team_keys.groupby(["cohort_slug", "match_id"]).cumcount()
    features = features.merge(team_keys, on=["cohort_slug", "match_id", "team_id"], how="left")
    return features


def attach_feature_labels(pred: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
    label_cols = [
        "cohort_slug",
        "match_id",
        "team_id",
        "phase_slot",
        "team_idx",
        "final_rank_norm",
        "final_top5",
        "final_top10",
    ]
    labels = features[label_cols].drop_duplicates(["cohort_slug", "match_id", "team_id", "phase_slot"])
    out = pred.merge(labels, on=["cohort_slug", "match_id", "team_id", "phase_slot"], how="left")
    before = len(out)
    out = out.dropna(subset=["team_idx", "final_rank_norm", "final_top5", "final_top10"]).copy()
    dropped = before - len(out)
    if dropped:
        print(f"Dropped {dropped} prediction rows without feature labels.")
    for col in ["phase_slot", "cut_phase", "team_idx", "hazard_true", "final_top5", "final_top10"]:
        out[col] = pd.to_numeric(out[col], errors="coerce").astype(int)
    out["final_rank_norm"] = pd.to_numeric(out["final_rank_norm"], errors="coerce")
    out["hazard_prob"] = pd.to_numeric(out["hazard_prob"], errors="coerce")
    return out


def model_predictions_to_score_rows(pred: pd.DataFrame) -> pd.DataFrame:
    trajectories = build_trajectories(pred)
    trajectories["risk_score"] = pd.to_numeric(trajectories["hazard_prob"], errors="coerce")
    trajectories["advantage_score"] = pd.to_numeric(trajectories["dynamic_win_prob"], errors="coerce")
    if "observed_readout" not in trajectories.columns:
        trajectories["observed_readout"] = 1
    return trajectories[
        [
            "cohort_slug",
            "model_variant",
            "match_id",
            "team_idx",
            "team_id",
            "phase_slot",
            "cut_phase",
            "hazard_true",
            "risk_score",
            "advantage_score",
            "final_rank_norm",
            "final_top5",
            "final_top10",
            "observed_readout",
        ]
    ].copy()


def heuristic_score_rows(features: pd.DataFrame, *, random_seed: int) -> pd.DataFrame:
    base_cols = [
        "cohort_slug",
        "match_id",
        "team_idx",
        "team_id",
        "phase_slot",
        "cut_phase",
        "hazard_true",
        "final_rank_norm",
        "final_top5",
        "final_top10",
    ]
    base = features[base_cols].copy()
    base["observed_readout"] = 1

    alive = _to_numeric(features["alive_players_end"])
    alive_avg = _to_numeric(features["alive_players_avg"])
    mean_health = _to_numeric(features["mean_health_avg"])
    min_health = _to_numeric(features["min_health_min"])
    health_capital = alive * mean_health + 0.25 * min_health + 5.0 * alive_avg

    zone_pressure = (
        _robust_z(features["outside_safe_ratio"])
        + 0.50 * _robust_z(features["outside_safe_duration_s"])
        + 0.25 * _robust_z(features["zone_entry_time"])
        + 0.25 * _robust_z(features["zone_gap_end_m"])
        - 0.50 * _robust_z(features["safe_alignment"])
    )

    random_risk = [
        _stable_unit(random_seed, "risk", row.cohort_slug, row.match_id, row.phase_slot, row.team_id)
        for row in base.itertuples(index=False)
    ]
    random_advantage = [
        _stable_unit(random_seed, "adv", row.cohort_slug, row.match_id, row.phase_slot, row.team_id)
        for row in base.itertuples(index=False)
    ]

    frames: list[pd.DataFrame] = []
    for variant, risk_score, advantage_score in [
        ("random_alive_board", pd.Series(random_risk, index=base.index), pd.Series(random_advantage, index=base.index)),
        ("health_alive_count_board", -health_capital, health_capital),
        ("zone_distance_board", zone_pressure, -zone_pressure),
    ]:
        out = base.copy()
        out["model_variant"] = variant
        out["risk_score"] = pd.to_numeric(risk_score, errors="coerce")
        out["advantage_score"] = pd.to_numeric(advantage_score, errors="coerce")
        frames.append(out)
    return pd.concat(frames, ignore_index=True)


def evaluate_board_scores(score_rows: pd.DataFrame, *, top_k: int = TOP_K) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    score_rows = score_rows.copy()
    if "observed_readout" not in score_rows.columns:
        score_rows["observed_readout"] = 1
    score_rows["phase_slot"] = pd.to_numeric(score_rows["phase_slot"], errors="coerce").astype(int)
    for col in ["hazard_true", "risk_score", "advantage_score", "final_top5", "final_top10", "final_rank_norm"]:
        score_rows[col] = pd.to_numeric(score_rows[col], errors="coerce")
    score_rows = score_rows.sort_values(
        ["cohort_slug", "model_variant", "match_id", "team_id", "phase_slot"]
    ).copy()
    team_group = score_rows.groupby(["cohort_slug", "model_variant", "match_id", "team_id"], sort=False)
    score_rows["risk_score_delta"] = team_group["risk_score"].diff()
    score_rows["advantage_score_delta"] = team_group["advantage_score"].diff()

    group_cols = ["cohort_slug", "model_variant", "match_id", "phase_slot"]
    for key, phase in score_rows.groupby(group_cols, sort=False):
        cohort, variant, match_id, phase_slot = key
        observed = phase[
            (phase["observed_readout"].fillna(1).astype(int) == 1)
            & phase["risk_score"].notna()
            & phase["hazard_true"].notna()
        ].copy()
        advantage = phase[
            phase["advantage_score"].notna()
            & phase["final_top5"].notna()
            & phase["final_top10"].notna()
            & phase["final_rank_norm"].notna()
        ].copy()
        if observed.empty or advantage.empty:
            continue

        risk_sorted = observed.sort_values("risk_score", ascending=False)
        risk_labels = risk_sorted["hazard_true"].astype(int).to_numpy()
        risk_top = risk_sorted.iloc[0]
        has_elimination = int(risk_labels.sum()) > 0

        adv_sorted = advantage.sort_values("advantage_score", ascending=False)
        adv_top = adv_sorted.iloc[0]
        topk_adv = adv_sorted.head(top_k)
        adv_topk_rel = adv_sorted["final_top10"].astype(int).to_numpy()

        risk_delta = observed[observed["risk_score_delta"].notna()].copy()
        risk_delta_metrics = _rank_metrics(
            risk_delta,
            score_col="risk_score_delta",
            label_col="hazard_true",
            top_k=top_k,
        )
        adv_delta = advantage[advantage["advantage_score_delta"].notna()].copy()
        if adv_delta.empty:
            adv_delta_top = None
            adv_delta_ndcg = float("nan")
        else:
            adv_delta_sorted = adv_delta.sort_values("advantage_score_delta", ascending=False)
            adv_delta_top = adv_delta_sorted.iloc[0]
            adv_delta_ndcg = _ndcg_at(adv_delta_sorted["final_top10"].astype(int).to_numpy(), top_k)

        rows.append(
            {
                "cohort_slug": cohort,
                "model_variant": variant,
                "board_label": VARIANT_LABELS.get(str(variant), str(variant)),
                "match_id": match_id,
                "phase_slot": int(phase_slot),
                "at_risk_rows": int(len(observed)),
                "next_phase_eliminations": int(risk_labels.sum()),
                "risk_top_team_id": str(risk_top["team_id"]),
                "risk_top_score": float(risk_top["risk_score"]),
                "risk_top1_hit": float(int(risk_top["hazard_true"])) if has_elimination else float("nan"),
                "risk_top3_hit": _hit_at(risk_labels, 3),
                f"risk_top{top_k}_hit": _hit_at(risk_labels, top_k),
                "risk_mrr": _reciprocal_rank(risk_labels),
                "risk_precision_at_3": _precision_at(risk_labels, 3),
                f"risk_precision_at_{top_k}": _precision_at(risk_labels, top_k),
                "adv_top_team_id": str(adv_top["team_id"]),
                "adv_top_score": float(adv_top["advantage_score"]),
                "adv_top_final_top5": int(adv_top["final_top5"]),
                "adv_top_final_top10": int(adv_top["final_top10"]),
                "adv_top_rank_norm": float(adv_top["final_rank_norm"]),
                f"adv_top{top_k}_contains_top5": float(topk_adv["final_top5"].astype(int).max()),
                f"adv_top{top_k}_contains_top10": float(topk_adv["final_top10"].astype(int).max()),
                f"adv_top{top_k}_top10_precision": float(topk_adv["final_top10"].astype(int).mean()),
                f"adv_ndcg_top{top_k}_top10": _ndcg_at(adv_topk_rel, top_k),
                "risk_delta_top_team_id": (
                    str(risk_delta.sort_values("risk_score_delta", ascending=False).iloc[0]["team_id"])
                    if not risk_delta.empty
                    else ""
                ),
                "risk_delta_top1_hit": risk_delta_metrics["top1_hit"],
                "risk_delta_top3_hit": risk_delta_metrics["top3_hit"],
                f"risk_delta_top{top_k}_hit": risk_delta_metrics[f"top{top_k}_hit"],
                "risk_delta_mrr": risk_delta_metrics["mrr"],
                "adv_delta_top_team_id": str(adv_delta_top["team_id"]) if adv_delta_top is not None else "",
                "adv_delta_top_final_top5": (
                    int(adv_delta_top["final_top5"]) if adv_delta_top is not None else float("nan")
                ),
                "adv_delta_top_final_top10": (
                    int(adv_delta_top["final_top10"]) if adv_delta_top is not None else float("nan")
                ),
                f"adv_delta_ndcg_top{top_k}_top10": adv_delta_ndcg,
                "leader_changed_from_prev_phase": np.nan,
            }
        )

    board = pd.DataFrame(rows)
    if board.empty:
        return board
    board = board.sort_values(["cohort_slug", "model_variant", "match_id", "phase_slot"]).reset_index(drop=True)
    leader_change = (
        board.groupby(["cohort_slug", "model_variant", "match_id"])["adv_top_team_id"]
        .apply(lambda s: s.astype(str) != s.astype(str).shift())
        .reset_index(level=[0, 1, 2], drop=True)
    )
    board["leader_changed_from_prev_phase"] = leader_change.astype(int)
    first_rows = ~board.duplicated(["cohort_slug", "model_variant", "match_id"])
    board.loc[first_rows, "leader_changed_from_prev_phase"] = 0
    return board


def summarize_board_scores(board: pd.DataFrame, *, top_k: int = TOP_K) -> tuple[pd.DataFrame, pd.DataFrame]:
    if board.empty:
        return pd.DataFrame(), pd.DataFrame()
    metric_cols = [
        "risk_top1_hit",
        "risk_top3_hit",
        f"risk_top{top_k}_hit",
        "risk_mrr",
        "risk_precision_at_3",
        f"risk_precision_at_{top_k}",
        "adv_top_final_top5",
        "adv_top_final_top10",
        f"adv_top{top_k}_contains_top5",
        f"adv_top{top_k}_contains_top10",
        f"adv_top{top_k}_top10_precision",
        f"adv_ndcg_top{top_k}_top10",
        "risk_delta_top1_hit",
        "risk_delta_top3_hit",
        f"risk_delta_top{top_k}_hit",
        "risk_delta_mrr",
        "adv_delta_top_final_top5",
        "adv_delta_top_final_top10",
        f"adv_delta_ndcg_top{top_k}_top10",
        "leader_changed_from_prev_phase",
    ]
    by_cohort = (
        board.groupby(["cohort_slug", "model_variant", "board_label"])
        .agg(
            phase_boards=("phase_slot", "count"),
            matches=("match_id", "nunique"),
            mean_at_risk_rows=("at_risk_rows", "mean"),
            mean_next_phase_eliminations=("next_phase_eliminations", "mean"),
            **{col: (col, "mean") for col in metric_cols},
        )
        .reset_index()
    )
    aggregate = (
        by_cohort.groupby(["model_variant", "board_label"])
        .agg(
            cohorts=("cohort_slug", "nunique"),
            phase_boards=("phase_boards", "sum"),
            median_risk_top1_hit=("risk_top1_hit", "median"),
            median_risk_top3_hit=("risk_top3_hit", "median"),
            median_risk_mrr=("risk_mrr", "median"),
            median_adv_top_final_top5=("adv_top_final_top5", "median"),
            median_adv_top_final_top10=("adv_top_final_top10", "median"),
            median_adv_top5_contains_top10=(f"adv_top{top_k}_contains_top10", "median"),
            median_adv_ndcg_top5_top10=(f"adv_ndcg_top{top_k}_top10", "median"),
            median_risk_delta_top1_hit=("risk_delta_top1_hit", "median"),
            median_risk_delta_mrr=("risk_delta_mrr", "median"),
            median_adv_delta_top_final_top10=("adv_delta_top_final_top10", "median"),
            median_adv_delta_ndcg_top5_top10=(f"adv_delta_ndcg_top{top_k}_top10", "median"),
            mean_leader_change_rate=("leader_changed_from_prev_phase", "mean"),
        )
        .reset_index()
        .sort_values(["median_risk_top1_hit", "median_adv_top_final_top10"], ascending=False)
    )
    return by_cohort, aggregate


def build_sign_count(
    by_cohort: pd.DataFrame,
    *,
    proposed_variant: str = PROPOSED_VARIANT,
    metrics: list[str] | None = None,
) -> pd.DataFrame:
    metrics = metrics or BOARD_METRICS
    proposed = by_cohort[by_cohort["model_variant"] == proposed_variant]
    rows: list[dict[str, Any]] = []
    for baseline in sorted(set(by_cohort["model_variant"]) - {proposed_variant}):
        baseline_df = by_cohort[by_cohort["model_variant"] == baseline]
        joined = proposed.merge(
            baseline_df,
            on="cohort_slug",
            suffixes=("_proposed", "_baseline"),
            how="inner",
        )
        for metric in metrics:
            p_col = f"{metric}_proposed"
            b_col = f"{metric}_baseline"
            if p_col not in joined.columns or b_col not in joined.columns:
                continue
            diff = pd.to_numeric(joined[p_col], errors="coerce") - pd.to_numeric(joined[b_col], errors="coerce")
            diff = diff.dropna()
            rows.append(
                {
                    "baseline_variant": baseline,
                    "baseline_label": VARIANT_LABELS.get(baseline, baseline),
                    "metric": metric,
                    "cohorts_compared": int(len(diff)),
                    "lstm_wins": int((diff > 1e-12).sum()),
                    "ties": int((diff.abs() <= 1e-12).sum()),
                    "lstm_losses": int((diff < -1e-12).sum()),
                    "median_lstm_minus_baseline": float(diff.median()) if not diff.empty else float("nan"),
                    "mean_lstm_minus_baseline": float(diff.mean()) if not diff.empty else float("nan"),
                }
            )
    return pd.DataFrame(rows)


def to_markdown_table(df: pd.DataFrame, *, max_rows: int | None = None) -> str:
    if df.empty:
        return "_No rows._"
    display = df.head(max_rows).copy() if max_rows else df.copy()
    for col in display.columns:
        if pd.api.types.is_float_dtype(display[col]):
            display[col] = display[col].map(lambda v: "" if pd.isna(v) else f"{float(v):.4f}")
        else:
            display[col] = display[col].map(lambda v: "" if pd.isna(v) else str(v))
    lines = [
        "| " + " | ".join(display.columns) + " |",
        "| " + " | ".join("---" for _ in display.columns) + " |",
    ]
    for row in display.itertuples(index=False):
        lines.append("| " + " | ".join(str(v) for v in row) + " |")
    return "\n".join(lines)


def write_example_figure(score_rows: pd.DataFrame, out_path: Path, *, top_k: int = TOP_K) -> dict[str, Any]:
    proposed = score_rows[score_rows["model_variant"] == PROPOSED_VARIANT].copy()
    proposed = proposed[
        (proposed["observed_readout"].fillna(1).astype(int) == 1)
        & proposed["risk_score"].notna()
        & proposed["advantage_score"].notna()
    ]
    if proposed.empty:
        return {}
    board_sizes = (
        proposed.groupby(["cohort_slug", "match_id", "phase_slot"])
        .agg(rows=("team_id", "nunique"), eliminations=("hazard_true", "sum"))
        .reset_index()
    )
    candidates = board_sizes[(board_sizes["rows"] >= top_k) & (board_sizes["eliminations"] > 0)]
    if candidates.empty:
        candidates = board_sizes
    chosen = candidates.sort_values(["cohort_slug", "phase_slot", "rows"], ascending=[True, True, False]).iloc[0]
    subset = proposed[
        (proposed["cohort_slug"] == chosen["cohort_slug"])
        & (proposed["match_id"] == chosen["match_id"])
        & (proposed["phase_slot"] == chosen["phase_slot"])
    ].copy()
    risk = subset.sort_values("risk_score", ascending=False).head(top_k)
    adv = subset.sort_values("advantage_score", ascending=False).head(top_k)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    axes[0].barh(risk["team_id"].astype(str), risk["risk_score"].astype(float), color="#b04747")
    axes[0].invert_yaxis()
    axes[0].set_title("Top risk board")
    axes[0].set_xlabel("risk score")
    axes[0].set_ylabel("team")
    axes[1].barh(adv["team_id"].astype(str), adv["advantage_score"].astype(float), color="#3d6f88")
    axes[1].invert_yaxis()
    axes[1].set_title("Top advantage board")
    axes[1].set_xlabel("survival share")
    axes[1].set_ylabel("team")
    fig.suptitle(
        f"{chosen['cohort_slug']} | phase {int(chosen['phase_slot'])} | {str(chosen['match_id'])[:8]}"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return {
        "cohort_slug": str(chosen["cohort_slug"]),
        "match_id": str(chosen["match_id"]),
        "phase_slot": int(chosen["phase_slot"]),
        "path": str(out_path),
    }


def _claim_boundary_note(aggregate: pd.DataFrame, sign_count: pd.DataFrame) -> str:
    simple = {"random_alive_board", "health_alive_count_board", "zone_distance_board", "sagong_prior_score_hgb_board"}
    risk_wins = sign_count[
        sign_count["baseline_variant"].isin(simple) & sign_count["metric"].eq("risk_top1_hit")
    ]
    simple_wins = int(risk_wins["lstm_wins"].sum()) if not risk_wins.empty else 0
    simple_losses = int(risk_wins["lstm_losses"].sum()) if not risk_wins.empty else 0
    pressure = aggregate[aggregate["model_variant"] == "pressure_habitat_hgb_board"]
    lstm = aggregate[aggregate["model_variant"] == PROPOSED_VARIANT]
    pressure_stronger = False
    if not pressure.empty and not lstm.empty:
        pressure_stronger = bool(
            float(pressure.iloc[0]["median_risk_top1_hit"]) > float(lstm.iloc[0]["median_risk_top1_hit"])
        )
    if pressure_stronger:
        return (
            "Pressure-habitat HGB is stronger on median risk top-1, so the manuscript should frame "
            "the LSTM board as the realtime readout form rather than the globally strongest board scorer."
        )
    if simple_wins > simple_losses:
        return (
            "The proposed board beats the simple/prior-style baselines on risk top-1 in more cohort comparisons "
            "than it loses; keep the board claim, with the HGB comparison reported explicitly."
        )
    return (
        "The comparison is mixed against simple/prior-style boards; keep RQ4 as a board-construction result "
        "and avoid a broad superiority claim."
    )


def write_report(
    path: Path,
    *,
    board: pd.DataFrame,
    by_cohort: pd.DataFrame,
    aggregate: pd.DataFrame,
    sign_count: pd.DataFrame,
    figure_info: dict[str, Any],
    manifest: dict[str, Any],
) -> None:
    primary_cols = [
        "model_variant",
        "board_label",
        "cohorts",
        "phase_boards",
        "median_risk_top1_hit",
        "median_risk_top3_hit",
        "median_risk_mrr",
        "median_adv_top_final_top5",
        "median_adv_top_final_top10",
        "median_adv_ndcg_top5_top10",
        "median_risk_delta_top1_hit",
        "median_adv_delta_top_final_top10",
    ]
    sign_cols = [
        "baseline_label",
        "metric",
        "cohorts_compared",
        "lstm_wins",
        "ties",
        "lstm_losses",
        "median_lstm_minus_baseline",
    ]
    cohort_cols = [
        "cohort_slug",
        "model_variant",
        "phase_boards",
        "risk_top1_hit",
        "risk_top3_hit",
        "risk_mrr",
        "adv_top_final_top10",
        f"adv_ndcg_top{TOP_K}_top10",
        "risk_delta_top1_hit",
        "adv_delta_top_final_top10",
    ]
    lines = [
        "# RQ4 Board-Baseline Comparison Result",
        "",
        "Date: 2026-07-06",
        "Class: `ExperimentResult`",
        "Lifecycle: `analysis_complete_bounded`",
        "",
        "## Purpose",
        "",
        "This report compares the phase-transition risk/advantage board against the frozen",
        "random, health/alive-count, zone-distance, Sagong-style, official-raw, and",
        "pressure-habitat baselines.",
        "",
        "## Aggregate",
        "",
        to_markdown_table(aggregate[primary_cols]),
        "",
        "## Sign Count",
        "",
        "Positive values mean the LSTM realtime board is better than the named baseline.",
        "",
        to_markdown_table(sign_count[sign_cols]),
        "",
        "## Cohort Summary",
        "",
        to_markdown_table(by_cohort[cohort_cols]),
        "",
        "## Example Board Figure",
        "",
        f"- Path: `{figure_info.get('path', '')}`",
        f"- Cohort: `{figure_info.get('cohort_slug', '')}`",
        f"- Match: `{figure_info.get('match_id', '')}`",
        f"- Phase: `{figure_info.get('phase_slot', '')}`",
        "",
        "## Claim Boundary",
        "",
        _claim_boundary_note(aggregate, sign_count),
        "",
        "Do not claim that this validates a separate 10-second streaming deployment.",
        "The comparison uses phase-cut board rows and saved phase-hazard predictions.",
        "",
        "## Manifest",
        "",
        "```json",
        json.dumps(manifest, indent=2, default=str),
        "```",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    lstm_scores = load_lstm_score_rows(args.lstm_trajectories)
    prior_pred = load_prior_predictions(args.prior_root, PRIOR_MODEL_MAP)
    match_to_cohort = build_match_cohort_map(lstm_scores, prior_pred)
    match_ids = sorted(set(match_to_cohort))
    features = prepare_feature_rows(
        feature_csv=args.feature_csv,
        match_to_cohort=match_to_cohort,
        match_ids=match_ids,
        refresh=args.refresh_features,
    )

    prior_pred = attach_feature_labels(prior_pred, features)
    prior_scores = model_predictions_to_score_rows(prior_pred)
    heuristic_scores = heuristic_score_rows(features, random_seed=args.random_seed)
    score_rows_raw = pd.concat([heuristic_scores, prior_scores, lstm_scores], ignore_index=True)
    score_rows, common_universe = filter_common_score_universe(score_rows_raw)

    board = evaluate_board_scores(score_rows, top_k=args.top_k)
    by_cohort, aggregate = summarize_board_scores(board, top_k=args.top_k)
    sign_count = build_sign_count(by_cohort)
    figure_info = write_example_figure(
        score_rows,
        out_dir / "rq4_board_baseline_example_board.png",
        top_k=args.top_k,
    )

    score_path = out_dir / "rq4_board_baseline_score_rows.csv"
    board_path = out_dir / "rq4_board_baseline_board_rows.csv"
    by_cohort_path = out_dir / "rq4_board_baseline_by_cohort.csv"
    aggregate_path = out_dir / "rq4_board_baseline_aggregate.csv"
    sign_path = out_dir / "rq4_board_baseline_sign_count.csv"
    manifest_path = out_dir / "rq4_board_baseline_manifest.json"
    report_path = out_dir / "RQ4_BOARD_BASELINE_COMPARISON_RESULT_2026-07-06.md"

    score_rows.to_csv(score_path, index=False)
    board.to_csv(board_path, index=False)
    by_cohort.to_csv(by_cohort_path, index=False)
    aggregate.to_csv(aggregate_path, index=False)
    sign_count.to_csv(sign_path, index=False)

    manifest = {
        "lstm_trajectories": str(args.lstm_trajectories.resolve()),
        "prior_root": str(args.prior_root.resolve()),
        "feature_csv": str(args.feature_csv.resolve()),
        "top_k": int(args.top_k),
        "random_seed": int(args.random_seed),
        "variants": VARIANT_LABELS,
        "outputs": {
            "score_rows": str(score_path),
            "board_rows": str(board_path),
            "by_cohort": str(by_cohort_path),
            "aggregate": str(aggregate_path),
            "sign_count": str(sign_path),
            "example_figure": figure_info.get("path"),
            "report": str(report_path),
        },
        "coverage": {
            "score_rows_raw": int(len(score_rows_raw)),
            "score_rows": int(len(score_rows)),
            "board_rows": int(len(board)),
            "cohorts": int(board["cohort_slug"].nunique()) if not board.empty else 0,
            "matches": int(board["match_id"].nunique()) if not board.empty else 0,
            "common_universe": common_universe,
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    write_report(
        report_path,
        board=board,
        by_cohort=by_cohort,
        aggregate=aggregate,
        sign_count=sign_count,
        figure_info=figure_info,
        manifest=manifest,
    )
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lstm-trajectories", type=Path, default=DEFAULT_LSTM_TRAJECTORIES)
    parser.add_argument("--prior-root", type=Path, default=DEFAULT_PRIOR_ROOT)
    parser.add_argument("--feature-csv", type=Path, default=DEFAULT_FEATURE_CACHE)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--top-k", type=int, default=TOP_K)
    parser.add_argument("--random-seed", type=int, default=20260706)
    parser.add_argument("--refresh-features", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), indent=2, default=str))
