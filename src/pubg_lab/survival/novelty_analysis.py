"""Novelty diagnostics for the PUBG STDB model.

Outputs:
* phase-wise uplift tables for STDB features and dynamic graph edges
* group-level permutation importance for the full tabular node-feature model
* selected feature-level permutation importance

The purpose is to turn the modeling results into paper-ready evidence:
where the proposed representation helps, and which feature families carry the
signal.
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
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from pubg_lab.survival.feature_ablation import (
    COMBAT_COLS,
    EXPOSURE_COLS,
    GLOBAL_COLS,
    LOOT_COLS,
    PRIOR_SCORE_COLS,
    STATE_COLS,
    TRAJECTORY_COLS,
    _available,
    _binary_metrics,
    _classifier,
    _rows_to_df,
    add_prior_features,
    filter_to_aligned_keys,
    fit_prior_transform,
    make_feature_configs,
)
from pubg_lab.survival.baselines import load_phase_rows
from model_stgcn.splits import load_or_create_split
from pubg_lab.data.stdb.queries import list_phase_eligible_match_ids

logger = logging.getLogger("model_stgcn.novelty_analysis")


SELECTED_FEATURES = [
    "safe_alignment",
    "outside_safe_ratio",
    "path_near_enemy_ratio",
    "min_nearest_enemy_m",
    "avg_enemy_density_100m",
    "path_length_m",
    "rotation_timing",
    "sinuosity",
    "boundary_distance_slope",
    "alive_players_end",
    "damage_taken",
    "damage_dealt",
    "pickup_count",
    "vehicle_ratio",
    "team_spread_avg_m",
    "combat_partner_count",
]


def _metric_pivot(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df


def compute_phase_deltas(
    *,
    feature_phase_path: str | Path,
    graph_phase_path: str | Path,
) -> pd.DataFrame:
    feature_phase = _metric_pivot(feature_phase_path)
    graph_phase = _metric_pivot(graph_phase_path)
    rows: list[dict[str, Any]] = []

    def add_delta(df: pd.DataFrame, left: str, right: str, label: str) -> None:
        left_df = df[df["model"] == left].set_index("phase_idx")
        right_df = df[df["model"] == right].set_index("phase_idx")
        common = sorted(set(left_df.index) & set(right_df.index))
        for phase_idx in common:
            l = left_df.loc[phase_idx]
            r = right_df.loc[phase_idx]
            rows.append(
                {
                    "comparison": label,
                    "left_model": left,
                    "right_model": right,
                    "phase_idx": int(phase_idx),
                    "n": int(l["n"]),
                    "pos_rate": float(l["pos_rate"]),
                    "delta_auc": float(l["auc"] - r["auc"]),
                    "delta_ap": float(l["ap"] - r["ap"]),
                    "bce_improvement": float(r["bce"] - l["bce"]),
                    "acc_delta": float(l["acc"] - r["acc"]),
                    "left_auc": float(l["auc"]),
                    "right_auc": float(r["auc"]),
                    "left_ap": float(l["ap"]),
                    "right_ap": float(r["ap"]),
                    "left_bce": float(l["bce"]),
                    "right_bce": float(r["bce"]),
                }
            )

    add_delta(
        feature_phase,
        "conventional_plus_stdb_spatial",
        "conventional",
        "STDB spatial features over conventional telemetry",
    )
    add_delta(
        feature_phase,
        "stdb_spatial",
        "prior_scores",
        "STDB spatial-only over prior-score baseline",
    )
    add_delta(
        feature_phase,
        "full_node_features",
        "prior_raw",
        "Full node features over prior raw baseline",
    )
    add_delta(
        graph_phase,
        "dynamic_full_e12",
        "dynamic_none_e12",
        "Dynamic heterogeneous graph over no-graph neural model",
    )
    return pd.DataFrame(rows)


def _mean_metrics_for_prediction(model, X: pd.DataFrame, y: np.ndarray) -> dict[str, float | int]:
    prob = model.predict_proba(X)[:, 1]
    return _binary_metrics(y, prob)


def _permute_group(df: pd.DataFrame, cols: list[str], rng: np.random.Generator) -> pd.DataFrame:
    if not cols:
        return df
    out = df.copy()
    perm = rng.permutation(len(out))
    for col in cols:
        out[col] = out[col].to_numpy()[perm]
    return out


def permutation_importance(
    *,
    model,
    test_df: pd.DataFrame,
    feature_cols: list[str],
    groups: dict[str, list[str]],
    repeats: int,
    seed: int,
) -> tuple[pd.DataFrame, dict[str, float | int]]:
    rng = np.random.default_rng(seed)
    X = test_df[feature_cols]
    y = test_df["hazard_next"].astype(int).to_numpy()
    base = _mean_metrics_for_prediction(model, X, y)
    rows: list[dict[str, Any]] = []

    for name, raw_cols in groups.items():
        cols = _available(raw_cols, X)
        if not cols:
            continue
        deltas = []
        for _ in range(repeats):
            permuted = _permute_group(X, cols, rng)
            metrics = _mean_metrics_for_prediction(model, permuted, y)
            deltas.append(
                {
                    "auc_drop": float(base["auc"] - metrics["auc"]),
                    "ap_drop": float(base["ap"] - metrics["ap"]),
                    "bce_increase": float(metrics["bce"] - base["bce"]),
                    "acc_drop": float(base["acc"] - metrics["acc"]),
                }
            )
        delta_df = pd.DataFrame(deltas)
        row = {
            "group": name,
            "n_features": len(cols),
            "features": ",".join(cols),
            "base_auc": base["auc"],
            "base_ap": base["ap"],
            "base_bce": base["bce"],
            "base_acc": base["acc"],
        }
        for metric in ["auc_drop", "ap_drop", "bce_increase", "acc_drop"]:
            row[f"{metric}_mean"] = float(delta_df[metric].mean())
            row[f"{metric}_std"] = float(delta_df[metric].std(ddof=0))
        rows.append(row)

    return pd.DataFrame(rows).sort_values("auc_drop_mean", ascending=False), base


def _plot_outputs(
    out_dir: Path, phase_delta: pd.DataFrame, group_imp: pd.DataFrame, feature_imp: pd.DataFrame
) -> list[str]:
    outputs: list[str] = []
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - optional artifact
        logger.warning("skip plots: %s", exc)
        return outputs

    def savefig(name: str) -> None:
        path = out_dir / name
        plt.tight_layout()
        plt.savefig(path, dpi=180)
        plt.close()
        outputs.append(str(path))

    plt.figure(figsize=(9, 5))
    for comparison in [
        "STDB spatial features over conventional telemetry",
        "Dynamic heterogeneous graph over no-graph neural model",
        "Full node features over prior raw baseline",
    ]:
        sub = phase_delta[phase_delta["comparison"] == comparison].sort_values("phase_idx")
        plt.plot(sub["phase_idx"], sub["delta_auc"], marker="o", label=comparison)
    plt.axhline(0, color="#444444", linewidth=0.8)
    plt.xlabel("Phase")
    plt.ylabel("AUC uplift")
    plt.title("Phase-wise novelty uplift")
    plt.legend(fontsize=8)
    savefig("phase_auc_uplift.png")

    top_group = group_imp.sort_values("auc_drop_mean", ascending=True)
    plt.figure(figsize=(8, 4.8))
    plt.barh(top_group["group"], top_group["auc_drop_mean"], color="#2f5f8f")
    plt.xlabel("AUC drop after group permutation")
    plt.title("Feature group importance")
    savefig("group_importance_auc.png")

    top_feature = feature_imp.sort_values("auc_drop_mean", ascending=True).tail(12)
    plt.figure(figsize=(8, 5.2))
    plt.barh(top_feature["group"], top_feature["auc_drop_mean"], color="#6f8f2f")
    plt.xlabel("AUC drop after feature permutation")
    plt.title("Selected feature importance")
    savefig("selected_feature_importance_auc.png")

    return outputs


def run(args: argparse.Namespace) -> dict[str, Any]:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    phase_delta = compute_phase_deltas(
        feature_phase_path=args.feature_phase_metrics,
        graph_phase_path=args.graph_phase_metrics,
    )

    ids = list_phase_eligible_match_ids(args.map, args.mode, args.limit)
    train_ids, _val_ids, test_ids = load_or_create_split(
        ids,
        split_json=args.split_json,
        seed=args.seed,
        val_frac=args.val_frac,
        test_frac=args.test_frac,
        metadata={"source": "model_stgcn.novelty_analysis", "limit": args.limit},
    )
    logger.info("split total=%d train=%d test=%d", len(ids), len(train_ids), len(test_ids))

    train_rows, _ = load_phase_rows(train_ids)
    test_rows, _ = load_phase_rows(test_ids)
    train_df = _rows_to_df(train_rows)
    test_df = _rows_to_df(test_rows)
    prior_transform = fit_prior_transform(train_df)
    train_df = add_prior_features(train_df, prior_transform)
    test_df = add_prior_features(test_df, prior_transform)
    test_df, alignment = filter_to_aligned_keys(test_df, args.align_to)
    logger.info("rows train=%d test=%d alignment=%s", len(train_df), len(test_df), alignment)

    feature_cols = make_feature_configs(train_df)["full_node_features"]
    model = _classifier(feature_cols)
    logger.info("fit full node feature model for permutation importance")
    model.fit(train_df[feature_cols], train_df["hazard_next"].astype(int).to_numpy())

    groups = {
        "prior_scores": PRIOR_SCORE_COLS,
        "global_context": GLOBAL_COLS,
        "team_state": STATE_COLS,
        "trajectory": TRAJECTORY_COLS,
        "exposure": EXPOSURE_COLS,
        "combat": COMBAT_COLS,
        "loot": LOOT_COLS,
    }
    group_imp, base_metrics = permutation_importance(
        model=model,
        test_df=test_df,
        feature_cols=feature_cols,
        groups=groups,
        repeats=args.repeats,
        seed=args.seed,
    )
    feature_groups = {
        feature: [feature] for feature in SELECTED_FEATURES if feature in feature_cols
    }
    feature_imp, _ = permutation_importance(
        model=model,
        test_df=test_df,
        feature_cols=feature_cols,
        groups=feature_groups,
        repeats=args.repeats,
        seed=args.seed + 17,
    )

    phase_path = out_dir / "phase_novelty_deltas.csv"
    group_path = out_dir / "group_permutation_importance.csv"
    feature_path = out_dir / "selected_feature_permutation_importance.csv"
    summary_path = out_dir / "novelty_analysis_summary.json"
    phase_delta.to_csv(phase_path, index=False)
    group_imp.to_csv(group_path, index=False)
    feature_imp.to_csv(feature_path, index=False)
    plot_paths = _plot_outputs(out_dir, phase_delta, group_imp, feature_imp)

    summary = {
        "split": {
            "total": len(ids),
            "train": len(train_ids),
            "test": len(test_ids),
        },
        "alignment": alignment,
        "base_full_node_metrics": base_metrics,
        "outputs": {
            "phase_novelty_deltas": str(phase_path),
            "group_permutation_importance": str(group_path),
            "selected_feature_permutation_importance": str(feature_path),
            "plots": plot_paths,
        },
        "top_group_importance_by_auc_drop": group_imp.head(10).to_dict(orient="records"),
        "top_selected_feature_importance_by_auc_drop": feature_imp.head(12).to_dict(
            orient="records"
        ),
        "phase_deltas": phase_delta.to_dict(orient="records"),
    }
    summary_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    logger.info("wrote %s", summary_path)
    logger.info(
        "group importance=%s",
        group_imp[["group", "auc_drop_mean", "ap_drop_mean", "bce_increase_mean"]].to_dict(
            orient="records"
        ),
    )
    logger.info(
        "selected feature importance=%s",
        feature_imp[["group", "auc_drop_mean", "ap_drop_mean", "bce_increase_mean"]]
        .head(12)
        .to_dict(orient="records"),
    )
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
        "--align-to",
        default="model_stgcn/checkpoints/phase_result_analysis/hazard_predictions_test_aligned.csv",
    )
    ap.add_argument(
        "--feature-phase-metrics",
        default="model_stgcn/checkpoints/feature_ablation/feature_ablation_phase_metrics.csv",
    )
    ap.add_argument(
        "--graph-phase-metrics",
        default="model_stgcn/checkpoints/phase_result_analysis/hazard_phase_metrics.csv",
    )
    ap.add_argument("--out-dir", default="model_stgcn/checkpoints/novelty_analysis")
    ap.add_argument("--repeats", type=int, default=5)
    return ap.parse_args()


if __name__ == "__main__":
    run(parse_args())
