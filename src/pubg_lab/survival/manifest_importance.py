"""Manifest-V2 phase-band and permutation-importance diagnostics.

This is the paper-facing companion to `feature_ablation --config-style manifest`.
It keeps the same split/alignment, fits the selected manifest config
(default: A11 full representation), and reports:

* early/mid/late metrics from the manifest feature-ablation predictions
* family-level permutation importance for F0/F5/F6/F7
* individual feature permutation importance for the selected model
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

from pubg_lab.survival.baselines import CATEGORICAL_COLUMNS
from pubg_lab.survival.feature_ablation import (
    V2_MANIFEST_PATH,
    _available,
    _binary_metrics,
    _classifier,
    _load_manifest,
    _rows_to_df,
    filter_to_aligned_keys,
    load_phase_rows_from_view,
    make_manifest_feature_configs,
)
from model_stgcn.splits import load_or_create_split
from pubg_lab.data.stdb.queries import list_phase_feature_match_ids

logger = logging.getLogger("model_stgcn.manifest_importance")

DEFAULT_SPLIT = "model_stgcn/checkpoints/full_db_reexperiment/split_phase_all_current_seed42.json"
DEFAULT_ALIGN = (
    "model_stgcn/checkpoints/full_db_reexperiment/v2_node_features/"
    "phase_result_analysis/hazard_predictions_test_aligned.csv"
)
DEFAULT_PREDICTIONS = (
    "model_stgcn/checkpoints/feature_ablation_full_db_manifest_v2/feature_ablation_predictions.csv"
)


def _dedupe(cols: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for col in cols:
        if col not in seen:
            out.append(col)
            seen.add(col)
    return out


def _phase_band(phase_idx: int) -> str:
    if phase_idx <= 2:
        return "early_1_2"
    if phase_idx <= 5:
        return "mid_3_5"
    return "late_6_7"


def _write_phase_band_metrics(prediction_path: Path, out_path: Path) -> pd.DataFrame:
    preds = pd.read_csv(prediction_path)
    preds["phase_band"] = preds["phase_idx"].astype(int).map(_phase_band)
    rows: list[dict[str, Any]] = []
    for (model, band), g in preds.groupby(["model", "phase_band"], dropna=False):
        row = {"model": model, "phase_band": band}
        row.update(_binary_metrics(g["y"].to_numpy(), g["prob"].to_numpy()))
        row["n_matches"] = int(g["match_id"].nunique())
        rows.append(row)
    out = pd.DataFrame(rows).sort_values(["model", "phase_band"])
    out.to_csv(out_path, index=False)
    return out


def _permute_group(df: pd.DataFrame, cols: list[str], rng: np.random.Generator) -> pd.DataFrame:
    out = df.copy()
    perm = rng.permutation(len(out))
    for col in cols:
        out[col] = out[col].to_numpy()[perm]
    return out


def _predict_metrics(model, X: pd.DataFrame, y: np.ndarray) -> dict[str, float | int]:
    prob = model.predict_proba(X)[:, 1]
    return _binary_metrics(y, prob)


def permutation_importance(
    *,
    model,
    X: pd.DataFrame,
    y: np.ndarray,
    groups: dict[str, list[str]],
    repeats: int,
    seed: int,
) -> tuple[pd.DataFrame, dict[str, float | int]]:
    rng = np.random.default_rng(seed)
    base = _predict_metrics(model, X, y)
    rows: list[dict[str, Any]] = []
    for name, raw_cols in groups.items():
        cols = _available(raw_cols, X)
        if not cols:
            continue
        repeat_rows: list[dict[str, float]] = []
        for _ in range(repeats):
            metrics = _predict_metrics(model, _permute_group(X, cols, rng), y)
            repeat_rows.append(
                {
                    "auc_drop": float(base["auc"] - metrics["auc"]),
                    "ap_drop": float(base["ap"] - metrics["ap"]),
                    "bce_increase": float(metrics["bce"] - base["bce"]),
                    "acc_drop": float(base["acc"] - metrics["acc"]),
                }
            )
        delta = pd.DataFrame(repeat_rows)
        row: dict[str, Any] = {
            "group": name,
            "n_features": len(cols),
            "features": ",".join(cols),
            "base_auc": base["auc"],
            "base_ap": base["ap"],
            "base_bce": base["bce"],
            "base_acc": base["acc"],
        }
        for metric in ("auc_drop", "ap_drop", "bce_increase", "acc_drop"):
            row[f"{metric}_mean"] = float(delta[metric].mean())
            row[f"{metric}_std"] = float(delta[metric].std(ddof=0))
        rows.append(row)
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values("auc_drop_mean", ascending=False)
    return out, base


def _family_groups(manifest: dict[str, Any], config_cols: list[str]) -> dict[str, list[str]]:
    families = manifest["families"]
    out: dict[str, list[str]] = {}
    for family in (
        "f0_official_raw_phase",
        "f5_stdb_spatial",
        "f6_network_phase_role_tier1",
        "f7_habitat_pressure_tier1",
    ):
        cols = [c for c in families[family].get("columns", []) if c in config_cols]
        if cols:
            out[family] = cols
    return out


def run(args: argparse.Namespace) -> dict[str, Any]:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = _load_manifest(args.manifest)
    ids = list_phase_feature_match_ids(args.map, args.mode, args.limit)
    train_ids, _val_ids, test_ids = load_or_create_split(
        ids,
        split_json=args.split_json,
        seed=args.seed,
        val_frac=args.val_frac,
        test_frac=args.test_frac,
        metadata={"source": "model_stgcn.manifest_importance", "limit": args.limit},
    )
    logger.info("split total=%d train=%d test=%d", len(ids), len(train_ids), len(test_ids))

    train_rows, _ = load_phase_rows_from_view(train_ids, args.feature_view)
    test_rows, _ = load_phase_rows_from_view(test_ids, args.feature_view)
    train_df = _rows_to_df(train_rows)
    test_df = _rows_to_df(test_rows)
    test_df, alignment = filter_to_aligned_keys(test_df, args.align_to)
    logger.info("rows train=%d test=%d alignment=%s", len(train_df), len(test_df), alignment)

    configs = make_manifest_feature_configs(train_df, manifest_path=args.manifest)
    if args.config not in configs:
        raise SystemExit(f"config not available: {args.config}")
    feature_cols = configs[args.config]
    categorical_cols = _dedupe(list(manifest.get("categorical_columns", [])) + CATEGORICAL_COLUMNS)

    model = _classifier(feature_cols, categorical_cols)
    logger.info("fit %s rows=%d features=%d", args.config, len(train_df), len(feature_cols))
    model.fit(train_df[feature_cols], train_df["hazard_next"].astype(int).to_numpy())
    X_test = test_df[feature_cols]
    y_test = test_df["hazard_next"].astype(int).to_numpy()

    family_imp, base_metrics = permutation_importance(
        model=model,
        X=X_test,
        y=y_test,
        groups=_family_groups(manifest, feature_cols),
        repeats=args.repeats,
        seed=args.seed,
    )
    feature_imp, _ = permutation_importance(
        model=model,
        X=X_test,
        y=y_test,
        groups={col: [col] for col in feature_cols},
        repeats=args.feature_repeats,
        seed=args.seed + 17,
    )

    band_path = out_dir / "manifest_phase_band_metrics.csv"
    family_path = out_dir / "a11_family_permutation_importance.csv"
    feature_path = out_dir / "a11_feature_permutation_importance.csv"
    summary_path = out_dir / "manifest_importance_summary.json"

    phase_band = _write_phase_band_metrics(Path(args.predictions), band_path)
    family_imp.to_csv(family_path, index=False)
    feature_imp.to_csv(feature_path, index=False)

    summary = {
        "config": args.config,
        "split": {"total": len(ids), "train": len(train_ids), "test": len(test_ids)},
        "alignment": alignment,
        "base_metrics": base_metrics,
        "outputs": {
            "phase_band_metrics": str(band_path),
            "family_permutation_importance": str(family_path),
            "feature_permutation_importance": str(feature_path),
        },
        "top_family_importance": family_imp.head(10).to_dict(orient="records"),
        "top_feature_importance": feature_imp.head(args.top_features).to_dict(orient="records"),
        "phase_band_focus": phase_band[
            phase_band["model"].isin(
                ["a0_official_raw", "a7_raw_plus_habitat", "a11_full_representation"]
            )
        ].to_dict(orient="records"),
    }
    summary_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    logger.info("wrote %s", summary_path)
    logger.info(
        "family importance=%s",
        family_imp[["group", "auc_drop_mean", "ap_drop_mean", "bce_increase_mean"]].to_dict(
            orient="records"
        ),
    )
    logger.info(
        "top features=%s",
        feature_imp[["group", "auc_drop_mean", "ap_drop_mean", "bce_increase_mean"]]
        .head(args.top_features)
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
    ap.add_argument("--split-json", default=DEFAULT_SPLIT)
    ap.add_argument("--align-to", default=DEFAULT_ALIGN)
    ap.add_argument("--feature-view", default="stdb.v_team_phase_feature_v2")
    ap.add_argument("--manifest", type=Path, default=V2_MANIFEST_PATH)
    ap.add_argument("--predictions", default=DEFAULT_PREDICTIONS)
    ap.add_argument("--config", default="a11_full_representation")
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--feature-repeats", type=int, default=1)
    ap.add_argument("--top-features", type=int, default=20)
    ap.add_argument("--out-dir", default="model_stgcn/checkpoints/manifest_importance_v2")
    return ap.parse_args()


if __name__ == "__main__":
    run(parse_args())
